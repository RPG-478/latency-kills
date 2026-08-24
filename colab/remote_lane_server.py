"""One Llama 3.1 8B motor lane for a paid Google Colab T4 runtime.

The server keeps the fixed V4 system prefix in a GPU KV cache and evaluates only
the tiny changing observation suffix.  It deliberately accepts one request at a
time: three notebooks provide three physical lanes without same-GPU contention.
"""

from __future__ import annotations

import os
import re
import secrets
import threading
import time
from typing import Any

import torch
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


MODEL_ID = os.environ.get("LATENCY_KILLS_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LANE_NAME = os.environ.get("LATENCY_KILLS_LANE_NAME", "colab-t4").strip()
BEARER_TOKEN = os.environ.get("LATENCY_KILLS_LANE_TOKEN", "").strip()
CONSTRAIN_DIGITS = os.environ.get(
    "LATENCY_KILLS_CONSTRAIN_DIGITS", "0"
).strip().lower() in {"1", "true", "yes"}
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()

if len(BEARER_TOKEN) < 24:
    raise RuntimeError("LATENCY_KILLS_LANE_TOKEN must be an ephemeral 24+ char token")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is required to download the gated official Llama weights")
if not torch.cuda.is_available():
    raise RuntimeError("remote_lane_server requires a CUDA GPU")


V4_SYSTEM = (
    "Reply with exactly one ASCII digit and nothing else. Apply the first true "
    "row only: v=0=>4; v=1 and a<=0=>0; v=1 and x<-220=>2; "
    "v=1 and -220<=x<-80=>1; v=1 and -80<=x<=80=>5; "
    "v=1 and 80<x<=220=>3; v=1 and x>220=>4. "
    "Important: every v=0 input is 4, never 0. Examples: "
    "v=0 x=9999 a=10=>4; v=1 x=0 a=0=>0; v=1 x=-350 a=10=>2; "
    "v=1 x=0 a=10=>5; v=1 x=350 a=10=>4. /no_think"
)


class MotorRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=160)
    observation: str = Field(min_length=9, max_length=80)


class VagoTextRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=160)
    system_prompt: str = Field(min_length=20, max_length=8_000)
    user_content: str = Field(min_length=100, max_length=12_000)
    max_new_tokens: int = Field(default=200, ge=1, le=200)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


def _authorize(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {BEARER_TOKEN}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def _observation(text: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"v=([01]) x=(-?\d+) a=(-?\d+)", text)
    if match is None:
        raise HTTPException(status_code=422, detail="invalid observation grammar")
    visible, x, ammo = (int(value) for value in match.groups())
    if not -10_000 <= x <= 10_000 or not -999 <= ammo <= 999:
        raise HTTPException(status_code=422, detail="observation value out of range")
    return visible, x, ammo


_quantization = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
_load_started = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    quantization_config=_quantization,
    device_map="auto",
    dtype=torch.float16,
    attn_implementation="sdpa",
)
model.eval()
LOAD_SECONDS = time.perf_counter() - _load_started
os.environ.pop("HF_TOKEN", None)
HF_TOKEN = ""


def _chat_ids(observation: str) -> torch.Tensor:
    batch = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": V4_SYSTEM},
            {"role": "user", "content": observation},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return batch["input_ids"].to(model.device)


_prefix_batch = tokenizer.apply_chat_template(
    [{"role": "system", "content": V4_SYSTEM}],
    tokenize=True,
    add_generation_prompt=False,
    return_tensors="pt",
    return_dict=True,
)
_prefix_ids = _prefix_batch["input_ids"].to(model.device)
_prefix_len = int(_prefix_ids.shape[-1])
_full_check = _chat_ids("v=1 x=0 a=10")
if not torch.equal(_full_check[:, :_prefix_len], _prefix_ids):
    raise RuntimeError("chat template system prefix is not cacheable")

_digit_ids = [tokenizer.encode(str(value), add_special_tokens=False)[0] for value in range(6)]
if any(len(tokenizer.encode(str(value), add_special_tokens=False)) != 1 for value in range(6)):
    raise RuntimeError("motor digits must each be one tokenizer token")

with torch.inference_mode():
    torch.cuda.synchronize()
    _cache = model(
        input_ids=_prefix_ids,
        attention_mask=torch.ones_like(_prefix_ids),
        use_cache=True,
    ).past_key_values
    torch.cuda.synchronize()
if not hasattr(_cache, "crop"):
    raise RuntimeError("this Transformers cache cannot be restored after a suffix")

_inference_lock = threading.Lock()
_vago_cached_system_prompt: str | None = None
_vago_cached_prefix_ids: torch.Tensor | None = None
_vago_cached_prefix: Any | None = None


def _infer(observation: str) -> tuple[str, float, int]:
    started = time.perf_counter()
    full_ids = _chat_ids(observation)
    suffix = full_ids[:, _prefix_len:]
    suffix_len = int(suffix.shape[-1])
    attention_mask = torch.ones(
        (1, _prefix_len + suffix_len),
        device=model.device,
        dtype=torch.long,
    )
    try:
        with torch.inference_mode():
            torch.cuda.synchronize()
            output = model(
                input_ids=suffix,
                attention_mask=attention_mask,
                past_key_values=_cache,
                use_cache=True,
            )
            next_logits = output.logits[0, -1]
            if CONSTRAIN_DIGITS:
                digit_logits = next_logits[_digit_ids]
                chosen_id = _digit_ids[int(digit_logits.argmax().item())]
            else:
                chosen_id = int(next_logits.argmax().item())
            torch.cuda.synchronize()
    finally:
        _cache.crop(_prefix_len)
    compute_ms = (time.perf_counter() - started) * 1000.0
    decoded = tokenizer.decode([chosen_id], skip_special_tokens=True).strip()
    if len(decoded) != 1 or decoded not in "012345":
        raise HTTPException(status_code=422, detail="model did not emit a motor digit")
    return decoded, compute_ms, suffix_len


def _vago_prefix(system_prompt: str) -> tuple[torch.Tensor, Any]:
    """Cache exactly one VAGO system prefix; requests are serialized per lane."""

    global _vago_cached_prefix, _vago_cached_prefix_ids, _vago_cached_system_prompt
    if (
        _vago_cached_system_prompt == system_prompt
        and _vago_cached_prefix_ids is not None
        and _vago_cached_prefix is not None
    ):
        return _vago_cached_prefix_ids, _vago_cached_prefix

    prefix_batch = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}],
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
    )
    prefix_ids = prefix_batch["input_ids"].to(model.device)
    with torch.inference_mode():
        prefix_cache = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
        ).past_key_values
    if not hasattr(prefix_cache, "crop"):
        raise RuntimeError("this Transformers cache cannot restore a VAGO prefix")
    _vago_cached_system_prompt = system_prompt
    _vago_cached_prefix_ids = prefix_ids
    _vago_cached_prefix = prefix_cache
    return prefix_ids, prefix_cache


_VAGO_ACTIONS = ("shoot", "move_forward", "turn_left", "turn_right")
_VAGO_BUTTONS = {
    "shoot": (1, 0, 0, 0),
    "move_forward": (0, 1, 0, 0),
    "turn_left": (0, 0, 1, 0),
    "turn_right": (0, 0, 0, 1),
}


def _parse_vago_action(text: str) -> tuple[str, tuple[int, int, int, int]]:
    action_text = text.strip().lower()
    lines = [line.strip() for line in action_text.split("\n") if line.strip()]
    if lines:
        action_text = lines[-1]
    buttons = [0, 0, 0, 0]
    parsed: list[str] = []
    for action in _VAGO_ACTIONS:
        if action not in action_text:
            continue
        buttons = [
            max(left, right)
            for left, right in zip(buttons, _VAGO_BUTTONS[action], strict=True)
        ]
        parsed.append(action)
    if parsed:
        return "+".join(parsed), tuple(buttons)
    return "move_forward", _VAGO_BUTTONS["move_forward"]


def _infer_vago_text(
    request: VagoTextRequest,
) -> tuple[str, tuple[int, int, int, int], str, float, int, int, int, int]:
    started = time.perf_counter()
    prefix_ids, vago_cache = _vago_prefix(request.system_prompt)
    prefix_tokens = int(prefix_ids.shape[-1])
    batch = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_content},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = batch["input_ids"].to(model.device)
    if not torch.equal(input_ids[:, :prefix_tokens], prefix_ids):
        raise HTTPException(
            status_code=422, detail="VAGO system prefix is not cacheable"
        )
    suffix_ids = input_ids[:, prefix_tokens:]
    suffix_tokens = int(suffix_ids.shape[-1])
    prompt_tokens = prefix_tokens + suffix_tokens
    attention_mask = torch.ones(
        (1, prompt_tokens), device=model.device, dtype=torch.long
    )
    with torch.inference_mode():
        torch.cuda.synchronize()
        completion_token_ids: list[int] = []
        try:
            output = model(
                input_ids=suffix_ids,
                attention_mask=attention_mask,
                past_key_values=vago_cache,
                use_cache=True,
            )
            for _ in range(request.max_new_tokens):
                logits = output.logits[:, -1, :]
                if request.temperature > 0:
                    probabilities = torch.softmax(logits / request.temperature, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                token_id = int(next_token.item())
                completion_token_ids.append(token_id)
                if token_id == tokenizer.eos_token_id:
                    break
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((1, 1), device=model.device, dtype=torch.long),
                    ),
                    dim=-1,
                )
                output = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=vago_cache,
                    use_cache=True,
                )
        finally:
            vago_cache.crop(prefix_tokens)
        torch.cuda.synchronize()
    completion_ids = torch.tensor(completion_token_ids, device=model.device)
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    compute_ms = (time.perf_counter() - started) * 1000.0
    action, buttons = _parse_vago_action(completion)
    return (
        action,
        buttons,
        completion[:1_000],
        compute_ms,
        prompt_tokens,
        int(completion_ids.shape[-1]),
        prefix_tokens,
        suffix_tokens,
    )


app = FastAPI(title="Latency Kills Remote T4 Lane", docs_url=None, redoc_url=None)


@app.get("/health", dependencies=[Depends(_authorize)])
def health() -> dict[str, Any]:
    return {
        "ready": True,
        "lane": LANE_NAME,
        "model": MODEL_ID,
        "gpu": torch.cuda.get_device_name(0),
        "quantization": "bitsandbytes NF4, float16 compute",
        "constrained_digits": CONSTRAIN_DIGITS,
        "prefix_tokens": _prefix_len,
        "load_seconds": round(LOAD_SECONDS, 3),
        "input_modes": ["v4-structured", "vago-cloud-text"],
    }


@app.post("/motor", dependencies=[Depends(_authorize)])
def motor(request: MotorRequest) -> dict[str, Any]:
    _observation(request.observation)
    queued_at = time.perf_counter()
    with _inference_lock:
        acquired_at = time.perf_counter()
        token, compute_ms, suffix_tokens = _infer(request.observation)
    return {
        "request_id": request.request_id,
        "token": token,
        "model": MODEL_ID,
        "lane": LANE_NAME,
        "queue_ms": (acquired_at - queued_at) * 1000.0,
        "compute_ms": compute_ms,
        "suffix_tokens": suffix_tokens,
        "constrained_digits": CONSTRAIN_DIGITS,
    }


@app.post("/vago-text", dependencies=[Depends(_authorize)])
def vago_text(request: VagoTextRequest) -> dict[str, Any]:
    queued_at = time.perf_counter()
    with _inference_lock:
        acquired_at = time.perf_counter()
        (
            action,
            buttons,
            completion,
            compute_ms,
            prompt_tokens,
            completion_tokens,
            prefix_tokens,
            suffix_tokens,
        ) = _infer_vago_text(request)
    return {
        "request_id": request.request_id,
        "action": action,
        "buttons": buttons,
        "completion": completion,
        "model": MODEL_ID,
        "lane": LANE_NAME,
        "queue_ms": (acquired_at - queued_at) * 1000.0,
        "compute_ms": compute_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "temperature": request.temperature,
        "max_new_tokens": request.max_new_tokens,
    }
