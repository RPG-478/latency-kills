from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .openrouter import StreamResult
from .vago_text import parse_vago_cloud_action


DEFAULT_LANE_ENV = "LATENCY_KILLS_REMOTE_LANES"


class RemoteLaneFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RemoteLaneConfig:
    name: str
    endpoint: str
    bearer_token: str


@dataclass(frozen=True, slots=True)
class VagoTextDecision:
    request_id: str
    action: str
    buttons: tuple[int, int, int, int]
    completion: str
    lane: str
    model: str | None
    arrived_at: float
    wire_ms: float
    server_compute_ms: float | None
    server_queue_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(slots=True)
class _LaneStats:
    requests: int = 0
    failures: int = 0
    wire_ms: list[float] = field(default_factory=list)
    server_compute_ms: list[float] = field(default_factory=list)
    server_queue_ms: list[float] = field(default_factory=list)


def load_remote_lane_configs(
    *,
    config_file: Path | None = None,
    env_name: str = DEFAULT_LANE_ENV,
) -> tuple[RemoteLaneConfig, ...]:
    """Load ephemeral endpoints without putting bearer tokens on the command line."""

    if config_file is not None:
        if not config_file.is_file():
            raise ValueError(f"remote lane config does not exist: {config_file}")
        raw = config_file.read_text(encoding="utf-8-sig")
        source = str(config_file)
    else:
        raw = os.environ.get(env_name, "")
        source = f"environment variable {env_name}"
    if not raw.strip():
        raise ValueError(f"remote lane configuration is absent from {source}")

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"remote lane configuration in {source} is not JSON") from error
    rows = decoded.get("lanes") if isinstance(decoded, dict) else decoded
    if not isinstance(rows, list) or not rows:
        raise ValueError("remote lane configuration needs a non-empty lanes list")
    if len(rows) > 16:
        raise ValueError("remote lane configuration supports at most sixteen lanes")

    configs: list[RemoteLaneConfig] = []
    seen_names: set[str] = set()
    seen_endpoints: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"remote lane {index} must be an object")
        name = str(row.get("name", f"lane-{index}")).strip()
        endpoint = str(row.get("endpoint", "")).strip().rstrip("/")
        inline_token = str(row.get("token", "")).strip()
        token_env = str(row.get("token_env", "")).strip()
        if inline_token and token_env:
            raise ValueError(f"remote lane {name} sets both token and token_env")
        bearer_token = inline_token or os.environ.get(token_env, "").strip()

        if not name or len(name) > 80:
            raise ValueError(f"remote lane {index} has an invalid name")
        if name in seen_names:
            raise ValueError(f"duplicate remote lane name: {name}")
        _validate_endpoint(endpoint, lane=name)
        if endpoint in seen_endpoints:
            raise ValueError(f"duplicate remote lane endpoint: {name}")
        if not bearer_token:
            token_source = token_env or "token"
            raise ValueError(f"remote lane {name} has no bearer token in {token_source}")
        if len(bearer_token) < 24:
            raise ValueError(f"remote lane {name} bearer token is unexpectedly short")

        seen_names.add(name)
        seen_endpoints.add(endpoint)
        configs.append(
            RemoteLaneConfig(
                name=name,
                endpoint=endpoint,
                bearer_token=bearer_token,
            )
        )
    return tuple(configs)


class RemoteLanePoolClient:
    """One persistent HTTP connection and one in-flight request per GPU lane."""

    def __init__(
        self,
        configs: tuple[RemoteLaneConfig, ...],
        *,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not configs:
            raise ValueError("at least one remote lane is required")
        # Probe/event code records ``client.model`` for every pilot.  Remote
        # lanes are a pool rather than one hosted model, but still need a
        # stable, non-secret label for the shared instrumentation path.
        self.model = "remote-lane-pool"
        self.configs = configs
        self._clients = tuple(
            httpx.AsyncClient(
                base_url=config.endpoint,
                http2=True,
                timeout=httpx.Timeout(
                    connect=min(10.0, timeout_seconds),
                    read=timeout_seconds,
                    write=min(10.0, timeout_seconds),
                    pool=min(10.0, timeout_seconds),
                ),
                headers={
                    "Authorization": f"Bearer {config.bearer_token}",
                    "Content-Type": "application/json",
                    "X-Title": "Latency Kills",
                },
                transport=transport,
            )
            for config in configs
        )
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for index in range(len(configs)):
            self._available.put_nowait(index)
        self._stats = [_LaneStats() for _ in configs]
        self._request_sequence = 0

    @property
    def lane_count(self) -> int:
        return len(self.configs)

    async def aclose(self) -> None:
        await asyncio.gather(*(client.aclose() for client in self._clients))

    async def warmup(self) -> list[dict[str, Any]]:
        started = time.monotonic()

        async def check(index: int) -> dict[str, Any]:
            config = self.configs[index]
            try:
                response = await self._clients[index].get("/health")
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError, OSError) as error:
                raise RemoteLaneFailure(
                    f"remote lane {config.name} health check failed: "
                    f"{type(error).__name__}"
                ) from error
            if not isinstance(payload, dict) or payload.get("ready") is not True:
                raise RemoteLaneFailure(f"remote lane {config.name} is not ready")
            return {
                "name": config.name,
                "model": _optional_text(payload.get("model")),
                "gpu": _optional_text(payload.get("gpu")),
                "quantization": _optional_text(payload.get("quantization")),
                "constrained_digits": bool(payload.get("constrained_digits")),
                "prefix_tokens": _nonnegative_int(payload.get("prefix_tokens")),
            }

        rows = await asyncio.gather(*(check(i) for i in range(self.lane_count)))
        elapsed_ms = (time.monotonic() - started) * 1000.0
        for row in rows:
            row["parallel_health_wall_ms"] = elapsed_ms
        return rows

    async def stream_motor(
        self,
        *,
        observation_text: str,
        run_id: str,
        observation_seq: int,
        on_visible: Callable[[str, float], None],
    ) -> StreamResult:
        lane_index = await self._available.get()
        config = self.configs[lane_index]
        client = self._clients[lane_index]
        stats = self._stats[lane_index]
        self._request_sequence += 1
        request_id = f"{run_id}-{observation_seq}-{self._request_sequence}"
        started = time.monotonic()
        try:
            response = await client.post(
                "/motor",
                json={
                    "request_id": request_id,
                    "observation": observation_text,
                },
            )
            arrived_at = time.monotonic()
            if response.status_code >= 400:
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as error:
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned invalid JSON"
                ) from error
            if not isinstance(payload, dict):
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned a non-object response"
                )
            token = _optional_text(payload.get("token"))
            if token is None or len(token) != 1 or token not in "012345":
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned an invalid motor token"
                )

            wire_ms = (arrived_at - started) * 1000.0
            compute_ms = _finite_nonnegative(payload.get("compute_ms"))
            queue_ms = _finite_nonnegative(payload.get("queue_ms"))
            stats.requests += 1
            stats.wire_ms.append(wire_ms)
            if compute_ms is not None:
                stats.server_compute_ms.append(compute_ms)
            if queue_ms is not None:
                stats.server_queue_ms.append(queue_ms)

            on_visible(token, arrived_at)
            usage: dict[str, Any] = {
                "remote_lane": config.name,
                "server_compute_ms": compute_ms,
                "server_queue_ms": queue_ms,
                "suffix_tokens": _nonnegative_int(payload.get("suffix_tokens")),
                "constrained_digits": bool(payload.get("constrained_digits")),
            }
            return StreamResult(
                response_id=_optional_text(payload.get("request_id")) or request_id,
                reported_model=_optional_text(payload.get("model")),
                provider=f"remote-colab/{config.name}",
                reasoning_types=(),
                raw_reasoning_chars=0,
                visible_chars=1,
                first_byte_ms=wire_ms,
                first_reasoning_ms=None,
                first_visible_ms=wire_ms,
                total_ms=(time.monotonic() - started) * 1000.0,
                usage=usage,
            )
        except (httpx.HTTPError, OSError) as error:
            stats.failures += 1
            raise RemoteLaneFailure(
                f"remote lane {config.name} request failed: {type(error).__name__}"
            ) from error
        finally:
            self._available.put_nowait(lane_index)

    async def decide_vago_text(
        self,
        *,
        system_prompt: str,
        user_content: str,
        run_id: str,
        observation_seq: int,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> VagoTextDecision:
        """Run VAGO's published Cloud-LLM text contract on one free GPU."""

        if not 1 <= max_new_tokens <= 200:
            raise ValueError("VAGO max_new_tokens must be between 1 and 200")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("VAGO temperature must be between 0 and 2")
        if not system_prompt.strip() or not user_content.strip():
            raise ValueError("VAGO prompt content cannot be empty")

        lane_index = await self._available.get()
        config = self.configs[lane_index]
        client = self._clients[lane_index]
        stats = self._stats[lane_index]
        self._request_sequence += 1
        request_id = f"{run_id}-{observation_seq}-{self._request_sequence}"
        started = time.monotonic()
        try:
            response = await client.post(
                "/vago-text",
                json={
                    "request_id": request_id,
                    "system_prompt": system_prompt,
                    "user_content": user_content,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
            )
            arrived_at = time.monotonic()
            if response.status_code >= 400:
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as error:
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned invalid JSON"
                ) from error
            if not isinstance(payload, dict):
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned a non-object response"
                )

            action = _optional_text(payload.get("action"))
            raw_buttons = payload.get("buttons")
            if action is None or not isinstance(raw_buttons, list | tuple):
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned an invalid VAGO action"
                )
            canonical_action, canonical_buttons = parse_vago_cloud_action(action)
            try:
                buttons = tuple(int(value) for value in raw_buttons)
            except (TypeError, ValueError) as error:
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned invalid VAGO buttons"
                ) from error
            if (
                action != canonical_action
                or len(buttons) != 4
                or buttons != canonical_buttons
            ):
                stats.failures += 1
                raise RemoteLaneFailure(
                    f"remote lane {config.name} returned inconsistent VAGO controls"
                )

            wire_ms = (arrived_at - started) * 1000.0
            compute_ms = _finite_nonnegative(payload.get("compute_ms"))
            queue_ms = _finite_nonnegative(payload.get("queue_ms"))
            stats.requests += 1
            stats.wire_ms.append(wire_ms)
            if compute_ms is not None:
                stats.server_compute_ms.append(compute_ms)
            if queue_ms is not None:
                stats.server_queue_ms.append(queue_ms)
            completion = payload.get("completion")
            if not isinstance(completion, str):
                completion = ""
            return VagoTextDecision(
                request_id=_optional_text(payload.get("request_id")) or request_id,
                action=action,
                buttons=canonical_buttons,
                completion=completion[:1_000],
                lane=config.name,
                model=_optional_text(payload.get("model")),
                arrived_at=arrived_at,
                wire_ms=wire_ms,
                server_compute_ms=compute_ms,
                server_queue_ms=queue_ms,
                prompt_tokens=_nonnegative_int(payload.get("prompt_tokens")),
                completion_tokens=_nonnegative_int(payload.get("completion_tokens")),
            )
        except (httpx.HTTPError, OSError) as error:
            stats.failures += 1
            raise RemoteLaneFailure(
                f"remote lane {config.name} request failed: {type(error).__name__}"
            ) from error
        finally:
            self._available.put_nowait(lane_index)

    def snapshot(self) -> dict[str, Any]:
        return {
            "lane_count": self.lane_count,
            "lanes": [
                {
                    "name": config.name,
                    "requests": stats.requests,
                    "failures": stats.failures,
                    "wire_ms": _latency_summary(stats.wire_ms),
                    "server_compute_ms": _latency_summary(
                        stats.server_compute_ms
                    ),
                    "server_queue_ms": _latency_summary(stats.server_queue_ms),
                }
                for config, stats in zip(self.configs, self._stats, strict=True)
            ],
        }


def _validate_endpoint(endpoint: str, *, lane: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"remote lane {lane} endpoint must be an https URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            f"remote lane {lane} endpoint cannot contain credentials, query, or fragment"
        )
    if parsed.path not in {"", "/"}:
        raise ValueError(f"remote lane {lane} endpoint must not contain a path")


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": ordered[p95_index],
    }


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _finite_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _nonnegative_int(value: Any) -> int | None:
    number = _finite_nonnegative(value)
    return None if number is None else int(number)
