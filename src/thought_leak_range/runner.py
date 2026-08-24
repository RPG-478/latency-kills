from __future__ import annotations

import asyncio
import hashlib
import json
import math
import secrets
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TextIO

from PIL import Image, ImageDraw
import vizdoom as vzd

from .arena import Observation, PracticeRange
from .clock_thread import (
    ClockDecision,
    DecisionMailbox,
    LatestObservationMailbox,
    PlayerClockThread,
)
from .council import (
    LAUNCH_ORDER,
    SPECIALISTS,
    MotorCouncilArbiter,
    SpecialistBitParser,
)
from .motor_token import (
    MotorTick,
    MotorToken,
    MotorTokenArbiter,
    MotorTokenParser,
)
from .openrouter import BudgetExceeded, OpenRouterReasoningClient, StreamResult
from .remote_lanes import RemoteLanePoolClient
from .protocol import (
    Action,
    DirectBitParser,
    DirectShotArbiter,
    DirectShotParser,
    FireGateParser,
    LeaseArbiter,
    MotorFrameParser,
    ThoughtCommitParser,
)


@dataclass(slots=True)
class RunMetrics:
    accepted_markers: int = 0
    rejected_markers: Counter[str] = field(default_factory=Counter)
    marker_latency_ms: list[float] = field(default_factory=list)
    actions: Counter[str] = field(default_factory=Counter)
    request_errors: int = 0
    completed_requests: int = 0
    coalesced_observations: int = 0
    gate_armed_ticks: int = 0
    direct_fire_decisions: int = 0
    direct_wait_decisions: int = 0
    direct_correct_decisions: int = 0
    direct_incorrect_decisions: int = 0
    direct_shots_executed: int = 0
    direct_hits: int = 0
    direct_damage: int = 0
    council_votes: int = 0
    council_claims: int = 0
    council_correct_votes: int = 0
    council_incorrect_votes: int = 0
    council_conflicts: int = 0
    council_selected: Counter[str] = field(default_factory=Counter)
    council_shots_executed: int = 0
    motor_token_decisions: int = 0
    motor_token_correct: int = 0
    motor_token_incorrect: int = 0
    motor_token_queued_decisions: int = 0
    motor_token_queued_fire_decisions: int = 0
    motor_token_committed_decisions: int = 0
    motor_token_superseded_before_commit: int = 0
    motor_token_selected: Counter[str] = field(default_factory=Counter)
    motor_token_ticks: Counter[str] = field(default_factory=Counter)
    motor_token_loop_calls: int = 0
    motor_token_game_ticks: set[int] = field(default_factory=set)
    motor_token_preemptions: int = 0
    motor_token_direction_rejections: Counter[str] = field(default_factory=Counter)
    motor_token_fire_decisions: int = 0
    motor_token_fire_ticks: int = 0
    motor_token_fire_loop_calls: int = 0
    motor_token_fire_game_ticks: set[int] = field(default_factory=set)
    motor_token_ammo_decrements: int = 0
    motor_token_hits: int = 0
    motor_token_damage: int = 0
    motor_token_native_expiry_violations: int = 0
    motor_token_native_expiry_overrun_ticks: int = 0
    motor_token_native_expiry_max_overrun_ticks: int = 0
    observed_ammo_decrements: int = 0
    observed_ammo_increases: int = 0
    last_observed_ammo: int | None = None
    budget_guard_stopped: bool = False


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    passed: bool
    marker_action: str | None
    expected_action: str
    semantically_correct: bool
    marker_latency_ms: float | None
    stream: StreamResult
    specialist_results: dict[str, object] | None = None


def _track_observed_ammo(metrics: RunMetrics, observation: Observation) -> Observation:
    """Track ammo changes across every observation, including held-action shots."""

    previous = metrics.last_observed_ammo
    if previous is not None:
        if observation.ammo < previous:
            metrics.observed_ammo_decrements += previous - observation.ammo
        elif observation.ammo > previous:
            metrics.observed_ammo_increases += observation.ammo - previous
    metrics.last_observed_ammo = observation.ammo
    return observation


class RunArtifacts:
    def __init__(self, *, base_dir: Path, run_id: str, save_thoughts: bool) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.directory = base_dir.resolve() / f"{stamp}-{run_id[:8]}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started_at = time.monotonic()
        self.events_path = self.directory / "events.jsonl"
        self.summary_path = self.directory / "summary.json"
        self.gif_path = self.directory / "episode.gif"
        self.thoughts_path = self.directory / "thoughts.jsonl"
        self._events = self.events_path.open("w", encoding="utf-8", newline="\n")
        self._thoughts: TextIO | None = (
            self.thoughts_path.open("w", encoding="utf-8", newline="\n")
            if save_thoughts
            else None
        )

    def event(self, kind: str, **data: object) -> None:
        row = {
            "t_ms": round((time.monotonic() - self.started_at) * 1000.0, 3),
            "kind": kind,
            **data,
        }
        self._events.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def thought(self, *, obs: int, text: str, source: str = "reasoning.text") -> None:
        if self._thoughts is None:
            return
        row = {
            "t_ms": round((time.monotonic() - self.started_at) * 1000.0, 3),
            "obs": obs,
            "source": source,
            "text": text,
        }
        self._thoughts.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._thoughts.flush()

    def write_summary(self, summary: dict[str, object]) -> None:
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self._events.close()
        if self._thoughts is not None:
            self._thoughts.close()


class ReplayRecorder:
    def __init__(self, *, every_ticks: int = 4) -> None:
        self.every_ticks = every_ticks
        self.frames: list[Image.Image] = []

    def maybe_add(self, *, tick: int, frame, action: Action, obs: int) -> None:
        if frame is None or tick % self.every_ticks != 0:
            return
        if len(frame.shape) == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        image = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 18), fill=(0, 0, 0))
        draw.text((4, 4), f"OBS {obs:03d}  MOTOR {action.value}", fill=(0, 255, 120))
        self.frames.append(image)

    def save(self, path: Path) -> bool:
        if not self.frames:
            return False
        self.frames[0].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=114,
            loop=0,
            optimize=False,
        )
        return True


class MockReasoningPilot:
    def __init__(self, *, tap_mode: str = "marker") -> None:
        self.tap_mode = tap_mode

    async def think(
        self,
        *,
        observation: Observation,
        run_id: str,
        on_reasoning,
        on_visible,
        specialist: Action | None = None,
        blackboard: str = "",
    ) -> StreamResult:
        started = time.monotonic()
        action = _rule_action(observation)
        if self.tap_mode in {"direct-motor", "direct-motor-lite"}:
            token = (
                _motor_token_lite_rule(observation)
                if self.tap_mode == "direct-motor-lite"
                else _motor_token_rule(observation)
            )
            await asyncio.sleep(0.025)
            arrived = time.monotonic()
            on_visible(_motor_wire_value(token, tap_mode=self.tap_mode), arrived)
            finished = time.monotonic()
            return StreamResult(
                response_id=f"mock-{observation.seq}-{token.name.lower()}",
                reported_model=f"mock/{self.tap_mode}-token",
                provider="local",
                reasoning_types=(),
                raw_reasoning_chars=0,
                visible_chars=1,
                first_byte_ms=(arrived - started) * 1000.0,
                first_reasoning_ms=None,
                first_visible_ms=(arrived - started) * 1000.0,
                total_ms=(finished - started) * 1000.0,
                usage={},
            )
        if self.tap_mode == "four-agent":
            if specialist is None:
                raise ValueError("four-agent requests require a specialist")
            bit = "1" if _council_rule_action(observation) is specialist else "0"
            await asyncio.sleep(0.025)
            arrived = time.monotonic()
            on_visible(bit, arrived)
            finished = time.monotonic()
            return StreamResult(
                response_id=f"mock-{observation.seq}-{specialist.value.lower()}",
                reported_model="mock/four-agent-bit-stream",
                provider="local",
                reasoning_types=(),
                raw_reasoning_chars=0,
                visible_chars=1,
                first_byte_ms=(arrived - started) * 1000.0,
                first_reasoning_ms=None,
                first_visible_ms=(arrived - started) * 1000.0,
                total_ms=(finished - started) * 1000.0,
                usage={},
            )
        if self.tap_mode in {"direct-shot", "direct-bit"}:
            nonce = _direct_nonce(run_id=run_id, obs=observation.seq)
            bit = "1" if _direct_rule_action(observation) is Action.FIRE else "0"
            if self.tap_mode == "direct-bit":
                await asyncio.sleep(0.025)
                arrived = time.monotonic()
                on_visible(bit, arrived)
                finished = time.monotonic()
                return StreamResult(
                    response_id=f"mock-{observation.seq}",
                    reported_model="mock/visible-bit-stream",
                    provider="local",
                    reasoning_types=(),
                    raw_reasoning_chars=0,
                    visible_chars=1,
                    first_byte_ms=(arrived - started) * 1000.0,
                    first_reasoning_ms=None,
                    first_visible_ms=(arrived - started) * 1000.0,
                    total_ms=(finished - started) * 1000.0,
                    usage={},
                )
            header = f"!{nonce}:{bit}!"
            first_at: float | None = None
            for chunk in (header[:5], header[5:]):
                await asyncio.sleep(0.025)
                arrived = time.monotonic()
                first_at = first_at or arrived
                on_reasoning(chunk, arrived)
            on_visible(".", time.monotonic())
            finished = time.monotonic()
            return StreamResult(
                response_id=f"mock-{observation.seq}",
                reported_model="mock/reasoning-stream",
                provider="local",
                reasoning_types=("reasoning.text",),
                raw_reasoning_chars=len(header),
                visible_chars=1,
                first_byte_ms=(first_at - started) * 1000.0 if first_at else None,
                first_reasoning_ms=(first_at - started) * 1000.0 if first_at else None,
                first_visible_ms=(finished - started) * 1000.0,
                total_ms=(finished - started) * 1000.0,
                usage={},
            )
        marker = (
            f"[[ACT run={run_id} obs={observation.seq} "
            f"ttl=220 action={action.value}]]"
        )
        chunks = [
            "Maybe FIRE, but the word FIRE is not a commitment. ",
            "The sentence says do not FIRE yet. Committing now: ",
            (
                "So trigger is ARMED. "
                if observation.target_visible and observation.ammo > 0
                else "So trigger is SAFE. "
            ),
            f"So action is {action.value}. ",
            marker[:17],
            marker[17:-2],
            marker[-2:],
        ]
        first_at: float | None = None
        for chunk in chunks:
            await asyncio.sleep(0.025)
            arrived = time.monotonic()
            first_at = first_at or arrived
            on_reasoning(chunk, arrived)
        on_visible(".", time.monotonic())
        finished = time.monotonic()
        return StreamResult(
            response_id=f"mock-{observation.seq}",
            reported_model="mock/reasoning-stream",
            provider="local",
            reasoning_types=("reasoning.text",),
            raw_reasoning_chars=sum(len(chunk) for chunk in chunks),
            visible_chars=1,
            first_byte_ms=(first_at - started) * 1000.0 if first_at else None,
            first_reasoning_ms=(first_at - started) * 1000.0 if first_at else None,
            first_visible_ms=(finished - started) * 1000.0,
            total_ms=(finished - started) * 1000.0,
            usage={},
        )


class OpenRouterPilot:
    def __init__(
        self,
        client: OpenRouterReasoningClient,
        *,
        tap_mode: str = "marker",
        direct_bit_reasoning: bool = False,
    ) -> None:
        self.client = client
        self.tap_mode = tap_mode
        self.direct_bit_reasoning = direct_bit_reasoning

    async def think(
        self,
        *,
        observation: Observation,
        run_id: str,
        on_reasoning,
        on_visible,
        specialist: Action | None = None,
        blackboard: str = "",
    ) -> StreamResult:
        return await self.client.stream(
            messages=_motor_messages(
                observation=observation,
                run_id=run_id,
                tap_mode=self.tap_mode,
                specialist=specialist,
                blackboard=blackboard,
            ),
            on_reasoning=on_reasoning,
            on_visible=on_visible,
            # The direct protocol allows a decision on a later standalone line.
            # Stopping at the first newline made compliant model output impossible
            # whenever even one line of reasoning preceded the decision.
            # Some providers begin visible output with a newline. The parser can
            # ignore whitespace, while an API newline stop would erase the bit.
            stop=None,
            temperature=(
                0.0
                if self.tap_mode
                in {
                    "direct-shot",
                    "direct-bit",
                    "four-agent",
                    "direct-motor",
                    "direct-motor-lite",
                }
                else None
            ),
            reasoning_enabled=(
                self.tap_mode
                not in {
                    "direct-bit",
                    "four-agent",
                    "direct-motor",
                    "direct-motor-lite",
                }
                or self.direct_bit_reasoning
            ),
        )


class RemoteLanePilot:
    """Drive the same V4 policy through one persistent connection per GPU."""

    def __init__(
        self,
        client: RemoteLanePoolClient,
        *,
        tap_mode: str = "direct-motor",
    ) -> None:
        if tap_mode != "direct-motor":
            raise ValueError("remote GPU lanes currently support direct-motor only")
        self.client = client
        self.tap_mode = tap_mode

    async def think(
        self,
        *,
        observation: Observation,
        run_id: str,
        on_reasoning,
        on_visible,
        specialist: Action | None = None,
        blackboard: str = "",
    ) -> StreamResult:
        if specialist is not None or blackboard:
            raise ValueError("remote GPU lanes do not support council requests")
        messages = _motor_messages(
            observation=observation,
            run_id=run_id,
            tap_mode=self.tap_mode,
        )
        return await self.client.stream_motor(
            observation_text=messages[-1]["content"],
            run_id=run_id,
            observation_seq=observation.seq,
            on_visible=on_visible,
        )


async def probe_raw_reasoning(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    show_thoughts: bool,
    tap_mode: str,
    probe_case: str = "fire",
    motor_probe_profile: str = "smoke",
) -> ProbeOutcome:
    if tap_mode in {"direct-motor", "direct-motor-lite"}:
        return await _probe_direct_motor_suite(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            show_thoughts=show_thoughts,
            tap_mode=tap_mode,
            probe_profile=motor_probe_profile,
        )
    cases = {
        "fire": (True, 0.0, 10),
        "left": (True, -0.25, 10),
        "right": (True, 0.25, 10),
        "edge-fire-left": (True, -0.08, 10),
        "edge-wait-left": (True, -0.081, 10),
        "edge-fire-right": (True, 0.08, 10),
        "edge-wait-right": (True, 0.081, 10),
        "no-target": (False, None, 10),
        "no-ammo": (True, 0.0, 0),
    }
    if probe_case not in cases:
        raise ValueError(f"unknown probe case: {probe_case}")
    visible, dx, ammo = cases[probe_case]
    observation = Observation(
        seq=0,
        captured_at=time.monotonic(),
        target_visible=visible,
        target_id=0 if visible else None,
        target_name="ProbeDummy" if visible else None,
        target_dx=dx,
        target_width=0.2 if visible else None,
        health=100,
        ammo=ammo,
        kills=0,
        hits=0,
        damage=0,
    )
    if tap_mode == "four-agent":
        return await _probe_four_agent(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            show_thoughts=show_thoughts,
            observation=observation,
            probe_case=probe_case,
        )
    parser = _make_parser(run_id=run_id, obs=0, tap_mode=tap_mode)
    frames = []

    def on_reasoning(text: str, arrived_at: float) -> None:
        artifacts.thought(obs=0, text=text)
        if show_thoughts:
            print(f"[probe thought] {text}", end="", flush=True)
        if tap_mode != "direct-bit":
            frames.extend(parser.feed(text, now=arrived_at))

    def on_visible(text: str, arrived_at: float) -> None:
        artifacts.thought(obs=0, text=text, source="visible")
        if tap_mode == "direct-bit":
            frames.extend(parser.feed(text, now=arrived_at))

    artifacts.event(
        "probe_started",
        model=getattr(pilot, "client", None) and pilot.client.model,
        tap_mode=tap_mode,
    )
    stream = await pilot.think(
        observation=observation,
        run_id=run_id,
        on_reasoning=on_reasoning,
        on_visible=on_visible,
    )
    if show_thoughts:
        print()
    expected_action = _direct_rule_action(observation)
    semantically_correct = bool(frames) and frames[0].action is expected_action
    passed = bool(frames) and (
        stream.visible_chars > 0 if tap_mode == "direct-bit" else stream.has_raw_reasoning
    ) and (semantically_correct if tap_mode in {"direct-shot", "direct-bit"} else True)
    latency = (
        (frames[0].received_at - observation.captured_at) * 1000.0
        if frames
        else None
    )
    artifacts.event(
        "probe_finished",
        passed=passed,
        marker_action=frames[0].action.value if frames else None,
        expected_action=expected_action.value,
        semantically_correct=semantically_correct,
        probe_case=probe_case,
        marker_latency_ms=latency,
        **_stream_log(stream),
    )
    return ProbeOutcome(
        passed=passed,
        marker_action=frames[0].action.value if frames else None,
        expected_action=expected_action.value,
        semantically_correct=semantically_correct,
        marker_latency_ms=latency,
        stream=stream,
    )


async def _probe_four_agent(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    show_thoughts: bool,
    observation: Observation,
    probe_case: str,
) -> ProbeOutcome:
    expected = _council_rule_action(observation)
    votes: dict[Action, object] = {}
    streams: dict[Action, StreamResult] = {}
    blackboard = "o=-1 p=0000 e=W"

    async def run_one(specialist: Action) -> None:
        parser = SpecialistBitParser(
            expected_run_id=run_id,
            expected_obs=observation.seq,
            specialist=specialist,
        )

        def on_reasoning(text: str, arrived_at: float) -> None:
            artifacts.thought(
                obs=0,
                text=text,
                source=f"probe.{specialist.value}.reasoning.text",
            )

        def on_visible(text: str, arrived_at: float) -> None:
            artifacts.thought(
                obs=0,
                text=text,
                source=f"probe.{specialist.value}.visible",
            )
            parsed = parser.feed(text, now=arrived_at)
            if parsed:
                votes[specialist] = parsed[0]
            if show_thoughts:
                print(f"[probe {specialist.value}] {text}", flush=True)

        streams[specialist] = await pilot.think(
            observation=observation,
            run_id=run_id,
            on_reasoning=on_reasoning,
            on_visible=on_visible,
            specialist=specialist,
            blackboard=blackboard,
        )

    artifacts.event(
        "probe_started",
        model=getattr(pilot, "client", None) and pilot.client.model,
        tap_mode="four-agent",
        specialists=[action.value for action in SPECIALISTS],
    )
    await asyncio.gather(*(run_one(specialist) for specialist in LAUNCH_ORDER))
    specialist_results: dict[str, object] = {}
    semantically_correct = len(votes) == len(SPECIALISTS)
    for specialist in SPECIALISTS:
        vote = votes.get(specialist)
        claimed = getattr(vote, "claimed", None)
        correct = claimed is (specialist is expected)
        semantically_correct = semantically_correct and correct
        stream = streams[specialist]
        specialist_results[specialist.value] = {
            "claimed": claimed,
            "correct": correct,
            "latency_ms": (
                (vote.received_at - observation.captured_at) * 1000.0
                if vote is not None
                else None
            ),
            **_stream_log(stream),
        }
    expected_vote = votes.get(expected)
    expected_stream = streams[expected]
    latency = (
        (expected_vote.received_at - observation.captured_at) * 1000.0
        if expected_vote is not None
        else None
    )
    passed = semantically_correct and all(
        stream.visible_chars > 0 for stream in streams.values()
    )
    artifacts.event(
        "probe_finished",
        passed=passed,
        marker_action=expected.value if getattr(expected_vote, "claimed", False) else None,
        expected_action=expected.value,
        semantically_correct=semantically_correct,
        probe_case=probe_case,
        marker_latency_ms=latency,
        specialist_results=specialist_results,
    )
    return ProbeOutcome(
        passed=passed,
        marker_action=expected.value if getattr(expected_vote, "claimed", False) else None,
        expected_action=expected.value,
        semantically_correct=semantically_correct,
        marker_latency_ms=latency,
        stream=expected_stream,
        specialist_results=specialist_results,
    )


async def _probe_direct_motor_suite(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    show_thoughts: bool,
    tap_mode: str = "direct-motor",
    probe_profile: str = "smoke",
) -> ProbeOutcome:
    captured_at = time.monotonic()
    if tap_mode == "direct-motor-lite":
        if probe_profile != "smoke":
            raise ValueError("direct-motor-lite only supports the smoke probe")
        cases = (
            (MotorToken.WAIT, True, 0.0, 0),
            (MotorToken.LEFT_HOLD, True, -0.35, 10),
            (MotorToken.RIGHT_HOLD, False, None, 10),
            (MotorToken.FIRE, True, 0.0, 10),
        )
    elif probe_profile == "smoke":
        cases = (
            (MotorToken.WAIT, True, 0.0, 0),
            (MotorToken.LEFT_SHORT, True, -0.15, 10),
            (MotorToken.LEFT_LONG, True, -0.35, 10),
            (MotorToken.RIGHT_SHORT, True, 0.15, 10),
            (MotorToken.RIGHT_LONG, False, None, 10),
            (MotorToken.FIRE, True, 0.0, 10),
        )
    elif probe_profile == "holdout":
        cases = (
            (MotorToken.WAIT, True, -0.317, 0),
            (MotorToken.LEFT_SHORT, True, -0.137, 10),
            (MotorToken.LEFT_LONG, True, -0.437, 10),
            (MotorToken.RIGHT_SHORT, True, 0.137, 10),
            (MotorToken.RIGHT_LONG, False, None, 10),
            (MotorToken.FIRE, True, 0.037, 10),
        )
    else:
        raise ValueError(f"unknown direct-motor probe profile: {probe_profile}")
    frames: dict[MotorToken, object] = {}
    streams: dict[MotorToken, StreamResult] = {}

    async def run_one(
        seq: int,
        expected: MotorToken,
        visible: bool,
        dx: float | None,
        ammo: int,
    ) -> None:
        observation = Observation(
            seq=seq,
            captured_at=captured_at,
            target_visible=visible,
            target_id=seq if visible else None,
            target_name="ProbeDummy" if visible else None,
            target_dx=dx,
            target_width=0.2 if visible else None,
            health=100,
            ammo=ammo,
            kills=0,
            hits=0,
            damage=0,
        )
        parser = MotorTokenParser(
            expected_run_id=run_id,
            expected_obs=seq,
            allowed_tokens=_allowed_motor_tokens(tap_mode),
            token_aliases=_motor_token_aliases(tap_mode),
        )

        def on_reasoning(text: str, arrived_at: float) -> None:
            artifacts.thought(
                obs=seq,
                text=text,
                source=f"probe.{expected.name}.reasoning.text",
            )

        def on_visible(text: str, arrived_at: float) -> None:
            artifacts.thought(
                obs=seq,
                text=text,
                source=f"probe.{expected.name}.visible",
            )
            parsed = parser.feed(text, now=arrived_at)
            if parsed:
                frames[expected] = parsed[0]
            if show_thoughts:
                print(f"[probe {expected.name}] {text}", flush=True)

        streams[expected] = await pilot.think(
            observation=observation,
            run_id=run_id,
            on_reasoning=on_reasoning,
            on_visible=on_visible,
        )

    artifacts.event(
        "probe_started",
        model=getattr(pilot, "client", None) and pilot.client.model,
        tap_mode=tap_mode,
        probe_profile=probe_profile,
        tokens=[expected.name for expected, *_ in cases],
    )
    await asyncio.gather(
        *(
            run_one(seq, expected, visible, dx, ammo)
            for seq, (expected, visible, dx, ammo) in enumerate(cases)
        )
    )
    suite_results: dict[str, object] = {}
    latencies: list[float] = []
    semantically_correct = len(frames) == len(cases)
    for expected, _visible, _dx, _ammo in cases:
        frame = frames.get(expected)
        actual = getattr(frame, "token", None)
        correct = actual is expected
        semantically_correct = semantically_correct and correct
        latency = (
            (frame.received_at - captured_at) * 1000.0
            if frame is not None
            else None
        )
        if latency is not None:
            latencies.append(latency)
        suite_results[expected.name] = {
            "expected_token": expected.value,
            "actual_token": actual.value if actual is not None else None,
            "correct": correct,
            "latency_ms": latency,
            **_stream_log(streams[expected]),
        }
    passed = semantically_correct and all(
        stream.visible_chars > 0 for stream in streams.values()
    )
    latency = max(latencies) if latencies else None
    artifacts.event(
        "probe_finished",
        passed=passed,
        marker_action=f"ALL_{len(cases)}" if passed else None,
        expected_action=f"ALL_{len(cases)}",
        semantically_correct=semantically_correct,
        probe_profile=probe_profile,
        marker_latency_ms=latency,
        specialist_results=suite_results,
    )
    return ProbeOutcome(
        passed=passed,
        marker_action=f"ALL_{len(cases)}" if passed else None,
        expected_action=f"ALL_{len(cases)}",
        semantically_correct=semantically_correct,
        marker_latency_ms=latency,
        stream=streams[MotorToken.FIRE],
        specialist_results=suite_results,
    )


async def run_practice_range(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    duration_seconds: float,
    observation_interval: float,
    lanes: int,
    request_limit: int,
    visible: bool,
    seed: int,
    show_thoughts: bool,
    tap_mode: str,
    scenario: str,
    world_clock: str = "unpaused",
    motor_body: str = "legacy",
    direct_max_age_ms: int = 300,
    direct_aim_assist: bool = False,
    council_movement_ttl_ms: int = 600,
    council_fire_max_age_ms: int = 300,
    motor_token_max_age_ms: int = 400,
    vago_frame_skip: int = 1,
    vago_flat_pulse: bool = False,
    motor_flat_pulse_ticks: int | None = None,
    clock_capture_frames: bool = False,
) -> dict[str, object]:
    if duration_seconds <= 0 or observation_interval <= 0:
        raise ValueError("duration and observation interval must be positive")
    if not 1 <= lanes <= 16:
        raise ValueError("lanes must be between one and sixteen")
    if request_limit < 0:
        raise ValueError("request limit cannot be negative")
    if world_clock not in {"unpaused", "vago-sync", "clock-thread"}:
        raise ValueError("world clock must be unpaused, vago-sync, or clock-thread")
    if motor_body not in {"legacy", "tick-lease", "clock-thread"}:
        raise ValueError("motor body must be legacy, tick-lease, or clock-thread")
    if not 1 <= vago_frame_skip <= 8:
        raise ValueError("VAGO frame skip must be between one and eight")
    if world_clock != "vago-sync" and vago_frame_skip != 1:
        raise ValueError("VAGO frame skip requires the vago-sync world clock")
    if world_clock != "vago-sync" and vago_flat_pulse:
        raise ValueError("VAGO flat pulse requires the vago-sync world clock")
    if world_clock != "clock-thread" and motor_flat_pulse_ticks is not None:
        raise ValueError("motor flat pulse ticks require the clock-thread world clock")
    if world_clock != "clock-thread" and clock_capture_frames:
        raise ValueError("clock frame capture requires the clock-thread world clock")
    if world_clock == "clock-thread":
        if tap_mode not in {"direct-motor", "direct-motor-lite"}:
            raise ValueError("clock-thread currently requires a direct-motor mode")
        if motor_body != "clock-thread":
            raise ValueError("clock-thread requires the clock-thread motor body")
        return await _run_player_clock_thread_motor_range(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            duration_seconds=duration_seconds,
            observation_interval=observation_interval,
            lanes=lanes,
            request_limit=request_limit,
            visible=visible,
            seed=seed,
            show_thoughts=show_thoughts,
            scenario=scenario,
            motor_token_max_age_ms=motor_token_max_age_ms,
            tap_mode=tap_mode,
            motor_flat_pulse_ticks=motor_flat_pulse_ticks,
            clock_capture_frames=clock_capture_frames,
        )
    if motor_body == "clock-thread":
        raise ValueError("clock-thread motor body requires world-clock clock-thread")
    if world_clock == "vago-sync":
        if tap_mode not in {"direct-motor", "direct-motor-lite"}:
            raise ValueError("vago-sync currently requires direct-motor V4")
        return await _run_vago_sync_motor_range(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            duration_seconds=duration_seconds,
            request_limit=request_limit,
            visible=visible,
            seed=seed,
            show_thoughts=show_thoughts,
            scenario=scenario,
            configured_lanes=lanes,
            motor_token_max_age_ms=motor_token_max_age_ms,
            motor_body=motor_body,
            tap_mode=tap_mode,
            frame_skip=vago_frame_skip,
            flat_pulse=vago_flat_pulse,
        )

    metrics = RunMetrics()
    direct_mode = tap_mode in {"direct-shot", "direct-bit"}
    council_mode = tap_mode == "four-agent"
    motor_token_mode = tap_mode in {"direct-motor", "direct-motor-lite"}
    if council_mode and lanes < len(SPECIALISTS):
        raise ValueError("four-agent mode needs at least four concurrent lanes")
    if motor_token_mode:
        arbiter = MotorTokenArbiter(
            run_id=run_id,
            maximum_age_ms=motor_token_max_age_ms,
            game_tick_lease=motor_body == "tick-lease",
        )
    elif council_mode:
        arbiter = MotorCouncilArbiter(
            run_id=run_id,
            movement_ttl_ms=council_movement_ttl_ms,
            fire_max_age_ms=council_fire_max_age_ms,
        )
    elif direct_mode:
        arbiter = DirectShotArbiter(
            run_id=run_id, maximum_age_ms=direct_max_age_ms
        )
    else:
        arbiter = LeaseArbiter(run_id=run_id)
    recorder = ReplayRecorder()
    tasks: dict[asyncio.Task[StreamResult], int] = {}
    observations: dict[int, Observation] = {}
    launched = 0
    latest_obs = 0
    previous_action = Action.WAIT
    previous_gate: bool | None = None
    previous_target_visible: bool | None = None
    last_observation: Observation | None = None
    initialization_started = time.monotonic()
    launch_stopped = False
    game_loop_duration_ms = 0.0
    held_motor_obs: int | None = None
    held_motor_action: Action | None = None
    held_motor_set_game_tick: int | None = None
    held_motor_expires_at_game_tick: int | None = None
    held_motor_overrun_recorded = False

    def record_native_expiry_overrun(current_game_tick: int) -> None:
        """Invalidate tick-lease runs when ASYNC_PLAYER held an action too long."""

        nonlocal held_motor_overrun_recorded
        if (
            not motor_token_mode
            or motor_body != "tick-lease"
            or held_motor_action in {None, Action.WAIT}
            or held_motor_expires_at_game_tick is None
            or held_motor_set_game_tick is None
            or held_motor_overrun_recorded
            or current_game_tick <= held_motor_expires_at_game_tick
        ):
            return
        overrun_ticks = current_game_tick - held_motor_expires_at_game_tick
        held_native_ticks = current_game_tick - held_motor_set_game_tick
        held_motor_overrun_recorded = True
        metrics.motor_token_native_expiry_violations += 1
        metrics.motor_token_native_expiry_overrun_ticks += overrun_ticks
        metrics.motor_token_native_expiry_max_overrun_ticks = max(
            metrics.motor_token_native_expiry_max_overrun_ticks,
            overrun_ticks,
        )
        artifacts.event(
            "motor_token_native_expiry_overrun",
            obs=held_motor_obs,
            action=held_motor_action.value,
            set_game_tick=held_motor_set_game_tick,
            expires_at_game_tick=held_motor_expires_at_game_tick,
            observed_game_tick=current_game_tick,
            held_native_ticks=held_native_ticks,
            overrun_ticks=overrun_ticks,
            comparison_valid=False,
        )

    artifacts.event(
        "range_started",
        duration_seconds=duration_seconds,
        observation_interval=observation_interval,
        lanes=lanes,
        request_limit=request_limit,
        visible=visible,
        seed=seed,
        tap_mode=tap_mode,
        scenario=scenario,
        world_clock=world_clock,
        motor_body=motor_body,
        clock_backend="vizdoom-async-player",
        direct_max_age_ms=direct_max_age_ms if direct_mode else None,
        direct_aim_assist=direct_aim_assist if direct_mode else None,
        council_movement_ttl_ms=(
            council_movement_ttl_ms if council_mode else None
        ),
        council_fire_max_age_ms=(
            council_fire_max_age_ms if council_mode else None
        ),
        motor_token_max_age_ms=(
            motor_token_max_age_ms if motor_token_mode else None
        ),
    )

    with PracticeRange(
        visible=visible,
        seed=seed,
        episode_timeout_seconds=duration_seconds + 1.0,
        scenario=scenario,
        async_player=True,
    ) as arena:
        started = time.monotonic()
        initial_combat = _track_observed_ammo(metrics, arena.observe(seq=0))
        next_observation_at = started
        next_tick_at = started
        artifacts.event(
            "arena_initialized",
            initialization_ms=(started - initialization_started) * 1000.0,
        )
        try:
            while not arena.finished and time.monotonic() - started < duration_seconds:
                now = time.monotonic()
                launch_stopped = _harvest_finished(
                    tasks=tasks,
                    artifacts=artifacts,
                    metrics=metrics,
                ) or launch_stopped

                live_state = (
                    _track_observed_ammo(metrics, arena.observe(seq=latest_obs))
                    if tap_mode
                    in {
                        "fire-gate",
                        "direct-shot",
                        "direct-bit",
                        "four-agent",
                        "direct-motor",
                        "direct-motor-lite",
                    }
                    else None
                )
                target_edge = False
                if live_state is not None:
                    last_observation = live_state
                    if previous_target_visible is None:
                        previous_target_visible = live_state.target_visible
                        artifacts.event(
                            "target_visibility",
                            visible=live_state.target_visible,
                            target_name=live_state.target_name,
                        )
                    elif live_state.target_visible != previous_target_visible:
                        previous_target_visible = live_state.target_visible
                        target_edge = True
                        next_observation_at = now
                        artifacts.event(
                            "target_visibility",
                            visible=live_state.target_visible,
                            target_name=live_state.target_name,
                            target_dx=live_state.target_dx,
                        )

                # Consume a fresh one-shot command at this tick boundary before
                # capturing a newer cloud observation. The old ordering captured
                # first and retroactively cancelled a decision that had already
                # arrived in time for this physical tick.
                executed_direct_frame = None
                executed_motor_tick: MotorTick | None = None
                if direct_mode:
                    assert isinstance(arbiter, DirectShotArbiter)
                    executed_direct_frame = arbiter.take_fire(now=now)

                if (
                    now >= next_observation_at
                    and launched < request_limit
                    and not launch_stopped
                ):
                    waiting_without_target = (
                        tap_mode in {"fire-gate", "direct-shot", "direct-bit"}
                        and live_state is not None
                        and not live_state.target_visible
                        and not target_edge
                    )
                    if waiting_without_target:
                        # One local visibility edge can wake the cloud brain. Do
                        # not spend every request saying SAFE to an empty room.
                        next_observation_at = now + observation_interval
                    elif (
                        len(tasks) + (len(SPECIALISTS) if council_mode else 1)
                        <= lanes
                        and launched
                        + (len(SPECIALISTS) if council_mode else 1)
                        <= request_limit
                    ):
                        latest_obs += 1
                        observation = _track_observed_ammo(
                            metrics, arena.observe(seq=latest_obs)
                        )
                        last_observation = observation
                        observations[latest_obs] = observation
                        if direct_mode:
                            assert isinstance(arbiter, DirectShotArbiter)
                            cancelled = arbiter.note_observation(latest_obs)
                            if cancelled is not None:
                                artifacts.event(
                                    "direct_shot_cancelled",
                                    obs=cancelled.obs,
                                    reason="newer_observation_captured",
                                )
                        blackboard = ""
                        if council_mode:
                            assert isinstance(arbiter, MotorCouncilArbiter)
                            blackboard = arbiter.blackboard()
                            arbiter.note_observation(
                                latest_obs, captured_at=observation.captured_at
                            )
                            artifacts.event(
                                "council_blackboard",
                                obs=latest_obs,
                                previous=blackboard,
                            )
                        artifacts.event("observation", **asdict(observation))
                        if motor_token_mode:
                            assert isinstance(arbiter, MotorTokenArbiter)
                            task = asyncio.create_task(
                                _run_motor_token_request(
                                    pilot=pilot,
                                    observation=observation,
                                    run_id=run_id,
                                    arbiter=arbiter,
                                    artifacts=artifacts,
                                    metrics=metrics,
                                    show_thoughts=show_thoughts,
                                    tap_mode=tap_mode,
                                )
                            )
                            tasks[task] = latest_obs
                            launched += 1
                        elif council_mode:
                            for specialist in LAUNCH_ORDER:
                                task = asyncio.create_task(
                                    _run_council_request(
                                        pilot=pilot,
                                        observation=observation,
                                        run_id=run_id,
                                        specialist=specialist,
                                        blackboard=blackboard,
                                        arbiter=arbiter,
                                        artifacts=artifacts,
                                        metrics=metrics,
                                        show_thoughts=show_thoughts,
                                    )
                                )
                                tasks[task] = latest_obs
                                launched += 1
                        else:
                            task = asyncio.create_task(
                                _run_thought_request(
                                    pilot=pilot,
                                    observation=observation,
                                    run_id=run_id,
                                    arbiter=arbiter,
                                    artifacts=artifacts,
                                    metrics=metrics,
                                    show_thoughts=show_thoughts,
                                    tap_mode=tap_mode,
                                )
                            )
                            tasks[task] = latest_obs
                            launched += 1
                    else:
                        metrics.coalesced_observations += 1
                        artifacts.event("observation_coalesced", reason="all_lanes_busy")
                    # Set this after capture/task creation. Native screen reads can
                    # take hundreds of ms on their first call.
                    next_observation_at = time.monotonic() + observation_interval

                if motor_token_mode:
                    assert isinstance(arbiter, MotorTokenArbiter)
                    executed_motor_tick = arbiter.take_tick(
                        now=now, game_tick=arena.ticks
                    )
                    if executed_motor_tick is not None and live_state is not None:
                        source = observations.get(executed_motor_tick.frame.obs)
                        rejection = (
                            _stale_direction_reason(
                                executed_motor_tick.frame.token,
                                source,
                                live_state,
                            )
                            if source is not None
                            else None
                        )
                        if rejection is not None:
                            metrics.motor_token_direction_rejections[rejection] += 1
                            artifacts.event(
                                "motor_token_direction_rejected",
                                reason=rejection,
                                obs=executed_motor_tick.frame.obs,
                                token=executed_motor_tick.frame.token.name,
                                source_target_id=source.target_id,
                                source_target_dx=source.target_dx,
                                current_target_id=live_state.target_id,
                                current_target_dx=live_state.target_dx,
                                execute_game_tick=arena.ticks,
                            )
                            executed_motor_tick = replace(
                                executed_motor_tick,
                                action=Action.WAIT,
                            )
                    if (
                        executed_motor_tick is not None
                        and executed_motor_tick.preempted is not None
                    ):
                        metrics.motor_token_preemptions += 1
                        artifacts.event(
                            "motor_token_preempted",
                            game_tick=arena.ticks,
                            previous_obs=executed_motor_tick.preempted.obs,
                            next_obs=executed_motor_tick.frame.obs,
                        )
                    action = (
                        executed_motor_tick.action
                        if executed_motor_tick is not None
                        else Action.WAIT
                    )
                elif council_mode:
                    assert isinstance(arbiter, MotorCouncilArbiter)
                    action = arbiter.take_action(now=now)
                elif direct_mode:
                    assert isinstance(arbiter, DirectShotArbiter)
                    if executed_direct_frame is not None and live_state is not None:
                        action = Action.FIRE if live_state.ammo > 0 else Action.WAIT
                        if action is Action.WAIT:
                            artifacts.event(
                                "direct_shot_blocked",
                                obs=executed_direct_frame.obs,
                                reason="no_ammo",
                            )
                            executed_direct_frame = None
                    elif direct_aim_assist:
                        assert live_state is not None
                        action = _tracking_action(live_state)
                    else:
                        action = Action.WAIT
                elif tap_mode == "fire-gate":
                    assert isinstance(arbiter, LeaseArbiter)
                    lease_action = arbiter.current_action(now=now)
                    gate_armed = lease_action is Action.FIRE
                    if gate_armed != previous_gate:
                        artifacts.event("trigger_gate", armed=gate_armed)
                        previous_gate = gate_armed
                    if gate_armed:
                        metrics.gate_armed_ticks += 1
                    assert live_state is not None
                    action = _spinal_action(
                        observation=live_state,
                        trigger_armed=gate_armed,
                    )
                else:
                    assert isinstance(arbiter, LeaseArbiter)
                    lease_action = arbiter.current_action(now=now)
                    action = lease_action
                metrics.actions[action.value] += 1
                if action is not previous_action:
                    artifacts.event("action_changed", action=action.value)
                    previous_action = action
                recorder.maybe_add(
                    tick=arena.ticks,
                    frame=arena.frame(),
                    action=action,
                    obs=latest_obs,
                )
                execute_game_tick = arena.ticks
                record_native_expiry_overrun(execute_game_tick)
                step_started_at = time.monotonic()
                reward = arena.step(action)
                if executed_motor_tick is not None:
                    token = executed_motor_tick.frame.token
                    metrics.motor_token_ticks[token.name] += 1
                    metrics.motor_token_loop_calls += 1
                    metrics.motor_token_game_ticks.add(execute_game_tick)
                    if executed_motor_tick.committed:
                        metrics.motor_token_committed_decisions += 1
                        metrics.motor_token_superseded_before_commit += (
                            executed_motor_tick.superseded_before_commit
                        )
                        metrics.motor_token_selected[token.name] += 1
                        artifacts.event(
                            "motor_token_committed",
                            obs=executed_motor_tick.frame.obs,
                            obs_game_tick=executed_motor_tick.frame.obs_game_tick,
                            execute_game_tick=execute_game_tick,
                            token=token.value,
                            token_name=token.name,
                            superseded_before_commit=(
                                executed_motor_tick.superseded_before_commit
                            ),
                        )
                        if token is MotorToken.FIRE:
                            metrics.motor_token_fire_decisions += 1
                    if action is Action.FIRE:
                        metrics.motor_token_fire_loop_calls += 1
                        metrics.motor_token_fire_game_ticks.add(execute_game_tick)
                        assert live_state is not None
                        before = live_state
                        after = _track_observed_ammo(
                            metrics, arena.observe(seq=latest_obs)
                        )
                        last_observation = after
                        source = observations[executed_motor_tick.frame.obs]
                        hit_delta = max(0, after.hits - before.hits)
                        damage_delta = max(0, after.damage - before.damage)
                        metrics.motor_token_fire_ticks += 1
                        metrics.motor_token_ammo_decrements += max(
                            0, before.ammo - after.ammo
                        )
                        artifacts.event(
                            "motor_token_fire_executed",
                            source_obs=executed_motor_tick.frame.obs,
                            source_game_tick=executed_motor_tick.frame.obs_game_tick,
                            execute_game_tick=execute_game_tick,
                            source_token=token.name,
                            source_target_id=source.target_id,
                            source_target_dx=source.target_dx,
                            source_age_ms=(
                                step_started_at - source.captured_at
                            )
                            * 1000.0,
                            token_to_fire_ms=(
                                step_started_at
                                - executed_motor_tick.frame.received_at
                            )
                            * 1000.0,
                            current_target_visible=before.target_visible,
                            current_target_id=before.target_id,
                            current_target_dx=before.target_dx,
                            ammo_before=before.ammo,
                            ammo_after=after.ammo,
                            ammo_decrement=max(0, before.ammo - after.ammo),
                            hits_before=before.hits,
                            hits_after=after.hits,
                            hit_delta=hit_delta,
                            damage_before=before.damage,
                            damage_after=after.damage,
                            damage_delta=damage_delta,
                            kills_before=before.kills,
                            kills_after=after.kills,
                            reward=reward,
                        )
                    if motor_body == "tick-lease":
                        if (
                            action is Action.WAIT
                            or executed_motor_tick.expires_at_game_tick is None
                        ):
                            held_motor_obs = None
                            held_motor_action = None
                            held_motor_set_game_tick = None
                            held_motor_expires_at_game_tick = None
                            held_motor_overrun_recorded = False
                        elif held_motor_obs != executed_motor_tick.frame.obs:
                            held_motor_obs = executed_motor_tick.frame.obs
                            held_motor_action = action
                            held_motor_set_game_tick = execute_game_tick
                            held_motor_expires_at_game_tick = (
                                executed_motor_tick.expires_at_game_tick
                            )
                            held_motor_overrun_recorded = False
                elif motor_token_mode and motor_body == "tick-lease":
                    held_motor_obs = None
                    held_motor_action = None
                    held_motor_set_game_tick = None
                    held_motor_expires_at_game_tick = None
                    held_motor_overrun_recorded = False
                if council_mode and action is Action.FIRE:
                    metrics.council_shots_executed += 1
                if executed_direct_frame is not None:
                    before = live_state
                    after = _track_observed_ammo(
                        metrics, arena.observe(seq=latest_obs)
                    )
                    last_observation = after
                    source = observations[executed_direct_frame.obs]
                    hit_delta = max(0, after.hits - before.hits)
                    damage_delta = max(0, after.damage - before.damage)
                    metrics.direct_shots_executed += 1
                    artifacts.event(
                        "direct_shot_executed",
                        source_obs=executed_direct_frame.obs,
                        source_target_id=source.target_id,
                        source_target_dx=source.target_dx,
                        source_age_ms=(step_started_at - source.captured_at) * 1000.0,
                        marker_to_fire_ms=(
                            step_started_at - executed_direct_frame.received_at
                        )
                        * 1000.0,
                        current_target_visible=before.target_visible,
                        current_target_id=before.target_id,
                        current_target_dx=before.target_dx,
                        ammo_before=before.ammo,
                        ammo_after=after.ammo,
                        hits_before=before.hits,
                        hits_after=after.hits,
                        hit_delta=hit_delta,
                        damage_before=before.damage,
                        damage_after=after.damage,
                        damage_delta=damage_delta,
                        kills_before=before.kills,
                        kills_after=after.kills,
                        reward=reward,
                    )
                # basic.cfg has living_reward=-1 on every tick. Keep the JSONL
                # useful by recording only rewards beyond that heartbeat.
                if reward not in {0.0, -1.0}:
                    artifacts.event(
                        "non_living_reward",
                        value=reward,
                        total=arena.total_reward,
                    )

                if arena.async_player:
                    # The native ASYNC_PLAYER clock already caught up during
                    # the step above. Do not replay every missed scheduler
                    # slot in a burst after a long Cloud wait.
                    next_tick_at = time.monotonic() + 1.0 / 35.0
                else:
                    next_tick_at += 1.0 / 35.0
                await asyncio.sleep(max(0.0, next_tick_at - time.monotonic()))
        finally:
            # Capture active wall time before cancelling streams and encoding the
            # replay. The old metric accidentally included GIF finalization.
            game_loop_duration_ms = (time.monotonic() - started) * 1000.0
            arbiter.panic_release()
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            episode_finished = arena.finished
            try:
                final_observation = _track_observed_ammo(
                    metrics, arena.observe(seq=latest_obs + 1)
                )
            except vzd.ViZDoomError:
                final_observation = last_observation
            record_native_expiry_overrun(arena.ticks)
            total_reward = arena.total_reward
            ticks = arena.ticks

    if direct_mode and final_observation is not None:
        metrics.direct_hits = max(0, final_observation.hits - initial_combat.hits)
        metrics.direct_damage = max(
            0, final_observation.damage - initial_combat.damage
        )
    if motor_token_mode and final_observation is not None:
        metrics.motor_token_hits = max(
            0, final_observation.hits - initial_combat.hits
        )
        metrics.motor_token_damage = max(
            0, final_observation.damage - initial_combat.damage
        )

    gif_written = recorder.save(artifacts.gif_path)
    marker_latency = metrics.marker_latency_ms
    summary: dict[str, object] = {
        "run_id": run_id,
        "tap_mode": tap_mode,
        "scenario": scenario,
        "seed": seed,
        "world_clock": world_clock,
        "motor_body": motor_body,
        "clock_backend": "vizdoom-async-player",
        "duration_ms": round(game_loop_duration_ms, 3),
        "simulation_duration_ms": round(ticks / 35.0 * 1000.0, 3),
        "ticks": ticks,
        "episode_finished": episode_finished,
        "requests_launched": launched,
        "requests_completed": metrics.completed_requests,
        "request_errors": metrics.request_errors,
        "accepted_markers": metrics.accepted_markers,
        "rejected_markers": dict(metrics.rejected_markers),
        "marker_latency_ms": marker_latency,
        "mean_marker_latency_ms": (
            sum(marker_latency) / len(marker_latency) if marker_latency else None
        ),
        "actions_by_tick": dict(metrics.actions),
        "coalesced_observations": metrics.coalesced_observations,
        "gate_armed_ticks": metrics.gate_armed_ticks,
        "direct_fire_decisions": metrics.direct_fire_decisions,
        "direct_wait_decisions": metrics.direct_wait_decisions,
        "direct_correct_decisions": metrics.direct_correct_decisions,
        "direct_incorrect_decisions": metrics.direct_incorrect_decisions,
        "direct_shots_executed": metrics.direct_shots_executed,
        "direct_hits": metrics.direct_hits,
        "direct_damage": metrics.direct_damage,
        "council_votes": metrics.council_votes,
        "council_claims": metrics.council_claims,
        "council_correct_votes": metrics.council_correct_votes,
        "council_incorrect_votes": metrics.council_incorrect_votes,
        "council_conflicts": metrics.council_conflicts,
        "council_selected": dict(metrics.council_selected),
        "council_shots_executed": metrics.council_shots_executed,
        "motor_token_decisions": metrics.motor_token_decisions,
        "motor_token_correct": metrics.motor_token_correct,
        "motor_token_incorrect": metrics.motor_token_incorrect,
        "motor_token_queued_decisions": metrics.motor_token_queued_decisions,
        "motor_token_queued_fire_decisions": metrics.motor_token_queued_fire_decisions,
        "motor_token_committed_decisions": metrics.motor_token_committed_decisions,
        "motor_token_superseded_before_commit": metrics.motor_token_superseded_before_commit,
        "motor_token_selected": dict(metrics.motor_token_selected),
        "motor_token_ticks": dict(metrics.motor_token_ticks),
        "motor_token_loop_calls": metrics.motor_token_loop_calls,
        "motor_token_game_ticks": sorted(metrics.motor_token_game_ticks),
        "motor_token_unique_game_ticks": len(metrics.motor_token_game_ticks),
        "motor_token_preemptions": metrics.motor_token_preemptions,
        "motor_token_direction_rejections": dict(
            metrics.motor_token_direction_rejections
        ),
        "motor_token_fire_decisions": metrics.motor_token_fire_decisions,
        "motor_token_fire_ticks": metrics.motor_token_fire_ticks,
        "motor_token_fire_loop_calls": metrics.motor_token_fire_loop_calls,
        "motor_token_fire_game_ticks": sorted(metrics.motor_token_fire_game_ticks),
        "motor_token_unique_fire_game_ticks": len(metrics.motor_token_fire_game_ticks),
        "motor_token_ammo_decrements": metrics.motor_token_ammo_decrements,
        "motor_token_hits": metrics.motor_token_hits,
        "motor_token_damage": metrics.motor_token_damage,
        "motor_token_native_expiry_violations": metrics.motor_token_native_expiry_violations,
        "motor_token_native_expiry_overrun_ticks": metrics.motor_token_native_expiry_overrun_ticks,
        "motor_token_native_expiry_max_overrun_ticks": metrics.motor_token_native_expiry_max_overrun_ticks,
        "episode_ammo_delta": (
            initial_combat.ammo - final_observation.ammo
            if final_observation is not None
            else None
        ),
        "episode_ammo_delta_valid_for_scenario": scenario == "defend_the_center",
        "observed_ammo_decrements": metrics.observed_ammo_decrements,
        "observed_ammo_increases": metrics.observed_ammo_increases,
        "comparison_valid": not (
            motor_body == "tick-lease"
            and metrics.motor_token_native_expiry_violations > 0
        ),
        "invalid_reasons": (
            ["native_action_expiry_overrun"]
            if motor_body == "tick-lease"
            and metrics.motor_token_native_expiry_violations > 0
            else []
        ),
        "budget_guard_stopped": metrics.budget_guard_stopped,
        "total_reward": total_reward,
        "final_observation": asdict(final_observation) if final_observation else None,
        "episode_gif": str(artifacts.gif_path) if gif_written else None,
    }
    artifacts.event("range_finished", **summary)
    return summary


async def _run_player_clock_thread_motor_range(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    duration_seconds: float,
    observation_interval: float,
    lanes: int,
    request_limit: int,
    visible: bool,
    seed: int,
    show_thoughts: bool,
    scenario: str,
    motor_token_max_age_ms: int,
    tap_mode: str,
    motor_flat_pulse_ticks: int | None,
    clock_capture_frames: bool,
) -> dict[str, object]:
    """Run formal D: PLAYER owns the native clock; asyncio owns only requests."""

    metrics = RunMetrics()
    decision_mailbox = DecisionMailbox(run_id=run_id)
    observation_mailbox = LatestObservationMailbox()
    clock = PlayerClockThread(
        decision_mailbox=decision_mailbox,
        observation_mailbox=observation_mailbox,
        duration_seconds=duration_seconds,
        observation_interval=observation_interval,
        visible=visible,
        seed=seed,
        scenario=scenario,
        motor_token_max_age_ms=motor_token_max_age_ms,
        flat_pulse_ticks=motor_flat_pulse_ticks,
        capture_frames=clock_capture_frames,
    )
    recorder = ReplayRecorder()
    tasks: dict[asyncio.Task[StreamResult], int] = {}
    launched_observations: set[int] = set()
    launched = 0
    stop_launching = False
    started = time.monotonic()

    artifacts.event(
        "range_started",
        run_id=run_id,
        tap_mode=tap_mode,
        scenario=scenario,
        world_clock="clock-thread",
        motor_body="clock-thread",
        clock_backend="vizdoom-player-clock-thread",
        configured_lanes=lanes,
        motor_token_max_age_ms=motor_token_max_age_ms,
        motor_flat_pulse_ticks=motor_flat_pulse_ticks,
        formal_condition="D",
    )
    clock.start()

    try:
        while clock.is_alive():
            if tasks:
                stop_launching = _harvest_finished(
                    tasks=tasks,
                    artifacts=artifacts,
                    metrics=metrics,
                ) or stop_launching

            observation = observation_mailbox.latest()
            if (
                observation is not None
                and observation.seq not in launched_observations
                and len(tasks) < lanes
                and launched < request_limit
                and not stop_launching
            ):
                launched_observations.add(observation.seq)
                launched += 1
                task = asyncio.create_task(
                    _run_clock_thread_motor_request(
                        pilot=pilot,
                        observation=observation,
                        run_id=run_id,
                        decision_mailbox=decision_mailbox,
                        artifacts=artifacts,
                        metrics=metrics,
                        show_thoughts=show_thoughts,
                        tap_mode=tap_mode,
                    )
                )
                tasks[task] = observation.seq
            await asyncio.sleep(0.002)

        if tasks:
            _harvest_finished(
                tasks=tasks,
                artifacts=artifacts,
                metrics=metrics,
            )
    finally:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        clock.stop()
        await asyncio.to_thread(clock.join)

    result = clock.result
    for event in result.events:
        artifacts.event(
            event.kind,
            clock_t_ms=round(event.clock_t_ms, 3),
            **event.data,
        )

    for observation in result.observations:
        _track_observed_ammo(metrics, observation)
    for tick, frame, action, obs in result.frame_samples:
        recorder.maybe_add(tick=tick, frame=frame, action=action, obs=obs)

    initial_combat = result.initial_observation
    final_observation = result.final_observation
    stats = result.stats
    metrics.accepted_markers = stats.queued_decisions
    metrics.motor_token_queued_decisions = stats.queued_decisions
    metrics.motor_token_queued_fire_decisions = stats.queued_fire_decisions
    metrics.motor_token_committed_decisions = stats.committed_decisions
    metrics.motor_token_fire_decisions = stats.committed_fire_decisions
    metrics.motor_token_superseded_before_commit = stats.superseded_before_commit
    metrics.motor_token_selected = stats.selected
    metrics.motor_token_ticks = stats.token_ticks
    metrics.motor_token_game_ticks = stats.game_ticks
    metrics.motor_token_loop_calls = len(stats.game_ticks)
    metrics.motor_token_fire_game_ticks = stats.fire_game_ticks
    metrics.motor_token_fire_loop_calls = len(stats.fire_game_ticks)
    metrics.motor_token_preemptions = stats.preemptions
    metrics.motor_token_ammo_decrements = metrics.observed_ammo_decrements
    metrics.motor_token_hits = max(
        0, (final_observation.hits if final_observation else 0) - initial_combat.hits
    )
    metrics.motor_token_damage = max(
        0,
        (final_observation.damage if final_observation else 0)
        - initial_combat.damage,
    )
    for event in result.events:
        if event.kind == "motor_token_rejected":
            metrics.rejected_markers[str(event.data.get("reason", "unknown"))] += 1

    ticks = result.ticks
    runner_wall_ms = (time.monotonic() - started) * 1000.0
    requested_active_duration_ms = duration_seconds * 1000.0
    active_duration_complete = (
        result.episode_finished
        or result.active_wall_ms >= requested_active_duration_ms * 0.95
    )
    simulation_duration_ms = ticks / 35.0 * 1000.0
    effective_tick_hz = (
        ticks / (result.active_wall_ms / 1000.0)
        if result.active_wall_ms > 0
        else 0.0
    )
    # Dying completes the episode duration, but it cannot excuse a slowed
    # native clock. Parallel ViZDoom processes exposed that the old `or`
    # mislabeled 23-30 Hz deaths as valid 35 Hz trials.
    clock_rate_complete = effective_tick_hz >= 35.0 * 0.90
    comparison_valid = (
        active_duration_complete
        and clock_rate_complete
        and not clock.capture_frames
    )
    gif_written = recorder.save(artifacts.gif_path)
    marker_latency = metrics.marker_latency_ms
    summary: dict[str, object] = {
        "run_id": run_id,
        "tap_mode": tap_mode,
        "scenario": scenario,
        "seed": seed,
        "world_clock": "clock-thread",
        "motor_body": "clock-thread",
        "clock_backend": "vizdoom-player-clock-thread",
        "formal_condition": "D",
        "motor_flat_pulse_ticks": motor_flat_pulse_ticks,
        "duration_basis": "active_wall_time",
        "duration_ms": round(result.active_wall_ms, 3),
        "initialization_ms": round(result.initialization_ms, 3),
        "active_wall_ms": round(result.active_wall_ms, 3),
        "runner_wall_ms": round(runner_wall_ms, 3),
        "requested_active_duration_ms": round(requested_active_duration_ms, 3),
        "active_duration_complete": active_duration_complete,
        "initialization_excluded_from_episode_clock": True,
        "simulation_duration_ms": round(simulation_duration_ms, 3),
        "effective_tick_hz": round(effective_tick_hz, 3),
        "clock_rate_complete": clock_rate_complete,
        "ticks": ticks,
        "episode_finished": result.episode_finished,
        "configured_lanes": lanes,
        "effective_lanes": lanes,
        "requests_launched": launched,
        "requests_completed": metrics.completed_requests,
        "request_errors": metrics.request_errors,
        "accepted_markers": metrics.accepted_markers,
        "rejected_markers": dict(metrics.rejected_markers),
        "marker_latency_ms": marker_latency,
        "mean_marker_latency_ms": (
            sum(marker_latency) / len(marker_latency) if marker_latency else None
        ),
        "actions_by_tick": dict(stats.actions),
        "coalesced_observations": max(
            0, len(result.observations) - len(launched_observations)
        ),
        "motor_token_decisions": metrics.motor_token_decisions,
        "motor_token_correct": metrics.motor_token_correct,
        "motor_token_incorrect": metrics.motor_token_incorrect,
        "motor_token_queued_decisions": metrics.motor_token_queued_decisions,
        "motor_token_queued_fire_decisions": (
            metrics.motor_token_queued_fire_decisions
        ),
        "motor_token_committed_decisions": metrics.motor_token_committed_decisions,
        "motor_token_superseded_before_commit": (
            metrics.motor_token_superseded_before_commit
        ),
        "motor_token_selected": dict(metrics.motor_token_selected),
        "motor_token_ticks": dict(metrics.motor_token_ticks),
        "motor_token_loop_calls": metrics.motor_token_loop_calls,
        "motor_token_game_ticks": sorted(metrics.motor_token_game_ticks),
        "motor_token_unique_game_ticks": len(metrics.motor_token_game_ticks),
        "motor_token_preemptions": metrics.motor_token_preemptions,
        "motor_token_direction_rejections": dict(
            metrics.motor_token_direction_rejections
        ),
        "motor_token_fire_decisions": metrics.motor_token_fire_decisions,
        "motor_token_fire_ticks": len(metrics.motor_token_fire_game_ticks),
        "motor_token_fire_loop_calls": metrics.motor_token_fire_loop_calls,
        "motor_token_fire_game_ticks": sorted(metrics.motor_token_fire_game_ticks),
        "motor_token_unique_fire_game_ticks": len(
            metrics.motor_token_fire_game_ticks
        ),
        "motor_token_ammo_decrements": metrics.motor_token_ammo_decrements,
        "motor_token_hits": metrics.motor_token_hits,
        "motor_token_damage": metrics.motor_token_damage,
        "motor_token_native_expiry_violations": 0,
        "motor_token_native_expiry_overrun_ticks": 0,
        "motor_token_native_expiry_max_overrun_ticks": 0,
        "episode_ammo_delta": (
            initial_combat.ammo - final_observation.ammo
            if final_observation is not None
            else None
        ),
        "episode_ammo_delta_valid_for_scenario": scenario == "defend_the_center",
        "observed_ammo_decrements": metrics.observed_ammo_decrements,
        "observed_ammo_increases": metrics.observed_ammo_increases,
        "comparison_valid": comparison_valid,
        "invalid_reasons": [
            reason
            for reason, invalid in (
                ("active_duration_incomplete", not active_duration_complete),
                ("native_clock_rate_below_90_percent", not clock_rate_complete),
                ("observation_only_frame_capture", clock.capture_frames),
            )
            if invalid
        ],
        "frame_capture_enabled": clock.capture_frames,
        "native_action_expiry_model": "PLAYER_make_action_one_tick",
        "budget_guard_stopped": metrics.budget_guard_stopped,
        "total_reward": result.total_reward,
        "final_observation": (
            asdict(final_observation) if final_observation is not None else None
        ),
        "episode_gif": str(artifacts.gif_path) if gif_written else None,
    }
    artifacts.event("range_finished", **summary)
    return summary


async def _run_vago_sync_motor_range(
    *,
    pilot,
    run_id: str,
    artifacts: RunArtifacts,
    duration_seconds: float,
    request_limit: int,
    visible: bool,
    seed: int,
    show_thoughts: bool,
    scenario: str,
    configured_lanes: int,
    motor_token_max_age_ms: int,
    motor_body: str,
    tap_mode: str,
    frame_skip: int,
    flat_pulse: bool,
) -> dict[str, object]:
    """Run V4 behind VAGO's blocking synchronous world clock.

    There is only one effective request lane because no new observation exists
    while the current request is in flight. The first accepted visible motor
    token unfreezes the world for its bounded pulse; the remainder of the
    response is still awaited before the next observation is captured.
    """

    metrics = RunMetrics()
    arbiter = MotorTokenArbiter(
        run_id=run_id,
        maximum_age_ms=motor_token_max_age_ms,
        game_tick_lease=motor_body == "tick-lease",
    )
    recorder = ReplayRecorder()
    observations: dict[int, Observation] = {}
    target_ticks = max(1, math.ceil(duration_seconds * 35.0))
    launched = 0
    latest_obs = 0
    previous_action = Action.WAIT
    last_observation: Observation | None = None
    sync_fail_closed_wait_ticks = 0
    stop_reason = "target_simulation_time"
    initialization_started = time.monotonic()

    artifacts.event(
        "range_started",
        duration_seconds=duration_seconds,
        duration_basis="simulation_time",
        target_ticks=target_ticks,
        observation_interval=None,
        lanes=configured_lanes,
        effective_lanes=1,
        request_limit=request_limit,
        visible=visible,
        seed=seed,
        tap_mode=tap_mode,
        scenario=scenario,
        world_clock="vago-sync",
        clock_backend="vizdoom-player",
        motor_token_max_age_ms=motor_token_max_age_ms,
        motor_body=motor_body,
        frame_skip=frame_skip,
        flat_pulse=flat_pulse,
    )

    with PracticeRange(
        visible=visible,
        seed=seed,
        episode_timeout_seconds=duration_seconds + 1.0,
        scenario=scenario,
        async_player=False,
    ) as arena:
        started = time.monotonic()
        initial_combat = _track_observed_ammo(metrics, arena.observe(seq=0))
        artifacts.event(
            "arena_initialized",
            initialization_ms=(started - initialization_started) * 1000.0,
        )

        def execute_tick(motor_tick: MotorTick | None) -> None:
            nonlocal last_observation, previous_action
            action = motor_tick.action if motor_tick is not None else Action.WAIT
            live_state = _track_observed_ammo(
                metrics, arena.observe(seq=latest_obs)
            )
            last_observation = live_state
            metrics.actions[action.value] += 1
            if action is not previous_action:
                artifacts.event("action_changed", action=action.value)
                previous_action = action
            recorder.maybe_add(
                tick=arena.ticks,
                frame=arena.frame(),
                action=action,
                obs=latest_obs,
            )
            execute_game_tick = arena.ticks
            step_started_at = time.monotonic()
            native_ticks = min(frame_skip, target_ticks - arena.ticks)
            reward = arena.step(action, ticks=native_ticks)

            if motor_tick is not None:
                token = motor_tick.frame.token
                metrics.motor_token_ticks[token.name] += 1
                metrics.motor_token_loop_calls += 1
                metrics.motor_token_game_ticks.add(execute_game_tick)
                if motor_tick.committed:
                    metrics.motor_token_committed_decisions += 1
                    metrics.motor_token_superseded_before_commit += (
                        motor_tick.superseded_before_commit
                    )
                    metrics.motor_token_selected[token.name] += 1
                    artifacts.event(
                        "motor_token_committed",
                        obs=motor_tick.frame.obs,
                        obs_game_tick=motor_tick.frame.obs_game_tick,
                        execute_game_tick=execute_game_tick,
                        token=token.value,
                        token_name=token.name,
                        superseded_before_commit=motor_tick.superseded_before_commit,
                    )
                    if token is MotorToken.FIRE:
                        metrics.motor_token_fire_decisions += 1
                if motor_tick.preempted is not None:
                    metrics.motor_token_preemptions += 1
                    artifacts.event(
                        "motor_token_preempted",
                        game_tick=arena.ticks,
                        previous_obs=motor_tick.preempted.obs,
                        next_obs=motor_tick.frame.obs,
                    )
                if action is Action.FIRE:
                    metrics.motor_token_fire_loop_calls += 1
                    metrics.motor_token_fire_game_ticks.add(execute_game_tick)
                    after = _track_observed_ammo(
                        metrics, arena.observe(seq=latest_obs)
                    )
                    last_observation = after
                    source = observations[motor_tick.frame.obs]
                    hit_delta = max(0, after.hits - live_state.hits)
                    damage_delta = max(0, after.damage - live_state.damage)
                    metrics.motor_token_fire_ticks += 1
                    metrics.motor_token_ammo_decrements += max(
                        0, live_state.ammo - after.ammo
                    )
                    artifacts.event(
                        "motor_token_fire_executed",
                        source_obs=motor_tick.frame.obs,
                        source_game_tick=motor_tick.frame.obs_game_tick,
                        execute_game_tick=execute_game_tick,
                        source_token=token.name,
                        source_target_id=source.target_id,
                        source_target_dx=source.target_dx,
                        source_age_ms=(step_started_at - source.captured_at) * 1000.0,
                        token_to_fire_ms=(
                            step_started_at - motor_tick.frame.received_at
                        )
                        * 1000.0,
                        current_target_visible=live_state.target_visible,
                        current_target_id=live_state.target_id,
                        current_target_dx=live_state.target_dx,
                        ammo_before=live_state.ammo,
                        ammo_after=after.ammo,
                        ammo_decrement=max(0, live_state.ammo - after.ammo),
                        hits_before=live_state.hits,
                        hits_after=after.hits,
                        hit_delta=hit_delta,
                        damage_before=live_state.damage,
                        damage_after=after.damage,
                        damage_delta=damage_delta,
                        kills_before=live_state.kills,
                        kills_after=after.kills,
                        reward=reward,
                        world_clock="vago-sync",
                    )

            if reward not in {0.0, -1.0}:
                artifacts.event(
                    "non_living_reward",
                    value=reward,
                    total=arena.total_reward,
                )

        while (
            not arena.finished
            and arena.ticks < target_ticks
            and launched < request_limit
        ):
            latest_obs += 1
            observation = _track_observed_ammo(
                metrics, arena.observe(seq=latest_obs)
            )
            last_observation = observation
            observations[latest_obs] = observation
            artifacts.event("observation", **asdict(observation))

            ticks_before_wait = arena.ticks
            artifacts.event(
                "sync_world_wait_started",
                obs=latest_obs,
                game_ticks=ticks_before_wait,
            )
            accepted_event = asyncio.Event()
            launched += 1
            request_task = asyncio.create_task(
                _run_motor_token_request(
                    pilot=pilot,
                    observation=observation,
                    run_id=run_id,
                    arbiter=arbiter,
                    artifacts=artifacts,
                    metrics=metrics,
                    show_thoughts=show_thoughts,
                    accepted_event=accepted_event,
                    tap_mode=tap_mode,
                )
            )
            accepted_wait = asyncio.create_task(accepted_event.wait())
            executed_ticks = 0
            budget_stopped = False
            try:
                await asyncio.wait(
                    {request_task, accepted_wait},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                decision_received = accepted_event.is_set()
                artifacts.event(
                    "sync_world_wait_finished",
                    obs=latest_obs,
                    game_ticks=arena.ticks,
                    game_tick_delta=arena.ticks - ticks_before_wait,
                    decision_received=decision_received,
                )

                if decision_received:
                    while not arena.finished and arena.ticks < target_ticks:
                        motor_tick = arbiter.take_tick(
                            now=time.monotonic(), game_tick=arena.ticks
                        )
                        if motor_tick is None:
                            break
                        execute_tick(motor_tick)
                        executed_ticks += 1
                        if flat_pulse:
                            arbiter.panic_release()
                            break

                await request_task
                metrics.completed_requests += 1
            except BudgetExceeded as error:
                metrics.budget_guard_stopped = True
                stop_reason = "budget_guard"
                budget_stopped = True
                artifacts.event(
                    "request_error",
                    obs=latest_obs,
                    error=type(error).__name__,
                    detail=str(error),
                )
            except Exception as error:
                metrics.request_errors += 1
                artifacts.event(
                    "request_error",
                    obs=latest_obs,
                    error=type(error).__name__,
                    detail=str(error),
                )
            finally:
                if not accepted_wait.done():
                    accepted_wait.cancel()
                await asyncio.gather(accepted_wait, return_exceptions=True)

            if budget_stopped:
                if not request_task.done():
                    request_task.cancel()
                    await asyncio.gather(request_task, return_exceptions=True)
                break

            if executed_ticks == 0 and not arena.finished and arena.ticks < target_ticks:
                sync_fail_closed_wait_ticks += 1
                artifacts.event(
                    "sync_fail_closed_wait",
                    obs=latest_obs,
                    reason="no_fresh_valid_motor_token",
                )
                execute_tick(None)

        game_loop_duration_ms = (time.monotonic() - started) * 1000.0
        arbiter.panic_release()
        episode_finished = arena.finished
        if episode_finished:
            stop_reason = "episode_finished"
        elif arena.ticks >= target_ticks:
            stop_reason = "target_simulation_time"
        elif launched >= request_limit and stop_reason != "budget_guard":
            stop_reason = "request_limit"
        try:
            final_observation = _track_observed_ammo(
                metrics, arena.observe(seq=latest_obs + 1)
            )
        except vzd.ViZDoomError:
            final_observation = last_observation
        total_reward = arena.total_reward
        ticks = arena.ticks

    if final_observation is not None:
        metrics.motor_token_hits = max(
            0, final_observation.hits - initial_combat.hits
        )
        metrics.motor_token_damage = max(
            0, final_observation.damage - initial_combat.damage
        )

    gif_written = recorder.save(artifacts.gif_path)
    marker_latency = metrics.marker_latency_ms
    summary: dict[str, object] = {
        "run_id": run_id,
        "tap_mode": tap_mode,
        "scenario": scenario,
        "seed": seed,
        "world_clock": "vago-sync",
        "motor_body": motor_body,
        "clock_backend": "vizdoom-player",
        "duration_basis": "simulation_time",
        "requested_simulation_duration_ms": duration_seconds * 1000.0,
        "duration_ms": round(game_loop_duration_ms, 3),
        "simulation_duration_ms": round(ticks / 35.0 * 1000.0, 3),
        "target_ticks": target_ticks,
        "vago_frame_skip": frame_skip,
        "vago_flat_pulse": flat_pulse,
        "ticks": ticks,
        "stop_reason": stop_reason,
        "episode_finished": episode_finished,
        "configured_lanes": configured_lanes,
        "effective_lanes": 1,
        "requests_launched": launched,
        "requests_completed": metrics.completed_requests,
        "request_errors": metrics.request_errors,
        "accepted_markers": metrics.accepted_markers,
        "rejected_markers": dict(metrics.rejected_markers),
        "marker_latency_ms": marker_latency,
        "mean_marker_latency_ms": (
            sum(marker_latency) / len(marker_latency) if marker_latency else None
        ),
        "actions_by_tick": dict(metrics.actions),
        "coalesced_observations": 0,
        "motor_token_decisions": metrics.motor_token_decisions,
        "motor_token_correct": metrics.motor_token_correct,
        "motor_token_incorrect": metrics.motor_token_incorrect,
        "motor_token_queued_decisions": metrics.motor_token_queued_decisions,
        "motor_token_queued_fire_decisions": metrics.motor_token_queued_fire_decisions,
        "motor_token_committed_decisions": metrics.motor_token_committed_decisions,
        "motor_token_superseded_before_commit": metrics.motor_token_superseded_before_commit,
        "motor_token_selected": dict(metrics.motor_token_selected),
        "motor_token_ticks": dict(metrics.motor_token_ticks),
        "motor_token_loop_calls": metrics.motor_token_loop_calls,
        "motor_token_game_ticks": sorted(metrics.motor_token_game_ticks),
        "motor_token_unique_game_ticks": len(metrics.motor_token_game_ticks),
        "motor_token_preemptions": metrics.motor_token_preemptions,
        "motor_token_fire_decisions": metrics.motor_token_fire_decisions,
        "motor_token_fire_ticks": metrics.motor_token_fire_ticks,
        "motor_token_fire_loop_calls": metrics.motor_token_fire_loop_calls,
        "motor_token_fire_game_ticks": sorted(metrics.motor_token_fire_game_ticks),
        "motor_token_unique_fire_game_ticks": len(metrics.motor_token_fire_game_ticks),
        "motor_token_ammo_decrements": metrics.motor_token_ammo_decrements,
        "motor_token_hits": metrics.motor_token_hits,
        "motor_token_damage": metrics.motor_token_damage,
        "motor_token_native_expiry_violations": metrics.motor_token_native_expiry_violations,
        "motor_token_native_expiry_overrun_ticks": metrics.motor_token_native_expiry_overrun_ticks,
        "motor_token_native_expiry_max_overrun_ticks": metrics.motor_token_native_expiry_max_overrun_ticks,
        "episode_ammo_delta": (
            initial_combat.ammo - final_observation.ammo
            if final_observation is not None
            else None
        ),
        "episode_ammo_delta_valid_for_scenario": scenario == "defend_the_center",
        "observed_ammo_decrements": metrics.observed_ammo_decrements,
        "observed_ammo_increases": metrics.observed_ammo_increases,
        "comparison_valid": not (
            motor_body == "tick-lease"
            and metrics.motor_token_native_expiry_violations > 0
        ),
        "invalid_reasons": (
            ["native_action_expiry_overrun"]
            if motor_body == "tick-lease"
            and metrics.motor_token_native_expiry_violations > 0
            else []
        ),
        "sync_fail_closed_wait_ticks": sync_fail_closed_wait_ticks,
        "budget_guard_stopped": metrics.budget_guard_stopped,
        "total_reward": total_reward,
        "final_observation": asdict(final_observation) if final_observation else None,
        "episode_gif": str(artifacts.gif_path) if gif_written else None,
    }
    artifacts.event("range_finished", **summary)
    return summary


async def _run_thought_request(
    *,
    pilot,
    observation: Observation,
    run_id: str,
    arbiter: LeaseArbiter | DirectShotArbiter,
    artifacts: RunArtifacts,
    metrics: RunMetrics,
    show_thoughts: bool,
    tap_mode: str,
) -> StreamResult:
    parser = _make_parser(
        run_id=run_id,
        obs=observation.seq,
        tap_mode=tap_mode,
    )
    direct_mode = tap_mode in {"direct-shot", "direct-bit"}

    def process_motor_text(
        text: str, arrived_at: float, *, source_channel: str
    ) -> None:
        for frame in parser.feed(text, now=arrived_at):
            expected_action = (
                _direct_rule_action(observation) if direct_mode else None
            )
            if direct_mode:
                assert isinstance(arbiter, DirectShotArbiter)
                decision = arbiter.offer(
                    frame,
                    captured_at=observation.captured_at,
                    now=arrived_at,
                )
            else:
                assert isinstance(arbiter, LeaseArbiter)
                decision = arbiter.offer(frame, now=arrived_at)
            latency = (arrived_at - observation.captured_at) * 1000.0
            if decision.accepted:
                metrics.accepted_markers += 1
                metrics.marker_latency_ms.append(latency)
                if direct_mode:
                    if frame.action is Action.FIRE:
                        metrics.direct_fire_decisions += 1
                    else:
                        metrics.direct_wait_decisions += 1
                    if frame.action is expected_action:
                        metrics.direct_correct_decisions += 1
                    else:
                        metrics.direct_incorrect_decisions += 1
                print(
                    f"\n[MOTOR] obs={frame.obs} {frame.action.value} "
                    + (
                        f"one-shot latency={latency:.1f}ms"
                        if direct_mode
                        else f"ttl={frame.ttl_ms}ms latency={latency:.1f}ms"
                    ),
                    flush=True,
                )
            else:
                metrics.rejected_markers[decision.reason] += 1
            artifacts.event(
                "motor_marker",
                obs=frame.obs,
                action=frame.action.value,
                ttl_ms=frame.ttl_ms,
                latency_ms=latency,
                accepted=decision.accepted,
                reason=decision.reason,
                source_channel=source_channel,
                one_shot=direct_mode,
                expected_action=(
                    expected_action.value if expected_action is not None else None
                ),
                semantically_correct=(
                    frame.action is expected_action
                    if expected_action is not None
                    else None
                ),
            )

    def on_reasoning(text: str, arrived_at: float) -> None:
        artifacts.thought(obs=observation.seq, text=text)
        if show_thoughts:
            print(f"[obs {observation.seq:02d}] {text}", end="", flush=True)
        if tap_mode != "direct-bit":
            process_motor_text(text, arrived_at, source_channel="reasoning.text")

    def on_visible(text: str, arrived_at: float) -> None:
        artifacts.thought(obs=observation.seq, text=text, source="visible")
        if show_thoughts and tap_mode == "direct-bit":
            print(f"[obs {observation.seq:02d} visible] {text}", end="", flush=True)
        if tap_mode == "direct-bit":
            process_motor_text(text, arrived_at, source_channel="visible")

    artifacts.event("request_started", obs=observation.seq)
    result = await pilot.think(
        observation=observation,
        run_id=run_id,
        on_reasoning=on_reasoning,
        on_visible=on_visible,
    )
    if show_thoughts:
        print()
    artifacts.event("request_finished", obs=observation.seq, **_stream_log(result))
    return result


async def _run_council_request(
    *,
    pilot,
    observation: Observation,
    run_id: str,
    specialist: Action,
    blackboard: str,
    arbiter: MotorCouncilArbiter,
    artifacts: RunArtifacts,
    metrics: RunMetrics,
    show_thoughts: bool,
) -> StreamResult:
    parser = SpecialistBitParser(
        expected_run_id=run_id,
        expected_obs=observation.seq,
        specialist=specialist,
    )
    expected_action = _council_rule_action(observation)

    def on_reasoning(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source=f"{specialist.value}.reasoning.text",
        )

    def on_visible(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source=f"{specialist.value}.visible",
        )
        if show_thoughts:
            print(
                f"[obs {observation.seq:02d} {specialist.value}] {text}",
                end="",
                flush=True,
            )
        for vote in parser.feed(text, now=arrived_at):
            decision = arbiter.offer(vote, now=arrived_at)
            latency = (arrived_at - observation.captured_at) * 1000.0
            correct = vote.claimed is (specialist is expected_action)
            metrics.council_votes += 1
            if vote.claimed:
                metrics.council_claims += 1
            if correct:
                metrics.council_correct_votes += 1
            else:
                metrics.council_incorrect_votes += 1
            if decision.reason in {"conflicting_claim", "fire_preempted"}:
                metrics.council_conflicts += 1
            if decision.reason in {"selected", "fire_preempted"}:
                metrics.accepted_markers += 1
                metrics.marker_latency_ms.append(latency)
                metrics.council_selected[specialist.value] += 1
                print(
                    f"\n[COUNCIL] obs={vote.obs} {specialist.value} "
                    f"won latency={latency:.1f}ms reason={decision.reason}",
                    flush=True,
                )
            elif not decision.accepted:
                metrics.rejected_markers[decision.reason] += 1
            artifacts.event(
                "council_vote",
                obs=vote.obs,
                specialist=specialist.value,
                claimed=vote.claimed,
                expected_specialist=expected_action.value,
                semantically_correct=correct,
                latency_ms=latency,
                accepted=decision.accepted,
                reason=decision.reason,
                selected_action=(
                    decision.selected_action.value
                    if decision.selected_action is not None
                    else None
                ),
                blackboard=blackboard,
            )

    artifacts.event(
        "request_started",
        obs=observation.seq,
        specialist=specialist.value,
        blackboard=blackboard,
    )
    result = await pilot.think(
        observation=observation,
        run_id=run_id,
        on_reasoning=on_reasoning,
        on_visible=on_visible,
        specialist=specialist,
        blackboard=blackboard,
    )
    if show_thoughts:
        print()
    artifacts.event(
        "request_finished",
        obs=observation.seq,
        specialist=specialist.value,
        **_stream_log(result),
    )
    return result


async def _run_clock_thread_motor_request(
    *,
    pilot,
    observation: Observation,
    run_id: str,
    decision_mailbox: DecisionMailbox,
    artifacts: RunArtifacts,
    metrics: RunMetrics,
    show_thoughts: bool,
    tap_mode: str,
) -> StreamResult:
    """Parse a motor token and hand it to D's game thread mailbox."""

    parser = MotorTokenParser(
        expected_run_id=run_id,
        expected_obs=observation.seq,
        expected_game_tick=observation.game_tick,
        allowed_tokens=_allowed_motor_tokens(tap_mode),
        token_aliases=_motor_token_aliases(tap_mode),
    )
    expected = _motor_expected_token(observation, tap_mode=tap_mode)

    def on_reasoning(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source="motor_token.reasoning.text",
        )

    def on_visible(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source="motor_token.visible",
        )
        if show_thoughts:
            print(
                f"\n[PLAYER-CLOCK MOTOR] obs={observation.seq:03d} {text}",
                end="",
                flush=True,
            )
        for frame in parser.feed(text, now=arrived_at):
            decision = ClockDecision(
                frame=frame,
                captured_at=observation.captured_at,
                arrived_at=arrived_at,
            )
            decision_mailbox.submit(decision)
            latency = (arrived_at - observation.captured_at) * 1000.0
            correct = frame.token is expected
            metrics.motor_token_decisions += 1
            if correct:
                metrics.motor_token_correct += 1
            else:
                metrics.motor_token_incorrect += 1
            metrics.marker_latency_ms.append(latency)
            artifacts.event(
                "motor_token_submitted",
                obs=frame.obs,
                token=frame.token.value,
                token_name=frame.token.name,
                pulse_ticks=frame.token.pulse_ticks,
                latency_ms=latency,
                expected_token=expected.value,
                expected_token_name=expected.name,
                semantically_correct=correct,
                obs_game_tick=frame.obs_game_tick,
                clock_mailbox=True,
            )

    protocol = "motor4-hold5" if tap_mode == "direct-motor-lite" else "motor6"
    artifacts.event("request_started", obs=observation.seq, protocol=protocol)
    result = await pilot.think(
        observation=observation,
        run_id=run_id,
        on_reasoning=on_reasoning,
        on_visible=on_visible,
    )
    if show_thoughts:
        print()
    artifacts.event(
        "request_finished",
        obs=observation.seq,
        protocol=protocol,
        **_stream_log(result),
    )
    return result


async def _run_motor_token_request(
    *,
    pilot,
    observation: Observation,
    run_id: str,
    arbiter: MotorTokenArbiter,
    artifacts: RunArtifacts,
    metrics: RunMetrics,
    show_thoughts: bool,
    accepted_event: asyncio.Event | None = None,
    tap_mode: str = "direct-motor",
) -> StreamResult:
    parser = MotorTokenParser(
        expected_run_id=run_id,
        expected_obs=observation.seq,
        expected_game_tick=observation.game_tick,
        allowed_tokens=_allowed_motor_tokens(tap_mode),
        token_aliases=_motor_token_aliases(tap_mode),
    )
    expected = _motor_expected_token(observation, tap_mode=tap_mode)

    def on_reasoning(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source="motor_token.reasoning.text",
        )

    def on_visible(text: str, arrived_at: float) -> None:
        artifacts.thought(
            obs=observation.seq,
            text=text,
            source="motor_token.visible",
        )
        if show_thoughts:
            print(
                f"[obs {observation.seq:03d} motor] {text}",
                end="",
                flush=True,
            )
        for frame in parser.feed(text, now=arrived_at):
            decision = arbiter.offer(
                frame,
                captured_at=observation.captured_at,
                now=arrived_at,
                captured_game_tick=observation.game_tick,
            )
            latency = (arrived_at - observation.captured_at) * 1000.0
            correct = frame.token is expected
            metrics.motor_token_decisions += 1
            if correct:
                metrics.motor_token_correct += 1
            else:
                metrics.motor_token_incorrect += 1
            if decision.accepted:
                metrics.motor_token_queued_decisions += 1
                if frame.token is MotorToken.FIRE:
                    metrics.motor_token_queued_fire_decisions += 1
                if not arbiter.game_tick_lease:
                    metrics.motor_token_committed_decisions += 1
                    metrics.motor_token_selected[frame.token.name] += 1
                    if frame.token is MotorToken.FIRE:
                        metrics.motor_token_fire_decisions += 1
                metrics.accepted_markers += 1
                metrics.marker_latency_ms.append(latency)
                if accepted_event is not None:
                    accepted_event.set()
                if decision.preempted is not None:
                    metrics.motor_token_preemptions += 1
                print(
                    f"\n[MOTOR6] obs={frame.obs} token={frame.token.value}:"
                    f"{frame.token.name} ticks={frame.token.pulse_ticks} "
                    f"latency={latency:.1f}ms",
                    flush=True,
                )
            else:
                metrics.rejected_markers[decision.reason] += 1
            artifacts.event(
                "motor_token",
                obs=frame.obs,
                token=frame.token.value,
                token_name=frame.token.name,
                action=frame.token.action.value,
                pulse_ticks=frame.token.pulse_ticks,
                latency_ms=latency,
                accepted=decision.accepted,
                reason=decision.reason,
                expected_token=expected.value,
                expected_token_name=expected.name,
                semantically_correct=correct,
                obs_game_tick=frame.obs_game_tick,
                queued=decision.reason == "queued_for_game_tick",
                committed=decision.accepted and not arbiter.game_tick_lease,
                preempted_obs=(
                    decision.preempted.obs
                    if decision.preempted is not None
                    else None
                ),
                preempted_token=(
                    decision.preempted.token.name
                    if decision.preempted is not None
                    else None
                ),
            )

    artifacts.event("request_started", obs=observation.seq, protocol="motor6")
    result = await pilot.think(
        observation=observation,
        run_id=run_id,
        on_reasoning=on_reasoning,
        on_visible=on_visible,
    )
    if show_thoughts:
        print()
    artifacts.event(
        "request_finished",
        obs=observation.seq,
        protocol="motor6",
        **_stream_log(result),
    )
    return result


def _harvest_finished(
    *,
    tasks: dict[asyncio.Task[StreamResult], int],
    artifacts: RunArtifacts,
    metrics: RunMetrics,
) -> bool:
    stop_launching = False
    for task, obs in list(tasks.items()):
        if not task.done():
            continue
        tasks.pop(task)
        try:
            task.result()
            metrics.completed_requests += 1
        except asyncio.CancelledError:
            artifacts.event("request_cancelled", obs=obs)
        except BudgetExceeded as error:
            metrics.budget_guard_stopped = True
            stop_launching = True
            artifacts.event(
                "request_guard_stopped",
                obs=obs,
                reason="cost_or_request_budget",
                message=str(error)[:1200],
            )
            print(f"[request guard stopped] {error}", file=sys.stderr, flush=True)
        except Exception as error:  # The body keeps all buttons safely released on failure.
            metrics.request_errors += 1
            artifacts.event(
                "request_error",
                obs=obs,
                error_type=type(error).__name__,
                message=str(error)[:1200],
            )
            print(f"[request {obs} failed] {error}", file=sys.stderr, flush=True)
    return stop_launching


def _stream_log(stream: StreamResult) -> dict[str, object]:
    return {
        "response_id": stream.response_id,
        "reported_model": stream.reported_model,
        "provider": stream.provider,
        "reasoning_types": stream.reasoning_types,
        "raw_reasoning_chars": stream.raw_reasoning_chars,
        "visible_chars": stream.visible_chars,
        "first_byte_ms": stream.first_byte_ms,
        "first_reasoning_ms": stream.first_reasoning_ms,
        "first_visible_ms": stream.first_visible_ms,
        "total_ms": stream.total_ms,
        "usage": stream.usage,
    }


def _motor_messages(
    *,
    observation: Observation,
    run_id: str,
    tap_mode: str,
    specialist: Action | None = None,
    blackboard: str = "",
) -> list[dict[str, str]]:
    if tap_mode == "direct-motor-lite":
        system = (
            "Reply with exactly one uppercase ASCII letter and nothing else. "
            "Apply the first matching rule only: (1) v=0=>R; "
            "(2) v=1 and a<=0=>W; (3) x<-80=>L; "
            "(4) -80<=x<=80=>F; (5) x>80=>R. "
            "Examples: v=1 x=-350 a=10=>L; v=1 x=0 a=10=>F. /no_think"
        )
        user = (
            f"v={int(observation.target_visible)} "
            f"x={_direct_bit_x(observation)} a={observation.ammo}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "direct-motor":
        system = (
            "Reply with exactly one ASCII digit and nothing else. Examples: "
            "v=0 x=9999 a=10 =>4; v=1 x=0 a=0 =>0; "
            "v=1 x=-150 a=10 =>1; v=1 x=-350 a=10 =>2; "
            "v=1 x=150 a=10 =>3; v=1 x=350 a=10 =>4; "
            "v=1 x=0 a=10 =>5. General rule: no target=>4; no ammo=>0; "
            "x<-220=>2; -220<=x<-80=>1; -80<=x<=80=>5; "
            "80<x<=220=>3; x>220=>4. /no_think"
        )
        user = (
            f"v={int(observation.target_visible)} "
            f"x={_direct_bit_x(observation)} a={observation.ammo}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "four-agent":
        if specialist is None:
            raise ValueError("four-agent messages require a specialist")
        specialist_instructions = {
            Action.WAIT: (
                "Reply 1 only when v=1 and a<=0; otherwise reply 0."
            ),
            Action.LEFT: (
                "Reply 1 only when v=1, a>0, and x<-80; otherwise reply 0."
            ),
            Action.RIGHT: (
                "If v=0 reply 1. If v=1, reply 1 only when a>0 and x>80. "
                "In every other case reply 0."
            ),
            Action.FIRE: (
                "Reply 1 only when v=1, a>0, and -80<=x<=80; otherwise reply 0."
            ),
        }
        system = (
            "Reply with exactly one ASCII digit and nothing else. "
            f"{specialist_instructions[specialist]} Ignore p,e,o. /no_think"
        )
        user = (
            f"v={int(observation.target_visible)} x={_direct_bit_x(observation)} "
            f"a={observation.ammo} {blackboard}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "direct-shot":
        nonce = _direct_nonce(run_id=run_id, obs=observation.seq)
        system = (
            "One offline FPS trigger choice. Set b=1 iff v=1, a>0, and "
            "-0.08<=x<=0.08; else b=0. In raw reasoning emit one standalone "
            "line made by concatenating !, code, :, b, ! with no spaces. Put no "
            "other text on that line. Emit it as early as possible, then stop. "
            "Final answer: ."
        )
        dx = "9" if observation.target_dx is None else f"{observation.target_dx:+.3f}"
        user = (
            f"code={nonce} v={int(observation.target_visible)} "
            f"x={dx} a={observation.ammo}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "direct-bit":
        system = (
            "Reply with exactly one ASCII digit and nothing else. Reply 1 only "
            "when v=1, a>0, and -80<=x<=80; otherwise reply 0. /no_think"
        )
        # Preserve the existing 0.001 observation precision, but avoid asking a
        # tiny model to compare signed decimal strings at the fire boundary.
        dx = _direct_bit_x(observation)
        user = (
            f"v={int(observation.target_visible)} x={dx} a={observation.ammo}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "fire-gate":
        system = """You control only the trigger authorization in an offline ViZDoom
practice room. A deterministic local spine continuously tracks the nearest moving enemy and
will fire only while your short lease is armed and the enemy is centered. You do not steer or
time individual bullets. Think briefly. If a target is visible and ammo is above zero, decide
ARMED; otherwise decide SAFE. The instant you decide, make the first complete sentence of raw
reasoning exactly "So trigger is ARMED." or "So trigger is SAFE." That sentence is the only
motor authorization. Stop immediately afterward. Final answer: one period."""
        user = "Current game observation:\n" + observation.prompt_text()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    if tap_mode == "thought-phrase":
        system = """You control only an offline ViZDoom practice room. Your private raw
reasoning is streamed directly into a motor sniffer; the final answer does not control the
game. Think as briefly as possible and do not restate the observation or rules. Apply exactly
these rules: no target or no ammo -> WAIT; target_dx < -0.10 -> LEFT; target_dx > +0.10 ->
RIGHT; otherwise -> FIRE. The instant you decide, make the first complete sentence of your
reasoning exactly "So action is ACTION." with ACTION replaced by WAIT, LEFT, RIGHT, or FIRE.
That sentence is the physical commitment. Stop immediately afterward. Final answer: one period."""
        user = "Current game observation:\n" + observation.prompt_text()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    system = f"""You are a motor cortex controlling only an offline ViZDoom practice room.
Your raw private reasoning is streamed to a deterministic motor parser. The parser does not
wait for your final answer. At the very beginning of reasoning, commit exactly one action by
writing one complete marker with this exact ASCII grammar:

[[ACT run={run_id} obs={observation.seq} ttl=220 action=ACTION]]

Replace ACTION with exactly one of WAIT, LEFT, RIGHT, FIRE. Do not quote, discuss, or emit an
example marker before the commitment. Never emit a second marker. Rules: if no target or no
ammo, WAIT; if target_dx < -0.10, LEFT; if target_dx > +0.10, RIGHT; otherwise FIRE. Natural
language is never executable in safe mode. Do not restate the observation or rules. If prose
appears in reasoning before the marker, its first conclusion must be the exact sentence
"So action is ACTION." Put the marker in reasoning, not only in the final answer. After the
marker, stop reasoning quickly and make the final answer a single period."""
    user = "Newest game observation:\n" + observation.prompt_text()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _rule_action(observation: Observation) -> Action:
    if (
        not observation.target_visible
        or observation.target_dx is None
        or observation.ammo <= 0
    ):
        return Action.WAIT
    if observation.target_dx < -0.10:
        return Action.LEFT
    if observation.target_dx > 0.10:
        return Action.RIGHT
    return Action.FIRE


def _direct_rule_action(
    observation: Observation, *, fire_window: float = 0.08
) -> Action:
    if (
        observation.target_visible
        and observation.target_dx is not None
        and observation.ammo > 0
        and abs(observation.target_dx) <= fire_window
    ):
        return Action.FIRE
    return Action.WAIT


def _council_rule_action(observation: Observation) -> Action:
    if not observation.target_visible or observation.target_dx is None:
        return Action.RIGHT
    if observation.ammo <= 0:
        return Action.WAIT
    if _direct_bit_x(observation) < -80:
        return Action.LEFT
    if _direct_bit_x(observation) > 80:
        return Action.RIGHT
    return Action.FIRE


def _motor_token_rule(observation: Observation) -> MotorToken:
    if not observation.target_visible or observation.target_dx is None:
        return MotorToken.RIGHT_LONG
    if observation.ammo <= 0:
        return MotorToken.WAIT
    x = _direct_bit_x(observation)
    if x < -220:
        return MotorToken.LEFT_LONG
    if x < -80:
        return MotorToken.LEFT_SHORT
    if x <= 80:
        return MotorToken.FIRE
    if x <= 220:
        return MotorToken.RIGHT_SHORT
    return MotorToken.RIGHT_LONG


def _stale_direction_reason(
    token: MotorToken,
    source: Observation,
    current: Observation,
    *,
    fire_window: float = 0.080,
) -> str | None:
    """Reject only an enemy-relative turn that has already reached its goal.

    Search turns captured with no visible target remain valid. When a visible
    target caused LEFT/RIGHT, the cloud result may arrive after that same enemy
    crossed into the firing window. Failing closed to WAIT at that boundary
    prevents a stale pulse from carrying the crosshair across the enemy; it
    does not locally choose a corrective direction or fire for the model.
    """

    if token.action not in {Action.LEFT, Action.RIGHT}:
        return None
    if not source.target_visible or source.target_id is None:
        return None
    if not current.target_visible or current.target_dx is None:
        return "target_lost"
    if current.target_id != source.target_id:
        return "target_changed"
    if token.action is Action.LEFT and current.target_dx >= -fire_window:
        return "entered_fire_window"
    if token.action is Action.RIGHT and current.target_dx <= fire_window:
        return "entered_fire_window"
    return None


def _motor_token_lite_rule(observation: Observation) -> MotorToken:
    """Four-choice V5 policy; directions hold until preempted or five ticks."""

    if not observation.target_visible or observation.target_dx is None:
        return MotorToken.RIGHT_HOLD
    if observation.ammo <= 0:
        return MotorToken.WAIT
    x = _direct_bit_x(observation)
    if x < -80:
        return MotorToken.LEFT_HOLD
    if x <= 80:
        return MotorToken.FIRE
    return MotorToken.RIGHT_HOLD


def _allowed_motor_tokens(tap_mode: str) -> frozenset[MotorToken] | None:
    if tap_mode == "direct-motor":
        return frozenset(
            {
                MotorToken.WAIT,
                MotorToken.LEFT_SHORT,
                MotorToken.LEFT_LONG,
                MotorToken.RIGHT_SHORT,
                MotorToken.RIGHT_LONG,
                MotorToken.FIRE,
            }
        )
    if tap_mode != "direct-motor-lite":
        return None
    return frozenset(
        {
            MotorToken.WAIT,
            MotorToken.LEFT_HOLD,
            MotorToken.RIGHT_HOLD,
            MotorToken.FIRE,
        }
    )


def _motor_token_aliases(tap_mode: str) -> dict[str, MotorToken] | None:
    if tap_mode != "direct-motor-lite":
        return None
    return {
        "W": MotorToken.WAIT,
        "L": MotorToken.LEFT_HOLD,
        "R": MotorToken.RIGHT_HOLD,
        "F": MotorToken.FIRE,
    }


def _motor_wire_value(token: MotorToken, *, tap_mode: str) -> str:
    aliases = _motor_token_aliases(tap_mode)
    if aliases is None:
        return token.value
    return next(digit for digit, mapped in aliases.items() if mapped is token)


def _motor_expected_token(
    observation: Observation, *, tap_mode: str
) -> MotorToken:
    if tap_mode == "direct-motor-lite":
        return _motor_token_lite_rule(observation)
    return _motor_token_rule(observation)


def _direct_bit_x(observation: Observation) -> int:
    """Encode dx as signed thousandths without rounding an outside point inward."""

    if observation.target_dx is None:
        return 9999
    dx = observation.target_dx
    magnitude = math.ceil(abs(dx) * 1000.0 - 1e-9)
    return -magnitude if dx < 0 else magnitude


def _direct_nonce(*, run_id: str, obs: int) -> str:
    material = f"{run_id}:{obs}:direct-shot".encode("ascii")
    return hashlib.blake2s(material, digest_size=4).hexdigest()


def make_run_id() -> str:
    return secrets.token_hex(8)


def _make_parser(*, run_id: str, obs: int, tap_mode: str):
    if tap_mode == "marker":
        return MotorFrameParser(expected_run_id=run_id, expected_obs=obs)
    if tap_mode == "thought-phrase":
        return ThoughtCommitParser(expected_run_id=run_id, expected_obs=obs)
    if tap_mode == "fire-gate":
        return FireGateParser(expected_run_id=run_id, expected_obs=obs)
    if tap_mode == "direct-shot":
        return DirectShotParser(
            expected_run_id=run_id,
            expected_obs=obs,
            expected_nonce=_direct_nonce(run_id=run_id, obs=obs),
        )
    if tap_mode == "direct-bit":
        return DirectBitParser(expected_run_id=run_id, expected_obs=obs)
    raise ValueError(f"unknown tap mode: {tap_mode}")


def _tracking_action(
    observation: Observation, *, aim_deadzone: float = 0.08
) -> Action:
    """Aim-assist movement only; this function can never pull the trigger."""

    if not observation.target_visible or observation.target_dx is None:
        return Action.RIGHT
    if observation.target_dx < -aim_deadzone:
        return Action.LEFT
    if observation.target_dx > aim_deadzone:
        return Action.RIGHT
    return Action.WAIT


def _spinal_action(
    *,
    observation: Observation,
    trigger_armed: bool,
    aim_deadzone: float = 0.08,
) -> Action:
    if not observation.target_visible or observation.target_dx is None:
        # In defend_the_center enemies may approach from behind. A blind body
        # slowly scans instead of waiting to be bitten; the LLM still owns only
        # the trigger authorization.
        return Action.RIGHT
    if observation.target_dx < -aim_deadzone:
        return Action.LEFT
    if observation.target_dx > aim_deadzone:
        return Action.RIGHT
    return Action.FIRE if trigger_armed and observation.ammo > 0 else Action.WAIT
