from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .openrouter import (
    DEFAULT_MODEL,
    BudgetExceeded,
    CostBudget,
    OpenRouterReasoningClient,
    load_api_key,
)
from .runner import (
    MockReasoningPilot,
    OpenRouterPilot,
    RemoteLanePilot,
    RunArtifacts,
    make_run_id,
    probe_raw_reasoning,
    run_practice_range,
)
from .remote_lanes import (
    DEFAULT_LANE_ENV,
    RemoteLaneFailure,
    RemoteLanePoolClient,
    load_remote_lane_configs,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thought-leak-range",
        description=(
            "Leak streamed reasoning markers into an offline ViZDoom practice room."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    mock = subparsers.add_parser("mock", help="run without network or API cost")
    _add_range_arguments(mock, default_requests=24)

    live = subparsers.add_parser(
        "live", help="probe and run an OpenRouter raw-reasoning stream"
    )
    _add_range_arguments(live, default_requests=6)
    live.add_argument("--env-file", type=Path)
    live.add_argument("--model", default=DEFAULT_MODEL)
    live.add_argument(
        "--reasoning-effort",
        choices=("low", "high", "max"),
        default="low",
    )
    live.add_argument("--max-tokens", type=_bounded_int(16, 512), default=32)
    live.add_argument("--max-usd", type=_positive_float, default=0.005)
    live.add_argument(
        "--provider-sort",
        choices=("latency", "throughput", "price"),
        default="latency",
    )
    live.add_argument(
        "--provider",
        action="append",
        default=[],
        help="optional OpenRouter provider slug; repeat to set routing order",
    )
    live.add_argument(
        "--no-provider-fallback",
        action="store_false",
        dest="provider_allow_fallbacks",
        default=True,
        help="fail instead of silently changing away from --provider",
    )
    live.add_argument(
        "--preferred-max-latency",
        type=_positive_float,
        default=0.2,
        help="preferred provider p50 time-to-first-token in seconds",
    )
    live.add_argument(
        "--session-id",
        help="stable OpenRouter cache/sticky-routing session key",
    )
    live.add_argument(
        "--direct-bit-keep-reasoning",
        action="store_true",
        help=(
            "keep low-effort reasoning before the visible bit for endpoints "
            "such as GPT-OSS where reasoning cannot be disabled"
        ),
    )
    live.add_argument(
        "--probe-only",
        action="store_true",
        help="measure one synthetic centered-target decision without opening ViZDoom",
    )
    live.add_argument(
        "--probe-case",
        choices=(
            "fire",
            "left",
            "right",
            "edge-fire-left",
            "edge-wait-left",
            "edge-fire-right",
            "edge-wait-right",
            "no-target",
            "no-ammo",
        ),
        default="fire",
        help="synthetic decision case used by the startup probe",
    )

    remote = subparsers.add_parser(
        "remote-live",
        help="run V4 through one persistent connection per remote GPU lane",
    )
    _add_range_arguments(remote, default_requests=180)
    remote.set_defaults(tap_mode="direct-motor")
    remote.add_argument(
        "--lane-config",
        type=Path,
        help=(
            "ignored JSON file containing ephemeral remote endpoints; "
            f"otherwise read {DEFAULT_LANE_ENV}"
        ),
    )
    remote.add_argument(
        "--lane-env",
        default=DEFAULT_LANE_ENV,
        help="environment variable containing remote lane JSON",
    )
    remote.add_argument(
        "--remote-timeout",
        type=_positive_float,
        default=15.0,
        help="per-request timeout for a remote GPU lane",
    )
    remote.add_argument(
        "--probe-only",
        action="store_true",
        help="validate all V4 motor cases without opening ViZDoom",
    )
    remote.add_argument(
        "--probe-profile",
        choices=("smoke", "holdout"),
        default="holdout",
        help=(
            "holdout tests coordinates absent from the prompt examples; "
            "smoke retains the historical canonical-point wiring check"
        ),
    )
    return parser


def _add_range_arguments(
    parser: argparse.ArgumentParser, *, default_requests: int
) -> None:
    parser.add_argument("--duration", type=_positive_float, default=8.0)
    parser.add_argument(
        "--observation-interval", type=_positive_float, default=0.30
    )
    parser.add_argument("--lanes", type=_bounded_int(1, 16), default=3)
    parser.add_argument(
        "--max-requests", type=_bounded_int(1, 800), default=default_requests
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--show-thoughts", action="store_true")
    parser.add_argument("--save-thoughts", action="store_true")
    parser.add_argument(
        "--tap-mode",
        choices=(
            "marker",
            "thought-phrase",
            "fire-gate",
            "direct-shot",
            "direct-bit",
            "four-agent",
            "direct-motor",
            "direct-motor-lite",
        ),
        default="marker",
        help=(
            "marker is fail-closed; thought-phrase is the deliberately unsafe "
            "offline-only V0 sniffer; fire-gate lets the local spine track while "
            "the LLM only authorizes shooting; direct-shot maps one fresh raw "
            "decision to exactly one FIRE tick; direct-bit does the same from "
            "the first visible 1/0 without a textual nonce; four-agent races "
            "independent WAIT/LEFT/RIGHT/FIRE specialists over a shared blackboard; "
            "direct-motor lets one LLM choose a six-way action and pulse length; "
            "direct-motor-lite uses semantic W/L/R/F, three lanes, and "
            "preemptible five-tick direction holds"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=("basic", "defend_the_center"),
        default="basic",
    )
    parser.add_argument(
        "--direct-max-age-ms",
        type=_bounded_int(50, 2000),
        default=300,
        help="maximum observation age accepted by direct-shot",
    )
    parser.add_argument(
        "--direct-aim-assist",
        action="store_true",
        help="let the handwritten body track, but never decide when to fire",
    )
    parser.add_argument(
        "--council-movement-ttl-ms",
        type=_bounded_int(100, 2000),
        default=600,
        help="maximum lifetime of a four-agent WAIT/LEFT/RIGHT selection",
    )
    parser.add_argument(
        "--council-fire-max-age-ms",
        type=_bounded_int(50, 1000),
        default=300,
        help="maximum source-observation age for a four-agent FIRE claim",
    )
    parser.add_argument(
        "--motor-token-max-age-ms",
        type=_bounded_int(50, 1000),
        default=400,
        help="maximum source-observation age for a V4 direct motor token",
    )
    parser.add_argument(
        "--motor-body",
        choices=("legacy", "tick-lease", "clock-thread"),
        default="legacy",
        help=(
            "legacy keeps the old arbiter on the current ASYNC_PLAYER runner; "
            "formal PLAYER baseline B is the 6874fa3 worktree; tick-lease "
            "enables the experimental ASYNC one-commit-per-game-tick body; "
            "clock-thread is formal D"
        ),
    )
    parser.add_argument(
        "--world-clock",
        choices=("unpaused", "vago-sync", "clock-thread"),
        default="unpaused",
        help=(
            "unpaused keeps stepping at 35 Hz during cloud inference; vago-sync "
            "freezes direct-motor V4 until its next streamed motor token; "
            "clock-thread gives formal D a dedicated PLAYER clock thread"
        ),
    )
    parser.add_argument(
        "--vago-frame-skip",
        type=_bounded_int(1, 8),
        default=1,
        help=(
            "native tics advanced by each VAGO-sync motor pulse tick; "
            "4 applies VAGO benchmark-style holding to every V4 pulse tick"
        ),
    )
    parser.add_argument(
        "--vago-flat-pulse",
        action="store_true",
        help=(
            "execute exactly one frame-skipped chunk per VAGO-sync LLM decision, "
            "ignoring V4 SHORT/LONG pulse multiplication"
        ),
    )
    parser.add_argument(
        "--motor-flat-pulse-ticks",
        type=_bounded_int(1, 8),
        default=None,
        help=(
            "override every direct-motor token to this many native tics on "
            "the continuous clock-thread body"
        ),
    )
    parser.add_argument(
        "--clock-capture-frames",
        action="store_true",
        help=(
            "capture observation-only GIF frames on clock-thread; this adds "
            "screen reads and is not valid for formal timing comparisons"
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_DIR / "runs")


def main(argv: list[str] | None = None) -> None:
    _configure_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    motor_modes = {"direct-motor", "direct-motor-lite"}
    if args.world_clock in {"vago-sync", "clock-thread"} and args.tap_mode not in motor_modes:
        parser.error(
            f"--world-clock {args.world_clock} currently requires "
            "--tap-mode direct-motor or direct-motor-lite"
        )
    if args.world_clock == "clock-thread" and args.motor_body != "clock-thread":
        parser.error(
            "--world-clock clock-thread requires --motor-body clock-thread"
        )
    if args.motor_body == "clock-thread" and args.world_clock != "clock-thread":
        parser.error(
            "--motor-body clock-thread requires --world-clock clock-thread"
        )
    if args.vago_frame_skip != 1 and args.world_clock != "vago-sync":
        parser.error("--vago-frame-skip other than 1 requires --world-clock vago-sync")
    if args.vago_flat_pulse and args.world_clock != "vago-sync":
        parser.error("--vago-flat-pulse requires --world-clock vago-sync")
    if args.motor_flat_pulse_ticks is not None and args.world_clock != "clock-thread":
        parser.error("--motor-flat-pulse-ticks requires --world-clock clock-thread")
    if args.clock_capture_frames and args.world_clock != "clock-thread":
        parser.error("--clock-capture-frames requires --world-clock clock-thread")
    try:
        if args.command == "mock":
            result = asyncio.run(_run_mock(args))
        else:
            if not args.probe_only:
                minimum_requests = {
                    "four-agent": 5,
                    "direct-motor": 7,
                    "direct-motor-lite": 5,
                }.get(args.tap_mode, 2)
                if args.max_requests < minimum_requests:
                    parser.error(
                        f"{args.tap_mode} live needs at least "
                        f"{minimum_requests} requests for probe + game"
                    )
            if args.command == "live":
                result = asyncio.run(_run_live(args))
            else:
                result = asyncio.run(_run_remote(args))
    except (ValueError, RuntimeError, BudgetExceeded, RemoteLaneFailure) as error:
        parser.exit(2, f"thought-leak-range: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


async def _run_mock(args: argparse.Namespace) -> dict[str, object]:
    run_id = make_run_id()
    artifacts = RunArtifacts(
        base_dir=args.artifact_dir,
        run_id=run_id,
        save_thoughts=args.save_thoughts,
    )
    try:
        summary = await run_practice_range(
            pilot=MockReasoningPilot(tap_mode=args.tap_mode),
            run_id=run_id,
            artifacts=artifacts,
            duration_seconds=args.duration,
            observation_interval=args.observation_interval,
            lanes=args.lanes,
            request_limit=args.max_requests,
            visible=args.visible,
            seed=args.seed,
            show_thoughts=args.show_thoughts,
            tap_mode=args.tap_mode,
            scenario=args.scenario,
            world_clock=args.world_clock,
            motor_body=args.motor_body,
            direct_max_age_ms=args.direct_max_age_ms,
            direct_aim_assist=args.direct_aim_assist,
            council_movement_ttl_ms=args.council_movement_ttl_ms,
            council_fire_max_age_ms=args.council_fire_max_age_ms,
            motor_token_max_age_ms=args.motor_token_max_age_ms,
            vago_frame_skip=args.vago_frame_skip,
            vago_flat_pulse=args.vago_flat_pulse,
            motor_flat_pulse_ticks=args.motor_flat_pulse_ticks,
            clock_capture_frames=args.clock_capture_frames,
        )
        result = {
            "mode": "mock",
            "world_clock": args.world_clock,
            "motor_body": args.motor_body,
            "artifacts": str(artifacts.directory),
            "range": summary,
        }
        artifacts.write_summary(result)
        return result
    finally:
        artifacts.close()


async def _run_live(args: argparse.Namespace) -> dict[str, object]:
    api_key = load_api_key(env_file=args.env_file)
    run_id = make_run_id()
    artifacts = RunArtifacts(
        base_dir=args.artifact_dir,
        run_id=run_id,
        save_thoughts=args.save_thoughts,
    )
    budget = CostBudget(
        maximum_usd=args.max_usd,
        maximum_requests=args.max_requests,
    )
    client = OpenRouterReasoningClient(
        api_key=api_key,
        budget=budget,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_tokens=args.max_tokens,
        provider_sort=args.provider_sort,
        provider_order=tuple(args.provider),
        provider_allow_fallbacks=args.provider_allow_fallbacks,
        preferred_max_latency_seconds=args.preferred_max_latency,
        session_id=args.session_id,
    )
    pilot = OpenRouterPilot(
        client,
        tap_mode=args.tap_mode,
        direct_bit_reasoning=args.direct_bit_keep_reasoning,
    )
    try:
        warmup_ms = await client.warmup()
        artifacts.event("openrouter_warmup", latency_ms=warmup_ms, http2=True)
        probe = await probe_raw_reasoning(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            show_thoughts=args.show_thoughts,
            tap_mode=args.tap_mode,
            probe_case=args.probe_case,
        )
        if not probe.passed:
            failure = {
                "mode": "live",
                "status": "probe_failed_closed",
                "world_clock": args.world_clock,
                "artifacts": str(artifacts.directory),
                "warmup_ms": warmup_ms,
                "probe": _probe_dict(probe),
                "budget": budget.snapshot(),
            }
            artifacts.write_summary(failure)
            raise RuntimeError(
                "the required streamed motor decision was not observed; "
                f"tap_mode={args.tap_mode}, reasoning types={probe.stream.reasoning_types}. "
                "No game control granted."
            )

        if args.probe_only:
            result = {
                "mode": "live",
                "status": "probe_completed",
                "requested_model": args.model,
                "reported_model": probe.stream.reported_model,
                "provider": probe.stream.provider,
                "reasoning_effort": args.reasoning_effort,
                "provider_sort": args.provider_sort,
                "provider_order": args.provider,
                "provider_allow_fallbacks": args.provider_allow_fallbacks,
                "preferred_max_latency_seconds": args.preferred_max_latency,
                "session_id": args.session_id,
                "tap_mode": args.tap_mode,
                "world_clock": args.world_clock,
                "direct_bit_keep_reasoning": args.direct_bit_keep_reasoning,
                "artifacts": str(artifacts.directory),
                "warmup_ms": warmup_ms,
                "probe": _probe_dict(probe),
                "budget": budget.snapshot(),
            }
            artifacts.write_summary(result)
            return result

        remaining_requests = max(0, args.max_requests - budget.requests)
        range_summary = await run_practice_range(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            duration_seconds=args.duration,
            observation_interval=args.observation_interval,
            lanes=args.lanes,
            request_limit=remaining_requests,
            visible=args.visible,
            seed=args.seed,
            show_thoughts=args.show_thoughts,
            tap_mode=args.tap_mode,
            scenario=args.scenario,
            world_clock=args.world_clock,
            motor_body=args.motor_body,
            direct_max_age_ms=args.direct_max_age_ms,
            direct_aim_assist=args.direct_aim_assist,
            council_movement_ttl_ms=args.council_movement_ttl_ms,
            council_fire_max_age_ms=args.council_fire_max_age_ms,
            motor_token_max_age_ms=args.motor_token_max_age_ms,
            vago_frame_skip=args.vago_frame_skip,
            vago_flat_pulse=args.vago_flat_pulse,
            motor_flat_pulse_ticks=args.motor_flat_pulse_ticks,
            clock_capture_frames=args.clock_capture_frames,
        )
        result = {
            "mode": "live",
            "status": "completed",
            "requested_model": args.model,
            "reported_model": probe.stream.reported_model,
            "provider": probe.stream.provider,
            "reasoning_effort": args.reasoning_effort,
            "provider_sort": args.provider_sort,
            "provider_order": args.provider,
            "provider_allow_fallbacks": args.provider_allow_fallbacks,
            "preferred_max_latency_seconds": args.preferred_max_latency,
            "session_id": args.session_id,
            "tap_mode": args.tap_mode,
            "scenario": args.scenario,
            "seed": args.seed,
            "world_clock": args.world_clock,
            "motor_body": args.motor_body,
            "direct_max_age_ms": args.direct_max_age_ms,
            "direct_aim_assist": args.direct_aim_assist,
            "council_movement_ttl_ms": args.council_movement_ttl_ms,
            "council_fire_max_age_ms": args.council_fire_max_age_ms,
            "motor_token_max_age_ms": args.motor_token_max_age_ms,
            "direct_bit_keep_reasoning": args.direct_bit_keep_reasoning,
            "artifacts": str(artifacts.directory),
            "warmup_ms": warmup_ms,
            "probe": _probe_dict(probe),
            "range": range_summary,
            "budget": budget.snapshot(),
        }
        artifacts.write_summary(result)
        return result
    finally:
        await client.aclose()
        artifacts.close()


async def _run_remote(args: argparse.Namespace) -> dict[str, object]:
    configs = load_remote_lane_configs(
        config_file=args.lane_config,
        env_name=args.lane_env,
    )
    if args.tap_mode != "direct-motor":
        raise ValueError("remote-live currently requires --tap-mode direct-motor")
    if args.lanes != len(configs):
        raise ValueError(
            f"--lanes is {args.lanes}, but the remote configuration contains "
            f"{len(configs)} endpoints; use exactly one endpoint per lane"
        )

    run_id = make_run_id()
    artifacts = RunArtifacts(
        base_dir=args.artifact_dir,
        run_id=run_id,
        save_thoughts=args.save_thoughts,
    )
    client = RemoteLanePoolClient(
        configs,
        timeout_seconds=args.remote_timeout,
    )
    pilot = RemoteLanePilot(client, tap_mode=args.tap_mode)
    try:
        warmup_started = asyncio.get_running_loop().time()
        health = await client.warmup()
        warmup_ms = (asyncio.get_running_loop().time() - warmup_started) * 1000.0
        artifacts.event(
            "remote_lanes_ready",
            latency_ms=warmup_ms,
            lane_count=len(configs),
            lanes=health,
        )
        probe = await probe_raw_reasoning(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            show_thoughts=args.show_thoughts,
            tap_mode=args.tap_mode,
            probe_case="fire",
            motor_probe_profile=args.probe_profile,
        )
        if not probe.passed:
            failure = {
                "mode": "remote-live",
                "status": "probe_failed_closed",
                "world_clock": args.world_clock,
                "probe_profile": args.probe_profile,
                "artifacts": str(artifacts.directory),
                "warmup_ms": warmup_ms,
                "remote_health": health,
                "probe": _probe_dict(probe),
                "remote": client.snapshot(),
            }
            artifacts.write_summary(failure)
            raise RuntimeError(
                "the remote Llama policy failed the V4 semantic probe; "
                "no game control granted"
            )

        if args.probe_only:
            result = {
                "mode": "remote-live",
                "status": "probe_completed",
                "tap_mode": args.tap_mode,
                "world_clock": args.world_clock,
                "probe_profile": args.probe_profile,
                "artifacts": str(artifacts.directory),
                "warmup_ms": warmup_ms,
                "remote_health": health,
                "probe": _probe_dict(probe),
                "remote": client.snapshot(),
            }
            artifacts.write_summary(result)
            return result

        range_summary = await run_practice_range(
            pilot=pilot,
            run_id=run_id,
            artifacts=artifacts,
            duration_seconds=args.duration,
            observation_interval=args.observation_interval,
            lanes=args.lanes,
            request_limit=args.max_requests,
            visible=args.visible,
            seed=args.seed,
            show_thoughts=args.show_thoughts,
            tap_mode=args.tap_mode,
            scenario=args.scenario,
            world_clock=args.world_clock,
            motor_body=args.motor_body,
            direct_max_age_ms=args.direct_max_age_ms,
            direct_aim_assist=args.direct_aim_assist,
            council_movement_ttl_ms=args.council_movement_ttl_ms,
            council_fire_max_age_ms=args.council_fire_max_age_ms,
            motor_token_max_age_ms=args.motor_token_max_age_ms,
            vago_frame_skip=args.vago_frame_skip,
            vago_flat_pulse=args.vago_flat_pulse,
            motor_flat_pulse_ticks=args.motor_flat_pulse_ticks,
            clock_capture_frames=args.clock_capture_frames,
        )
        result = {
            "mode": "remote-live",
            "status": "completed",
            "tap_mode": args.tap_mode,
            "scenario": args.scenario,
            "seed": args.seed,
            "world_clock": args.world_clock,
            "motor_body": args.motor_body,
            "probe_profile": args.probe_profile,
            "configured_lanes": args.lanes,
            "observation_interval": args.observation_interval,
            "motor_token_max_age_ms": args.motor_token_max_age_ms,
            "artifacts": str(artifacts.directory),
            "warmup_ms": warmup_ms,
            "remote_health": health,
            "probe": _probe_dict(probe),
            "range": range_summary,
            "remote": client.snapshot(),
        }
        artifacts.write_summary(result)
        return result
    finally:
        await client.aclose()
        artifacts.close()


def _probe_dict(probe) -> dict[str, object]:
    return {
        "passed": probe.passed,
        "marker_action": probe.marker_action,
        "expected_action": probe.expected_action,
        "semantically_correct": probe.semantically_correct,
        "marker_latency_ms": probe.marker_latency_ms,
        "reasoning_types": probe.stream.reasoning_types,
        "raw_reasoning_chars": probe.stream.raw_reasoning_chars,
        "visible_chars": probe.stream.visible_chars,
        "first_byte_ms": probe.stream.first_byte_ms,
        "first_reasoning_ms": probe.stream.first_reasoning_ms,
        "first_visible_ms": probe.stream.first_visible_ms,
        "total_ms": probe.stream.total_ms,
        "reported_model": probe.stream.reported_model,
        "provider": probe.stream.provider,
        "usage": probe.stream.usage,
        "specialist_results": probe.specialist_results,
    }


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _bounded_int(low: int, high: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return value

    return parse


def _configure_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main(sys.argv[1:])
