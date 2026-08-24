"""Map a remote V4 lane's actual motor-token decision boundaries.

This is a manual live diagnostic, not part of pytest.  It deliberately queries
each physical endpoint independently so a six-case startup smoke test cannot
stand in for generalization across the numeric decision boundaries.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from thought_leak_range.remote_lanes import (
    RemoteLanePoolClient,
    load_remote_lane_configs,
)


TOKEN_NAMES = {
    "0": "WAIT",
    "1": "LEFT_SHORT",
    "2": "LEFT_LONG",
    "3": "RIGHT_SHORT",
    "4": "RIGHT_LONG",
    "5": "FIRE",
}


def expected_token(*, visible: int, x: int, ammo: int) -> str:
    if visible == 0:
        return "4"
    if ammo <= 0:
        return "0"
    if x < -220:
        return "2"
    if x < -80:
        return "1"
    if x <= 80:
        return "5"
    if x <= 220:
        return "3"
    return "4"


async def sweep_lane(config, xs: list[int]) -> dict[str, object]:
    client = RemoteLanePoolClient((config,), timeout_seconds=30.0)
    rows: list[dict[str, object]] = []
    try:
        health = (await client.warmup())[0]
        cases = [(0, 9999, 10), (1, 0, 0)] + [(1, x, 10) for x in xs]
        for seq, (visible, x, ammo) in enumerate(cases):
            seen: list[str] = []
            result = await client.stream_motor(
                observation_text=f"v={visible} x={x} a={ammo}",
                run_id="boundary-sweep",
                observation_seq=seq,
                on_visible=lambda token, _arrived: seen.append(token),
            )
            actual = seen[0]
            expected = expected_token(visible=visible, x=x, ammo=ammo)
            rows.append(
                {
                    "visible": visible,
                    "x": x,
                    "ammo": ammo,
                    "expected": expected,
                    "expected_name": TOKEN_NAMES[expected],
                    "actual": actual,
                    "actual_name": TOKEN_NAMES[actual],
                    "correct": actual == expected,
                    "wire_ms": result.first_visible_ms,
                    "server_compute_ms": result.usage.get("server_compute_ms"),
                }
            )
        return {
            "lane": config.name,
            "health": health,
            "rows": rows,
            "correct": sum(bool(row["correct"]) for row in rows),
            "total": len(rows),
            "client": client.snapshot(),
        }
    finally:
        await client.aclose()


async def run(args: argparse.Namespace) -> dict[str, object]:
    configs = load_remote_lane_configs(config_file=args.lane_config)
    xs = args.xs or list(range(args.x_min, args.x_max + 1, args.x_step))
    lanes = await asyncio.gather(*(sweep_lane(config, xs) for config in configs))
    lane_agreement: dict[str, object] | None = None
    if len(lanes) > 1:
        reference = [row["actual"] for row in lanes[0]["rows"]]
        comparisons = [
            [row["actual"] for row in lane["rows"]] == reference
            for lane in lanes[1:]
        ]
        lane_agreement = {
            "reference_lane": lanes[0]["lane"],
            "compared_lanes": [lane["lane"] for lane in lanes[1:]],
            "all_outputs_identical": all(comparisons),
        }
    return {
        "experiment": "remote V4 continuous decision-boundary sweep",
        "x_min": args.x_min,
        "x_max": args.x_max,
        "x_step": args.x_step,
        "x_values": xs,
        "lanes": lanes,
        "lane_agreement": lane_agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-config", type=Path, required=True)
    parser.add_argument("--x-min", type=int, default=-500)
    parser.add_argument("--x-max", type=int, default=500)
    parser.add_argument("--x-step", type=int, default=20)
    parser.add_argument("--xs", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.xs is None and (args.x_min > args.x_max or args.x_step <= 0):
        parser.error("x range must be ascending with a positive step")
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for lane in result["lanes"]:
        print(f"{lane['lane']}: {lane['correct']}/{lane['total']} correct")


if __name__ == "__main__":
    main()
