"""Bundle and summarize a completed remote-policy ViZDoom seed sweep.

This is a manual research utility.  It reads only public-safe run artifacts;
remote endpoint URLs and bearer tokens are never present in those artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": statistics.median(values) if values else None,
        "p95": _nearest_rank(values, 0.95),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(runs_dir: Path, *, label: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    wire_ms: list[float] = []
    compute_ms: list[float] = []
    completion_tokens: list[float] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    actual_actions: Counter[str] = Counter()
    per_seed: list[dict[str, Any]] = []

    for directory in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        summary_path = directory / "summary.json"
        events_path = directory / "events.jsonl"
        if not summary_path.is_file() or not events_path.is_file():
            continue
        summary = _load_json(summary_path)
        records.append(summary)

        for line in events_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("kind") == "motor_token_submitted":
                expected = str(event["expected_token_name"])
                actual = str(event["token_name"])
                confusion[expected][actual] += 1
                actual_actions[actual] += 1
                latencies.append(float(event["latency_ms"]))
            elif event.get("kind") == "request_finished":
                if event.get("first_visible_ms") is not None:
                    wire_ms.append(float(event["first_visible_ms"]))
                usage = event.get("usage") or {}
                if usage.get("server_compute_ms") is not None:
                    compute_ms.append(float(usage["server_compute_ms"]))
                if usage.get("completion_tokens") is not None:
                    completion_tokens.append(float(usage["completion_tokens"]))

        range_summary = summary["range"]
        decisions = int(range_summary["motor_token_decisions"])
        correct = int(range_summary["motor_token_correct"])
        per_seed.append(
            {
                "seed": int(summary["seed"]),
                "run_directory": directory.name,
                "kills": int(range_summary["final_observation"]["kills"]),
                "valid": bool(range_summary["comparison_valid"]),
                "decisions": decisions,
                "semantic_correct": correct,
                "semantic_accuracy": correct / decisions if decisions else None,
                "request_errors": int(range_summary["request_errors"]),
                "effective_tick_hz": float(range_summary["effective_tick_hz"]),
                "duration_ms": int(range_summary["duration_ms"]),
            }
        )

    per_seed.sort(key=lambda row: row["seed"])
    records.sort(key=lambda row: int(row["seed"]))
    if not records:
        raise ValueError(f"no complete run directories found in {runs_dir}")
    seeds = [row["seed"] for row in per_seed]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate seeds in {runs_dir}: {seeds}")

    total_decisions = sum(row["decisions"] for row in per_seed)
    total_correct = sum(row["semantic_correct"] for row in per_seed)
    kills = [row["kills"] for row in per_seed]
    wrong_fire = sum(
        count
        for expected, actuals in confusion.items()
        if expected != "FIRE"
        for actual, count in actuals.items()
        if actual == "FIRE"
    )
    remote_health = records[0]["remote_health"]
    return {
        "experiment": label,
        "seeds": seeds,
        "conditions": {
            "scenario": records[0]["scenario"],
            "world_clock": records[0]["world_clock"],
            "motor_body": records[0]["motor_body"],
            "lanes": records[0]["configured_lanes"],
            "observation_interval_seconds": records[0]["observation_interval"],
            "motor_token_max_age_ms": records[0]["motor_token_max_age_ms"],
            "motor_flat_pulse_ticks": records[0]["range"][
                "motor_flat_pulse_ticks"
            ],
            "policy_ids": sorted(
                {row.get("policy_id") for row in remote_health if row.get("policy_id")}
            ),
            "quantizations": sorted(
                {
                    row.get("quantization")
                    for row in remote_health
                    if row.get("quantization")
                }
            ),
            "observation_encodings": sorted(
                {
                    row.get("observation_encoding")
                    for row in remote_health
                    if row.get("observation_encoding")
                }
            ),
            "motor_output_modes": sorted(
                {
                    row.get("motor_output_mode")
                    for row in remote_health
                    if row.get("motor_output_mode")
                }
            ),
        },
        "aggregate": {
            "runs": len(records),
            "valid_runs": sum(row["valid"] for row in per_seed),
            "kills_total": sum(kills),
            "kills_mean": statistics.fmean(kills),
            "kills_by_seed": kills,
            "decisions": total_decisions,
            "semantic_correct": total_correct,
            "semantic_incorrect": total_decisions - total_correct,
            "semantic_accuracy": (
                total_correct / total_decisions if total_decisions else None
            ),
            "wrong_fire_decisions": wrong_fire,
            "request_errors": sum(row["request_errors"] for row in per_seed),
            "mean_effective_tick_hz": statistics.fmean(
                row["effective_tick_hz"] for row in per_seed
            ),
            "decision_latency_ms": _distribution(latencies),
            "remote_wire_ms": _distribution(wire_ms),
            "server_compute_ms": _distribution(compute_ms),
            "completion_tokens": _distribution(completion_tokens),
            "actual_actions": dict(sorted(actual_actions.items())),
            "confusion_expected_to_actual": {
                expected: dict(sorted(actual.items()))
                for expected, actual in sorted(confusion.items())
            },
        },
        "per_seed": per_seed,
        "summaries": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.runs_dir, label=args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    aggregate = result["aggregate"]
    print(
        f"{aggregate['kills_total']} kills / {aggregate['runs']} runs; "
        f"{aggregate['decisions']} decisions; "
        f"semantic={aggregate['semantic_accuracy']:.2%}"
    )


if __name__ == "__main__":
    main()
