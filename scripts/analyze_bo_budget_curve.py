#!/usr/bin/env python3
"""Recover matched best-so-far BO curves from per-call measurement logs."""

import argparse
import csv
import json
from pathlib import Path


def load_records(path):
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument(
        "--checkpoints", nargs="+", type=int, default=[20, 40, 80, 160, 250]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = args.output or args.campaign / "bo_budget_curve.csv"
    rows = []
    for path in sorted(args.campaign.glob("*/*/*/logs/bo_evaluations_*.jsonl")):
        relative = path.relative_to(args.campaign).parts
        arm, app, repeat_dir = relative[:3]
        records = load_records(path)
        groups = {}
        for record in records:
            key = (record.get("trace", ""), str(record.get("capacitor", "")))
            groups.setdefault(key, []).append(record)
        for (trace, capacitor), group in groups.items():
            for checkpoint in args.checkpoints:
                eligible = [
                    record for record in group
                    if int(record["call"]) <= checkpoint
                    and float(record["error"]) <= float(record["error_bound"])
                    and bool(record.get("improves_original"))
                ]
                best = min(
                    eligible,
                    key=lambda record: (
                        float(record["power_cycle_ratio"]), float(record["error"])
                    ),
                    default=None,
                )
                rows.append({
                    "arm": arm,
                    "app": app,
                    "planner_repeat": repeat_dir.removeprefix("seed_"),
                    "trace": trace,
                    "capacitor": capacitor,
                    "bo_calls": checkpoint,
                    "improvement_found": best is not None,
                    "best_power_cycle_ratio": (
                        float(best["power_cycle_ratio"]) if best else 1.0
                    ),
                    "best_power_cycle_reduction": (
                        float(best["power_cycle_reduction"]) if best else 0.0
                    ),
                    "error": float(best["error"]) if best else 0.0,
                    "winning_call": int(best["call"]) if best else "",
                })

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "arm", "app", "planner_repeat", "trace", "capacitor", "bo_calls",
        "improvement_found", "best_power_cycle_ratio",
        "best_power_cycle_reduction", "error", "winning_call",
    ]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} budget-curve rows to {output}")


if __name__ == "__main__":
    main()
