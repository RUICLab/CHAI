#!/usr/bin/env python3
"""Verify that campaign repetitions contain distinct raw planner outputs."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--minimum", type=int, default=2)
    args = parser.parse_args()

    groups = defaultdict(list)
    for path in sorted(args.campaign.glob("*/*/*/logs/experiment_metrics_*.json")):
        arm, app, repeat_dir = path.relative_to(args.campaign).parts[:3]
        data = json.loads(path.read_text())
        groups[(arm, app)].append((
            repeat_dir,
            data.get("planner_response_sha256"),
            int(data.get("planner_response_count", 0)),
        ))

    failures = []
    for (arm, app), repeats in sorted(groups.items()):
        if len(repeats) < args.minimum:
            failures.append(f"{arm}/{app}: only {len(repeats)} repeat(s)")
            continue
        hashes = [digest for _repeat, digest, count in repeats if digest and count]
        if len(hashes) != len(repeats):
            failures.append(f"{arm}/{app}: missing raw planner response/hash")
        elif len(set(hashes)) != len(hashes):
            failures.append(f"{arm}/{app}: duplicate planner-response hashes")
        else:
            print(f"OK {arm}/{app}: {len(hashes)} distinct planner generations")

    if not groups:
        failures.append("no experiment metrics found")
    if failures:
        raise SystemExit("planner-repeat verification failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
