#!/usr/bin/env python3
"""Count false operator labels correctly, with a denominator that matches.

    python3 scripts/operator_label_audit.py experiment_results/<run_id>

Why this replaces the earlier count
-----------------------------------
The figure used to report 35 (arm B) vs 37 (arm C) "detected mislabels" and call
that the same mislabelling rate. Both halves of that were wrong.

`operator_conformance_violation_count` counts violation EVENTS, not distinct
mistakes. One operator declared on one function and retried twice counts twice:

    fft_radix2: [{attempt: 1, declared: [24]}, {attempt: 2, declared: [24]}]  -> 2

That would be survivable if every arm retried equally. They do not. Conformance
is ENFORCED in the CHAI arms and not in the baselines, so CHAI re-prompts after a
failed check and accumulates more events for the same underlying mistake --
2.47 attempts per violation in arm C against 1.84 in arm B. The event count
therefore rewards the arm that checks harder, which is exactly backwards.

There was also no denominator. 35 and 37 were bare totals; nothing said out of
how many opportunities.

What this script counts instead
-------------------------------
Numerator: distinct (benchmark, seed, function, operator id) tuples that ever
failed conformance. Retries collapse; two different bad operators on one
function stay two.

Denominator: `functions_attempted`, summed over cells. Every arm records it and
every arm gets 42, so the rate is comparable across arms.

The obvious denominator -- operators actually declared -- is NOT usable here:
`approved_plans` is empty in all 18 baseline cells and present in only 15 of 18
CHAI cells, so a declared-operator rate would be computable for one side of the
comparison only.

Shipped labels are re-derived from the accepted artifact, not from any in-process
verdict, because that is the number a user is actually exposed to.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.operator_conformance import nonconforming_operator_ids  # noqa: E402

ARMS = [
    ("checkmate_baseline", "A", "CheckMate T1-T20, no control plane"),
    ("operator_only_plus10", "B", "CheckMate + 10 operators, no control plane"),
    ("chai_plus10", "C", "CHAI, same 30 operators, control plane ON"),
    ("chai_baseline_ops", "D", "CHAI T1-T20, control plane ON"),
]


def audit_arm(run_dir, arm):
    events = attempts = 0
    distinct = set()
    retries = []
    by_seed = defaultdict(set)

    pattern = os.path.join(run_dir, arm, "*", "*", "logs",
                           "experiment_metrics_*.json")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as handle:
            metrics = json.load(handle)
        # <run>/<arm>/<app>/<seed>/logs/<file>.json
        parts = path.split(os.sep)
        app, seed = parts[-4], parts[-3]
        assert seed.startswith("seed"), f"unexpected layout: {path}"

        events += metrics.get("operator_conformance_violation_count") or 0
        attempts += metrics.get("functions_attempted") or 0

        for function, records in (
                metrics.get("operator_conformance_violations") or {}).items():
            retries.append(len(records))
            for record in records:
                for tid in record.get("declared") or []:
                    distinct.add((app, seed, function, tid))
                    by_seed[seed].add((app, function, tid))

    shipped = 0
    plans = 0
    plans_affected = 0
    accepted = os.path.join(run_dir, arm, "*", "*", "outputs",
                            "accepted_approximation_plans_*.json")
    for path in sorted(glob.glob(accepted)):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("plans") if isinstance(payload, dict) else payload
        for entry in entries or []:
            plans += 1
            code = entry.get("completeFunction") or ""
            declared = set(entry.get("technique_numbers") or [])
            if not (code and declared):
                continue
            bad = nonconforming_operator_ids(code, declared)
            shipped += len(bad)
            if bad:
                plans_affected += 1

    return {
        "violation_events": events,
        "distinct_invalid_ids": len(distinct),
        "functions_attempted": attempts,
        "invalid_ids_per_attempt": (len(distinct) / attempts) if attempts else None,
        "attempts_per_violation": (sum(retries) / len(retries)) if retries else None,
        "distinct_by_seed": {s: len(v) for s, v in sorted(by_seed.items())},
        "accepted_plans": plans,
        "shipped_invalid_labels": shipped,
        "plans_carrying_a_false_label": plans_affected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--json", dest="json_out",
                        default="reports/operator_label_audit.json")
    args = parser.parse_args()

    report = {"run_dir": args.run_dir, "arms": {}}
    for arm, tag, label in ARMS:
        report["arms"][arm] = dict(audit_arm(args.run_dir, arm),
                                   tag=tag, label=label)

    print("=" * 96)
    print("FALSE OPERATOR LABELS  —  distinct mistakes, matched denominator")
    print("=" * 96)
    print(f"  {'arm':<6}{'events':>8}{'distinct':>10}{'attempts':>10}"
          f"{'per attempt':>13}{'retries':>9}{'shipped':>9}{'of plans':>10}")
    for arm, tag, _ in ARMS:
        r = report["arms"][arm]
        print(f"  {tag:<6}{r['violation_events']:>8}"
              f"{r['distinct_invalid_ids']:>10}{r['functions_attempted']:>10}"
              f"{r['invalid_ids_per_attempt']:>13.3f}"
              f"{r['attempts_per_violation']:>9.2f}"
              f"{r['shipped_invalid_labels']:>9}"
              f"{r['accepted_plans']:>10}")

    b = report["arms"]["operator_only_plus10"]
    c = report["arms"]["chai_plus10"]
    print("\n  Per seed (distinct invalid ids, 6 benchmarks each):")
    print(f"    B  {b['distinct_by_seed']}")
    print(f"    C  {c['distinct_by_seed']}")
    print("\n  The event counts that used to be reported as a 'same mislabelling")
    print(f"  rate' were {b['violation_events']} vs {c['violation_events']}. They are not comparable: arm C")
    print(f"  retries {c['attempts_per_violation']:.2f} times per violation against arm B's "
          f"{b['attempts_per_violation']:.2f},")
    print("  because only C enforces the check.")
    print(f"\n  Corrected: B makes {b['distinct_invalid_ids']} distinct false labels in "
          f"{b['functions_attempted']} function attempts "
          f"({b['invalid_ids_per_attempt']:.2f} each),")
    print(f"  C makes {c['distinct_invalid_ids']} in {c['functions_attempted']} "
          f"({c['invalid_ids_per_attempt']:.2f} each).")

    # State the direction the data shows, rather than asserting one. Overlapping
    # per-seed ranges mean the arms are indistinguishable, not that CHAI wins.
    bs = sorted(b["distinct_by_seed"].values())
    cs = sorted(c["distinct_by_seed"].values())
    print(f"  Per-seed spread: B {bs}, C {cs}.")
    if bs and cs and (max(bs) >= min(cs) and max(cs) >= min(bs)):
        print("  The ranges overlap, so the two planners mislabel at rates this")
        print("  run cannot separate. Do not claim CHAI's model makes fewer mistakes.")
    elif c["invalid_ids_per_attempt"] > b["invalid_ids_per_attempt"]:
        print("  CHAI's planner mislabels MORE often than the baseline's.")
    else:
        print("  CHAI's planner mislabels less often, but see the per-seed spread")
        print("  before treating that as a real effect.")
    print(f"\n  What actually differs is delivery: {b['shipped_invalid_labels']} false labels reach the user")
    print(f"  from B (in {b['plans_carrying_a_false_label']} of {b['accepted_plans']} accepted plans), "
          f"{c['shipped_invalid_labels']} from C (of {c['accepted_plans']}).")
    print("  That zero is an ENFORCEMENT outcome: prune_unimplemented_operator_claims")
    print("  strips disproven labels before the artifact is written, and it is gated")
    print("  CHAI-only, so it is true by construction.")

    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
