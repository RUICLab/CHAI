#!/usr/bin/env python3
"""Compare plan QUALITY across arms, independently of the performance outcome.

    python3 scripts/plan_quality_report.py experiment_results/<run_id>

Why this exists
---------------
The headline objective (error and power cycles) answers "did the tuned program
run better". It cannot answer "did the planner make a better plan", and on a
benchmark with no exploitable approximation the two questions have completely
different answers: every arm can tie at the identity point while their plans
differ enormously in quality.

Accepted-plan properties below are re-derived from the ACCEPTED artifact rather
than trusted from an in-process verdict. Raw and removed-label counts are the
explicit exception: they are generation-process measures written before CHAI
mutates the artifact. Keeping the two sources separate is what prevents a clean
post-pruning artifact from being misreported as a mistake-free model output.

  hardware-infeasible  operators declared that the target cannot execute
                       (needs an FPU/SIMD the chip lacks). This is the direct
                       measure of hardware-awareness.
  inapplicable         operators declared that the registry says do not apply
                       to this benchmark at all
  raw invalid labels   model declarations the code does not implement, before
                       control-plane pruning (attempt-level process measure)
  labels removed       invalid labels stripped by CHAI while salvaging code
  unsupported shipped  declared operators absent from the accepted artifact
  per-function menus   distinct operator menus across functions; an arm with
                       no per-function selection has exactly one
  knobs/function       plan dimensionality, which the search budget must cover
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.hardware_profile import load_hardware_profile          # noqa: E402
from lib.technique_registry import get_all_techniques           # noqa: E402
from utils.knob_metadata import (                               # noqa: E402
    KnobMetadataError,
    normalize_knob_metadata,
)
from utils.operator_conformance import (                        # noqa: E402
    nonconforming_operator_ids,
)

ARMS = [
    ("checkmate_baseline", "A  CheckMate T1-T20, no control plane"),
    ("operator_only_plus10", "B  CheckMate+10 T1-T30, no control plane"),
    ("chai_plus10", "C  CHAI T1-T30, control plane ON"),
    ("chai_baseline_ops", "D  CHAI T1-T20, control plane ON"),
]


def _load(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def assess_cell(cell_dir, hardware):
    """Derive plan-quality dimensions from one run's accepted artifact."""
    accepted = sorted(glob.glob(os.path.join(
        cell_dir, "outputs", "accepted_approximation_plans_*.json")))
    if not accepted:
        return None
    payload = _load(accepted[0])
    plans = (payload or {}).get("plans") if isinstance(payload, dict) else payload
    if not isinstance(plans, list) or not plans:
        return None

    metrics_files = sorted(glob.glob(os.path.join(
        cell_dir, "logs", "experiment_metrics_*.json")))
    metrics = _load(metrics_files[0]) if metrics_files else {}
    app = (metrics or {}).get("app", "")

    registry = get_all_techniques()
    infeasible = []
    inapplicable = []
    false_claims = []
    knob_counts = []

    for entry in plans:
        name = entry.get("functionName", "?")
        code = entry.get("completeFunction") or ""
        declared = set(entry.get("technique_numbers") or [])

        for tid in sorted(declared):
            technique = registry.get(tid)
            if technique is None:
                continue
            if technique.requires_fpu and not hardware.has_fpu:
                infeasible.append((name, tid, technique.name, "needs FPU"))
            elif technique.requires_simd and not hardware.has_simd:
                infeasible.append((name, tid, technique.name, "needs SIMD"))
            applicable = technique.applicable_apps
            if applicable is not None and app not in applicable:
                inapplicable.append((name, tid, technique.name))

        if code and declared:
            bad = nonconforming_operator_ids(code, declared)
            for tid in sorted(bad):
                false_claims.append((name, tid))

        try:
            variables, _ranges, _steps = normalize_knob_metadata(entry)
            knob_counts.append(len(variables))
        except KnobMetadataError:
            knob_counts.append(0)

    operator_sets = (metrics or {}).get("function_operator_sets") or {}
    distinct_menus = len({tuple(sorted(v)) for v in operator_sets.values()}) \
        if operator_sets else 1

    return {
        "functions": len(plans),
        "hardware_infeasible": infeasible,
        "inapplicable": inapplicable,
        "false_claims": false_claims,
        "raw_false_claims": (metrics or {}).get(
            "raw_operator_claim_violation_count"
        ),
        "raw_claims_checked": (metrics or {}).get(
            "raw_operator_claims_checked"
        ),
        "removed_false_claims": (metrics or {}).get(
            "removed_operator_claim_count"
        ),
        "distinct_menus": distinct_menus,
        "per_function_selection": bool(operator_sets),
        "knobs_total": sum(knob_counts),
        "knobs_max": max(knob_counts) if knob_counts else 0,
        "identity_verified": (metrics or {}).get("identity_gate_mode"),
        "target_build": (metrics or {}).get("target_build_gate_mode"),
        "inert_knob_rejections": (metrics or {}).get("inert_knob_rejections"),
    }


def _fmt_spread(values):
    """Render repeated observations as median plus range, never a lone number.

    The planner is not reproducible at temperature 0, so one observation of a
    cell fixes no magnitude. Showing the spread keeps that visible instead of
    letting a single sample masquerade as a measurement.
    """
    if not values:
        return "  —"
    values = sorted(values)
    low, high = values[0], values[-1]
    mid = values[len(values) // 2]
    if low == high:
        return f"{mid}"
    return f"{mid} [{low}-{high}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--hardware", default="config/hardware_msp430.json")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    hardware = load_hardware_profile(args.hardware)
    report = {"run_dir": args.run_dir, "hardware": hardware.name, "apps": {}}

    apps = sorted({
        app
        for arm, _ in ARMS
        for app in (os.listdir(os.path.join(args.run_dir, arm))
                    if os.path.isdir(os.path.join(args.run_dir, arm)) else [])
        if os.path.isdir(os.path.join(args.run_dir, arm, app))
    })

    print("=" * 104)
    print(f"PLAN QUALITY  —  target {hardware.summary()}")
    print("=" * 104)
    print("  Properties of the plan itself, re-derived from each run's accepted")
    print("  artifact. Repeated cells show  median [min-max]  across repeats.")

    totals = {arm: {"infeasible": 0, "false": 0, "plans": 0, "cells": 0,
                    "raw_false": 0, "raw_checked": 0, "raw_measured": 0,
                    "removed": 0,
                    "removed_measured": 0}
              for arm, _ in ARMS}

    for app in apps:
        print(f"\n{app}")
        print(f"  {'arm':<40} {'reps':>4} {'plans':>10} {'HW-infeas':>11} "
              f"{'shipped bad':>12} {'menus':>7}")
        print("  " + "-" * 92)
        report["apps"][app] = {}
        for arm, label in ARMS:
            cells = sorted(glob.glob(os.path.join(args.run_dir, arm, app, "*")))
            observations = []
            for cell in cells:
                if not os.path.isdir(cell):
                    continue
                assessment = assess_cell(cell, hardware)
                if assessment is not None:
                    observations.append(assessment)
            if not observations:
                print(f"  {label:<40} {len(cells):>4} {'no accepted plan':>10}")
                report["apps"][app][arm] = {"repeats": 0}
                continue

            plans = [o["functions"] for o in observations]
            infeasible = [len(o["hardware_infeasible"]) for o in observations]
            false_claims = [len(o["false_claims"]) for o in observations]
            menus = [o["distinct_menus"] for o in observations]

            totals[arm]["infeasible"] += sum(infeasible)
            totals[arm]["false"] += sum(false_claims)
            totals[arm]["plans"] += sum(plans)
            totals[arm]["cells"] += len(observations)
            for observation in observations:
                raw = observation["raw_false_claims"]
                raw_checked = observation["raw_claims_checked"]
                removed = observation["removed_false_claims"]
                if raw is not None and raw_checked is not None:
                    totals[arm]["raw_false"] += raw
                    totals[arm]["raw_checked"] += raw_checked
                    totals[arm]["raw_measured"] += 1
                if removed is not None:
                    totals[arm]["removed"] += removed
                    totals[arm]["removed_measured"] += 1

            report["apps"][app][arm] = {
                "repeats": len(observations),
                "plans": plans,
                "hardware_infeasible": infeasible,
                "false_claims": false_claims,
                "raw_false_claims": [
                    o["raw_false_claims"] for o in observations
                ],
                "raw_claims_checked": [
                    o["raw_claims_checked"] for o in observations
                ],
                "removed_false_claims": [
                    o["removed_false_claims"] for o in observations
                ],
                "distinct_menus": menus,
                "detail": [
                    {"hardware_infeasible": o["hardware_infeasible"],
                     "false_claims": o["false_claims"]}
                    for o in observations
                ],
            }
            print(f"  {label:<40} {len(observations):>4} "
                  f"{_fmt_spread(plans):>10} {_fmt_spread(infeasible):>11} "
                  f"{_fmt_spread(false_claims):>12} {_fmt_spread(menus):>7}")

        for arm, label in ARMS:
            entry = report["apps"][app].get(arm) or {}
            for observation in entry.get("detail", []):
                for fname, tid, tname, reason in observation["hardware_infeasible"]:
                    print(f"      {label[:1]}  {fname}: T{tid} {tname} — {reason}")

    print("\n" + "=" * 104)
    print("POOLED ACROSS ALL BENCHMARKS AND REPEATS")
    print("=" * 104)
    print(f"  {'arm':<40} {'cells':>6} {'plans':>7} {'HW-infeasible':>15} "
          f"{'raw bad/all*':>12} {'removed*':>10} {'shipped bad':>12}")
    print("  " + "-" * 108)
    for arm, label in ARMS:
        t = totals[arm]
        raw = (f"{t['raw_false']}/{t['raw_checked']}"
               if t["raw_measured"] == t["cells"] else "n/a")
        removed = (str(t["removed"]) if t["removed_measured"] == t["cells"]
                   else "n/a")
        print(f"  {label:<40} {t['cells']:>6} {t['plans']:>7} "
              f"{t['infeasible']:>15} {raw:>12} {removed:>10} "
              f"{t['false']:>12}")
    report["pooled"] = totals

    print("\n" + "=" * 100)
    print("READING THIS TABLE")
    print("=" * 100)
    print("  HW-infeasible is the direct hardware-awareness measure: an operator")
    print("  the target physically cannot run. CHAI's gatekeeper removes these")
    print("  before the model ever sees them, so a non-zero count in an arm with")
    print("  the control plane ON would be a defect, not a design choice.")
    print()
    print("  * raw bad and removed are available only in campaigns run after the")
    print("  pre-pruning instrumentation was added. They are attempt-level counts.")
    print("  shipped bad is re-derived from accepted artifacts and is a LOWER")
    print("  BOUND: a checker cannot prove a transformation is genuinely present.")
    print("  A zero for CHAI is an enforcement outcome, not evidence that its")
    print("  underlying model or planner originated no false labels.")
    print()
    print("  menus = 1 means every function was offered the identical operator")
    print("  list. Greater than 1 means the plan was specialised per function.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=str)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
