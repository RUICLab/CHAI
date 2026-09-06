#!/usr/bin/env python3
"""Aggregate a 2x2 ablation into main effects, an interaction, and validity rates.

    python3 scripts/analyze_ablation.py experiment_results/<run_id>

Design
------
    A checkmate_baseline    T1-T20, control plane off
    B operator_only_plus10  T1-T30, control plane off
    C chai_plus10           T1-T30, control plane on
    D chai_baseline_ops     T1-T20, control plane on

The earlier version reported simple contrasts (B-A, C-B, D-A). In a 2x2 those
are *simple effects*, each estimated at one level of the other factor. The
correct summaries average over both levels:

    operator main effect  = ((B - A) + (C - D)) / 2
    control  main effect  = ((D - A) + (C - B)) / 2
    interaction           = (C - B) - (D - A)

Validity rules enforced here
----------------------------
* A cell whose ``status.json`` reports a non-zero exit is invalid.
* A cell missing a verified MSP430 target build is invalid when the run
  requested the strict gate.
* A cell in which a canonical hand-written backend replaced generated code is
  reported separately and excluded from planner comparisons.
* Cells that produced no valid plan or no improvement are NOT dropped: they are
  counted in the validity rates, and effect sizes are computed on paired seeds
  only, where every arm produced a comparable number for that (app, seed).
* A run with fewer than the expected seeds is reported as incomplete.
"""

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys

ARMS = [
    ("checkmate_baseline", "A  T1-T20, control off"),
    ("operator_only_plus10", "B  T1-T30, control off"),
    ("chai_plus10", "C  T1-T30, control ON"),
    ("chai_baseline_ops", "D  T1-T20, control ON"),
]
ARM_IDS = [arm for arm, _ in ARMS]


# ---------------------------------------------------------------------------
#  Cell loading
# ---------------------------------------------------------------------------

def _load_json(path):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def read_cell(cell_dir):
    """Return one (app, arm, seed) cell's outcome and validity."""
    cell = {
        "dir": cell_dir,
        "valid": False,
        "invalid_reason": None,
        "exit_code": None,
        "objective": None,
        "error": None,
        "power_cycles": None,
        "power_cycle_ratio": None,
        "bo_outcome": None,
        "identity_verified": None,
        "produced_configuration": False,
        "canonical_backend": False,
        "target_build_verified": None,
        "api_calls": None,
        "max_prompt_tokens": None,
    }

    status = _load_json(os.path.join(cell_dir, "status.json"))
    if status is None:
        cell["invalid_reason"] = "no status.json (cell never ran)"
        return cell
    cell["exit_code"] = status.get("exit_code")
    if cell["exit_code"] not in (0, None):
        cell["invalid_reason"] = f"runner exit {cell['exit_code']}"
        return cell

    logs = os.path.join(cell_dir, "logs")
    metric_files = sorted(glob.glob(os.path.join(logs, "experiment_metrics_*.json")))
    if not metric_files:
        cell["invalid_reason"] = "no experiment_metrics_*.json"
        return cell
    if len(metric_files) > 1:
        cell["invalid_reason"] = (
            f"{len(metric_files)} metrics files present; cannot attribute the "
            "result to one configuration"
        )
        return cell
    metrics = _load_json(metric_files[0]) or {}

    cell["api_calls"] = metrics.get("api_calls")
    cell["conformance_violations"] = metrics.get(
        "operator_conformance_violation_count", 0
    )
    cell["conformance_enforced"] = metrics.get("operator_conformance_enforced")
    cell["functions_attempted"] = metrics.get("functions_attempted", 0)
    cell["functions_succeeded"] = metrics.get("functions_succeeded", 0)
    measured = metrics.get("context_budget_measured") or {}
    cell["max_prompt_tokens"] = measured.get("max_prompt_tokens")

    if metrics.get("canonical_backends") == "on" or metrics.get(
        "canonical_backend_used"
    ):
        cell["canonical_backend"] = True

    gate = metrics.get("target_build_gate_mode")
    statuses = (metrics.get("target_build_status") or {}).values()
    verified = [s.get("status") for s in statuses] if statuses else []
    if gate == "strict":
        cell["target_build_verified"] = bool(verified) and all(
            s == "passed" for s in verified
        )
    elif gate in ("advisory", "off", None):
        cell["target_build_verified"] = None if not verified else all(
            s == "passed" for s in verified
        )

    if metrics.get("run_outcome") == "no_valid_plan":
        cell["valid"] = True
        cell["bo_outcome"] = "no_valid_plan"
        return cell

    best_files = sorted(glob.glob(os.path.join(logs, "best_knobs_*.csv")))
    if not best_files:
        cell["valid"] = True
        cell["bo_outcome"] = "no_result_file"
        return cell
    if len(best_files) > 1:
        cell["invalid_reason"] = f"{len(best_files)} best_knobs files present"
        return cell

    with open(best_files[0], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        cell["valid"] = True
        cell["bo_outcome"] = "no_valid_configuration"
        return cell
    if len(rows) > 1:
        cell["invalid_reason"] = (
            f"{len(rows)} result rows; this analyser expects one operating "
            "point per cell (use one trace x one capacitor per run)"
        )
        return cell

    row = rows[0]

    def number(*keys):
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except ValueError:
                    return None
        return None

    cell["valid"] = True
    cell["produced_configuration"] = True
    cell["error"] = number("error")
    cell["power_cycles"] = number("power_cycles", "checkpoints")
    cell["power_cycle_ratio"] = number("power_cycle_ratio", "checkpoint_ratio")
    cell["objective"] = number("optimization_metric")
    cell["bo_outcome"] = row.get("bo_outcome")
    cell["identity_verified"] = row.get("identity_verified")
    return cell


def collect(root, expected_seeds=None):
    data = {}
    for arm in ARM_IDS:
        arm_dir = os.path.join(root, arm)
        if not os.path.isdir(arm_dir):
            continue
        for app in sorted(os.listdir(arm_dir)):
            app_dir = os.path.join(arm_dir, app)
            if not os.path.isdir(app_dir):
                continue
            for seed_dir in sorted(os.listdir(app_dir)):
                if not seed_dir.startswith("seed_"):
                    continue
                cell = read_cell(os.path.join(app_dir, seed_dir))
                cell["seed"] = seed_dir.replace("seed_", "")
                data.setdefault(app, {}).setdefault(arm, {})[cell["seed"]] = cell
    return data


# ---------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------

def mean_ci(values, confidence=0.95):
    """Return ``(mean, half_width)``; half-width is ``None`` below 2 samples.

    Uses a normal approximation, which is all a handful of seeds supports. It is
    reported so the reader can see the spread, not to claim significance.
    """
    if not values:
        return None, None
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, None
    stderr = statistics.stdev(values) / math.sqrt(len(values))
    critical = 1.96 if confidence == 0.95 else 1.64
    return mean, critical * stderr


def paired_seeds(app_data, arms_required):
    """Seeds for which every required arm produced a comparable objective."""
    seed_sets = []
    for arm in arms_required:
        cells = app_data.get(arm, {})
        seed_sets.append({
            seed for seed, cell in cells.items()
            if cell["valid"] and cell["objective"] is not None
            and not cell["canonical_backend"]
        })
    if not seed_sets:
        return []
    return sorted(set.intersection(*seed_sets))


def fmt(value, digits=4):
    return "  n/a" if value is None else f"{value:.{digits}f}"


def fmt_ci(mean, half):
    if mean is None:
        return "n/a"
    if half is None:
        return f"{mean:+.4f}"
    return f"{mean:+.4f} +/-{half:.4f}"


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--expected-seeds", type=int, default=None,
                        help="Required seed count per cell; defaults to the manifest.")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.exit(f"No such campaign directory: {args.run_dir}")

    manifest = _load_json(os.path.join(args.run_dir, "manifest.json")) or {}
    expected_seeds = args.expected_seeds or len(manifest.get("seeds", []) or []) or None

    data = collect(args.run_dir)
    if not data:
        sys.exit(f"No arm directories found under {args.run_dir}")

    report = {
        "run_dir": args.run_dir,
        "manifest": manifest,
        "expected_seeds": expected_seeds,
        "apps": {},
        "effects": {},
        "warnings": [],
    }

    if manifest.get("canonical_backends") == "on":
        report["warnings"].append(
            "Campaign ran with canonical_backends=on: results measure "
            "hand-written benchmark-specific backends, not the planner."
        )
    if manifest.get("target_build_gate") != "strict":
        report["warnings"].append(
            f"target_build_gate={manifest.get('target_build_gate')!r}: candidates "
            "were not required to pass a verified MSP430 build."
        )
    dirty = (manifest.get("provenance") or {}).get("git_dirty_files")
    if dirty:
        report["warnings"].append(
            f"{dirty} uncommitted file(s) at run time; config_hash "
            f"{(manifest.get('provenance') or {}).get('config_hash')} pins what ran."
        )

    # ---- Per-benchmark validity and outcomes ------------------------------
    print("=" * 86)
    print("CELL VALIDITY AND OUTCOMES")
    print("=" * 86)
    for app in sorted(data):
        print(f"\n{app}")
        print(f"  {'arm':<26} {'cells':>5} {'valid':>6} {'config':>7} "
              f"{'objective (mean +/- 95% CI)':>30}")
        print("  " + "-" * 80)
        report["apps"][app] = {}
        for arm, label in ARMS:
            cells = list(data[app].get(arm, {}).values())
            if not cells:
                print(f"  {label:<26} {'not run':>5}")
                continue
            valid = [c for c in cells if c["valid"]]
            invalid = [c for c in cells if not c["valid"]]
            with_config = [c for c in valid if c["produced_configuration"]]
            objectives = [c["objective"] for c in with_config
                          if c["objective"] is not None and not c["canonical_backend"]]
            mean, half = mean_ci(objectives)
            outcomes = sorted({c["bo_outcome"] or "invalid" for c in cells})

            report["apps"][app][arm] = {
                "cells": len(cells),
                "valid": len(valid),
                "invalid": [
                    {"seed": c["seed"], "reason": c["invalid_reason"]} for c in invalid
                ],
                "produced_configuration": len(with_config),
                "objective_mean": mean,
                "objective_ci95": half,
                "outcomes": outcomes,
                "canonical_backend_cells": sum(1 for c in cells if c["canonical_backend"]),
                "max_prompt_tokens": max(
                    (c["max_prompt_tokens"] for c in cells if c["max_prompt_tokens"]),
                    default=None,
                ),
                "operator_mislabels": sum(
                    c.get("conformance_violations") or 0 for c in valid
                ),
                "functions_attempted": sum(
                    c.get("functions_attempted") or 0 for c in valid
                ),
                "functions_succeeded": sum(
                    c.get("functions_succeeded") or 0 for c in valid
                ),
            }
            print(f"  {label:<26} {len(cells):>5} {len(valid):>6} "
                  f"{len(with_config):>7} {fmt_ci(mean, half):>30}")
            for cell in invalid:
                print(f"      seed {cell['seed']}: INVALID — {cell['invalid_reason']}")
            if expected_seeds and len(cells) < expected_seeds:
                print(f"      INCOMPLETE: {len(cells)}/{expected_seeds} seeds")
                report["warnings"].append(
                    f"{app}/{arm}: only {len(cells)}/{expected_seeds} seeds present"
                )
            backend_cells = [c for c in cells if c["canonical_backend"]]
            if backend_cells:
                print(f"      {len(backend_cells)} cell(s) used a canonical "
                      "hand-written backend — excluded from planner effects")
            if arm.startswith("chai"):
                unverified = [
                    c for c in valid
                    if c["target_build_verified"] is False
                    and c["produced_configuration"]
                ]
                if unverified:
                    print(f"      {len(unverified)} cell(s) without a verified "
                          "target build")

    # ---- Operator provenance -----------------------------------------------
    print("\n" + "=" * 86)
    print("OPERATOR PROVENANCE  (declared operators vs. what the code implements)")
    print("=" * 86)
    print("  Measured in every arm; ENFORCED only where the control plane is on.")
    print(f"\n  {'benchmark':<18} {'arm':<26} {'enforced':>9} {'mislabels':>10} "
          f"{'funcs ok':>10}")
    print("  " + "-" * 78)
    for app in sorted(data):
        for arm, label in ARMS:
            summary_row = report["apps"].get(app, {}).get(arm)
            if not summary_row:
                continue
            cells = [c for c in data[app].get(arm, {}).values() if c["valid"]]
            enforced = any(c.get("conformance_enforced") for c in cells)
            print(f"  {app:<18} {label:<26} {str(bool(enforced)):>9} "
                  f"{summary_row['operator_mislabels']:>10} "
                  f"{summary_row['functions_succeeded']}/"
                  f"{summary_row['functions_attempted']:<8}")
    print("\n  A high unenforced mislabel count is evidence FOR the control plane:")
    print("  it is the amount of false operator provenance the baseline emits and")
    print("  reports as fact.")

    # ---- 2x2 effects on paired seeds --------------------------------------
    print("\n" + "=" * 86)
    print("2x2 EFFECTS ON PAIRED SEEDS  (negative = improvement in the objective)")
    print("=" * 86)
    print(f"\n  {'benchmark':<20} {'n':>3} {'operator main':>18} "
          f"{'control main':>18} {'interaction':>16}")
    print("  " + "-" * 80)

    operator_all, control_all, interaction_all = [], [], []

    for app in sorted(data):
        seeds = paired_seeds(data[app], ARM_IDS)
        if not seeds:
            print(f"  {app:<20} {0:>3}   no seed produced a comparable objective "
                  "in all four arms")
            report["effects"][app] = {"paired_seeds": 0}
            continue

        per_seed = {"operator": [], "control": [], "interaction": []}
        for seed in seeds:
            a = data[app]["checkmate_baseline"][seed]["objective"]
            b = data[app]["operator_only_plus10"][seed]["objective"]
            c = data[app]["chai_plus10"][seed]["objective"]
            d = data[app]["chai_baseline_ops"][seed]["objective"]
            per_seed["operator"].append(((b - a) + (c - d)) / 2)
            per_seed["control"].append(((d - a) + (c - b)) / 2)
            per_seed["interaction"].append((c - b) - (d - a))

        op_mean, op_half = mean_ci(per_seed["operator"])
        ct_mean, ct_half = mean_ci(per_seed["control"])
        ix_mean, ix_half = mean_ci(per_seed["interaction"])

        operator_all.extend(per_seed["operator"])
        control_all.extend(per_seed["control"])
        interaction_all.extend(per_seed["interaction"])

        report["effects"][app] = {
            "paired_seeds": len(seeds),
            "seeds": seeds,
            "operator_main_effect": {"mean": op_mean, "ci95": op_half,
                                     "per_seed": per_seed["operator"]},
            "control_main_effect": {"mean": ct_mean, "ci95": ct_half,
                                    "per_seed": per_seed["control"]},
            "interaction": {"mean": ix_mean, "ci95": ix_half,
                            "per_seed": per_seed["interaction"]},
        }
        print(f"  {app:<20} {len(seeds):>3} {fmt_ci(op_mean, op_half):>18} "
              f"{fmt_ci(ct_mean, ct_half):>18} {fmt_ci(ix_mean, ix_half):>16}")

    print("  " + "-" * 80)
    op_mean, op_half = mean_ci(operator_all)
    ct_mean, ct_half = mean_ci(control_all)
    ix_mean, ix_half = mean_ci(interaction_all)
    print(f"  {'pooled':<20} {len(operator_all):>3} {fmt_ci(op_mean, op_half):>18} "
          f"{fmt_ci(ct_mean, ct_half):>18} {fmt_ci(ix_mean, ix_half):>16}")

    report["effects"]["_pooled"] = {
        "n": len(operator_all),
        "operator_main_effect": {"mean": op_mean, "ci95": op_half},
        "control_main_effect": {"mean": ct_mean, "ci95": ct_half},
        "interaction": {"mean": ix_mean, "ci95": ix_half},
    }

    # ---- Interpretation ---------------------------------------------------
    print("\n" + "=" * 86)
    print("INTERPRETATION")
    print("=" * 86)
    print("  operator main effect : ((B-A) + (C-D)) / 2  — the ten added operators")
    print("  control main effect  : ((D-A) + (C-B)) / 2  — the hardware-aware control plane")
    print("  interaction          : (C-B) - (D-A)        — do the two compound?")
    print()
    if ct_mean is None or op_mean is None:
        print("  Not enough paired data to state a conclusion.")
    elif ct_mean < op_mean:
        print("  The control plane shows the larger improvement. The claim that the")
        print("  result is not merely the ten added operators is SUPPORTED by these")
        if ct_half is not None and abs(ct_mean) < ct_half:
            print("  point estimates, but the control effect's CI spans zero — the")
            print("  effect is not distinguishable from noise at this seed count.")
        else:
            print("  data.")
    else:
        print("  The operator catalogue shows an improvement at least as large as the")
        print("  control plane. These data do NOT support the claim that the control")
        print("  plane is what carries the result.")

    if report["warnings"]:
        print("\n" + "=" * 86)
        print("WARNINGS")
        print("=" * 86)
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
