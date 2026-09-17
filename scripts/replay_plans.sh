#!/usr/bin/env bash
#
# Re-run already-generated plans through BayesOpt + the Fused simulator.
#
#   bash scripts/replay_plans.sh <sweep_dir> [run_id]
#
# Generation-only sweeps produced accepted plans that cost real model calls but
# were never measured for performance: they skipped BO and Fused entirely. Every
# accepted artifact already carries what BO consumes -- completeFunction,
# knobVariables, knobRanges, knobStepSize -- so the plans can be replayed
# without spending a single new model call.
#
# This is also the only way to answer "does CHAI produce FASTER code": every
# previous performance run used the 7B, which mostly could not produce a valid
# plan at all, so there was nothing to tune.
#
# Uses main.py's existing --no_llm / --prerun_dir path; no code changes.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SWEEP="${1:?usage: replay_plans.sh <sweep_dir> [run_id]}"
RUN_ID="${2:-replay_$(date +%Y%m%d_%H%M%S)}"
TRACE="${TRACE:-traces/RF_2.csv}"
CAPACITORS="${CAPACITORS:-68e-6}"
HARDWARE="${HARDWARE:-config/hardware_msp430.json}"
BO_CALLS="${BO_CALLS:-40}"
# CheckMate used 0.50; this fork tightened it to 0.30. At 0.30 every approximate
# configuration on ar-iclib was rejected and only the identity point survived,
# so the bound itself may be what suppresses any performance result.
# utils/error_analyzer.py gives every benchmark its own bound with a written
# justification, and that per-benchmark value overrides the global
# CHAI_ERROR_BOUND. Forwarding only the global one silently left ar-iclib at
# 0.30 while the run claimed to be testing 0.50, so send the per-benchmark
# variable too.
ERROR_BOUND="${ERROR_BOUND:-}"
# A comparison needs the same number of evaluations in every arm. Dimension-
# scaled budgets are useful for exploratory tuning, but gave the unrestricted
# 19-knob plan 82 calls while a 5-knob CHAI plan received 40.
AUTO_BUDGET="${AUTO_BUDGET:-0}"
# The objective adds error/bound on top of a HARD reject at error > bound, so
# error is charged twice. A candidate must satisfy error/bound + cycle_ratio < 1
# to beat doing nothing, which discarded a verified 73.7% power-cycle saving at
# error 0.418 under a 0.5 bound. Set to 0 for the constrained formulation the
# function name promises: minimize power cycles subject to error <= bound.
# Applies to every arm identically.
OBJ_ERROR_WEIGHT="${OBJ_ERROR_WEIGHT:-}"
BO_SEED_BASE="${BO_SEED_BASE:-2026}"
# Optional whitespace-separated filters make a single-cell smoke test possible
# without copying or editing the source sweep.
REPLAY_ARMS="${REPLAY_ARMS:-}"
REPLAY_APPS="${REPLAY_APPS:-}"
REPLAY_SEEDS="${REPLAY_SEEDS:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-chai-dev:local}"
OUT="experiment_results/$RUN_ID"
STAGE=".replay_stage"

arm_flags() {
  case "$1" in
    checkmate_baseline)   echo "--operator_catalog checkmate_20      --hardware $HARDWARE --hardware_control prompt_only" ;;
    operator_only_plus10) echo "--operator_catalog checkmate_plus_10 --hardware $HARDWARE --hardware_control prompt_only" ;;
    chai_plus10)          echo "--operator_catalog checkmate_plus_10 --hardware $HARDWARE --hardware_control chai" ;;
    chai_baseline_ops)    echo "--operator_catalog checkmate_20      --hardware $HARDWARE --hardware_control chai" ;;
  esac
}

mkdir -p "$OUT"
python3 - "$OUT/manifest.json" "$SWEEP/manifest.json" <<PYEOF
import datetime, json, os, subprocess, sys
source = {}
if os.path.exists(sys.argv[2]):
    source = json.load(open(sys.argv[2]))
json.dump({
    "run_id": "$RUN_ID",
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "design": "performance replay of generation-only plans",
    "replayed_from": "$SWEEP",
    "apps": source.get("apps", []),
    "arms": source.get("arms", []),
    "source_planner_repeats": source.get("planner_repeats", source.get("seeds", [])),
    "bo_seed_base": int("$BO_SEED_BASE"),
    "bo_calls": int("$BO_CALLS"),
    "bo_auto_budget": "$AUTO_BUDGET" not in ("0", "false", "no", "off"),
    "generation_only": False,
    "trace": "$TRACE",
    "capacitors": "$CAPACITORS".split(","),
    "hardware": "$HARDWARE",
    "target_build_gate": source.get("target_build_gate", "strict"),
    "canonical_backends": source.get("canonical_backends", "off"),
    "error_bound_override": "$ERROR_BOUND" or None,
    "objective_error_weight": "$OBJ_ERROR_WEIGHT" or None,
    "source_provenance": source.get("provenance", {}),
    "replay_provenance": {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_dirty_files": len(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines()),
        "docker_image": "$DOCKER_IMAGE",
    },
}, open(sys.argv[1], "w"), indent=2)
PYEOF
echo "Replaying plans from $SWEEP -> $OUT"
echo "BO calls: $BO_CALLS   trace: $TRACE   capacitor: $CAPACITORS"
echo

total=0; replayed=0; skipped=0
for planfile in "$SWEEP"/*/*/seed_*/outputs/accepted_approximation_plans_*.json; do
  [ -e "$planfile" ] || continue
  # Parse relative to the sweep root, not absolute field positions: the sweep
  # directory itself contributes a path component, which shifted every field by
  # one and passed the arm name to --bm_name.
  rel="${planfile#"$SWEEP"/}"
  arm="$(echo "$rel" | awk -F/ '{print $1}')"
  app="$(echo "$rel" | awk -F/ '{print $2}')"
  seed="$(echo "$rel" | awk -F/ '{print $3}')"
  seed_number="${seed#seed_}"
  case " $REPLAY_ARMS " in *" $arm "*) ;; *) if [ -n "$REPLAY_ARMS" ]; then continue; fi ;; esac
  case " $REPLAY_APPS " in *" $app "*) ;; *) if [ -n "$REPLAY_APPS" ]; then continue; fi ;; esac
  case " $REPLAY_SEEDS " in *" $seed_number "*) ;; *) if [ -n "$REPLAY_SEEDS" ]; then continue; fi ;; esac
  total=$((total+1))
  if [ "$seed_number" -ge 1000 ] 2>/dev/null; then
    bo_seed="$seed_number"
  else
    bo_seed=$((BO_SEED_BASE + seed_number - 1))
  fi
  source_cell="$(dirname "$(dirname "$planfile")")"
  dest="$OUT/$arm/$app/$seed"
  mkdir -p "$dest/logs" "$dest/outputs"

  # Skip cells that produced no usable plan; there is nothing to tune.
  n_plans=$(python3 -c "
import json,sys
try:
    d=json.load(open('$planfile'))
    p=d.get('plans') if isinstance(d,dict) else d
    print(len([x for x in (p or []) if x.get('knobVariables')]))
except Exception: print(0)")
  if [ "$n_plans" = "0" ]; then
    cp -r "$source_cell/outputs/." "$dest/outputs/" 2>/dev/null || true
    for metric in "$source_cell"/logs/experiment_metrics_*.json; do
      [ -e "$metric" ] && cp "$metric" "$dest/logs/"
    done
    python3 - "$dest/status.json" <<PYEOF
import datetime, json, sys
json.dump({"app": "$app", "arm": "$arm", "seed": $seed_number,
           "bo_seed": $bo_seed, "exit_code": 0, "state": "completed",
           "reason": "source cell produced no tunable plan",
           "replayed_from": "$planfile",
           "completed_at": datetime.datetime.utcnow().isoformat() + "Z"},
          open(sys.argv[1], "w"), indent=2)
PYEOF
    skipped=$((skipped+1))
    continue
  fi

  # Stage the saved plan where --prerun_dir expects it.
  rm -rf "$STAGE"; mkdir -p "$STAGE/approximated_functions"
  python3 - "$planfile" "$STAGE/approximated_functions/apx_all.json" <<'PYEOF'
import json, sys
raw = json.load(open(sys.argv[1]))
plans = raw.get("plans") if isinstance(raw, dict) else raw
# BayesOpt consumes a bare list of plan entries.
json.dump([p for p in plans if p.get("knobVariables")], open(sys.argv[2], "w"), indent=2)
PYEOF

  echo "[$(date +%H:%M:%S)] $arm / $app / $seed  ($n_plans function(s), BO seed $bo_seed)"
  rm -rf logs outputs; mkdir -p logs outputs
  set +e
  docker run --rm --label chai.experiment=true \
    -e "CHAI_BO_CALLS=$BO_CALLS" -e "CHAI_BO_SEED=$bo_seed" \
    -e "CHAI_REQUIRE_IMPROVEMENT=1" -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
    -e "CHAI_BO_AUTO_BUDGET=$AUTO_BUDGET" \
    ${OBJ_ERROR_WEIGHT:+-e "CHAI_OBJECTIVE_ERROR_WEIGHT=$OBJ_ERROR_WEIGHT"} \
    ${ERROR_BOUND:+-e "CHAI_ERROR_BOUND=$ERROR_BOUND"} \
    ${ERROR_BOUND:+-e "CHAI_ERROR_BOUND_$(echo "$app" | tr 'a-z-' 'A-Z_')=$ERROR_BOUND"} \
    -v "$PWD:/workspace" -w /workspace \
    "$DOCKER_IMAGE" python3 -u main.py \
      --bm_name "$app" --trace "$TRACE" --capacitors "$CAPACITORS" \
      --no_llm --prerun_dir "$STAGE" \
      $(arm_flags "$arm") \
    > "$dest/console.log" 2>&1
  code=$?
  set -e
  cp -r logs/. "$dest/logs/" 2>/dev/null || true
  cp -r outputs/. "$dest/outputs/" 2>/dev/null || true
  # Generation-time validity/provenance metrics remain applicable to the saved
  # plan. main.py intentionally skips regenerating them under --no_llm.
  for metric in "$source_cell"/logs/experiment_metrics_*.json; do
    [ -e "$metric" ] && cp "$metric" "$dest/logs/"
  done
  python3 - "$dest/status.json" <<PYEOF
import json, sys, datetime
json.dump({"app": "$app", "arm": "$arm", "seed": $seed_number,
           "bo_seed": $bo_seed, "exit_code": $code,
           "state": "completed" if $code == 0 else "failed",
           "replayed_from": "$planfile",
           "completed_at": datetime.datetime.utcnow().isoformat() + "Z"},
          open(sys.argv[1], "w"), indent=2)
PYEOF
  replayed=$((replayed+1))
  echo "    exit=$code"
done

rm -rf "$STAGE"
echo
echo "Replayed $replayed of $total cells ($skipped had no usable plan) -> $OUT"
echo "Analyse: python3 scripts/analyze_ablation.py $OUT"
