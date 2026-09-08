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
ERROR_BOUND="${ERROR_BOUND:-}"
AUTO_BUDGET="${AUTO_BUDGET:-1}"
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
echo "Replaying plans from $SWEEP -> $OUT"
echo "BO calls: $BO_CALLS   trace: $TRACE   capacitor: $CAPACITORS"
echo

total=0; replayed=0; skipped=0
for planfile in "$SWEEP"/*/*/seed_*/outputs/accepted_approximation_plans_*.json; do
  [ -e "$planfile" ] || continue
  total=$((total+1))
  arm="$(echo "$planfile" | awk -F/ '{print $2}')"
  app="$(echo "$planfile" | awk -F/ '{print $3}')"
  seed="$(echo "$planfile" | awk -F/ '{print $4}')"

  # Skip cells that produced no usable plan; there is nothing to tune.
  n_plans=$(python3 -c "
import json,sys
try:
    d=json.load(open('$planfile'))
    p=d.get('plans') if isinstance(d,dict) else d
    print(len([x for x in (p or []) if x.get('knobVariables')]))
except Exception: print(0)")
  if [ "$n_plans" = "0" ]; then skipped=$((skipped+1)); continue; fi

  dest="$OUT/$arm/$app/$seed"
  mkdir -p "$dest/logs" "$dest/outputs"

  # Stage the saved plan where --prerun_dir expects it.
  rm -rf "$STAGE"; mkdir -p "$STAGE/approximated_functions"
  python3 - "$planfile" "$STAGE/approximated_functions/apx_all.json" <<'PYEOF'
import json, sys
raw = json.load(open(sys.argv[1]))
plans = raw.get("plans") if isinstance(raw, dict) else raw
# BayesOpt consumes a bare list of plan entries.
json.dump([p for p in plans if p.get("knobVariables")], open(sys.argv[2], "w"), indent=2)
PYEOF

  echo "[$(date +%H:%M:%S)] $arm / $app / $seed  ($n_plans function(s))"
  rm -rf logs outputs; mkdir -p logs outputs
  set +e
  docker run --rm --label chai.experiment=true \
    -e "CHAI_BO_CALLS=$BO_CALLS" -e "CHAI_BO_SEED=2026" \
    -e "CHAI_REQUIRE_IMPROVEMENT=1" -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
    -e "CHAI_BO_AUTO_BUDGET=$AUTO_BUDGET" \
    ${ERROR_BOUND:+-e "CHAI_ERROR_BOUND=$ERROR_BOUND"} \
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
  python3 - "$dest/status.json" <<PYEOF
import json, sys, datetime
json.dump({"app": "$app", "arm": "$arm", "seed": "$seed", "exit_code": $code,
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
