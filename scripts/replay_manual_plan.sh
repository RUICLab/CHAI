#!/usr/bin/env bash
# Replay one explicitly labeled manual/Codex implementation without any LLM.
set -euo pipefail
export LC_ALL=C
export LANG=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PLAN="${1:?usage: replay_manual_plan.sh <plan.json> <app> [run_id]}"
APP="${2:?usage: replay_manual_plan.sh <plan.json> <app> [run_id]}"
RUN_ID="${3:-manual_replay_$(date +%Y%m%d_%H%M%S)}"
BO_CALLS="${BO_CALLS:-250}"
BO_SEED="${BO_SEED:-2026}"
TRACE="${TRACE:-traces/RF_2.csv}"
CAPACITORS="${CAPACITORS:-68e-6}"
HARDWARE="${HARDWARE:-config/hardware_msp430.json}"
DOCKER_IMAGE="${DOCKER_IMAGE:-chai-dev:local}"
OUT="experiment_results/$RUN_ID/codex_implementation/$APP/seed_$BO_SEED"
STAGE="$(mktemp -d "$REPO_ROOT/.manual_replay.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT INT TERM
mkdir -p "$STAGE/approximated_functions" "$OUT/logs" "$OUT/outputs"
cp "$PLAN" "$STAGE/approximated_functions/apx_all.json"
STAGE_REL="${STAGE#"$REPO_ROOT/"}"

python3 - "$PLAN" <<'PYEOF'
import json, sys
from utils.knob_metadata import validate_plan_entries
from utils.operator_conformance import validate_operator_conformance

plans = json.load(open(sys.argv[1]))
validate_plan_entries(plans)
for plan in plans:
    error = validate_operator_conformance(
        plan["completeFunction"], set(plan.get("technique_numbers", []))
    )
    if error:
        raise SystemExit(error)
print("manual plan passed metadata and operator-conformance validation")
PYEOF

rm -rf logs outputs
mkdir -p logs outputs
set +e
docker run --rm --label chai.experiment=true \
  -e "CHAI_BO_CALLS=$BO_CALLS" -e "CHAI_BO_SEED=$BO_SEED" \
  -e "CHAI_BO_AUTO_BUDGET=0" -e "CHAI_OBJECTIVE_ERROR_WEIGHT=0" \
  -e "CHAI_REQUIRE_IMPROVEMENT=1" -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
  -v "$REPO_ROOT:/workspace" -w /workspace \
  "$DOCKER_IMAGE" python3 -u main.py \
    --bm_name "$APP" --trace "$TRACE" --capacitors "$CAPACITORS" \
    --no_llm --prerun_dir "$STAGE_REL" \
    --hardware "$HARDWARE" --hardware_control chai \
    --operator_catalog checkmate_20 \
  > "$OUT/console.log" 2>&1
CODE=$?
set -e
cp -r logs/. "$OUT/logs/" 2>/dev/null || true
cp -r outputs/. "$OUT/outputs/" 2>/dev/null || true
python3 - "$OUT/status.json" <<PYEOF
import datetime, json, sys
json.dump({
  "app": "$APP", "arm": "codex_implementation", "bo_seed": $BO_SEED,
  "bo_calls": $BO_CALLS, "exit_code": $CODE,
  "state": "completed" if $CODE == 0 else "failed",
  "plan": "$PLAN", "planner": "Qwen-32B", "implementer": "Codex",
  "completed_at": datetime.datetime.utcnow().isoformat() + "Z"
}, open(sys.argv[1], "w"), indent=2)
PYEOF
echo "exit=$CODE -> $OUT"
exit "$CODE"
