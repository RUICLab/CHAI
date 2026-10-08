#!/usr/bin/env bash
# One fresh planner repetition x six apps x four arms = 24 preflight cells.
# Each cell runs 250 BO calls; analyze_bo_budget_curve.py recovers matched
# 20/40/80/160/250 best-so-far curves without rerunning the planner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SEEDS="${SEEDS:-2026}"
export BO_CALLS="${BO_CALLS:-250}"
export BO_AUTO_BUDGET=0
export GENERATION_ONLY=0
export OBJ_ERROR_WEIGHT="${OBJ_ERROR_WEIGHT:-0}"
export ALLOW_DIRTY_TREE="${ALLOW_DIRTY_TREE:-1}"
export RUN_ID="${RUN_ID:-preflight_$(date +%Y%m%d_%H%M%S)}"

"$SCRIPT_DIR/run_ablation_campaign.sh"
python3 "$SCRIPT_DIR/analyze_bo_budget_curve.py" \
  "$SCRIPT_DIR/../experiment_results/$RUN_ID"
