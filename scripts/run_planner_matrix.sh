#!/usr/bin/env bash
#
# The 2x2 that separates model size from planner configuration.
#
#                 our settings          CheckMate settings
#   7B            (already collected)   variant A
#   32B           variant B             variant C
#
# "CheckMate settings" are upstream SYSNET-LUMS/CheckMate's: error bound 0.50,
# no technique-summary context, no few-shot examples, and a retry budget that
# does not give up after three attempts. Upstream ran those with a frontier
# model and reported good results; this fork changed all of them at once AND
# dropped to a 7B, so a failure cannot currently be attributed to either.
#
# Runs sequentially: one Ollama model is resident at a time, so overlapping
# sweeps would thrash the GPU rather than finish sooner.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

APPS="${APPS:-ar-iclib sobel-iclib fft-iclib lqi-iclib}"
SEEDS="${SEEDS:-1 2 3}"
TUNNEL="${OLLAMA_SSH_TUNNEL:-nsola5@131.193.50.170}"
STAMP="$(date +%Y%m%d_%H%M%S)"

run_variant() {  # run_variant <name> <model> <checkmate_settings>
  local name="$1" model="$2" cm="$3"
  local rid="matrix_${name}_${STAMP}"
  echo "=============================================================="
  echo "VARIANT $name   model=$model  checkmate_settings=$cm"
  echo "=============================================================="
  GENERATION_ONLY=1 CHECKMATE_SETTINGS="$cm" MODEL="$model" \
  APPS="$APPS" SEEDS="$SEEDS" BO_CALLS=8 BO_AUTO_BUDGET=0 \
  OLLAMA_SSH_TUNNEL="$TUNNEL" RUN_ID="$rid" \
    bash scripts/run_ablation_campaign.sh 2>&1 | tail -5
  echo "$rid" >> /tmp/chai_matrix_runs
  echo
}

: > /tmp/chai_matrix_runs
run_variant "32b_ourprompts"  "chai-qwen2.5-coder:32b-24k" 0
run_variant "32b_checkmate"   "chai-qwen2.5-coder:32b-24k" 1

echo "=============================================================="
echo "MATRIX COMPLETE. Runs:"
cat /tmp/chai_matrix_runs
echo
echo "Compare with:"
while read -r r; do echo "  python3 scripts/plan_quality_report.py experiment_results/$r"; done < /tmp/chai_matrix_runs
