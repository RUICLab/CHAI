#!/usr/bin/env bash
#
# Four-arm ablation over the six benchmarks.
#
#   A  checkmate_baseline    T1-T20   no control plane   <- CheckMate-style baseline
#   B  operator_only_plus10  T1-T30   no control plane
#   C  chai_plus10           T1-T30   full control plane
#   D  chai_baseline_ops     T1-T20   full control plane
#
#   operator main effect  = ((B-A) + (C-D)) / 2
#   control  main effect  = ((D-A) + (C-B)) / 2
#   interaction           = (C-B) - (D-A)
#
# Every arm is held to the identical frozen target list AND code summary,
# objective, per-benchmark error bound, trace, capacitor, seed, and BO budget.
# Canonical hand-written backends are OFF: a win produced by a benchmark-specific
# hand-written implementation measures that implementation, not the planner.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APPS="${APPS:-ar-iclib sobel-iclib stringsearch-iclib fft-iclib lqi-iclib bc-iclib}"
ARMS="${ARMS:-checkmate_baseline operator_only_plus10 chai_plus10 chai_baseline_ops}"
SEEDS="${SEEDS:-2026 2027 2028}"
BO_CALLS="${BO_CALLS:-40}"
MODEL="${MODEL:-chai-qwen2.5-coder:7b-24k}"
TRACE="${TRACE:-traces/RF_2.csv}"
CAPACITORS="${CAPACITORS:-68e-6}"
HARDWARE="${HARDWARE:-config/hardware_msp430.json}"
RUN_ID="${RUN_ID:-ablation_$(date +%Y%m%d_%H%M%S)}"
DOCKER_IMAGE="${DOCKER_IMAGE:-chai-dev:local}"
# Where inference runs.
#   Local  : http://host.docker.internal:11434/v1
#   Remote : start `ssh -N -L 11435:localhost:11434 <user>@<gpu-host>` first,
#            then point this at http://host.docker.internal:11435/v1
# Sustained 7B inference on a laptop thermally shuts the machine down mid-run,
# which silently truncates a campaign, so prefer the tunnel for real runs.
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434/v1}"
OLLAMA_SSH_TUNNEL="${OLLAMA_SSH_TUNNEL:-}"   # e.g. nsola5@131.193.50.170
OLLAMA_REMOTE_PORT="${OLLAMA_REMOTE_PORT:-11434}"
OLLAMA_LOCAL_PORT="${OLLAMA_LOCAL_PORT:-11435}"
TARGET_BUILD_GATE="${TARGET_BUILD_GATE:-strict}"
CANONICAL_BACKENDS="${CANONICAL_BACKENDS:-off}"
# Smoke runs may pin a small fixed budget; publishable runs must not.
BO_AUTO_BUDGET="${BO_AUTO_BUDGET:-1}"
# Plan quality is settled at generation time. Skipping search and simulation
# makes a six-benchmark sweep cheap enough to REPEAT, which is what the
# measurement actually needs: the planner is not reproducible at temperature 0,
# so one run of a cell fixes no magnitude. Under GENERATION_ONLY the SEEDS list
# is a list of repeat samples rather than optimizer seeds.
GENERATION_ONLY="${GENERATION_ONLY:-0}"

RESULTS_ROOT="experiment_results/$RUN_ID"
# Shared across campaigns by default: re-freezing per run costs model calls and
# risks the arms of two campaigns being compared against different target lists.
TARGETS_DIR="${TARGETS_DIR:-experiment_results/_frozen_targets}"
mkdir -p "$RESULTS_ROOT" "$TARGETS_DIR"

fail() { echo "ERROR: $*" >&2; exit 1; }

TUNNEL_PID=""
cleanup() {
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "${CAFFEINATE_PID:-}" ] && kill "$CAFFEINATE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ -n "$OLLAMA_SSH_TUNNEL" ]; then
  echo "[tunnel] forwarding localhost:$OLLAMA_LOCAL_PORT -> $OLLAMA_SSH_TUNNEL:$OLLAMA_REMOTE_PORT"
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 -N \
      -L "${OLLAMA_LOCAL_PORT}:localhost:${OLLAMA_REMOTE_PORT}" "$OLLAMA_SSH_TUNNEL" &
  TUNNEL_PID=$!
  for _ in $(seq 1 20); do
    curl -s --max-time 2 "http://localhost:${OLLAMA_LOCAL_PORT}/api/tags" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -s --max-time 5 "http://localhost:${OLLAMA_LOCAL_PORT}/api/tags" >/dev/null 2>&1 \
    || fail "SSH tunnel to $OLLAMA_SSH_TUNNEL did not expose Ollama on port $OLLAMA_REMOTE_PORT"
  OLLAMA_BASE_URL="http://host.docker.internal:${OLLAMA_LOCAL_PORT}/v1"
  echo "[tunnel] up (pid $TUNNEL_PID); inference goes to $OLLAMA_SSH_TUNNEL"
fi

# A campaign is long. Idle sleep kills the container mid-cell and leaves a
# half-written result that looks like a legitimate failure.
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i -w $$ &
  CAFFEINATE_PID=$!
  echo "[power] idle sleep inhibited for the duration of this campaign"
fi

arm_flags() {
  case "$1" in
    checkmate_baseline)   echo "--operator_catalog checkmate_20        --hardware $HARDWARE --hardware_control prompt_only" ;;
    operator_only_plus10) echo "--operator_catalog checkmate_plus_10   --hardware $HARDWARE --hardware_control prompt_only" ;;
    chai_plus10)          echo "--operator_catalog checkmate_plus_10   --hardware $HARDWARE --hardware_control chai" ;;
    chai_baseline_ops)    echo "--operator_catalog checkmate_20        --hardware $HARDWARE --hardware_control chai" ;;
    *) fail "unknown arm: $1" ;;
  esac
}

# ---------------------------------------------------------------------------
# Provenance. A result that cannot be tied to an exact image, model, commit and
# configuration is not reproducible, and silent drift between arms is exactly
# the failure mode this campaign exists to rule out.
# ---------------------------------------------------------------------------
# Docker Desktop stops on its own (sleep, updates, memory pressure) and a
# campaign that starts a minute later then dies on the first cell. Start it and
# wait, rather than failing a multi-hour run for a daemon that takes 10s.
if ! docker info >/dev/null 2>&1; then
  echo "[docker] daemon not responding; attempting to start it"
  if command -v open >/dev/null 2>&1; then open -a Docker >/dev/null 2>&1 || true; fi
  for _ in $(seq 1 60); do
    docker info >/dev/null 2>&1 && break
    sleep 2
  done
fi
docker info >/dev/null 2>&1 || fail "docker daemon is not available"

# `docker info` succeeding does NOT mean the image store is loaded: Docker
# Desktop answers the daemon socket before it has finished indexing images, so
# an inspect issued immediately after reports the image as missing and killed
# two campaigns here. Wait for the image itself, not just the daemon.
for _ in $(seq 1 60); do
  docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1 && break
  sleep 2
done

IMAGE_DIGEST="$(docker image inspect "$DOCKER_IMAGE" --format '{{.Id}}' 2>/dev/null)" \
  || fail "docker image $DOCKER_IMAGE not found after waiting for the image store"
MODEL_DIGEST="$(docker run --rm --network host "$DOCKER_IMAGE" \
  python3 -c "
import json,urllib.request
u='${OLLAMA_BASE_URL}'.replace('/v1','')+'/api/show'
r=urllib.request.Request(u,data=json.dumps({'model':'${MODEL}'}).encode(),
                         headers={'Content-Type':'application/json'},method='POST')
d=json.load(urllib.request.urlopen(r,timeout=20))
print(d.get('details',{}).get('parent_model') or d.get('digest') or
      d.get('model_info',{}).get('general.basename','unknown'))
print('PARAMS:'+' '.join(d.get('parameters','').split()))
" 2>/dev/null | tr '\n' ' ')" || fail "cannot reach Ollama at $OLLAMA_BASE_URL"
[ -n "$MODEL_DIGEST" ] || fail "Ollama did not describe model $MODEL"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repo')"
GIT_DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
CONFIG_HASH="$(cat main.py lib/bo.py lib/plan_contract.py lib/technique_registry.py \
  utils/optimization_metrics.py utils/error_analyzer.py utils/operator_conformance.py \
  prompts/*.txt prompts/FewShotExamples/*.txt "$HARDWARE" 2>/dev/null \
  | shasum -a 256 | cut -c1-16)"
BENCH_HASH="$(find benchmark_applications -name '*.c' -o -name '*.h' | sort \
  | xargs shasum -a 256 2>/dev/null | shasum -a 256 | cut -c1-16)"

python3 - "$RESULTS_ROOT/manifest.json" <<PYEOF
import json, sys, datetime
json.dump({
    "run_id": "$RUN_ID",
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "design": "2x2: operator catalogue (T1-T20 vs T1-T30) x control plane (off vs on)",
    "apps": "$APPS".split(),
    "arms": "$ARMS".split(),
    "seeds": [int(s) for s in "$SEEDS".split()],
    "bo_calls": $BO_CALLS,
    "bo_auto_budget": "$BO_AUTO_BUDGET" not in ("0","false","no","off"),
    "generation_only": "$GENERATION_ONLY" not in ("0","false","no","off"),
    "trace": "$TRACE",
    "capacitors": "$CAPACITORS".split(","),
    "hardware": "$HARDWARE",
    "canonical_backends": "$CANONICAL_BACKENDS",
    "target_build_gate": "$TARGET_BUILD_GATE",
    "provenance": {
        "docker_image": "$DOCKER_IMAGE",
        "docker_image_digest": "$IMAGE_DIGEST",
        "model": "$MODEL",
        "model_digest": "$MODEL_DIGEST".strip(),
        "ollama_base_url": "$OLLAMA_BASE_URL",
        "ollama_ssh_tunnel": "$OLLAMA_SSH_TUNNEL" or None,
        "git_commit": "$GIT_COMMIT",
        "git_dirty_files": $GIT_DIRTY,
        "config_hash": "$CONFIG_HASH",
        "benchmark_source_hash": "$BENCH_HASH",
    },
}, open(sys.argv[1], "w"), indent=2)
PYEOF

echo "=== Ablation campaign $RUN_ID ==="
echo "Apps   : $APPS"
echo "Arms   : $ARMS"
echo "Seeds  : $SEEDS"
echo "Image  : $IMAGE_DIGEST"
echo "Commit : $GIT_COMMIT (${GIT_DIRTY} dirty files)"
echo "Config : $CONFIG_HASH   Benchmarks: $BENCH_HASH"
[ "$GIT_DIRTY" != "0" ] && echo "NOTE: working tree is dirty; config_hash pins what actually ran."
echo

run_chai() {  # run_chai <console-log> <extra args...>
  local log="$1"; shift
  docker run --rm --network host \
    --label chai.experiment=true \
    -e "CHAI_BO_CALLS=$BO_CALLS" \
    -e "CHAI_BO_AUTO_BUDGET=$BO_AUTO_BUDGET" \
    -e "LLM_MODEL=$MODEL" \
    -e "API_NAME=Ollama" \
    -e "OLLAMA_BASE_URL=$OLLAMA_BASE_URL" \
    -e "OLLAMA_MIN_CONTEXT=18000" \
    -e "OLLAMA_MAX_TOKENS=2048" \
    -e "OLLAMA_MAX_RETRIES=0" \
    -e "CHAI_REQUIRE_IMPROVEMENT=1" \
    -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
    -v "$REPO_ROOT:/workspace" -w /workspace \
    "$DOCKER_IMAGE" python3 -u main.py "$@" > "$log" 2>&1
}

# ---------------------------------------------------------------------------
# Stage 1: freeze the target list AND the code summary, once per app.
# A failure here stops the campaign: running arms against differently-selected
# targets is the uncontrolled comparison this design exists to prevent.
# ---------------------------------------------------------------------------
for APP in $APPS; do
  TARGET_FILE="$TARGETS_DIR/${APP}.json"
  if [ -s "$TARGET_FILE" ]; then
    echo "[freeze] $APP: reusing $TARGET_FILE"
    continue
  fi
  echo "[freeze] $APP: selecting target functions once for all arms"
  if ! run_chai "$RESULTS_ROOT/freeze_${APP}.log" \
        --bm_name "$APP" --trace "$TRACE" --hardware "$HARDWARE" \
        --operator_catalog checkmate_plus_10 \
        --write_target_functions "$TARGET_FILE" --preflight_only; then
    tail -30 "$RESULTS_ROOT/freeze_${APP}.log" >&2
    fail "target freezing failed for $APP; campaign aborted"
  fi
  [ -s "$TARGET_FILE" ] || fail "target freezing produced no $TARGET_FILE"
  python3 -c "
import json,sys
d=json.load(open('$TARGET_FILE'))
t=d.get('target_functions',{})
if not any(str(v).lower()=='approximate' for v in t.values()):
    sys.exit('no approximable target frozen for $APP')
if not d.get('code_summary'):
    sys.exit('no code summary frozen for $APP')
print('  frozen:', sorted(k for k,v in t.items() if str(v).lower()=='approximate'))
" || fail "frozen target file for $APP is unusable"
done

# ---------------------------------------------------------------------------
# Stage 2: run every (app, arm, seed) cell.
# ---------------------------------------------------------------------------
TOTAL=0; OK=0
for APP in $APPS; do
  TARGET_FILE="$TARGETS_DIR/${APP}.json"
  for ARM in $ARMS; do
    FLAGS="$(arm_flags "$ARM")"
    for SEED in $SEEDS; do
      TOTAL=$((TOTAL+1))
      DEST="$RESULTS_ROOT/$ARM/$APP/seed_$SEED"
      mkdir -p "$DEST/logs" "$DEST/outputs"

      # Runs share the transient logs/ and outputs/ trees. Without clearing
      # them, a failed cell copies the PREVIOUS cell's artifacts and is scored
      # as if it had produced them.
      rm -rf logs outputs
      mkdir -p logs outputs

      GEN_ONLY_FLAG=""
      [ "$GENERATION_ONLY" != "0" ] && GEN_ONLY_FLAG="--generation_only"
      LABEL="seed"; [ "$GENERATION_ONLY" != "0" ] && LABEL="rep"
      echo "[$(date +%H:%M:%S)] $ARM / $APP / $LABEL $SEED"
      set +e
      # shellcheck disable=SC2086
      docker run --rm --network host \
        --label chai.experiment=true \
        -e "CHAI_BO_CALLS=$BO_CALLS" -e "CHAI_BO_AUTO_BUDGET=$BO_AUTO_BUDGET" \
        -e "CHAI_BO_SEED=$SEED" \
        -e "LLM_MODEL=$MODEL" -e "API_NAME=Ollama" \
        -e "OLLAMA_BASE_URL=$OLLAMA_BASE_URL" \
        -e "OLLAMA_MIN_CONTEXT=18000" -e "OLLAMA_MAX_TOKENS=2048" \
        -e "OLLAMA_MAX_RETRIES=0" \
        -e "CHAI_REQUIRE_IMPROVEMENT=1" -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
        -v "$REPO_ROOT:/workspace" -w /workspace \
        "$DOCKER_IMAGE" python3 -u main.py \
          --bm_name "$APP" --trace "$TRACE" --capacitors "$CAPACITORS" \
          --trace_approx 1 \
          --target_functions "$TARGET_FILE" \
          --canonical_backends "$CANONICAL_BACKENDS" \
          --target_build_gate "$TARGET_BUILD_GATE" \
          $GEN_ONLY_FLAG \
          $FLAGS \
        > "$DEST/console.log" 2>&1
      EXIT_CODE=$?
      set -e

      cp -r logs/. "$DEST/logs/" 2>/dev/null || true
      cp -r outputs/. "$DEST/outputs/" 2>/dev/null || true

      python3 - "$DEST/status.json" <<PYEOF
import json, sys, datetime
json.dump({
    "app": "$APP", "arm": "$ARM", "seed": $SEED,
    "exit_code": $EXIT_CODE,
    "state": "completed" if $EXIT_CODE == 0 else "failed",
    "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
    "config_hash": "$CONFIG_HASH",
    "docker_image_digest": "$IMAGE_DIGEST",
}, open(sys.argv[1], "w"), indent=2)
PYEOF
      [ "$EXIT_CODE" = "0" ] && OK=$((OK+1))
      echo "    exit=$EXIT_CODE  -> $DEST"
    done
  done
done

python3 - "$RESULTS_ROOT/current.json" <<PYEOF
import json, sys, datetime
json.dump({"state": "finished", "cells_total": $TOTAL, "cells_exit_zero": $OK,
           "finished_at": datetime.datetime.utcnow().isoformat() + "Z"},
          open(sys.argv[1], "w"), indent=2)
PYEOF

echo
echo "=== Campaign complete: $OK/$TOTAL cells exited zero -> $RESULTS_ROOT ==="
echo "Analyse with: python3 scripts/analyze_ablation.py $RESULTS_ROOT"
