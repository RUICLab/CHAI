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
# Canonical reference backends are OFF: a win produced by a benchmark-specific
# deterministic implementation measures that implementation, not the planner.

set -euo pipefail

# macOS does not provide the Linux-only C.UTF-8 locale. `shasum` is a Perl
# program and aborts during provenance hashing when Docker/setup exports that
# locale into the host shell, before the first experiment cell can run.
export LC_ALL=C
export LANG=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APPS="${APPS:-ar-iclib sobel-iclib stringsearch-iclib fft-iclib lqi-iclib bc-iclib}"
ARMS="${ARMS:-checkmate_baseline operator_only_plus10 chai_plus10 chai_baseline_ops}"
# These are independent end-to-end planner repetitions. The value also seeds
# BO, but API providers do not expose a controllable planner seed; do not call
# these three BO seeds over one fixed plan.
SEEDS="${SEEDS:-2026 2027 2028}"
# AR's first observed improvement arrived at call 49 and its best at 201, so a
# 40-call primary budget is known to be underpowered. Preflight at 250 and
# recover the 20/40/80/160/250 budget curve from the per-call log.
BO_CALLS="${BO_CALLS:-250}"
BUDGET_CHECKPOINTS="${BUDGET_CHECKPOINTS:-20 40 80 160 250}"
MODEL="${MODEL:-chai-qwen2.5-coder:7b-24k}"
# Provider: Ollama (local/tunnelled) or an API model. An API model needs no
# tunnel, no GPU, and no Ollama context verification.
CHAI_PROVIDER="${CHAI_PROVIDER:-Ollama}"
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
# A remote GPU campaign must not silently become a CPU campaign after a CUDA
# fault. Set to 0 only for an explicitly labelled CPU diagnostic.
REQUIRE_REMOTE_GPU="${REQUIRE_REMOTE_GPU:-1}"
TARGET_BUILD_GATE="${TARGET_BUILD_GATE:-strict}"
CANONICAL_BACKENDS="${CANONICAL_BACKENDS:-off}"
# Fix total evaluation effort across arms. Auto-scaling by plan dimensionality
# would give larger plans more calls and confound planner quality with compute.
# A deliberate scaling study may override this explicitly.
BO_AUTO_BUDGET="${BO_AUTO_BUDGET:-0}"
# Plan quality is settled at generation time. Skipping search and simulation
# makes a six-benchmark sweep cheap enough to REPEAT, which is what the
# measurement actually needs: the planner is not reproducible at temperature 0,
# so one run of a cell fixes no magnitude. Under GENERATION_ONLY the SEEDS list
# is a list of repeat samples rather than optimizer seeds.
GENERATION_ONLY="${GENERATION_ONLY:-0}"

# Planner-condition knobs, so a variant runs without editing the repo (the
# container mounts it live). Defaults reproduce this fork's current settings;
# CHECKMATE_SETTINGS=1 reproduces upstream CheckMate's, which used a frontier
# model with none of the added prompt context and a 0.50 bound.
ERROR_BOUND="${ERROR_BOUND:-}"
# The BO objective adds a linear error term ON TOP of the hard bound
# rejection, so "do nothing" scores exactly 1.0 and an approximation using
# 84% of its error budget must also cut 84% of cycles merely to tie. Every
# performance campaign in this project reported "no improvement" partly
# because of it. Set OBJ_ERROR_WEIGHT=0 for the constrained formulation:
# minimise power cycles subject to error <= bound.
OBJ_ERROR_WEIGHT="${OBJ_ERROR_WEIGHT:-}"
APPROXIMATION_CONTEXT="${APPROXIMATION_CONTEXT:-}"
GIVE_LOOP_PERF_EXAMPLES="${GIVE_LOOP_PERF_EXAMPLES:-}"
GIVE_NEW_TECHNIQUE_EXAMPLES="${GIVE_NEW_TECHNIQUE_EXAMPLES:-}"
MAX_COMPILE_RETRIES="${MAX_COMPILE_RETRIES:-}"
CHECKMATE_SETTINGS="${CHECKMATE_SETTINGS:-0}"
if [ "$CHECKMATE_SETTINGS" != "0" ]; then
  ERROR_BOUND="${ERROR_BOUND:-0.50}"
  APPROXIMATION_CONTEXT=0
  GIVE_LOOP_PERF_EXAMPLES=0
  GIVE_NEW_TECHNIQUE_EXAMPLES=0
  MAX_COMPILE_RETRIES="${MAX_COMPILE_RETRIES:-10}"
fi

RESULTS_ROOT="experiment_results/$RUN_ID"
# Shared across campaigns by default: re-freezing per run costs model calls and
# risks the arms of two campaigns being compared against different target lists.
TARGETS_DIR="${TARGETS_DIR:-config/frozen_targets}"
mkdir -p "$RESULTS_ROOT" "$TARGETS_DIR"

fail() { echo "ERROR: $*" >&2; exit 1; }

# A provider key was exposed during setup. We cannot verify rotation from a
# secret value, so API campaigns require an explicit operator acknowledgement.
if [ "$CHAI_PROVIDER" != "Ollama" ] && [ "${CONFIRM_CREDENTIAL_ROTATION:-0}" != "1" ]; then
  fail "rotate the exposed API credential, then set CONFIRM_CREDENTIAL_ROTATION=1"
fi

TUNNEL_PID=""
cleanup() {
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "${TUNNEL_WATCHDOG_PID:-}" ] && kill "$TUNNEL_WATCHDOG_PID" 2>/dev/null || true
  [ -n "${CAFFEINATE_PID:-}" ] && kill "$CAFFEINATE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

tunnel_alive() {
  curl -s --max-time 5 "http://localhost:${OLLAMA_LOCAL_PORT}/api/tags" >/dev/null 2>&1
}

# Bind the forward on all interfaces, not just loopback. The health check and
# every cell run inside Docker, and on macOS Docker Desktop a container's
# `host.docker.internal` is 192.168.65.254 -- it cannot reach a tunnel bound
# only to 127.0.0.1, so the campaign aborted with "cannot reach Ollama"
# while the tunnel was demonstrably up and serving the host fine.
start_tunnel() {
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
      -o ServerAliveCountMax=3 -o TCPKeepAlive=yes -N \
      -L "0.0.0.0:${OLLAMA_LOCAL_PORT}:localhost:${OLLAMA_REMOTE_PORT}" "$OLLAMA_SSH_TUNNEL" &
  TUNNEL_PID=$!
  for _ in $(seq 1 25); do
    tunnel_alive && return 0
    sleep 1
  done
  return 1
}

# A single `ssh -N -L` is a single point of failure for a multi-hour campaign,
# and its failure mode is silent: every later cell gets "Connection refused"
# from Ollama, produces no plan, and is recorded as though the planner had
# nothing to say. A dropped tunnel once corrupted 42 of 48 cells that way and
# nearly got reported as a result about prompt settings.
#
# So: supervise it, and check it before every cell.
if [ -n "$OLLAMA_SSH_TUNNEL" ] && [ "$CHAI_PROVIDER" = "Ollama" ]; then
  echo "[tunnel] forwarding localhost:$OLLAMA_LOCAL_PORT -> $OLLAMA_SSH_TUNNEL:$OLLAMA_REMOTE_PORT"
  start_tunnel || fail "SSH tunnel to $OLLAMA_SSH_TUNNEL did not expose Ollama"
  OLLAMA_BASE_URL="http://host.docker.internal:${OLLAMA_LOCAL_PORT}/v1"
  echo "[tunnel] up (pid $TUNNEL_PID); inference goes to $OLLAMA_SSH_TUNNEL"

  # Background supervisor: reconnect within seconds of a drop.
  (
    while kill -0 $$ 2>/dev/null; do
      sleep 20
      if ! tunnel_alive; then
        echo "[tunnel] DOWN - reconnecting" >&2
        kill "$TUNNEL_PID" 2>/dev/null || true
        start_tunnel && echo "[tunnel] reconnected" >&2
      fi
    done
  ) &
  TUNNEL_WATCHDOG_PID=$!
fi

# Fail a cell loudly when the model is unreachable, instead of letting it be
# scored as "the planner produced nothing".
require_model() {
  [ "$CHAI_PROVIDER" != "Ollama" ] && return 0
  local attempt
  for attempt in 1 2 3 4 5 6; do
    if [ -n "$OLLAMA_SSH_TUNNEL" ]; then
      tunnel_alive && return 0
      echo "[model] unreachable (attempt $attempt/6); restarting tunnel" >&2
      kill "$TUNNEL_PID" 2>/dev/null || true
      start_tunnel || true
    else
      curl -s --max-time 5 "${OLLAMA_BASE_URL%/v1}/api/tags" >/dev/null 2>&1 && return 0
    fi
    sleep 10
  done
  return 1
}

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

# Resolve the image id through `docker images`, not `docker image inspect`.
# Under Docker Desktop's containerd image store, `docker image inspect <tag>`
# reports "No such image" for an image that `docker images` lists and
# `docker run` starts perfectly well -- it resolves per-platform, and this
# image is linux/amd64 on an arm64 host. Two campaigns died on that false
# negative. `docker images` is the portable lookup, and it also gives the
# daemon time to finish indexing.
IMAGE_DIGEST=""
for _ in $(seq 1 60); do
  IMAGE_DIGEST="$(docker images --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>/dev/null \
    | awk -v want="$DOCKER_IMAGE" '$1 == want {print $2; exit}')"
  [ -n "$IMAGE_DIGEST" ] && break
  sleep 2
done
[ -n "$IMAGE_DIGEST" ] || fail "docker image $DOCKER_IMAGE not found in the image store"

# Prove the image actually runs before committing to a multi-hour campaign.
docker run --rm "$DOCKER_IMAGE" true >/dev/null 2>&1 \
  || fail "docker image $DOCKER_IMAGE is present but will not run"
if [ "$CHAI_PROVIDER" = "Ollama" ]; then
  # The payload must NOT be indented: python3 -c rejects leading whitespace on
  # a top-level statement with IndentationError. It used to be, and 2>/dev/null
  # swallowed the traceback, so a pure syntax error was reported as
  # "cannot reach Ollama" -- blaming the network for a bug that never opened a
  # socket. Keep stderr, and show it when the check fails.
  HEALTH_ERR="$RESULTS_ROOT/ollama_healthcheck.err"
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
" 2>"$HEALTH_ERR" | tr '\n' ' ' | tr -d '"')" \
    || fail "cannot reach Ollama at $OLLAMA_BASE_URL: $(tail -3 "$HEALTH_ERR" 2>/dev/null | tr '\n' ' ')"
  [ -n "$MODEL_DIGEST" ] || fail "Ollama did not describe model $MODEL"

  if [ -n "$OLLAMA_SSH_TUNNEL" ] && [ "$REQUIRE_REMOTE_GPU" != "0" ]; then
    GPU_HEALTH_ERR="$RESULTS_ROOT/ollama_gpu_healthcheck.err"
    GPU_BYTES="$(docker run --rm --network host "$DOCKER_IMAGE" \
python3 -c "
import json,urllib.request
base='${OLLAMA_BASE_URL}'.replace('/v1','')
req=urllib.request.Request(
    base+'/api/generate',
    data=json.dumps({'model':'${MODEL}','prompt':'Reply with OK',
                     'stream':False,
                     'options':{'num_predict':1,'num_ctx':18000}}).encode(),
    headers={'Content-Type':'application/json'}, method='POST')
json.load(urllib.request.urlopen(req,timeout=600))
loaded=json.load(urllib.request.urlopen(base+'/api/ps',timeout=20)).get('models',[])
match=next((m for m in loaded if m.get('name') == '${MODEL}' or
            m.get('model') == '${MODEL}'), {})
print(int(match.get('size_vram') or 0))
" 2>"$GPU_HEALTH_ERR")" \
      || fail "remote Ollama inference health check failed: $(tail -3 "$GPU_HEALTH_ERR" 2>/dev/null | tr '\n' ' ')"
    [ "$GPU_BYTES" -gt 0 ] 2>/dev/null \
      || fail "remote Ollama loaded $MODEL with size_vram=$GPU_BYTES; refusing silent CPU fallback"
    echo "[model] remote GPU acceleration verified (size_vram=$GPU_BYTES bytes)"
  fi
else
  # An API model has no local digest; the model string is the identifier.
  MODEL_DIGEST="api:$CHAI_PROVIDER:$MODEL"
fi

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
    "planner_repeats": [int(s) for s in "$SEEDS".split()],
    "repeat_semantics": "fresh end-to-end generation; numeric value also seeds BO, not the API planner",
    "bo_calls": $BO_CALLS,
    "bo_budget_checkpoints": [int(s) for s in "$BUDGET_CHECKPOINTS".split()],
    "bo_auto_budget": "$BO_AUTO_BUDGET" not in ("0","false","no","off"),
    "generation_only": "$GENERATION_ONLY" not in ("0","false","no","off"),
    "planner_settings": {
        "checkmate_settings": "$CHECKMATE_SETTINGS" != "0",
        "error_bound": "$ERROR_BOUND" or "default 0.30",
        "objective_error_weight": "$OBJ_ERROR_WEIGHT" or "default 1.0",
        "approximation_context": "$APPROXIMATION_CONTEXT" or "default 1",
        "loop_perf_examples": "$GIVE_LOOP_PERF_EXAMPLES" or "default 1",
        "new_technique_examples": "$GIVE_NEW_TECHNIQUE_EXAMPLES" or "default 1",
        "max_compile_retries": "$MAX_COMPILE_RETRIES" or "default 3",
    },
    "trace": "$TRACE",
    "capacitors": "$CAPACITORS".split(","),
    "hardware": "$HARDWARE",
    "canonical_backends": "$CANONICAL_BACKENDS",
    "target_build_gate": "$TARGET_BUILD_GATE",
    "provenance": {
        "docker_image": "$DOCKER_IMAGE",
        "docker_image_digest": "$IMAGE_DIGEST",
        "model": "$MODEL",
    "provider": "$CHAI_PROVIDER",
        # MODEL_DIGEST is interpolated into a Python string literal below, so
        # any double quote in it terminates that literal and corrupts the rest
        # of the dict. DeepSeek declares stop tokens ("User:", "Assistant:")
        # and broke the manifest here; qwen declares none, which is why this
        # stayed latent. Quotes are stripped at capture time.
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

# The container mounts this repo live, so any edit during a campaign lands
# mid-run and the cells before and after it are not the same experiment. One
# campaign was already invalidated that way. Record the tree state and refuse
# to start dirty unless explicitly overridden.
if [ "$GIT_DIRTY" != "0" ] && [ "${ALLOW_DIRTY_TREE:-0}" = "0" ]; then
  echo "ERROR: $GIT_DIRTY uncommitted file(s). The container mounts this repo," >&2
  echo "so editing during the run splits the campaign across two versions." >&2
  echo "Commit first, or set ALLOW_DIRTY_TREE=1 to accept that risk." >&2
  exit 1
fi

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
    --env-file .env \
    -e "CHAI_BO_CALLS=$BO_CALLS" \
    -e "CHAI_BO_AUTO_BUDGET=$BO_AUTO_BUDGET" \
    -e "LLM_MODEL=$MODEL" \
    -e "CHAI_API_NAME=$CHAI_PROVIDER" \
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
# Stage 1: validate the checked-in source-derived target list and code summary.
# The campaign must never ask a model to choose its own evaluation targets.
# ---------------------------------------------------------------------------
for APP in $APPS; do
  TARGET_FILE="$TARGETS_DIR/${APP}.json"
  [ -s "$TARGET_FILE" ] || fail "missing checked-in deterministic target file $TARGET_FILE"
  echo "[targets] $APP: validating $TARGET_FILE"
  python3 -c "
import json,sys
d=json.load(open('$TARGET_FILE'))
t=d.get('target_functions',{})
if not any(str(v).lower()=='approximate' for v in t.values()):
    sys.exit('no approximable target frozen for $APP')
if not d.get('code_summary'):
    sys.exit('no code summary frozen for $APP')
if not d.get('selection_policy'):
    sys.exit('no deterministic selection policy recorded for $APP')
print('  targets:', sorted(k for k,v in t.items() if str(v).lower()=='approximate'))
" || fail "deterministic target file for $APP is unusable"
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
      if ! require_model; then
        echo "    SKIPPED - model unreachable; recording an invalid cell" | tee "$DEST/console.log"
        python3 - "$DEST/status.json" <<PYEOF
import json, sys, datetime
json.dump({"app": "$APP", "arm": "$ARM", "seed": $SEED, "exit_code": 90,
           "state": "invalid", "reason": "model unreachable before the cell ran",
           "completed_at": datetime.datetime.utcnow().isoformat() + "Z"},
          open(sys.argv[1], "w"), indent=2)
PYEOF
        TOTAL=$((TOTAL))
        continue
      fi
      set +e
      # shellcheck disable=SC2086
      docker run --rm --network host \
        --label chai.experiment=true \
        -e "CHAI_BO_CALLS=$BO_CALLS" -e "CHAI_BO_AUTO_BUDGET=$BO_AUTO_BUDGET" \
        -e "CHAI_BO_SEED=$SEED" \
        --env-file .env \
        -e "LLM_MODEL=$MODEL" -e "CHAI_API_NAME=$CHAI_PROVIDER" \
        -e "OLLAMA_BASE_URL=$OLLAMA_BASE_URL" \
        ${CHAI_PROVIDER:+$([ "$CHAI_PROVIDER" = "Ollama" ] && echo "-e OLLAMA_MIN_CONTEXT=18000 -e OLLAMA_MAX_TOKENS=2048")} \
        -e "OLLAMA_MAX_RETRIES=0" \
        -e "CHAI_REQUIRE_IMPROVEMENT=1" -e "CHAI_IDENTITY_TOLERANCE=1e-6" \
        ${OBJ_ERROR_WEIGHT:+-e "CHAI_OBJECTIVE_ERROR_WEIGHT=$OBJ_ERROR_WEIGHT"} \
        ${ERROR_BOUND:+-e "CHAI_ERROR_BOUND=$ERROR_BOUND"} \
        ${ERROR_BOUND:+-e "CHAI_ERROR_BOUND_$(echo "$APP" | tr 'a-z-' 'A-Z_')=$ERROR_BOUND"} \
        ${APPROXIMATION_CONTEXT:+-e "CHAI_APPROXIMATION_CONTEXT=$APPROXIMATION_CONTEXT"} \
        ${GIVE_LOOP_PERF_EXAMPLES:+-e "CHAI_GIVE_LOOP_PERF_EXAMPLES=$GIVE_LOOP_PERF_EXAMPLES"} \
        ${GIVE_NEW_TECHNIQUE_EXAMPLES:+-e "CHAI_GIVE_NEW_TECHNIQUE_EXAMPLES=$GIVE_NEW_TECHNIQUE_EXAMPLES"} \
        ${MAX_COMPILE_RETRIES:+-e "CHAI_MAX_COMPILE_RETRIES=$MAX_COMPILE_RETRIES"} \
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
REPEAT_COUNT="$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')"
if [ "$REPEAT_COUNT" -gt 1 ]; then
  python3 scripts/verify_planner_repeats.py "$RESULTS_ROOT" --minimum "$REPEAT_COUNT" \
    || fail "planner repetitions were missing or not independent"
fi
echo "Analyse with: python3 scripts/analyze_ablation.py $RESULTS_ROOT"
