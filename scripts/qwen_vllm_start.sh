#!/usr/bin/env bash
# Idempotently launch the Qwen3-VL FP8 inference server on $QWEN_PORT (8000)
# via the upstream vLLM CLI with the flag set this DGX Spark (GB10 / sm_121)
# actually tolerates.
#
# Failure mode this script fixes:
#   The default vLLM 0.x launch path activates the FlashInfer CUTLASS
#   fused-MoE kernel for Qwen3-VL-30B-A3B-Instruct-FP8 and crashes during
#   JIT compile (exit 137 / SIGKILL). The flags below force the Triton MoE
#   backend, disable CUDA graphs (which the FlashInfer autotuner also trips
#   over), and disable the FlashInfer autotune pass entirely.
#
# Contract (mirrors vllm_start.sh for the Gemma BF16 server):
#   * health on http://127.0.0.1:${QWEN_PORT}/health AND /v1/models
#   * pid file at ${QWEN_PID_FILE} (default ./run/qwen.pid)
#   * stdout/err to ${QWEN_LOG} (default ./logs/qwen.log)
#
# The app's readiness probe (app.startup) checks /v1/models, so a startup
# that lands on /health but not /v1/models is considered NOT ready.

set -u
cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-30B-A3B-Instruct}"
QWEN_SERVED_NAME="${QWEN_SERVED_NAME:-qwen3_vl}"
QWEN_LOG="${QWEN_LOG:-./logs/qwen.log}"
QWEN_PID_FILE="${QWEN_PID_FILE:-./run/qwen.pid}"
HEALTH_TIMEOUT_SEC="${QWEN_HEALTH_TIMEOUT_SEC:-1800}"

# Operator-tunable. Defaults set to the working configuration on this
# DGX Spark; do not override unless you have re-verified the launch.
QWEN_MOE_BACKEND="${QWEN_MOE_BACKEND:-triton}"
QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-1}"     # 1 = pass --enforce-eager
QWEN_DISABLE_FLASHINFER_AUTOTUNE="${QWEN_DISABLE_FLASHINFER_AUTOTUNE:-1}"
QWEN_GPU_MEM_UTIL="${QWEN_GPU_MEM_UTIL:-0.70}"
# vLLM default for Qwen3-VL-30B is 262144 tokens; on this DGX Spark
# that needs 24 GiB of KV cache but the GPU only has ~22 GiB free
# after the model weights are loaded, so the server aborts with
# "ValueError: KV cache needed 24.0 GiB, available 21.84 GiB.".
# 65536 fits and is more than enough for a single SCO clip + POS bill
# turn. Operators with a larger card / different model can override
# by exporting QWEN_MAX_MODEL_LEN before launch.
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-65536}"

mkdir -p "$(dirname "$QWEN_LOG")" "$(dirname "$QWEN_PID_FILE")"

is_healthy() {
  # Two-stage probe: /health for liveness, /v1/models for readiness.
  # The OpenAI-compatible /v1/models endpoint only comes online after
  # the model is loaded, so it's the right gate for the app's chain
  # to start calling Qwen.
  curl -sf -o /dev/null --max-time 3 \
       "http://127.0.0.1:${QWEN_PORT}/health" 2>/dev/null \
    && curl -sf -o /dev/null --max-time 3 \
       "http://127.0.0.1:${QWEN_PORT}/v1/models" 2>/dev/null
}

if is_healthy; then
  echo "qwen: already healthy on :$QWEN_PORT (skipping launch)"
  exit 0
fi

# Pre-existing PID? Reap if stale, otherwise refuse to start a second
# server (the user explicitly asked us not to double-launch).
if [[ -f "$QWEN_PID_FILE" ]]; then
  OLD_PID="$(cat "$QWEN_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "qwen: PID file points at running process $OLD_PID but health" \
         "probe failed. NOT starting a duplicate. Run stop.sh / inspect" \
         "$QWEN_LOG before retrying." >&2
    exit 2
  fi
  rm -f "$QWEN_PID_FILE"
fi

if [[ ! -d ./venv ]]; then
  echo "ERROR: ./venv missing. Run install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ./venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Build the vLLM serve argv. Conditionals here so the operator can
# disable individual workarounds via the env vars above when a future
# vLLM build no longer needs them.
CMD=(vllm serve "$QWEN_MODEL_NAME"
     --host 0.0.0.0 --port "$QWEN_PORT"
     --served-model-name "$QWEN_SERVED_NAME"
     --moe-backend "$QWEN_MOE_BACKEND"
     --gpu-memory-utilization "$QWEN_GPU_MEM_UTIL"
     --no-enable-prefix-caching
)
if [[ "$QWEN_ENFORCE_EAGER" == "1" ]]; then
  CMD+=(--enforce-eager)
fi
if [[ "$QWEN_DISABLE_FLASHINFER_AUTOTUNE" == "1" ]]; then
  CMD+=(--no-enable-flashinfer-autotune)
fi
if [[ -n "$QWEN_MAX_MODEL_LEN" ]]; then
  CMD+=(--max-model-len "$QWEN_MAX_MODEL_LEN")
fi
if [[ -n "${QWEN_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  CMD+=($QWEN_EXTRA_ARGS)
fi

echo "qwen: launching: ${CMD[*]}" >&2
echo "qwen: launching on :$QWEN_PORT (logs -> $QWEN_LOG)"

(
  setsid "${CMD[@]}" < /dev/null
) >> "$QWEN_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$QWEN_PID_FILE"
echo "qwen: PID=$SERVER_PID  waiting up to ${HEALTH_TIMEOUT_SEC}s for /v1/models…"

for i in $(seq 1 "$HEALTH_TIMEOUT_SEC"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "qwen: process exited during startup (see $QWEN_LOG)" >&2
    tail -n 40 "$QWEN_LOG" >&2
    exit 1
  fi
  if is_healthy; then
    echo "qwen: healthy on :$QWEN_PORT after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "qwen: did not reach /v1/models within ${HEALTH_TIMEOUT_SEC}s — see $QWEN_LOG" >&2
exit 3
