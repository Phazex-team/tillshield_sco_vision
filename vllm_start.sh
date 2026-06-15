#!/usr/bin/env bash
# Idempotently launch the Gemma 4 inference server on $VLLM_PORT (8001).
#
# This DGX Spark / GB10 (sm_121) does not implement the FP4 conversion
# instruction (cvt.e2m1x2) that NVFP4 kernels require, so the original
# vLLM + bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 path produces NaN
# logits. We instead run google/gemma-4-26B-A4B-it (BF16) via the
# transformers-direct OpenAI-compat bridge in ``transformers_server.py``.
# See README "GB10 / sm_121 NVFP4 unsupported" for details.
#
# This script keeps the same contract as the prior vllm-based launcher:
#   * /health on http://127.0.0.1:${VLLM_PORT}
#   * pid file at ${PID_FILE} (default ./run/vllm.pid)
#   * stdout/err to ${VLLM_LOG} (default ./logs/vllm.log)
# so ./start.sh, ./stop.sh, ./status.sh and gemma_reasoner.py work unchanged.

set -u
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

VLLM_PORT="${VLLM_PORT:-8001}"
GEMMA_MODEL_NAME="${GEMMA_MODEL_NAME:-google/gemma-4-26B-A4B-it}"
VLLM_LOG="${VLLM_LOG:-./logs/vllm.log}"
PID_FILE="${PID_FILE:-./run/vllm.pid}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-1800}"   # BF16 weight load is slow

mkdir -p "$(dirname "$VLLM_LOG")" "$(dirname "$PID_FILE")"

is_healthy() {
  curl -sf -o /dev/null --max-time 3 \
       "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null
}

if is_healthy; then
  echo "vllm: already healthy on :${VLLM_PORT}"
  exit 0
fi

# Clean up a stale pid file.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "vllm: killing stale PID $OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

if [[ ! -d ./venv ]]; then
  echo "ERROR: ./venv missing. Run install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ./venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

echo "vllm: starting model=$GEMMA_MODEL_NAME on :$VLLM_PORT (logs -> $VLLM_LOG)"

(
  GEMMA_MODEL_NAME="$GEMMA_MODEL_NAME" VLLM_PORT="$VLLM_PORT" \
    setsid python transformers_server.py --port "$VLLM_PORT" \
                                          --model "$GEMMA_MODEL_NAME" \
    < /dev/null
) >>"$VLLM_LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo "vllm: PID=$SERVER_PID  waiting up to ${HEALTH_TIMEOUT_SEC}s for /health…"

for i in $(seq 1 "$HEALTH_TIMEOUT_SEC"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vllm: process exited during startup (see $VLLM_LOG)" >&2
    tail -n 30 "$VLLM_LOG" >&2
    exit 1
  fi
  if is_healthy; then
    echo "vllm: healthy on :$VLLM_PORT after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "vllm: timed out waiting for /health after ${HEALTH_TIMEOUT_SEC}s" >&2
echo "tail $VLLM_LOG:" >&2
tail -n 30 "$VLLM_LOG" >&2
exit 2
