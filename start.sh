#!/usr/bin/env bash
# Start the full v3 stack: vLLM (port 8001) + the FastAPI app (port 3902).
# Calls vllm_start.sh first and only proceeds once vLLM is /health=200.
set -u
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source ./.env; set +a
fi

APP_PORT="${APP_PORT:-3902}"
APP_LOG="${APP_LOG:-./logs/app.log}"
APP_PID_FILE="${APP_PID_FILE:-./run/app.pid}"

mkdir -p "$(dirname "$APP_LOG")" "$(dirname "$APP_PID_FILE")"

# Step 1: vLLM (idempotent; fast if already running).
echo "[start] bringing up vLLM..."
bash ./vllm_start.sh
RC=$?
if [[ $RC -ne 0 ]]; then
  echo "[start] vLLM failed to start (rc=$RC)" >&2
  exit $RC
fi

# Step 1b: Phoenix telemetry (optional; never blocks app boot).
echo "[start] bringing up phoenix..."
if ! bash ./phoenix_start.sh; then
  echo "[start] phoenix not healthy — continuing without tracing" >&2
fi

# Step 2: app.
if [[ -f "$APP_PID_FILE" ]]; then
  OLD_PID="$(cat "$APP_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "[start] app already running (PID=$OLD_PID, port=$APP_PORT)"
    exit 0
  fi
  rm -f "$APP_PID_FILE"
fi

if [[ ! -d ./venv ]]; then
  echo "ERROR: ./venv missing. Run install.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source ./venv/bin/activate

export FRAUD_CONFIG="${FRAUD_CONFIG:-./config.yaml}"
export APP_PORT
export VLLM_PORT="${VLLM_PORT:-8001}"
export VLLM_URL="${VLLM_URL:-http://localhost:${VLLM_PORT}}"

echo "[start] launching app on :$APP_PORT (logs -> $APP_LOG)"
nohup python app.py >>"$APP_LOG" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$APP_PID_FILE"
echo "[start] app PID=$APP_PID"

# Quick sanity wait.
for i in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${APP_PORT}/health"; then
    echo "[start] app healthy on :$APP_PORT (after ${i}s)"
    echo "[start] open http://localhost:$APP_PORT"
    exit 0
  fi
  sleep 1
done
echo "[start] app did not become healthy in 30s — see $APP_LOG" >&2
exit 2
