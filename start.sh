#!/usr/bin/env bash
# Start the full SCO Vision stack:
#   * Qwen3-VL vLLM    on :8000 (primary VLM — default)
#   * Gemma BF16       on :8001 (optional fallback; OFF by default)
#   * Phoenix          on :6006 (optional telemetry; OFF by default)
#   * FastAPI app      on :3902
#
# Idempotent. Re-running is safe — already-healthy services are reused.
#
# Operator switches (env or .env):
#   START_QWEN=1    (default 1)  — bring up Qwen on :8000
#   START_GEMMA=0   (default 0)  — bring up Gemma fallback on :8001
#   START_PHOENIX=0 (default 0)  — bring up Phoenix telemetry on :6006
#
# Why Qwen is default and Gemma is off:
#   Qwen3-VL has stronger temporal/video grounding than the local
#   Gemma BF16 path. For day-to-day SCO operation we want Qwen as the
#   sole VLM. The Gemma code paths stay in tree; flip START_GEMMA=1
#   AND set reasoning.fallback_provider: gemma in config.yaml to
#   activate the fallback.
set -u
cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a; source ./.env; set +a
fi

APP_PORT="${APP_PORT:-3902}"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_LOG="${APP_LOG:-./logs/app.log}"
APP_PID_FILE="${APP_PID_FILE:-./run/app.pid}"

START_QWEN="${START_QWEN:-1}"
START_GEMMA="${START_GEMMA:-0}"
START_PHOENIX="${START_PHOENIX:-0}"

mkdir -p "$(dirname "$APP_LOG")" "$(dirname "$APP_PID_FILE")"

# Step 1a: Qwen vLLM (primary VLM, default ON).
if [[ "$START_QWEN" == "1" ]]; then
  echo "[start] bringing up Qwen3-VL on :8000..."
  bash ./scripts/qwen_vllm_start.sh
  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "[start] Qwen failed to start (rc=$RC) — see logs/qwen.log" >&2
    exit $RC
  fi
else
  echo "[start] START_QWEN=0 — skipping Qwen launch (set to 1 to enable)"
fi

# Step 1b: Gemma BF16 (fallback VLM, OFF by default).
if [[ "$START_GEMMA" == "1" ]]; then
  echo "[start] bringing up Gemma BF16 fallback on :8001..."
  bash ./vllm_start.sh
  RC=$?
  if [[ $RC -ne 0 ]]; then
    echo "[start] Gemma fallback failed (rc=$RC) — continuing without it" >&2
  fi
else
  echo "[start] START_GEMMA=0 — Gemma fallback off (set START_GEMMA=1 AND"
  echo "[start]   reasoning.fallback_provider=gemma in config.yaml to enable)"
fi

# Step 1c: Phoenix telemetry (optional; never blocks app boot).
if [[ "$START_PHOENIX" == "1" ]]; then
  echo "[start] bringing up phoenix..."
  if ! bash ./phoenix_start.sh; then
    echo "[start] phoenix not healthy — continuing without tracing" >&2
  fi
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
export FRAUD_OFFLINE_MODE="${FRAUD_OFFLINE_MODE:-1}"
export APP_PORT
export VLLM_PORT="${VLLM_PORT:-8001}"
export VLLM_URL="${VLLM_URL:-http://localhost:${VLLM_PORT}}"

echo "[start] launching app on :$APP_PORT (logs -> $APP_LOG)"
# The product is the modular FastAPI app (app.main, served via
# scripts/run_app.py) — it mounts the v1 API routers and the
# static/review.html operator console. The legacy monolith app.py is
# kept only for reference and must NOT be used to serve: it has none of
# the v1 routers (cases, admin/cameras, ops, etc.).
nohup python scripts/run_app.py --host "$APP_HOST" --port "$APP_PORT" >>"$APP_LOG" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$APP_PID_FILE"
echo "[start] app PID=$APP_PID"

# Quick sanity wait. The modular app's health route is /api/v1/health.
for i in $(seq 1 30); do
  if curl -sf -o /dev/null --max-time 2 "http://127.0.0.1:${APP_PORT}/api/v1/health"; then
    echo "[start] app healthy on :$APP_PORT (after ${i}s)"
    echo "[start] reviewer UI:  http://localhost:${APP_PORT}/static/review.html"
    echo "[start] API docs:     http://localhost:${APP_PORT}/api/v1/docs"
    echo "[start] Qwen models:  http://localhost:8000/v1/models"
    if [[ "$START_GEMMA" == "1" ]]; then
      echo "[start] Gemma health: http://localhost:8001/health"
    fi
    exit 0
  fi
  sleep 1
done
echo "[start] app did not become healthy in 30s — see $APP_LOG" >&2
exit 2
