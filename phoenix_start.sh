#!/usr/bin/env bash
# Idempotently start the Arize Phoenix telemetry server on :6006.
# Mirrors the vllm_start.sh shape: pid in ./run/phoenix.pid,
# logs in ./logs/phoenix.log, exits 0 if already healthy.

set -u

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

PHOENIX_PORT="${PHOENIX_PORT:-6006}"
PHOENIX_LOG="${PHOENIX_LOG:-./logs/phoenix.log}"
PHOENIX_PID_FILE="${PHOENIX_PID_FILE:-./run/phoenix.pid}"
HEALTH_TIMEOUT_SEC="${PHOENIX_HEALTH_TIMEOUT_SEC:-30}"

mkdir -p "$(dirname "$PHOENIX_LOG")" "$(dirname "$PHOENIX_PID_FILE")"

is_healthy() {
  curl -sf -o /dev/null --max-time 2 \
       "http://127.0.0.1:${PHOENIX_PORT}/" 2>/dev/null
}

if is_healthy; then
  echo "phoenix: already healthy on :${PHOENIX_PORT}"
  exit 0
fi

# Clean up a stale pid file.
if [[ -f "$PHOENIX_PID_FILE" ]]; then
  OLD_PID="$(cat "$PHOENIX_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "phoenix: killing stale PID $OLD_PID"
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PHOENIX_PID_FILE"
fi

if [[ ! -d ./venv ]]; then
  echo "ERROR: ./venv missing. Run install.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source ./venv/bin/activate

echo "phoenix: starting on :$PHOENIX_PORT (logs -> $PHOENIX_LOG)"
nohup python -m phoenix.server.main serve --port "$PHOENIX_PORT" \
      >>"$PHOENIX_LOG" 2>&1 &
PHOENIX_PID=$!
echo "$PHOENIX_PID" > "$PHOENIX_PID_FILE"
echo "phoenix: PID=$PHOENIX_PID  waiting up to ${HEALTH_TIMEOUT_SEC}s for HTTP..."

for i in $(seq 1 "$HEALTH_TIMEOUT_SEC"); do
  if ! kill -0 "$PHOENIX_PID" 2>/dev/null; then
    echo "phoenix: process exited during startup (see $PHOENIX_LOG)" >&2
    tail -n 30 "$PHOENIX_LOG" >&2
    exit 1
  fi
  if is_healthy; then
    echo "phoenix: healthy on :$PHOENIX_PORT after ${i}s"
    exit 0
  fi
  sleep 1
done

echo "phoenix: timed out waiting for HTTP on :$PHOENIX_PORT" >&2
tail -n 30 "$PHOENIX_LOG" >&2
exit 2
