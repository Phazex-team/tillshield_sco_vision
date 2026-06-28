#!/usr/bin/env bash
# Gracefully stop the v3 app and the vLLM server. Touches NOTHING outside
# sco_vision.
set -u
cd "$(dirname "$0")"

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

APP_PID_FILE="${APP_PID_FILE:-./run/app.pid}"
VLLM_PID_FILE="${PID_FILE:-./run/vllm.pid}"           # Gemma BF16 server
QWEN_PID_FILE="${QWEN_PID_FILE:-./run/qwen.pid}"       # Qwen3-VL vLLM server
PHOENIX_PID_FILE="${PHOENIX_PID_FILE:-./run/phoenix.pid}"

stop_pid() {
  local label="$1" pid_file="$2"
  if [[ ! -f "$pid_file" ]]; then
    echo "[stop] $label: no PID file ($pid_file)"
    return 0
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $label: PID $pid not running"
    rm -f "$pid_file"; return 0
  fi
  echo "[stop] $label: SIGTERM PID=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] $label: still running, SIGKILL"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
  echo "[stop] $label: stopped"
}

stop_pid "app"     "$APP_PID_FILE"
stop_pid "phoenix" "$PHOENIX_PID_FILE"
stop_pid "qwen"    "$QWEN_PID_FILE"
stop_pid "gemma"   "$VLLM_PID_FILE"
echo "[stop] done"
