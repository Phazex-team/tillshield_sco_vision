#!/usr/bin/env bash
# Show running/stopped status of vLLM + v3 app, port bindings, GPU usage,
# and the last 10 lines of each log.
set -u
cd "$(dirname "$0")"

if [[ -f .env ]]; then set -a; source ./.env; set +a; fi

APP_PORT="${APP_PORT:-3902}"
VLLM_PORT="${VLLM_PORT:-8001}"
PHOENIX_PORT="${PHOENIX_PORT:-6006}"
APP_PID_FILE="${APP_PID_FILE:-./run/app.pid}"
VLLM_PID_FILE="${PID_FILE:-./run/vllm.pid}"
PHOENIX_PID_FILE="${PHOENIX_PID_FILE:-./run/phoenix.pid}"
APP_LOG="${APP_LOG:-./logs/app.log}"
VLLM_LOG="${VLLM_LOG:-./logs/vllm.log}"
PHOENIX_LOG="${PHOENIX_LOG:-./logs/phoenix.log}"

show_one() {
  local label="$1" pid_file="$2" port="$3"
  echo "==== $label ===="
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "  status : RUNNING (pid $pid)"
    else
      echo "  status : STOPPED (stale pid file)"
    fi
  else
    echo "  status : STOPPED (no pid file)"
  fi
  local code
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
              "http://127.0.0.1:${port}/health" 2>/dev/null || echo "---")
  echo "  http   : :${port} -> ${code}"
  if command -v ss >/dev/null; then
    ss -ltn "sport = :${port}" 2>/dev/null | tail -n +2 | head -1 \
      | awk '{print "  bind   : "$0}'
  fi
}

show_one "vLLM"    "$VLLM_PID_FILE"    "$VLLM_PORT"
show_one "Phoenix" "$PHOENIX_PID_FILE" "$PHOENIX_PORT"
show_one "App"     "$APP_PID_FILE"     "$APP_PORT"

echo
echo "==== GPU ===="
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.used,memory.free,memory.total \
             --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /'
  echo "  -- compute apps --"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
             --format=csv,noheader 2>/dev/null \
    | sed 's/^/  /'
else
  echo "  nvidia-smi not on PATH"
fi

echo
echo "==== last 10 lines: vLLM ($VLLM_LOG) ===="
[[ -f "$VLLM_LOG" ]] && tail -n 10 "$VLLM_LOG" | sed 's/^/  /' \
  || echo "  (no log)"
echo
echo "==== last 10 lines: Phoenix ($PHOENIX_LOG) ===="
[[ -f "$PHOENIX_LOG" ]] && tail -n 10 "$PHOENIX_LOG" | sed 's/^/  /' \
  || echo "  (no log)"
echo
echo "==== last 10 lines: App ($APP_LOG) ===="
[[ -f "$APP_LOG" ]] && tail -n 10 "$APP_LOG" | sed 's/^/  /' \
  || echo "  (no log)"
