#!/usr/bin/env bash
# Fresh-DGX installer for fraud_detection_v3.
#
# Idempotent. Run from inside the project directory.
#
# Steps:
#   1. apt: install ffmpeg, git, curl, python3-venv (only if missing)
#   2. create ./venv (Python 3.12)
#   3. install locked Python deps from ./wheelhouse (offline-first)
#   4. install ./Falcon-Perception (vendored) editable
#   5. create runtime dirs and starter .env
#   6. chmod +x *.sh
set -u
cd "$(dirname "$0")"

PY="${PY:-python3}"
echo "================================================================"
echo "fraud_detection_v3 installer"
echo "================================================================"

# ---- 1. apt -----------------------------------------------------------
need_apt=()
for pkg in ffmpeg git curl python3-venv; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    need_apt+=("$pkg")
  fi
done
if (( ${#need_apt[@]} )); then
  echo "[1/9] apt installing: ${need_apt[*]}"
  if [[ $(id -u) -ne 0 ]]; then SUDO=sudo; else SUDO=; fi
  $SUDO apt-get update -y
  $SUDO apt-get install -y "${need_apt[@]}"
else
  echo "[1/9] apt prereqs already present"
fi

# ---- 2. venv ----------------------------------------------------------
if [[ ! -d venv ]]; then
  echo "[2/9] creating venv ./venv"
  "$PY" -m venv venv
else
  echo "[2/9] venv already exists"
fi
# shellcheck disable=SC1091
source ./venv/bin/activate

LOCK_FILE="${LOCK_FILE:-./requirements.lock}"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-./wheelhouse}"

# ---- 3. requirements --------------------------------------------------
if [[ -d "$WHEELHOUSE_DIR" ]] && [[ -f "$LOCK_FILE" ]]; then
  echo "[3/6] pip install from local wheelhouse using $LOCK_FILE"
  if ! pip install --no-index --find-links "$WHEELHOUSE_DIR" -r "$LOCK_FILE"; then
    echo "ERROR: offline wheelhouse install failed." >&2
    echo "       This copy's ./wheelhouse is incomplete or does not match" >&2
    echo "       ./requirements.lock. Rebuild/copy the full wheelhouse and retry." >&2
    exit 1
  fi
else
  echo "ERROR: wheelhouse or requirements.lock missing." >&2
  echo "       Expected: $WHEELHOUSE_DIR and $LOCK_FILE" >&2
  exit 1
fi

# ---- 4. Falcon-Perception (vendored) ----------------------------------
if [[ -d Falcon-Perception ]]; then
  echo "[4/6] pip install -e ./Falcon-Perception --no-deps"
  pip install -e "./Falcon-Perception" --no-deps
else
  echo "ERROR: Falcon-Perception source missing at ./Falcon-Perception" >&2
  exit 1
fi

# ---- runtime dirs / env ----------------------------------------------
MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR/hf" logs run videos

echo
echo "[note] install.sh does not fetch model weights."
echo "       Runtime expects repo-local bundles under:"
echo "         ./models/hf/<repo>/<name>/<snapshot>/"
echo "       Verify when Gemma download/copy is complete:"
echo "         python scripts/verify_offline_bundle.py --production"
echo "         python scripts/verify_offline_python_env.py"
echo

# ---- perms -----------------------------------------------------------
echo "[5/6] chmod +x *.sh"
chmod +x ./*.sh

if [[ ! -f .env ]] && [[ -f .env.example ]]; then
  echo "[6/6] copied .env.example -> .env"
  cp .env.example .env
fi

echo
echo "================================================================"
echo "Install complete."
echo
echo "Next steps:"
echo "   1) finish/copy the Gemma bundle under ./models/hf/google/..."
echo "   2) edit ./config.yaml  (cameras, classifiers, RTSP URLs)"
echo "   3) bash ./start.sh     (boots Gemma BF16 server + app)"
echo "   4) open http://localhost:\${APP_PORT:-3902}"
echo "================================================================"
