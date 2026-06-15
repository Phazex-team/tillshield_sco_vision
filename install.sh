#!/usr/bin/env bash
# Fresh-DGX installer for fraud_detection_v3.
#
# Idempotent. Run from inside the project directory.
#
# Steps:
#   1. apt: install ffmpeg, git, curl, python3-venv (only if missing)
#   2. create ./venv (Python 3.12)
#   3. pip install -r requirements.txt inside venv
#   4. install ./Falcon-Perception (vendored) editable
#   5. apply source-level patches to vLLM's gemma4.py (see gemma4_patched.py)
#   6. download Falcon weights to ./models/falcon (HF cache)
#   7. download Gemma NVFP4 weights to ./models/gemma
#   8. chmod +x *.sh
#   9. write a starter .env if none present
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
python -m pip install --upgrade pip wheel setuptools >/dev/null

# ---- 3. requirements --------------------------------------------------
# GB10 (sm_121) needs torch nightly + matching vLLM nightly built for
# CUDA 12.9. They live on non-PyPI indexes; pass them via --extra-index-url.
if [[ -f requirements.txt ]]; then
  echo "[3/9] pip install -r requirements.txt (with torch + vLLM nightly indexes for cu130)"
  pip install \
    --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130/ \
    --pre \
    -r requirements.txt
else
  echo "ERROR: requirements.txt missing" >&2; exit 1
fi

# ---- 4. Falcon-Perception (vendored) ----------------------------------
# IMPORTANT: don't install with the [torch] extra -- that pulls
# torch>=2.11 and breaks the vLLM ABI match. The plain install brings
# in just the package + non-torch deps (datasets, einops, etc.).
if [[ -d Falcon-Perception ]]; then
  echo "[4/9] pip install -e ./Falcon-Perception (no [torch] extra to keep vLLM ABI)"
  pip install -e "./Falcon-Perception" --no-deps
  pip install datasets einops pycocotools scipy hf-transfer typeguard tyro || true
else
  echo "[4/9] Falcon-Perception not vendored locally; installing PyPI fallback"
  pip install falcon-perception --no-deps || true
fi

# ---- 5. apply source-level vLLM patches ------------------------------
# Two surgical edits to vLLM's gemma4.py are needed for the
# bg-digitalservices Gemma-4-26B-A4B-it-NVFP4 checkpoint to load:
#   (a) experts.{e}.{proj} -> moe.experts.{e}.{proj}  (key prefix)
#   (b) drop reduce_results=True from FusedMoE init (no longer accepted)
# See gemma4_patched.py for the full rationale. Idempotent.
echo "[5/9] applying source-level vLLM patches via gemma4_patched.py"
if ! python gemma4_patched.py apply; then
  echo "ERROR: gemma4_patched.py apply failed" >&2
  exit 1
fi

# ---- model dirs ------------------------------------------------------
MODEL_DIR="${MODEL_DIR:-./models}"
mkdir -p "$MODEL_DIR/hf" logs run videos

echo
echo "[note] install.sh no longer fetches model weights at install time."
echo "       Production runtime loads weights from repo-local"
echo "       ./models/hf/<repo>/<name>/<snapshot>/ — populate that bundle"
echo "       on a connected machine via:"
echo "         python scripts/prepare_offline_model_bundle.py --download-approved"
echo "       Then verify with:"
echo "         python scripts/verify_offline_bundle.py --production"
echo

# ---- perms -----------------------------------------------------------
echo "[6/7] chmod +x *.sh"
chmod +x ./*.sh

if [[ ! -f .env ]] && [[ -f .env.example ]]; then
  echo "[7/7] copied .env.example -> .env"
  cp .env.example .env
fi

echo
echo "================================================================"
echo "Install complete."
echo
echo "Next steps:"
echo "   1) edit ./config.yaml  (cameras, classifiers, RTSP URLs)"
echo "   2) bash ./start.sh     (boots vLLM + app)"
echo "   3) open http://localhost:\${APP_PORT:-3902}"
echo "================================================================"
