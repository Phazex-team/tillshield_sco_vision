# SCO Vision — container image (venv-mount runtime).
#
# WHY NOT pip-install in the image:
#   This code is pinned to torch==2.11.0+cu130 and vllm==0.19.2rc1.dev205,
#   both of which have already rotated OFF the nightly indexes (they keep
#   only ~2 weeks; today they serve torch 2.14.dev / vllm 0.23.dev). The
#   bundled ./wheelhouse is partial and has no vLLM wheel. So the exact
#   tested stack can NOT be reinstalled from the network.
#
#   The reproducible artifact that DOES exist is the 12 GB host ./venv,
#   already built for this GB10 / sm_121 box with the exact tested
#   torch + vLLM. We mount it at runtime (see docker-compose.yml) instead
#   of reinstalling.
#
# THE SYMLINK TRICK:
#   The venv's console scripts (vllm, uvicorn) and its Falcon-Perception
#   editable finder carry the ABSOLUTE path the venv was created under:
#   /home/fazil/workspace/projects/sco_vision. We recreate that path as a
#   symlink to /app, so every baked-in absolute reference resolves and the
#   venv works verbatim — no relocation, no reinstall.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# Runtime prereqs. python3.12 must match the venv's pyvenv.cfg (3.12.3 on
# Ubuntu 24.04). ffmpeg = evidence encode; curl = healthchecks; libGL/glib
# = opencv-python-headless. gcc/g++/python3-dev = Triton JIT-compiles vLLM
# kernels (e.g. Qwen3-VL's bilinear pos-embed) at load time and needs a
# host C compiler — without it EngineCore init fails. The venv scripts
# resolve python via /usr/bin/python3, which the python3 package provides.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        gcc g++ \
        ffmpeg git curl ca-certificates \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make the C compiler explicit for Triton's runtime build step.
ENV CC=gcc CXX=g++

# Make the venv's baked-in absolute path resolve to /app so its console
# scripts + editable installs work unchanged when ./venv is mounted.
RUN mkdir -p /home/fazil/workspace/projects \
    && ln -s /app /home/fazil/workspace/projects/sco_vision

WORKDIR /app

# Application code (the venv, models, and runtime dirs are bind-mounted).
COPY . .

# Default role is the FastAPI app; compose overrides per service.
CMD ["python", "scripts/run_app.py", "--host", "0.0.0.0", "--port", "3902"]
