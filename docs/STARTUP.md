# SCO Vision — Production Startup Guide

This document is the **operator-facing** runbook for booting and shutting
down the full SCO Vision stack. It supersedes the legacy refund-flow
boot notes in `README.md` for day-to-day SCO operation.

## Default mode (Qwen-only)

```bash
bash start.sh           # boots Qwen on :8000, then the app on :3902
bash stop.sh            # graceful stop (app, Qwen, Gemma if running, Phoenix)
```

`start.sh` is idempotent. Re-running it does not relaunch services that
already report healthy. `qwen_vllm_start.sh` refuses to double-launch a
second Qwen on the same port — if a stale PID file points at a process
that fails health, stop that process or remove `run/qwen.pid` before
retrying. The Gemma BF16 launcher follows the same contract.

### What comes up by default

| Service | Port | Default | Health probe |
|---|---|---|---|
| Qwen3-VL vLLM (primary VLM) | 8000 | **ON** | `/health` AND `/v1/models` contains `qwen3_vl` |
| FastAPI app | 3902 | **ON** | `/api/v1/health` |
| Gemma BF16 (fallback VLM) | 8001 | **OFF** | `/health` |
| Phoenix telemetry | 6006 | **OFF** | `/health` |

### URLs the operator opens

| Surface | URL |
|---|---|
| Reviewer UI | `http://localhost:3902/static/review.html` |
| API health | `http://localhost:3902/api/v1/health` |
| API docs (Swagger) | `http://localhost:3902/api/v1/docs` |
| Qwen models list | `http://localhost:8000/v1/models` |
| Gemma health (if enabled) | `http://localhost:8001/health` |
| Phoenix (if enabled) | `http://localhost:6006` |

## Operator switches

`start.sh` reads these env vars (settable in `.env` or the shell):

| Var | Default | Effect |
|---|---|---|
| `START_QWEN` | `1` | Bring up Qwen on `:8000`. Set `0` if Qwen is managed externally. |
| `START_GEMMA` | `0` | Bring up Gemma BF16 fallback on `:8001`. |
| `START_PHOENIX` | `0` | Bring up Phoenix telemetry on `:6006`. |
| `APP_PORT` | `3902` | App HTTP port. |
| `QWEN_PORT` | `8000` | Qwen vLLM port (consumed by `scripts/qwen_vllm_start.sh`). |

## Qwen launcher (working flag set)

`scripts/qwen_vllm_start.sh` launches Qwen3-VL with the **only** flag
combination known to work on this DGX Spark (GB10 / sm_121):

```
--moe-backend triton
--enforce-eager
--no-enable-flashinfer-autotune
```

Without those flags, the default vLLM path activates FlashInfer
CUTLASS fused-MoE for Qwen3-VL-30B-A3B-Instruct-FP8 and crashes during
JIT compile (exit 137 / SIGKILL). The launcher writes its PID to
`run/qwen.pid` and logs to `logs/qwen.log`. Health gate is two-stage:
`/health` must respond AND `/v1/models` must list the served name
(`qwen3_vl`).

Operator tunables (env or `.env`):

| Var | Default | Notes |
|---|---|---|
| `QWEN_MOE_BACKEND` | `triton` | Keep at `triton` until vLLM ships a working FlashInfer kernel for this SKU. |
| `QWEN_ENFORCE_EAGER` | `1` | Disables CUDA graphs. Set `0` only after re-verifying the launch. |
| `QWEN_DISABLE_FLASHINFER_AUTOTUNE` | `1` | Disables the JIT autotune that crashes the model load. |
| `QWEN_GPU_MEM_UTIL` | `0.70` | vLLM `--gpu-memory-utilization`. |
| `QWEN_MAX_MODEL_LEN` | (empty) | Override max context if needed; leave empty to take the model default. |
| `QWEN_EXTRA_ARGS` | (empty) | Free-form extra args appended to the `vllm serve` command. |

## Cold-start expectations

| Model | First-time load (cold) | Memory footprint |
|---|---|---|
| Qwen3-VL FP8 | 3–6 min via vLLM (kernel autotune + KV cache warmup) | model + KV ≈ 25–35 GiB on the GPU; ~60 GiB unified at peak |
| Gemma BF16 | ~4–5 min from disk to ready (1013-shard load) | model ≈ 50 GiB unified |

The app does NOT wait for Qwen to finish its KV-cache warmup. It waits
only for `/v1/models` to respond. First analyze_case after start may
incur a one-off kernel autotune.

## Vision provider chain — production defaults

The runtime provider chain is controlled by `config.yaml` under
`reasoning`:

```yaml
reasoning:
  primary_provider: qwen3_vl
  fallback_provider: null      # OFF — set to "gemma" to re-enable
  warm_fallback: false
```

### Why Qwen-only by default

- **Qwen3-VL** has stronger temporal/video grounding than the local
  Gemma BF16 path and produces cleaner v2 SCO basket-match JSON.
  Used as the sole VLM for day-to-day SCO operation.
- **Gemma BF16** code paths are preserved in tree (`gemma_reasoner.py`,
  `reasoning/providers/gemma.py`, `vllm_start.sh`,
  `transformers_server.py`) but the chain-level fallback is OFF by
  default so a Qwen failure surfaces a clear error in the reviewer UI
  instead of silently switching providers.
- To re-enable Gemma fallback: set `reasoning.fallback_provider: gemma`
  AND boot the Gemma server (`START_GEMMA=1 bash start.sh`).

### When Qwen is unavailable

- The app does NOT silently fall through to Gemma unless fallback is
  explicitly enabled.
- The case grid shows the case in `REVIEW` with `_chain_attempts`
  containing `qwen3_vl=err:...` and no successful provider entry.
- The reviewer UI's Model-claims panel renders the error reason so
  the operator can act (restart Qwen, enable Gemma, etc.).

## Smoke commands

```bash
# Quick startup smoke
bash start.sh

# Is Qwen up and serving the right model name?
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Is the app healthy?
curl -s http://localhost:3902/api/v1/health

# Is the reviewer UI loading?
curl -sI http://localhost:3902/static/review.html | head -1

# Memory state (psutil-backed; checks the app process pool)
curl -s http://localhost:3902/api/v1/memory | python3 -m json.tool

# Provider chain readiness
curl -s http://localhost:3902/api/v1/ops/status | python3 -c \
  'import sys, json; d=json.load(sys.stdin); print(json.dumps(d.get("qwen3_vl") or d.get("vlm"), indent=2))'
```

## Stop

```bash
bash stop.sh           # stops app, Qwen, Gemma (if running), Phoenix
```

`stop.sh` SIGTERMs each process, waits up to 10s, then SIGKILLs if
still alive. PID files in `run/` are cleared on success.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `qwen: did not reach /v1/models within Ns` | Cold load took longer than `QWEN_HEALTH_TIMEOUT_SEC` (default 1800s) | Bump the timeout, or check `logs/qwen.log` for the actual cause |
| Qwen exits 137 (SIGKILL) at JIT compile | Wrong flag set — FlashInfer autotune still on | Confirm `QWEN_DISABLE_FLASHINFER_AUTOTUNE=1` is in env, or run `scripts/qwen_vllm_start.sh` directly (it sets the right flags) |
| App refuses to call Qwen even though `:8000` is up | Stale memory_guard state from a previous run | Hit `/api/v1/memory` to read state; external HTTP providers should never be blocked but in-process providers are. See `app/memory_guard.py` and `reasoning/providers/chain.py` per-provider gate. |
| Case shows `Vision did not run / all providers errored` | Qwen down (no fallback configured) | Restart Qwen (`bash scripts/qwen_vllm_start.sh`) or temporarily enable Gemma fallback in config + start the Gemma server |
| Qwen launcher refuses to start a second instance | Existing PID file points at a healthy or stuck process | `bash stop.sh` first, or remove `run/qwen.pid` if you know the process is gone |

## What `start.sh` no longer does

- Does NOT launch the Gemma server by default. Old behaviour: always-on.
- Does NOT launch Phoenix by default. Old behaviour: always-on (`phoenix_start.sh`).
- Does NOT silently fall through to Gemma on Qwen failure. Old chain default was `gemma`; new default is `null` so a Qwen outage is visible to the reviewer.

The Gemma and Phoenix code paths are preserved verbatim — only the
default boot policy changed.
