# Fraud Detection v3 — Multi-Camera Video Reasoning

Fraud / safety / QC / shelf / access video reasoning over RTSP streams.
Falcon Perception (object detection) + Gemma 4 26B-A4B-it (NVFP4 video
reasoning, served by vLLM). Per-camera classifiers, per-camera prompts,
per-camera zones. FastAPI dashboard with live MJPEG and a per-session
evidence MP4.

```
RTSP cam ─► motion+ROI ─► clip ─► Falcon (objects) ─► vLLM/Gemma (judgment)
                                          │                  │
                                          └──── PNG snapshot + CSV row + MP4
```

## Hardware

| | |
|---|---|
| Reference platform | **NVIDIA DGX Spark (GB10 Grace Blackwell)** |
| Memory | 128 GB unified (LPDDR5x, 273 GB/s) |
| OS / arch | Ubuntu 24.04, aarch64 |
| CUDA | 13.0 |
| Python | 3.12 |

The whole stack also runs on any single-GPU x86 host with ~70 GB of
VRAM (Gemma NVFP4 is ~13 GB resident; Falcon ~6 GB; vLLM KV cache and
activations need the rest).

## Fresh install

```bash
git clone <this-repo> fraud_detection_v3
cd fraud_detection_v3
bash install.sh           # apt + venv + pip + model downloads
cp .env.example .env      # edit if you want non-default ports / paths
bash start.sh             # boots vLLM (8001) then app (3902)
open http://localhost:3902
```

`install.sh` is idempotent — re-running it is safe.

## Files

| Path | Purpose |
|---|---|
| `app.py` | FastAPI dashboard + REST endpoints |
| `monitor.py` | Pipeline (CameraWorker, InferenceWorker, SessionDispatcher) |
| `falcon_detector.py` | Falcon Perception wrapper (in-process, bf16, CUDA) |
| `gemma_reasoner.py` | vLLM HTTP client (Gemma 4 NVFP4) |
| `gemma4_patched.py` | Source-level patcher for vLLM's `gemma4.py` (Fix A + Fix B — see "Known vLLM Patches" below) |
| `classifiers.py` | **Single source of truth** for built-in classifier defaults |
| `frame_broker.py` | Per-camera MJPEG + last-detection registry |
| `session_logger.py` | CSV writer + retention janitor |
| `rtsp_reader.py` | OpenCV/FFMPEG RTSP capture with reconnect |
| `zone_trigger.py` | Customer-zone motion detector |
| `overlay.py` | OpenCV draw routines (zones, bboxes, banners) |
| `video_encoder.py` | ffmpeg subprocess H.264 evidence encoder |
| `config.yaml` | Cameras, zones, classifiers, settings, vLLM URL |
| `static/index.html` | Single-page dashboard |
| `vllm_start.sh` | Launch vLLM with the gemma4 patch applied |
| `start.sh` / `stop.sh` / `status.sh` | One-command lifecycle |
| `install.sh` | Fresh-DGX setup |
| `.env.example` | All overridable env vars |

## Configuration guide

`config.yaml` is the only file you normally touch. Each camera entry:

```yaml
cameras:
  - id: cam_01                       # short stable id (used in CSV + session ID prefix)
    name: "Return Counter — Qusais"  # human label shown on the dashboard
    rtsp_url: rtsp://...             # source stream
    classifier: fraud                # one of fraud/safety/manufacturing/shelf/access/custom
    token_budget: 1120               # 70/140/280/560/1120; empty -> classifier default
    cooldown_sec: 30                 # min gap between sessions on this camera
    zones:                           # PIXEL coords on the camera's native frame
      customer_zone: {x: 874, y: 29, w: 360, h: 607}
      staff_zone:    {x: 570, y: 4,  w: 303, h: 645}
    prompts:                         # per-camera overrides; "" = use classifier default
      falcon: ""
      gemma_system: ""
      gemma_user: ""
```

Add another camera: append another entry. No code changes needed.

### Adding a new classifier

Open `classifiers.py` and add a new key under `CLASSIFIERS`:

```python
"warehouse": {
    "display_label": "Warehouse Pick Audit",
    "color": "#0fbcb0",        # any CSS color (used on dashboard badges)
    "token_budget": 560,
    "falcon_prompt": "box, pallet, scanner, picker, cart",
    "gemma_system": "...",      # use {start_objects} / {action_objects}
    "gemma_user":   "...",
},
```

That's it. The new classifier:
- shows up in the dashboard's classifier dropdown (Config tab)
- is filterable in the Logs tab
- gets its own row in the daily report
- is referenced by setting `classifier: warehouse` on any camera

### Editing prompts per camera

In the Config tab, pick the camera, then edit Falcon / Gemma system /
Gemma user. **Empty = use the classifier default.** Non-empty wins.

A change takes effect on the next session.

## Ports

| Port | Service | Don't collide with |
|---|---|---|
| **3902** | v3 dashboard (this repo) | v1=3900, v2=3901, validator=39003 |
| **8001** | vLLM OpenAI server | ollama=11434, open_webui=3000 |
| **6006** | Phoenix tracing UI + OTLP/HTTP receiver | tensorboard=6006 (collides — see `.env`) |

Override in `.env` if you have a conflict.

## Logs and storage

| Path | What |
|---|---|
| `./logs/app.log` | App stdout/stderr |
| `./logs/vllm.log` | vLLM stdout/stderr |
| `./logs/sessions_YYYY-MM-DD.csv` | One row per analysed session |
| `./logs/report_YYYY-MM-DD.txt` | Daily summary (per-classifier breakdown) |
| `./logs/snapshots/` | Annotated PNG (full-res) per session |
| `./videos/` | Downscaled H.264 MP4 evidence per session |
| `./run/{vllm,app}.pid` | PID files used by stop.sh / status.sh |

Default retention: snapshots/MP4 deleted after 7 days, CSV/reports after
14. Tune in `config.yaml` (`settings.retention_days`).

## CSV columns

```
session_id, camera_id, camera, classifier, scenario_label,
start_time, end_time, duration_sec,
handover_occurred, item_presented, item_count,
customer_description, items_handed_over,
confidence, flag_for_review,
num_people, per_person_json, narrative,
objects_detected,
merged_from, merged_count,
snapshot_path, mp4_path
```

`session_id` format: `<CAM_PREFIX>-NNN` (e.g. `CAM01-017`). The prefix
is derived from `cameras[].id` so multi-camera setups don't collide.

## GB10 / sm_121 — NVFP4 hardware path is unsupported

> **Status (2026-04-29):** This DGX Spark / GB10 (Grace-Blackwell server,
> compute capability **sm_121**) does **not** implement the FP4
> conversion instruction NVFP4 kernels rely on. The system therefore
> runs Gemma 4 in **BF16** via `transformers_server.py`, not the NVFP4
> + vLLM path described in the original v3 plan. `vllm_start.sh` now
> launches the BF16 server; `gemma_reasoner.py` and the rest of the app
> are unchanged.

### What we tried

Every NVFP4 path available in vLLM 0.19.2-nightly was attempted:

| MoE backend | Result |
|---|---|
| `VLLM_CUTLASS` (default) | NaN logits — sm_120 kernel binaries miscompute on sm_121 |
| `MARLIN` | Gibberish output — Marlin's NVFP4 dequant assumes ≥0 scales; this checkpoint has signed scales |
| `EMULATION` | Numerically correct but **0.2 tok/s** (~5 min per session) |
| `FLASHINFER_TRTLLM` / `FLASHINFER_CUTEDSL` | Engine refuses init: kernel does not support cuda device |
| `FLASHINFER_CUTLASS` | Engine refuses init: kernel does not support GELU MoE activation |

A from-source rebuild of vLLM 0.19.1 with `TORCH_CUDA_ARCH_LIST=12.1`
also failed — at the PTX-assembly level, ~23×:

```
csrc/quantization/fp4/nvfp4_quant_kernels.cu →
ptxas error : Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'
```

`cvt.e2m1x2` is the FP4 (E2M1 packed-pair) conversion instruction every
NVFP4 dequant uses. ptxas in CUDA 13.0 will not emit it for `sm_121`.
This is an **ISA-level limitation of GB10**, not a vLLM-version or
kernel-binary issue — sm_120 (consumer Blackwell) and the larger
Blackwell server SKUs include this instruction; GB10 does not.

### Practical effect

* **Memory:** BF16 weights are ~52 GB resident vs ~16 GB for NVFP4.
  Stack peaks at ~85-110 GB used out of 121 GB unified pool.
* **Latency:** ~73 s per session (20 frames, sequential transformers,
  no batching) vs ~10-15 s the NVFP4 plan targeted. Adequate for 1-2
  cameras at ~60 s session cadence; would not scale to many cameras
  without a batch-capable inference engine.
* **Quality:** identical model weights and chat template, so output
  quality matches the NVFP4 plan (verified by 9/9 correct fraud-flag
  sessions on CAM01-446..454).

### To re-enable NVFP4 in the future

One of:

1. **NGC Blackwell-aware vLLM container** — NVIDIA may ship sm_121-
   compatible NVFP4 kernels there before the open toolchain.
2. **NVIDIA toolchain update.** Watch
   <https://developer.nvidia.com/cuda-toolkit> for a CUDA / nvcc
   release that adds a software fallback (or hardware support, if any
   GB10 stepping ever lands) for `cvt.e2m1x2` on sm_121.
3. **Re-quantize the checkpoint with non-negative scales** — would
   unblock the MARLIN MoE backend (~14 tok/s) without touching the
   hardware path.

When any of these lands, swap `vllm_start.sh` back to launching
`vllm serve bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 ...` (the
previous version is preserved in git history; the gemma4 patches
described below are still required for the NVFP4 checkpoint).

## Known vLLM Patches (NVFP4 — currently inactive)

These are kept for reference and to make the NVFP4 path easy to revive
once the ISA blocker above clears. They are *not* applied by the BF16
launcher and have no effect on the running stack today.

Two source-level edits are made to `vllm/model_executor/models/gemma4.py`
inside the venv when the NVFP4 path is in use. Both are applied
automatically by `install.sh` (which calls `python gemma4_patched.py
apply`). They are idempotent and a backup of the unpatched file is
written next to it as `gemma4.py.preupstreampatch.bak`.

### Working stack (NVFP4 — original target, currently blocked)

```
vLLM       0.19.2 nightly     (wheels.vllm.ai/nightly/cu130/)
torch      2.11.0 + cu130     (download.pytorch.org/whl/nightly/cu130)
torchvision 0.26.0 + cu130
transformers 5.6.2
```

### Fix A — MoE expert key prefix

```diff
- f"experts.{expert_id}.{proj_name}"
+ f"moe.experts.{expert_id}.{proj_name}"
```

vLLM's expert-weight loader walks `expert_params_mapping` looking for
checkpoint keys named `experts.{e}.{proj}` and remapping them onto the
fused `experts.w13_*` / `experts.w2_*` parameter names. The
`bg-digitalservices` NVFP4 checkpoint exposes the same weights under
`moe.experts.{e}.{proj}` (the parent block is named `moe`). Without the
prefix the loader silently misses every expert weight and the model
serves zeroed expert kernels.

### Fix B — drop `reduce_results=True` from `FusedMoE(...)`

```diff
  self.experts = FusedMoE(
      ...
-     reduce_results=True,
      renormalize=True,
      quant_config=quant_config,
      prefix=f"{prefix}.experts",
      ...
  )
```

vLLM nightly's `FusedMoE.__init__` no longer accepts `reduce_results`;
keeping the kwarg raises `TypeError: FusedMoE.__init__() got an
unexpected keyword argument 'reduce_results'` at engine init.

### Patch lifecycle

| Action | Command |
|---|---|
| Apply (idempotent) | `python gemma4_patched.py apply` |
| Verify (no writes) | `python gemma4_patched.py verify` |
| Restore from backup | `python gemma4_patched.py restore` |

`install.sh` runs `apply` automatically on a fresh venv. The current
`vllm_start.sh` is the BF16 launcher and does not run these patches —
they only matter if you switch back to the NVFP4 path (see "GB10 /
sm_121 — NVFP4 hardware path is unsupported" above).

## Troubleshooting

| Symptom | Try |
|---|---|
| `bash status.sh` shows vLLM = STOPPED | `tail -n 200 logs/vllm.log` — usually OOM, model not downloaded, or port collision. |
| App health is 200 but no live feed | Check the camera's RTSP URL in `/config` and use the Test Connection button in the Config tab. |
| `vllm: timed out waiting for /health` | BF16 weight load (~52 GB) takes 5-10 min on first run. Bump `HEALTH_TIMEOUT_SEC` in `.env` (default 1800). |
| Empty `content` from Gemma / `unparsable model output` in CSV | Almost certainly someone reverted to the NVFP4 path. Confirm `cat run/vllm.pid` points at `transformers_server.py`, not `vllm serve`. See "GB10 / sm_121 NVFP4 unsupported". |
| Dashboard says "loading models…" forever | Falcon weights are downloading. `tail -f logs/app.log`. |
| `TypeError: FusedMoE.__init__() got an unexpected keyword argument 'reduce_results'` | Fix B was not applied. Run `python gemma4_patched.py apply`. |
| `KeyError: 'layers.0.moe.moe.experts.0.down_proj.weight'` or empty / nonsense Gemma replies | Fix A was not applied. Run `python gemma4_patched.py apply`. |
| Session merges that shouldn't | Lower `settings.session_merge_similarity_threshold` (default 0.6). |
| Classifier change in UI doesn't pick new prompts | The Save button clears `prompts.*` overrides on classifier change so defaults take effect — re-save your custom prompts after switching classifiers if you want to keep them. |

## Stopping / restarting

```bash
bash stop.sh              # stops app + vLLM
bash start.sh             # restarts both
bash status.sh            # who's up, GPU usage, recent log tails
```

`stop.sh` only touches PIDs from `./run/*.pid`. It will not kill any
other Python / vLLM / app process you have running.

## Observability (Phoenix)

Every Falcon detect, Gemma reasoning call, and session completion is
traced via OpenTelemetry to a local Arize Phoenix server. Use it to
debug detection decisions: each Gemma span carries the full raw model
response (before JSON parsing), the parsed result fields, the
classifier and thinking-mode flag, the frame count, and the latency.

| Port | Service |
|---|---|
| **6006** | Phoenix UI + OTLP/HTTP `/v1/traces` receiver |

Toggle in `config.yaml`:

```yaml
observability:
  phoenix_enabled: true
  phoenix_url: http://localhost:6006
  phoenix_project: fraud_detection_v3
```

`bash start.sh` brings up Phoenix between vLLM and the app. Phoenix
failure does not block the app — tracing degrades to no-op. Open
http://localhost:6006 directly, or click the **🔍 Traces** button in the
dashboard header.

Span names you'll see:
- `falcon.detect` — one per `falcon.detect()` call (frame_type=start|action), with per-detection `det.{i}.label` / `det.{i}.bbox`.
- `gemma.reason` — one per Gemma call, with `raw_response`, `result.*`, `thinking`, `latency_ms`, `num_frames`, `token_budget`. Token usage counts are not captured (would require modifying the HTTP client internals).
- `session.complete` — one per analysed session.

Disable: set `phoenix_enabled: false`. The Phoenix server can also be
stopped independently (`bash stop.sh` stops it; PID is in
`./run/phoenix.pid`).

### Per-classifier reasoning controls

Each classifier has two new tunable knobs in `classifiers.py`:

| Knob | What it does |
|---|---|
| `enable_thinking` | If True, the chat template enables Gemma 4's thinking/reasoning mode for that classifier (sent as `chat_template_kwargs.enable_thinking` on the OpenAI request). Set False for fast binary checks. |
| `max_frames` | Per-classifier frame cap fed to Gemma. Falls back to `settings.gemma_video_max_seconds * gemma_video_fps` if missing. |

Both can be overridden per camera in `config.yaml`:

```yaml
cameras:
  - id: cam_01
    classifier: fraud
    enable_thinking: false   # override the classifier default
    max_frames: 10           # override the classifier default
```

The Config tab on the dashboard surfaces both as form controls.
`quick_describe()` (the cheap merge probe) always uses
`enable_thinking=false` regardless of classifier.

## What's NOT in v3 (yet)

- vLLM does not yet shard Gemma across multiple GPUs (the model fits on
  one). To multi-GPU, set `--tensor-parallel-size` via
  `VLLM_EXTRA_ARGS` in `.env`.
- Authentication. Bind the dashboard to localhost or front it with
  nginx + auth in production.
- Falcon detection still runs in-process under a single global lock —
  on hosts with several cameras + heavy traffic it will be the
  bottleneck before vLLM is.
