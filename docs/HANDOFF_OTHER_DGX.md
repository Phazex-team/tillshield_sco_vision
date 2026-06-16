# PhazeX Return/Refund Platform - Other DGX Handoff

This USB copy was prepared from:

- Source repo: `/home/fazil/workspace/projects/fraud_detection_v3`
- USB repo: `/media/fazil/PHAZEX_USB/fraud_detection_v3`
- Latest copied commit: `33273a2 Block M: close the 2 wiring/naming gaps Block L missed`
- USB filesystem: ext4, label `PHAZEX_USB`
- Copy policy: source/runtime repo copied, but oversized/unneeded generated folders were excluded.

## Product Goal

This app is now the local/offline PhazeX Retail Return/Refund Visual Evidence Investigation Platform.

It is not an automatic fraud accusation system. The platform links POS return/refund events to CCTV evidence, reconstructs the matching video window, runs perception/tracking/OCR/reasoning, packages evidence, and routes the case to human review.

Allowed outcomes only:

- `VERIFIED`
- `REVIEW`
- `HIGH_RISK_REVIEW`
- `INVALID_VIDEO`

Human review remains required for adverse business action.

## What Claude Implemented

Claude transformed the MVP into a production-oriented local/offline app in phases:

- Safety baseline and git history.
- Offline model registry and bundle verification.
- Offline Python wheelhouse verification.
- Qwen3-VL primary provider with Gemma fallback provider.
- Lazy provider loading so Qwen/Gemma are not loaded at app startup.
- Memory guard:
  - soft threshold: defer new inference
  - hard threshold: unload model providers and clear CUDA cache
  - emergency threshold: stop inference workers while keeping API/recorder alive if possible
- POS ingest and case creation.
- TillShield app-side integration:
  - `POST /api/v1/integrations/tillshield/transactions/event`
  - `POST /api/v1/integrations/tillshield/transactions/batch`
- Continuous CCTV segment recorder and immutable segment index.
- POS-to-video window builder using POS event time, not ingest time.
- Missing/corrupt/insufficient footage paths produce `INVALID_VIDEO`.
- Perception pipeline modules:
  - Falcon Perception adapter/client
  - SAM2 client/runtime checks
  - sampling
  - traditional tracking
  - temporal memory
  - OCR crop path
  - keyframes
- Evidence persistence:
  - detections
  - tracks
  - track observations
  - keyframes
  - OCR results
- Evidence package and basic evidence graph.
- Reviewer UI:
  - case queue
  - video playback
  - keyframes
  - timeline/model claims/limitations
  - reviewer action form
  - audit trail
- Admin prompt endpoint with banned phrase validation.
- Storage retention and disk safety:
  - raw unlinked CCTV retention: 10 hours
  - linked evidence preserved
  - low disk pauses recorder, keeps API/UI alive
- Startup checks for production/offline mode.
- Tests for the above.

## What Codex Verified

Codex was used as the verification lane, not the primary coding lane.

Important verified checkpoints:

- `python -m pytest tests/ -q`
  - latest verified source result before USB copy: `199 passed, 3 skipped`
- `scripts/verify_offline_bundle.py --production`
  - passed on the full source repo before USB copy when all required models were present
- `scripts/verify_offline_python_env.py`
  - passed
- `FRAUD_OFFLINE_MODE=1` startup checks
  - passed with provider chain `qwen3_vl -> gemma`
- Block K/L/M corrective audits found and forced fixes for false/incomplete Claude claims.

Key verified safety behavior:

- VLM-only physical-item claim cannot produce `VERIFIED`.
- Runtime perception shape using `track_id` can produce `VERIFIED` only when a real physical-item track reaches counter/staff.
- Persisted perception shape using `tracker_id` is also accepted.
- Legacy monitor path now accepts `legacy_review_only=True` cleanly and returns review-safe output.
- Evidence package literal file hash is exposed outside the self-verifying package hash.
- UI/static user-facing accusation language was checked.
- TillShield app-side tests passed.

## Models Bundled In This USB Copy

Included under `models/hf/`:

- `Qwen/Qwen3-VL-30B-A3B-Instruct`
- `facebook/sam2-hiera-large`
- `tiiuae/Falcon-Perception`

Not included because the 128 GB USB did not have enough usable space:

- `google/gemma-4-26B-A4B-it`

The missing Gemma folder is the main remaining copy item if full offline fallback is required.

## Folders Intentionally Excluded From USB

Excluded because they are generated, too large, not portable, or not needed:

- `models/hf/google/gemma-4-26B-A4B-it/`
- `venv/`
- `logs/`
- `videos/`
- `run/`
- `storage/`
- `__pycache__/`
- `.pytest_cache/`

`videos/` was explicitly not needed.

## Remaining Copy Item

For full production/offline verification on the other DGX, copy this folder separately into the same relative path:

```text
models/hf/google/gemma-4-26B-A4B-it/
```

Expected source path on this DGX:

```text
/home/fazil/workspace/projects/fraud_detection_v3/models/hf/google/gemma-4-26B-A4B-it/
```

Expected destination path on the other DGX:

```text
<repo>/models/hf/google/gemma-4-26B-A4B-it/
```

Approximate size: `49G`.

Without this folder:

- Qwen3-VL primary, SAM2, and Falcon Perception are present on USB.
- Gemma fallback is missing.
- Production offline bundle verification may fail until Gemma is restored or config/asset policy is intentionally changed and reverified.

## Other DGX Setup Order

On the other DGX:

1. Copy the USB repo folder to the target location.
2. Copy the missing Gemma folder into:

   ```text
   <repo>/models/hf/google/gemma-4-26B-A4B-it/
   ```

3. Create a fresh Python venv from the copied wheelhouse. Do not reuse the old `venv/`.
4. Run:

   ```bash
   python scripts/verify_offline_bundle.py --production
   python scripts/verify_offline_python_env.py
   FRAUD_OFFLINE_MODE=1 python - <<'PY'
   from app.startup import run_startup_checks
   print(run_startup_checks())
   PY
   ```

5. Then run tests if desired:

   ```bash
   python -m pytest tests/ -q
   ```

6. Start app:

   ```bash
   FRAUD_OFFLINE_MODE=1 python scripts/run_app.py --host 127.0.0.1 --port 3902
   ```

7. Start recorder separately when camera config is ready:

   ```bash
   python scripts/run_segment_recorder.py --segment-duration-sec 60
   ```

Reviewer UI:

```text
http://127.0.0.1:3902/review.html
```

OpenAPI:

```text
http://127.0.0.1:3902/api/v1/docs
```

## Important Caveats

- Do not call this automatic fraud detection.
- Do not allow user-facing fraud/accusation language.
- Do not run production without verifying offline bundle and Python env.
- Do not start large model inference until memory guard/status is checked.
- `venv/` was intentionally excluded; rebuild it from `wheelhouse/`.
- Old `logs/` and `videos/` were intentionally excluded.
- Advanced MLOps/evaluation remains deferred by decision:
  - labelled benchmark dataset program
  - benchmark dashboard
  - experiment/model/prompt registry UIs
  - long-term calibration studies

## Last Known Good Source Verification

Before USB transfer, the source repo verified with:

- Latest commit: `33273a2`
- Tests: `199 passed, 3 skipped`
- Offline bundle verifier: passed
- Offline Python env verifier: passed
- Startup checks: passed

On the USB copy, full production bundle verification will require the missing Gemma folder to be restored first.
