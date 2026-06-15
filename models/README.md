# Offline model bundle

This directory is the **single source of truth for model weights** when the
repo is copied between machines (e.g. shipped on USB to another DGX Spark).
Runtime loads weights from `./models/hf/<repo>/<name>/<snapshot>/` and
never from `~/.cache/huggingface` in production.

## Layout

```
models/
  hf/
    Qwen/Qwen3-VL-30B-A3B-Instruct/<snapshot>/      # primary verifier (when enabled)
    google/gemma-4-26B-A4B-it/<snapshot>/           # fallback reasoner
    tiiuae/Falcon-Perception/<snapshot>/            # object detection + OCR
    facebook/sam2-hiera-large/<snapshot>/           # segmentation (SAM 2)
  manifest.json
  README.md
```

The four directories above are the **required** production assets.
SAM 3 (`facebook/sam3-*`) is the preferred-upgrade segmenter and lives
under the same `models/hf/` tree once Meta publishes the checkpoint;
the registry tags it `optional` so it cannot block the bundle.
`tiiuae/Falcon-OCR` is also `optional` — the base `Falcon-Perception`
checkpoint performs OCR via natural-language queries per its upstream
README, so the dedicated OCR variant is a quality optimisation, not a
runtime requirement.

`manifest.json` records the snapshot hash, file count, total size, and
sha256 of every file for each model. `scripts/verify_offline_bundle.py`
reads it offline.

## Filling the bundle

The bundle is built **from this machine's HuggingFace cache** by
hardlinking blobs into `./models/hf/...`. Hardlinks share inodes on the
same filesystem, so the bundle adds zero disk pressure here — but the
hardlinked copy is what gets carried across when the directory is
`rsync`'d or `tar`'d to USB.

```
# from the repo root, with the venv active:
python scripts/prepare_offline_model_bundle.py            # hardlink
python scripts/prepare_offline_model_bundle.py --copy     # full copy
```

The script will refuse to follow symlinks that point outside the
HuggingFace cache root and refuses any network access. If a model is
missing from the cache, the script aborts with a clear error — fetch it
on the source machine first; **never** let the runtime download.

## Verifying after transfer

On the destination machine, before starting any service:

```
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/verify_offline_bundle.py
```

The verifier fails fast if:

- a required model is missing under `./models/`,
- the active config points at a `~/.cache/...` path that no longer exists,
- any tracked file's sha256 has drifted from the manifest,
- production mode is requested but a configured-active model is cache-only.

## Production vs. dev mode

The runtime picks weights via `app.config.resolve_model_path`:

| Mode                                            | Search order                                | On failure                |
|-------------------------------------------------|----------------------------------------------|---------------------------|
| **Development** (`FRAUD_OFFLINE_MODE` unset)    | `./models/hf/...` → cache `local_path`       | warn + use cache          |
| **Production / offline** (`FRAUD_OFFLINE_MODE=1`)| `./models/hf/...` only                       | raise `OfflineBundleError`|

Toggle production mode in `.env` or directly via the env var.
