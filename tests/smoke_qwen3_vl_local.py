"""Optional real-load smoke for the Qwen3-VL provider.

Gated by ``RUN_REAL_QWEN_SMOKE=1`` so it never runs in CI by default.
Loads Qwen3-VL from the **repo-local** bundle (``./models/hf/...``) and
runs one tiny image + text inference. Confirms the review-safe JSON
schema parses.

Usage:
    RUN_REAL_QWEN_SMOKE=1 python -m tests.smoke_qwen3_vl_local

Exit codes:
    0  load + one inference succeeded; JSON parsed
    1  load OK but inference produced an unparseable response
    2  load failed or no repo-local snapshot
    77 skipped (RUN_REAL_QWEN_SMOKE not set)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _gate() -> int:
    if os.environ.get("RUN_REAL_QWEN_SMOKE") != "1":
        print("RUN_REAL_QWEN_SMOKE != 1 — skipping the real load smoke.")
        return 77
    return 0


def _build_manifest():
    from PIL import Image

    from reasoning.providers.base import EvidenceManifest

    img = Image.new("RGB", (224, 224), color=(40, 90, 140))
    # Draw a coloured rectangle so the model has something to describe.
    for x in range(60, 160):
        for y in range(60, 160):
            img.putpixel((x, y), (220, 220, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return EvidenceManifest(
        case_id="smoke-001",
        camera_id="cam_test",
        window_start_ts="2026-06-15T14:00:00Z",
        window_end_ts="2026-06-15T14:00:01Z",
        frames=[{"frame_id": "f0", "ts": "2026-06-15T14:00:00Z",
                 "image_url": f"data:image/jpeg;base64,{b64}"}],
        user_prompt=(
            "Briefly describe the scene. Reply ONLY in JSON: "
            '{"narrative": "...", "confidence": "high|medium|low"}'
        ),
    )


def main() -> int:
    rc = _gate()
    if rc:
        return rc

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from app.config import load_config, resolve_model_path
    from reasoning.providers import get_provider

    cfg = load_config()
    qwen_cfg = cfg.models.get("qwen3_vl")
    if qwen_cfg is None:
        print("config has no qwen3_vl entry", file=sys.stderr)
        return 2

    # Force repo-local resolution; production mode forbids cache fallback.
    try:
        local_path = resolve_model_path(qwen_cfg, production_mode=True)
    except Exception as exc:
        print(f"qwen3_vl not present in offline bundle: {exc}",
              file=sys.stderr)
        print("Run scripts/prepare_offline_model_bundle.py first.",
              file=sys.stderr)
        return 2

    print(f"loading qwen3_vl from {local_path}")
    p = get_provider(
        "qwen3_vl",
        model_name=qwen_cfg.name,
        enabled=True,
        local_path=local_path,
        max_new_tokens=128,
    )

    t0 = time.time()
    result = p.analyze_evidence(_build_manifest())
    dt = time.time() - t0
    print(f"latency_ms={int(dt * 1000)}  error={result.error}")
    print(f"raw_text[:240] = {result.raw_text[:240]!r}")
    print(f"parsed keys    = {list(result.parsed)}")

    if result.error is not None:
        return 2
    if not result.parsed.get("narrative"):
        print("WARN: no narrative parsed", file=sys.stderr)
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
