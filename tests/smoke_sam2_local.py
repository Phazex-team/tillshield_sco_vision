"""Optional real-load smoke for SAM 2 from the repo-local bundle.

Gated by ``RUN_REAL_SAM2_SMOKE=1`` so it never runs in CI by default.
Confirms ``facebook/sam2-hiera-large`` weights can be loaded from
``./models/hf/`` with no network access.

Usage:
    RUN_REAL_SAM2_SMOKE=1 python -m tests.smoke_sam2_local
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    if os.environ.get("RUN_REAL_SAM2_SMOKE") != "1":
        print("RUN_REAL_SAM2_SMOKE != 1 — skipping the SAM 2 real-load smoke.")
        return 77

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    bundle = REPO_ROOT / "models" / "hf" / "facebook" / "sam2-hiera-large"
    if not bundle.is_dir():
        print(f"sam2 bundle missing at {bundle}", file=sys.stderr)
        return 2
    snaps = [p for p in bundle.iterdir() if p.is_dir()]
    if not snaps:
        print("no sam2 snapshot under bundle", file=sys.stderr)
        return 2
    snap = max(snaps, key=lambda p: sum(f.stat().st_size for f in p.rglob("*")
                                        if f.is_file()))
    print(f"sam2 snapshot: {snap}")

    # Lazy-import sam2 client. The perception/sam2_client wrapper does
    # not require the full sam2 python package to be installed; the
    # smoke only confirms the weights file actually opens.
    weights = list(snap.glob("*.pt")) + list(snap.glob("*.safetensors"))
    if not weights:
        print("no weight shards (.pt/.safetensors) in snapshot",
              file=sys.stderr)
        return 2
    print(f"weight shards: {[w.name for w in weights]}")

    try:
        import torch
        for w in weights:
            if w.suffix == ".pt":
                blob = torch.load(str(w), map_location="cpu",
                                  weights_only=False)
                if isinstance(blob, dict):
                    print(f"  loaded {w.name}: top-level keys = "
                          f"{list(blob)[:5]}")
                else:
                    print(f"  loaded {w.name}: type={type(blob).__name__}")
                break
    except Exception as exc:
        print(f"torch.load failed: {exc}", file=sys.stderr)
        return 2

    print("SMOKE OK (sam2 weights load from repo-local bundle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
