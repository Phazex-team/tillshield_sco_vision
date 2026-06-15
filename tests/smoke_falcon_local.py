"""Optional real-load smoke for Falcon Perception from the repo-local bundle.

Gated by ``RUN_REAL_FALCON_SMOKE=1``. Confirms
``tiiuae/Falcon-Perception`` weights can be loaded from ``./models/hf/``
with no network access.

Usage:
    RUN_REAL_FALCON_SMOKE=1 python -m tests.smoke_falcon_local
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    if os.environ.get("RUN_REAL_FALCON_SMOKE") != "1":
        print("RUN_REAL_FALCON_SMOKE != 1 — skipping the Falcon real-load smoke.")
        return 77

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    bundle = (REPO_ROOT / "models" / "hf" / "tiiuae" / "Falcon-Perception")
    if not bundle.is_dir():
        print(f"falcon bundle missing at {bundle}", file=sys.stderr)
        return 2
    snaps = [p for p in bundle.iterdir() if p.is_dir()]
    snap = max(snaps, key=lambda p: sum(f.stat().st_size for f in p.rglob("*")
                                        if f.is_file()))
    print(f"falcon snapshot: {snap}")

    # The existing FalconDetector adapter takes a model name or local
    # path; route it explicitly at the local snapshot. The smoke only
    # constructs the detector, so it confirms the snapshot is loadable
    # without spending GPU memory on a full inference pass.
    try:
        from falcon_detector import FalconDetector
        FalconDetector(str(snap))
    except Exception as exc:
        print(f"FalconDetector construction failed: {exc}",
              file=sys.stderr)
        return 2
    print("SMOKE OK (falcon weights load from repo-local bundle)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
