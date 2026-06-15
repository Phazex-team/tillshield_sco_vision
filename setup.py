"""v3 model warmup helper.

In v3 the heavy lifting is done by ``install.sh`` (apt + venv + pip +
``snapshot_download`` for both Falcon and Gemma NVFP4). This script is
kept as a thin convenience: it warms only the Falcon weights into the HF
cache; the Gemma weights are managed by vLLM at server start (or by
``install.sh``).

Run:
    python setup.py [--config config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    print("=" * 60)
    print("v3 setup -- pre-loading Falcon Perception only.")
    print("Gemma is served by vLLM and is downloaded by install.sh /")
    print("vllm_start.sh, NOT by this script.")
    print("=" * 60)

    t0 = time.time()
    print(f"\nFalcon Perception: {cfg['models']['falcon']['name']}")
    from falcon_detector import FalconDetector
    FalconDetector(cfg["models"]["falcon"]["name"])
    print(f"  done in {time.time() - t0:.1f}s")

    print("\nNext: bash ./start.sh")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted -- partial downloads remain cached and resume next run.")
        sys.exit(130)
