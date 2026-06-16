"""End-to-end smoke test: synthesise a tiny image, send it through
GemmaVideoReasoner.reason() to the running vLLM server, print the parsed
JSON. Exits non-zero on failure.

Run AFTER vLLM is healthy on $VLLM_PORT.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_test")


def _synthetic_frames(n: int = 4, w: int = 224, h: int = 224) -> list[Image.Image]:
    """Generate ``n`` distinct PIL frames so vLLM has something to reason on."""
    out = []
    for i in range(n):
        arr = np.full((h, w, 3), 32, dtype=np.uint8)
        # Draw a moving square so the frames are visibly different.
        x0 = 30 + i * 30
        arr[60:160, x0:x0 + 60, 0] = 220   # red square
        arr[80:140, 30:90, 1] = 200        # green box stays put
        out.append(Image.fromarray(arr, mode="RGB"))
    return out


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gemma_reasoner import GemmaVideoReasoner
    from classifiers import resolve_prompts

    vllm_url = os.environ.get("VLLM_URL", "http://localhost:8001")
    model = os.environ.get(
        "GEMMA_MODEL_NAME", "google/gemma-4-26B-A4B-it"
    )

    reasoner = GemmaVideoReasoner(
        model_name=model,
        max_tokens=256,
        temperature=0.0,
        max_video_frames=4,
        video_fps=1,
        vllm_url=vllm_url,
        request_timeout_sec=180,
        request_retries=1,
        request_retry_backoff_sec=2,
    )
    if not reasoner.health():
        print(f"FAIL: vLLM /health not ok at {vllm_url}", file=sys.stderr)
        return 2

    # Use the fraud classifier defaults.
    cam_cfg = {"classifier": "fraud", "prompts": {"falcon": "", "gemma_system": "",
                                                   "gemma_user": ""}}
    resolved = resolve_prompts(cam_cfg)
    print(f"--- prompts (fraud classifier, token_budget={resolved['token_budget']}) ---")
    print("system:\n", resolved["gemma_system"][:200], "...")
    print("user:\n",   resolved["gemma_user"][:200], "...")

    frames = _synthetic_frames()
    print(f"--- sending {len(frames)} synthetic frames to {vllm_url} ---")

    t0 = time.time()
    result = reasoner.reason(
        frames,
        start_objects="counter, sign",
        action_objects="counter, sign, red bag, green box",
        system_prompt=resolved["gemma_system"],
        user_prompt=resolved["gemma_user"],
        token_budget=resolved["token_budget"],
        classifier=resolved["classifier"],
    )
    dt = time.time() - t0

    print(f"--- reason() took {dt:.1f}s; {result.get('_num_frames', '?')} frames ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Schema sanity.
    expected_keys = {"handover_occurred", "item_count", "items_handed_over",
                     "customer_description", "narrative", "confidence",
                     "flag_for_review", "people", "item_presented",
                     "objects_detected"}
    missing = expected_keys - set(result.keys())
    if missing:
        print(f"FAIL: missing keys: {missing}", file=sys.stderr)
        return 3
    print("OK: schema matches v2 contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
