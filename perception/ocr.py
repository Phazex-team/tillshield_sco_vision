"""OCR pass on receipt/document/label crops (PRODUCTION_SPEC §10).

By the registry's role tag ``detector_and_ocr``, the same
``tiiuae/Falcon-Perception`` checkpoint already covers OCR via
natural-language queries — there is no separate ``Falcon-OCR``
dependency at runtime. This module reuses the configured FalconClient
to run an OCR pass on crops produced from receipt/document detections.

If FalconClient fails to load, we record the failure as a limitation
and return an empty result.
"""
from __future__ import annotations

import logging
from typing import Optional

from .falcon_client import FalconClient
from .schemas import Detection, OcrResult


log = logging.getLogger(__name__)


OCR_LABELS: tuple[str, ...] = (
    "receipt", "document", "paper", "label", "box", "package",
)


def crops_for_ocr(detections: list[Detection]) -> list[int]:
    """Return indices of detections that are OCR candidates."""
    candidates: list[int] = []
    for idx, det in enumerate(detections):
        label = (det.label or "").lower()
        if any(token in label for token in OCR_LABELS):
            candidates.append(idx)
    return candidates


def run_ocr(client: FalconClient,
            detections: list[Detection],
            frames_by_idx: dict[int, "PIL.Image.Image"],
            *,
            min_size: int = 40) -> list[OcrResult]:
    """Run OCR on each OCR-candidate detection. ``frames_by_idx`` maps
    frame_idx to the decoded PIL image so we can crop quickly."""
    candidates = crops_for_ocr(detections)
    if not candidates:
        return []
    try:
        client._ensure_loaded()
    except Exception as exc:
        log.warning("falcon OCR client unavailable: %s", exc)
        return []
    results: list[OcrResult] = []
    for idx in candidates:
        det = detections[idx]
        img = frames_by_idx.get(det.frame_idx)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        try:
            crop = img.crop((x1, y1, x2, y2))
            _, dets = client._detector.detect(crop, query="extract text")
            text = " ".join(getattr(d, "label", "") for d in dets).strip()
        except Exception:
            log.exception("OCR failed on detection %d", idx)
            continue
        if not text:
            continue
        results.append(OcrResult(
            frame_id=det.frame_id,
            bbox_xyxy=list(det.bbox_xyxy),
            text=text,
            confidence=0.0,
        ))
    return results
