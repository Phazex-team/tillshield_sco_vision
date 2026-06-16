"""Adapter around the existing FalconDetector.

The existing ``falcon_detector.FalconDetector`` is a thin in-process
wrapper around the tiiuae/Falcon-Perception checkpoint. This module
adds repo-local path resolution + a higher-level ``detect_on_frames``
helper that returns ``perception.schemas.Detection`` objects directly.

In production, falcon weights load from
``./models/hf/tiiuae/Falcon-Perception/<snapshot>/`` (resolved through
``app.config.resolve_model_path``). The tests stub this client entirely.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from PIL import Image

from .schemas import Detection


log = logging.getLogger(__name__)


class FalconClient:
    """Wrap FalconDetector with manifest-friendly outputs."""

    def __init__(self, *, model_path: Optional[str] = None,
                 model_name: str = "tiiuae/Falcon-Perception"):
        self.model_path = model_path
        self.model_name = model_name
        self._detector = None

    def _ensure_loaded(self):
        if self._detector is not None:
            return
        from falcon_detector import FalconDetector
        target = self.model_path or self.model_name
        self._detector = FalconDetector(target)

    # Falcon-Perception is a referring detector: it returns boxes that
    # match a query phrase but emits NO per-box class label. To get
    # labels the track-gating can use (``perception.temporal_memory``
    # keys on "item"/"product"/"receipt"/...), we run one query per
    # category and stamp the category name onto every box it returns.
    DEFAULT_CATEGORIES: dict[str, str] = {
        "item": ("item, product, merchandise, bag, shopping bag, box, "
                 "package, clothing, bottle, electronics, phone"),
        "receipt": "receipt, document, paper, invoice",
        "person": "person, hand, arm, cashier, customer",
    }

    def detect_on_frames(self,
                         frames: list[tuple[int, datetime, Image.Image]],
                         *,
                         query: Optional[str] = None,
                         categories: Optional[dict[str, str]] = None
                         ) -> list[Detection]:
        """Run category-aware detection on ``(frame_idx, ts, pil_image)``.

        For each frame and each ``{label: query}`` category, the matched
        boxes are returned as ``Detection``s labelled with the category
        name. ``query`` is accepted for backward compatibility (added to
        the categories under a generic "item" bucket). Exceptions are
        caught per (frame, category) so one decode failure never loses
        the rest of the window.
        """
        self._ensure_loaded()
        cats = dict(categories or self.DEFAULT_CATEGORIES)
        if query:
            cats.setdefault("item", query)
        results: list[Detection] = []
        for frame_idx, ts, img in frames:
            for label, cat_query in cats.items():
                try:
                    _, dets = self._detector.detect(img, query=cat_query)
                except Exception:
                    log.exception("falcon detect failed on frame %d (%s)",
                                  frame_idx, label)
                    continue
                for d in dets:
                    results.append(Detection(
                        label=label,
                        score=float(getattr(d, "score", 0.0)) or 0.5,
                        bbox_xyxy=[float(x) for x in d.bbox_px],
                        frame_id=f"frame_{frame_idx:06d}",
                        frame_idx=frame_idx,
                        ts=ts,
                        query=cat_query,
                    ))
        return results

    def unload(self) -> None:
        self._detector = None
