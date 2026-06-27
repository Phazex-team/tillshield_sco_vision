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

    # Keys that downstream consumers (decision policy, customer_present gate,
    # track-gating in perception.temporal_memory) rely on. Callers can ADD
    # new categories (e.g. POS-derived SKU queries) but must never overwrite
    # these. A custom categories dict that drops/replaces them would silently
    # break person detection or item gating elsewhere in the pipeline.
    RESERVED_CATEGORY_KEYS: frozenset[str] = frozenset({"item", "person", "receipt"})

    def detect_on_frames(self,
                         frames: list[tuple[int, datetime, Image.Image]],
                         *,
                         query: Optional[str] = None,
                         categories: Optional[dict[str, str]] = None,
                         roi_crop: Optional[tuple[int, int, int, int]] = None
                         ) -> list[Detection]:
        """Run category-aware detection on ``(frame_idx, ts, pil_image)``.

        For each frame and each ``{label: query}`` category, the matched
        boxes are returned as ``Detection``s labelled with the category
        name. ``query`` is accepted for backward compatibility (added to
        the categories under a generic "item" bucket). Exceptions are
        caught per (frame, category) so one decode failure never loses
        the rest of the window.

        ``categories`` ADDS to ``DEFAULT_CATEGORIES`` — it does not
        replace it. Reserved keys (``item``, ``person``, ``receipt``)
        cannot be overwritten via ``categories`` because downstream
        consumers (decision policy, customer-presence gate, track-gating)
        depend on them. Attempts to overwrite are logged and ignored.

        ``roi_crop`` (optional ``(x1, y1, x2, y2)`` in *full-frame*
        pixel coordinates) restricts Falcon to the cropped region.
        Detection ``bbox_xyxy`` values are translated back to full-frame
        coordinates before being returned, so every downstream consumer
        (tracker, SAM 2, OCR, decision policy, evidence package) keeps
        its full-frame coordinate contract.
        """
        self._ensure_loaded()
        # Always start from defaults so person/item/receipt detectors
        # never silently disappear when a scenario adds POS categories.
        cats: dict[str, str] = dict(self.DEFAULT_CATEGORIES)
        if categories:
            for k, v in categories.items():
                if k in self.RESERVED_CATEGORY_KEYS:
                    log.warning(
                        "FalconClient: refusing to overwrite reserved "
                        "category %r via categories= (defaults preserved). "
                        "Use query= to refine the 'item' default.", k)
                    continue
                cats[k] = v
        if query:
            cats.setdefault("item", query)
        ox, oy = 0, 0
        if roi_crop is not None:
            ox, oy = int(roi_crop[0]), int(roi_crop[1])
        results: list[Detection] = []
        for frame_idx, ts, img in frames:
            target_img = img
            if roi_crop is not None:
                # Clip the crop to the actual image bounds; Falcon would
                # otherwise mis-handle out-of-bounds crops.
                w, h = img.size
                cx1 = max(0, min(int(roi_crop[0]), w))
                cy1 = max(0, min(int(roi_crop[1]), h))
                cx2 = max(cx1, min(int(roi_crop[2]), w))
                cy2 = max(cy1, min(int(roi_crop[3]), h))
                if cx2 > cx1 and cy2 > cy1:
                    target_img = img.crop((cx1, cy1, cx2, cy2))
                    ox, oy = cx1, cy1
                else:
                    target_img = img
                    ox, oy = 0, 0
            for label, cat_query in cats.items():
                try:
                    _, dets = self._detector.detect(target_img,
                                                    query=cat_query)
                except Exception:
                    log.exception("falcon detect failed on frame %d (%s)",
                                  frame_idx, label)
                    continue
                for d in dets:
                    bx = [float(x) for x in d.bbox_px]
                    if ox or oy:
                        bx = [bx[0] + ox, bx[1] + oy,
                              bx[2] + ox, bx[3] + oy]
                    results.append(Detection(
                        label=label,
                        score=float(getattr(d, "score", 0.0)) or 0.5,
                        bbox_xyxy=bx,
                        frame_id=f"frame_{frame_idx:06d}",
                        frame_idx=frame_idx,
                        ts=ts,
                        query=cat_query,
                    ))
        return results

    def unload(self) -> None:
        self._detector = None
