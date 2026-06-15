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

    def detect_on_frames(self,
                         frames: list[tuple[int, datetime, Image.Image]],
                         *,
                         query: str) -> list[Detection]:
        """Run detection on a list of ``(frame_idx, ts, pil_image)``.

        Returns a flat ``Detection`` list. Catches exceptions per-frame
        so a single decoding failure doesn't lose the rest of the window.
        """
        self._ensure_loaded()
        results: list[Detection] = []
        for frame_idx, ts, img in frames:
            try:
                _, dets = self._detector.detect(img, query=query)
            except Exception:
                log.exception("falcon detect failed on frame %d", frame_idx)
                continue
            for d in dets:
                results.append(Detection(
                    label=getattr(d, "label", "object"),
                    score=float(getattr(d, "score", 0.0)),
                    bbox_xyxy=[float(x) for x in d.bbox_px],
                    frame_id=f"frame_{frame_idx:06d}",
                    frame_idx=frame_idx,
                    ts=ts,
                    query=query,
                ))
        return results

    def unload(self) -> None:
        self._detector = None
