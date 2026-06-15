"""SAM 2 client.

When the ``sam2`` python package is importable AND the repo-local
``facebook/sam2-hiera-large`` snapshot exists, this client wraps the
official SAM 2 image-predictor for candidate-item segmentation.

When either dependency is missing, ``has_capability()`` is False and
the pipeline records ``sam2_unavailable`` as a limitation rather than
silently failing the case. PRODUCTION_SPEC §10 requires the segmenter
but explicitly allows degraded operation when SAM 2 cannot be loaded;
the decision policy then refuses to upgrade to VERIFIED on perception
evidence alone.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .schemas import Detection, Mask


log = logging.getLogger(__name__)


class Sam2Client:
    def __init__(self, *, model_path: Optional[str] = None,
                 device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self._predictor = None
        self._load_err: Optional[str] = None

    # ------------------------------------------------------------------
    # Capability
    # ------------------------------------------------------------------

    def has_capability(self) -> bool:
        if self._predictor is not None:
            return True
        if self._load_err is not None:
            return False
        try:
            import sam2  # noqa: F401
        except ImportError as exc:
            self._load_err = f"sam2 package not installed: {exc}"
            return False
        if not self.model_path or not os.path.isdir(self.model_path):
            self._load_err = f"sam2 weights missing: {self.model_path!r}"
            return False
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        if not self.has_capability():
            raise RuntimeError(self._load_err or "sam2 unavailable")
        try:
            # The sam2 package exposes the predictor in different ways
            # across versions; we use the high-level entry point.
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        except Exception as exc:
            self._load_err = f"sam2 import failed: {exc}"
            raise
        self._predictor = SAM2ImagePredictor.from_pretrained(
            self.model_path, device=self.device)

    def segment(self,
                image,
                detections: list[Detection]) -> list[Mask]:
        """Return one Mask per detection. The mask payload is left as
        the predictor's raw output; callers that persist masks should
        write them to disk and pass the resulting URI back into the
        Mask object."""
        if not detections:
            return []
        try:
            self._ensure_loaded()
        except Exception as exc:
            log.warning("sam2 load failed; continuing without masks: %s",
                        exc)
            return []
        try:
            import numpy as np
            self._predictor.set_image(np.asarray(image.convert("RGB")))
        except Exception:
            log.exception("sam2 set_image failed")
            return []
        masks: list[Mask] = []
        for idx, det in enumerate(detections):
            try:
                m, scores, _ = self._predictor.predict(
                    box=np.array(det.bbox_xyxy, dtype=np.float32),
                    multimask_output=False,
                )
                score = float(scores[0]) if hasattr(scores, "__len__") \
                    and len(scores) else 0.0
                masks.append(Mask(detection_idx=idx, score=score,
                                  bbox_xyxy=list(det.bbox_xyxy)))
            except Exception:
                log.exception("sam2 predict failed on detection %d", idx)
        return masks

    def unload(self) -> None:
        self._predictor = None
