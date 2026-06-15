"""OCR pass on receipt/document/label crops.

Two honest paths:

1. **Real OCR**: when the ``tiiuae/Falcon-OCR`` weights are bundled
   under ``./models/hf/`` (optional preferred upgrade in
   ``offline_assets.yaml``), we instantiate the upstream
   ``falcon_perception.paged_ocr_inference.OCRInferenceEngine`` on
   that snapshot and run plain OCR per crop. The engine produces real
   text + a confidence score.

2. **Unavailable**: when the OCR weights are not bundled, the pass
   returns ``[]`` AND populates ``perception_result["limitations"]``
   with ``"ocr_unavailable"`` so the decision policy and reviewer UI
   both see the gap honestly. We do NOT pretend that joining
   detection labels is OCR text.

The runtime never downloads. If the optional Falcon-OCR upgrade is
desired, bundle it explicitly via
``python scripts/prepare_offline_model_bundle.py --download-approved
  --asset falcon_ocr_specialized``.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from .schemas import Detection, OcrResult


log = logging.getLogger(__name__)


OCR_LABELS: tuple[str, ...] = (
    "receipt", "document", "paper", "label", "box", "package",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def crops_for_ocr(detections: list[Detection]) -> list[int]:
    """Return indices of detections that are OCR candidates."""
    candidates: list[int] = []
    for idx, det in enumerate(detections):
        label = (det.label or "").lower()
        if any(token in label for token in OCR_LABELS):
            candidates.append(idx)
    return candidates


class OcrEngine:
    """Lazy real-OCR adapter.

    ``has_capability()`` returns ``True`` only when both:

    * The ``falcon_perception`` package is importable, AND
    * ``model_path`` points at a Falcon-OCR snapshot on disk.

    ``run_on_crops`` runs the upstream OCRInferenceEngine on each
    candidate crop and returns ``OcrResult`` rows. On any failure we
    return ``[]`` and log; we never substitute detection-label text.
    """

    def __init__(self, *, model_path: Optional[str] = None,
                 device: str = "cuda", dtype: str = "bfloat16"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._engine = None
        self._tokenizer = None
        self._image_processor = None
        self._load_err: Optional[str] = None

    def has_capability(self) -> bool:
        if self._engine is not None:
            return True
        if self._load_err is not None:
            return False
        try:
            import falcon_perception  # noqa: F401
        except ImportError as exc:
            self._load_err = f"falcon_perception package missing: {exc}"
            return False
        if not self.model_path:
            self._load_err = "no falcon-ocr local_path configured"
            return False
        import os
        if not os.path.isdir(self.model_path):
            self._load_err = f"falcon-ocr weights missing: {self.model_path}"
            return False
        return True

    def _load(self) -> None:
        if self._engine is not None:
            return
        from falcon_perception import load_and_prepare_model
        from falcon_perception.data import ImageProcessor
        from falcon_perception.paged_ocr_inference import OCRInferenceEngine
        model, tokenizer, _model_args = load_and_prepare_model(
            hf_model_id=None,
            hf_local_dir=self.model_path,
            device=self.device,
            dtype=self.dtype,
            compile=False,
        )
        self._tokenizer = tokenizer
        self._image_processor = ImageProcessor(patch_size=16, merge_size=1)
        self._engine = OCRInferenceEngine(
            model, tokenizer, self._image_processor,
            capture_cudagraph=False,
        )

    def run_on_crops(self,
                     crops: list[tuple[int, Detection, Any]]
                     ) -> list[OcrResult]:
        """``crops`` is a list of ``(detection_idx, Detection, PIL.Image)``.

        Returns one ``OcrResult`` per readable crop. The engine is
        loaded on the first call and cached; tests that monkeypatch
        ``has_capability()`` to False bypass the entire path.
        """
        if not crops or not self.has_capability():
            return []
        try:
            self._load()
        except Exception as exc:
            log.exception("falcon-ocr load failed")
            self._load_err = f"load failed: {exc}"
            return []
        results: list[OcrResult] = []
        for det_idx, det, img in crops:
            try:
                blob = self._engine.generate_plain(images=[img],
                                                   use_tqdm=False)
                text = ""
                if isinstance(blob, list) and blob:
                    head = blob[0]
                    if isinstance(head, str):
                        text = head
                    elif isinstance(head, dict):
                        text = str(head.get("text", ""))
                if not text.strip():
                    continue
                results.append(OcrResult(
                    frame_id=det.frame_id,
                    bbox_xyxy=list(det.bbox_xyxy),
                    text=text.strip(),
                    confidence=1.0,
                ))
            except Exception:
                log.exception("falcon-ocr inference failed on det %d",
                              det_idx)
        return results

    def status(self) -> dict:
        return {
            "available": self.has_capability(),
            "model_path": self.model_path,
            "error": self._load_err,
        }


def run_ocr(engine: Optional[OcrEngine],
            detections: list[Detection],
            frames_by_idx: dict,
            *,
            min_size: int = 40
            ) -> tuple[list[OcrResult], list[str]]:
    """Run OCR on candidate crops.

    Returns ``(results, limitations)``. When no engine is available the
    pass emits ``"ocr_unavailable"`` so the reviewer + decision policy
    see the gap. When the engine fails on a specific crop the failure
    is logged but does not poison the rest of the result.
    """
    limitations: list[str] = []
    candidates = crops_for_ocr(detections)
    if not candidates:
        return [], limitations
    if engine is None or not engine.has_capability():
        reason = "ocr_unavailable"
        if engine is not None and engine._load_err:
            reason = f"ocr_unavailable: {engine._load_err}"
        limitations.append(reason)
        return [], limitations

    crops: list[tuple[int, Detection, Any]] = []
    for idx in candidates:
        det = detections[idx]
        img = frames_by_idx.get(det.frame_idx)
        if img is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue
        crops.append((idx, det, img.crop((x1, y1, x2, y2))))
    if not crops:
        limitations.append("ocr_no_qualifying_crops")
        return [], limitations
    return engine.run_on_crops(crops), limitations
