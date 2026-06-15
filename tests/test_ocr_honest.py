"""OCR honesty tests.

The previous implementation called the Falcon detector with
query="extract text" and joined detection labels — that is NOT OCR.
The replacement must either:

* run the real ``falcon_perception.paged_ocr_inference.OCRInferenceEngine``
  on a bundled ``tiiuae/Falcon-OCR`` snapshot, OR
* honestly report OCR as unavailable + emit a structured limitation.

These tests pin both branches without loading any real model.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _det(label: str, bbox=(0, 0, 100, 100), frame_idx: int = 0):
    from perception.schemas import Detection
    return Detection(
        label=label, score=0.9, bbox_xyxy=list(bbox),
        frame_id=f"f{frame_idx:03d}", frame_idx=frame_idx,
        ts=datetime(2026, 6, 15, 14, 0, 0),
    )


def test_run_ocr_with_no_engine_emits_structured_limitation():
    from perception.ocr import run_ocr
    dets = [_det("receipt")]
    results, lims = run_ocr(None, dets, {0: object()})
    assert results == []
    assert any(l.startswith("ocr_unavailable") for l in lims)


def test_run_ocr_with_no_capability_emits_limitation():
    from perception.ocr import OcrEngine, run_ocr
    engine = OcrEngine(model_path="/definitely/not/here")
    assert engine.has_capability() is False
    dets = [_det("receipt")]
    results, lims = run_ocr(engine, dets, {0: object()})
    assert results == []
    assert any(l.startswith("ocr_unavailable") for l in lims)
    # The unavailable reason must surface the path so operators can fix.
    assert any("not/here" in l or "no falcon-ocr" in l for l in lims)


def test_run_ocr_no_candidates_returns_empty_no_limitation():
    from perception.ocr import run_ocr
    dets = [_det("hand")]
    results, lims = run_ocr(None, dets, {0: object()})
    assert results == []
    assert lims == []


def test_run_ocr_uses_engine_when_capability_present(monkeypatch):
    """When the engine reports has_capability=True, ``run_on_crops`` is
    actually called. We stub the engine so no real model loads."""
    from perception import ocr as ocr_mod
    from perception.schemas import OcrResult

    class _StubEngine:
        _load_err = None

        def has_capability(self):
            return True

        def run_on_crops(self, crops):
            return [OcrResult(frame_id=c[1].frame_id,
                              bbox_xyxy=list(c[1].bbox_xyxy),
                              text="REAL OCR TEXT",
                              confidence=0.95)
                    for c in crops]

    from PIL import Image
    frame = Image.new("RGB", (200, 200), "white")
    dets = [_det("receipt", bbox=(10, 10, 110, 110))]
    results, lims = ocr_mod.run_ocr(_StubEngine(), dets, {0: frame})
    assert lims == []
    assert results and results[0].text == "REAL OCR TEXT"
    assert results[0].confidence == 0.95


def test_pipeline_propagates_ocr_unavailable_limitation(monkeypatch):
    """When the pipeline runs with no OCR engine, the result's
    ``limitations`` field must carry ``ocr_unavailable`` so the case
    runner + decision policy see the gap."""
    from perception import pipeline as pl
    from perception.schemas import Detection
    from perception.sampling import SamplingPolicy
    from perception.temporal_memory import Zone

    base = datetime(2026, 6, 15, 14, 0, 0)
    detections = [Detection(label="receipt", score=0.9,
                            bbox_xyxy=[10, 10, 200, 200],
                            frame_id="f0", frame_idx=0, ts=base)]
    fake_frames = [(0, base, object())]

    class _StubFalcon:
        def detect_on_frames(self, frames, *, query):
            return detections
        def _ensure_loaded(self):
            pass

    class _NoSam2:
        def has_capability(self):
            return False
        def segment(self, *a, **k):
            return []

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=500, y=0, w=400, h=400)],
        falcon_client=_StubFalcon(),
        sam2_client=_NoSam2(),
        sampling=SamplingPolicy(),
        ocr_engine=None,
    )
    assert any(l.startswith("ocr_unavailable") for l in result["limitations"])
