"""Processing-timings legend — perception pipeline, case_runner,
evidence package, and review UI rendering.

The timings dict is advisory only. The decision policy never reads it
and tests must not pin specific durations. Tests pin *shape* (keys),
*honesty* (skipped stages omitted, not zeroed), *propagation*
(perception -> case_runner -> evidence package -> UI), and
*provenance* (actual VLM provider, not the configured primary).
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Perception pipeline returns timings_ms
# ---------------------------------------------------------------------

def test_run_perception_emits_timings_when_falcon_and_sam2_run(monkeypatch):
    """Both Falcon and SAM 2 execute → both stages must be timed."""
    from datetime import datetime as _dt
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = _dt(2026, 6, 17, 14, 0, 0)
    detections = [Detection(label="bag", score=0.9,
                             bbox_xyxy=[10, 10, 50, 50],
                             frame_id="f0", frame_idx=0, ts=base)]
    fake_frames = [(0, base, object())]

    class _Falcon:
        def detect_on_frames(self, frames, *, query, **kwargs):
            return detections
        def _ensure_loaded(self): pass

    class _Sam2:
        def has_capability(self): return True
        def segment(self, img, dets):
            return []  # empty masks list — call still timed

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    monkeypatch.setattr(pl, "run_ocr", lambda *a, **k: ([], []))

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=0, y=0, w=1000, h=1000)],
        falcon_client=_Falcon(),
        sam2_client=_Sam2(),
        sampling=SamplingPolicy(),
    )
    t = result["timings_ms"]
    # Required keys present.
    for k in ("sample_frames_ms", "falcon_ms", "sam2_ms",
              "tracker_ms", "ocr_ms", "keyframes_ms", "total_ms"):
        assert k in t, f"perception timings missing {k!r}: {t}"
    # All recorded values are non-negative ints.
    for k, v in t.items():
        assert isinstance(v, int) and v >= 0, (k, v)


def test_run_perception_omits_sam2_when_skipped(monkeypatch):
    """SAM 2 not capable → sam2_ms must be OMITTED, not zeroed."""
    from datetime import datetime as _dt
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy
    from perception.schemas import Detection
    from perception.temporal_memory import Zone

    base = _dt(2026, 6, 17, 14, 0, 0)
    detections = [Detection(label="bag", score=0.9,
                             bbox_xyxy=[10, 10, 50, 50],
                             frame_id="f0", frame_idx=0, ts=base)]
    fake_frames = [(0, base, object())]

    class _Falcon:
        def detect_on_frames(self, frames, *, query, **kwargs):
            return detections
        def _ensure_loaded(self): pass

    class _NoSam2:
        def has_capability(self): return False
        def segment(self, *a, **k): return []

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: fake_frames)
    monkeypatch.setattr(pl, "run_ocr", lambda *a, **k: ([], []))

    result = pl.run_perception_on_window(
        window_path="/tmp/fake.mp4", fps=25,
        zones=[Zone(name="counter_zone", x=0, y=0, w=1000, h=1000)],
        falcon_client=_Falcon(),
        sam2_client=_NoSam2(),
        sampling=SamplingPolicy(),
    )
    t = result["timings_ms"]
    assert "sam2_ms" not in t, \
        "skipped SAM 2 must not appear in timings (honesty rule)"
    # Other stages still present.
    for k in ("sample_frames_ms", "falcon_ms", "tracker_ms",
              "ocr_ms", "keyframes_ms", "total_ms"):
        assert k in t


def test_run_perception_no_frames_still_returns_total_ms(monkeypatch):
    """Empty window → only sample_frames_ms and total_ms are honest."""
    from datetime import datetime as _dt
    import perception.pipeline as pl
    from perception.sampling import SamplingPolicy

    monkeypatch.setattr(pl, "_sample_frames",
                        lambda *a, **k: [])
    result = pl.run_perception_on_window(
        window_path=None, fps=25, zones=[],
        falcon_client=None, sam2_client=None,
        sampling=SamplingPolicy(),
    )
    t = result["timings_ms"]
    assert "total_ms" in t and "sample_frames_ms" in t
    assert "falcon_ms" not in t
    assert "sam2_ms" not in t


# ---------------------------------------------------------------------
# case_runner persists processing_timings_ms + evidence package surfaces
# ---------------------------------------------------------------------

@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    import shutil as _shutil
    if _shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg required for segment recorder")
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))

    cfg_path = ROOT / "config.yaml"
    backup = tmp_path / "config_backup.yaml"
    shutil.copy(cfg_path, backup)

    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    try:
        from app.memory_guard import get_policy
        get_policy().reset_for_test()
    except Exception:
        pass
    # Shrink the correlation window so a few-second synthetic segment is
    # enough to reach the 80% coverage threshold — same trick the
    # ``test_runtime_real`` plumbing tests use.
    import pos.correlation as pc
    monkeypatch.setattr(pc, "PRE_ROLL_SEC", 2)
    monkeypatch.setattr(pc, "POST_ROLL_SEC", 2)
    yield ds, tmp_path
    shutil.copy(backup, cfg_path)


def _synthetic_frames(n, *, fps=25, start=None, size=(160, 120)):
    import numpy as np
    start = start or datetime(2026, 6, 17, 14, 0, 0)
    out = []
    dt = timedelta(seconds=1.0 / fps)
    for i in range(n):
        ts = start + dt * i
        frame = np.full((size[1], size[0], 3),
                         fill_value=(i * 5) % 255, dtype="uint8")
        out.append((ts, frame))
    return out


def _seed_segment_and_case(SM, storage_root, *, pos_event_at,
                            camera_id="cam_return_01"):
    """Record a real synthetic MP4 via ``SegmentRecorder`` (so the file
    AND index row are correct), open a POS event + Case at
    ``pos_event_at``, and return the case id."""
    from video.segment_recorder import RecorderConfig, SegmentRecorder
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
    from db.models import Case

    storage_root.mkdir(parents=True, exist_ok=True)
    seg_start = pos_event_at - timedelta(seconds=2)
    cfg = RecorderConfig(
        camera_id=camera_id, storage_root=storage_root,
        fps=25, width=160, height=120, segment_duration_sec=4,
    )
    rec = SegmentRecorder(cfg, session_factory=SM)
    rec.record_one_segment(_synthetic_frames(25 * 4, start=seg_start),
                            start_at=seg_start)

    with SM() as s:
        ingest_batch(s, PosBatchIn(
            source_system="test", store_id="store_1",
            received_at=pos_event_at,
            events=[PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-A", line_id="L1",
                event_type="SALE", pos_event_at=pos_event_at,
            )],
        ))
        s.commit()
        case = s.query(Case).first()
        return case.id


def _qwen_vlm_runner(session, case, window, manifest=None):
    return {
        "provider": "qwen3_vl",
        "model_name": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "model_snapshot": "Qwen3-VL-30B-A3B-Instruct-FP8",
        "parsed": {"handover_occurred": True,
                    "physical_item_presented": True,
                    "receipt_visible": True,
                    "confidence": "high",
                    "obstructed": False,
                    "camera_view_clear": True},
        "latency_ms": 17,
        "error": None,
    }


def _real_perception(session, case, window):
    return {
        "detections": [{"label": "bag", "score": 0.9,
                         "bbox_xyxy": [10, 10, 30, 30],
                         "frame_id": "frame_000000", "frame_idx": 0,
                         "ts": datetime.now().isoformat()}],
        "tracks": [{"track_id": "t1", "label": "bag",
                     "first_seen_ts": datetime.now().isoformat(),
                     "last_seen_ts": datetime.now().isoformat(),
                     "detections": [0],
                     "zones": ["counter_zone"],
                     "events": ["entered_counter_zone",
                                "handover_candidate"],
                     "physical_item_candidate": True,
                     "receipt_candidate": False,
                     "confidence": 0.9}],
        "keyframes": [{"frame_id": "frame_000000", "frame_idx": 0,
                        "ts": datetime.now().isoformat(),
                        "role": "first_appearance"}],
        "ocr": [], "limitations": [], "obstructed": False,
        "timings_ms": {
            "sample_frames_ms": 3, "falcon_ms": 11,
            "tracker_ms": 0, "ocr_ms": 0, "keyframes_ms": 0,
            "total_ms": 15,
        },
    }


def test_case_runner_persists_processing_timings_into_vlm_run(fresh_db):
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)

    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_real_perception,
                     vlm_runner=_qwen_vlm_runner)

    from db.models import VlmRun
    with SM() as s:
        run = s.query(VlmRun).filter(VlmRun.case_id == case_id).first()
    assert run is not None
    pt = (run.input_manifest or {}).get("processing_timings_ms")
    assert isinstance(pt, dict), "processing_timings_ms missing"
    for k in ("total_ms", "window_resolution_ms", "window_build_ms",
              "perception_total_ms", "manifest_frame_extract_ms",
              "vlm_roi_prepare_ms", "vlm_total_ms",
              "decision_ms", "package_write_ms"):
        assert k in pt, f"missing top-level timing {k!r}: {pt}"
    # Perception sub-dict propagated from the perception runner.
    assert isinstance(pt.get("perception"), dict)
    assert pt["perception"]["falcon_ms"] == 11
    # VLM fingerprint reflects the actual provider that returned a
    # result (here Qwen, but the same field powers the fallback case
    # below).
    assert pt["vlm"]["provider"] == "qwen3_vl"
    assert pt["vlm"]["status"] == "SUCCEEDED"


def test_case_runner_records_actual_provider_when_qwen_falls_back(fresh_db):
    """When ChainProvider falls back from Qwen to Gemma, the persisted
    vlm.provider must be ``gemma`` — NOT the configured primary."""
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)

    def _fallback_vlm_runner(session, case, window, manifest=None):
        # Mimic what _adapt_vlm_result(ChainProvider) returns AFTER the
        # chain has fallen back to Gemma: provider is "gemma".
        return {
            "provider": "gemma",
            "model_name": "google/gemma-4-26B-A4B-it",
            "model_snapshot": None,
            "parsed": {"handover_occurred": True,
                        "physical_item_presented": True,
                        "receipt_visible": False,
                        "confidence": "medium",
                        "obstructed": False,
                        "camera_view_clear": True,
                        "_chain_attempts": ["qwen3_vl=raised:RuntimeError",
                                            "gemma=ok"]},
            "latency_ms": 540,
            "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_real_perception,
                     vlm_runner=_fallback_vlm_runner)

    from db.models import VlmRun
    with SM() as s:
        run = s.query(VlmRun).filter(VlmRun.case_id == case_id).first()
    pt = (run.input_manifest or {}).get("processing_timings_ms")
    assert pt is not None
    assert pt["vlm"]["provider"] == "gemma", \
        "fallback case must surface actual provider, not configured primary"
    assert pt["vlm"]["status"] == "SUCCEEDED"


def test_evidence_package_exposes_processing_timings_ms(fresh_db):
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)

    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_real_perception,
                     vlm_runner=_qwen_vlm_runner)

    from evidence.package import latest_package_for_case
    with SM() as s:
        pkg = latest_package_for_case(s, case_id)
    assert pkg["reasoning"], "expected at least one reasoning entry"
    entry = pkg["reasoning"][0]
    assert "processing_timings_ms" in entry, \
        "package reasoning entry must surface processing_timings_ms"
    pt = entry["processing_timings_ms"]
    assert isinstance(pt, dict)
    assert pt["vlm"]["provider"] == "qwen3_vl"
    # The package must NOT expose the full input_manifest (which would
    # leak prompt/usage internals). Spot-check by absence of base64
    # image data on the package JSON.
    import json
    blob = json.dumps(pkg)
    assert "data:image/jpeg;base64," not in blob


# ---------------------------------------------------------------------
# Review UI structure
# ---------------------------------------------------------------------

def test_review_ui_renders_timings_legend_from_processing_timings_ms():
    src = (ROOT / "static" / "review.html").read_text()
    # Section header + read-only note.
    assert "Processing timings" in src
    assert "Decision policy never reads these" in src
    # JS reads from the backend-recorded field, never computes.
    assert "processing_timings_ms" in src
    assert "renderTimingsLegend" in src
    # Provider/latency rows present.
    assert "VLM provider used" in src
    assert "VLM latency" in src
    # Honest dash for missing values.
    assert "&mdash;" in src or "—" in src


def test_review_ui_does_not_compute_timings_in_js():
    """The brief says: do not compute timings in JS. Make sure we
    never call ``Date.now()`` / ``performance.now()`` in the
    timings rendering."""
    src = (ROOT / "static" / "review.html").read_text()
    # Allow performance.now() nowhere in the file (it isn't used).
    assert "performance.now(" not in src
    # Ensure renderTimingsLegend doesn't substract dates.
    legend_start = src.index("function renderTimingsLegend")
    legend_end = src.index("function kvSet", legend_start)
    legend_js = src[legend_start:legend_end]
    assert "Date.now(" not in legend_js
    assert "performance.now(" not in legend_js


def test_review_ui_renders_dash_when_timings_missing(monkeypatch):
    """When the package has no processing_timings_ms key the JS must
    still render the legend with `—` placeholders rather than crashing.
    We assert this by inspecting the source path of the renderer."""
    src = (ROOT / "static" / "review.html").read_text()
    legend_start = src.index("function renderTimingsLegend")
    legend_end = src.index("function kvSet", legend_start)
    legend_js = src[legend_start:legend_end]
    # The fmtMs helper unconditionally falls back to the dash span.
    assert "color:var(--muted)" in legend_js
    assert "fmtMs" in legend_js


# ---------------------------------------------------------------------
# GET /api/v1/cases/{case_id}/processing-timings
# ---------------------------------------------------------------------

@pytest.fixture
def client(fresh_db):
    """TestClient over the same isolated DB as the fresh_db fixture."""
    from fastapi.testclient import TestClient
    from app.main import create_app
    return TestClient(create_app())


def test_processing_timings_endpoint_returns_final_timings(client, fresh_db):
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)
    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_real_perception,
                     vlm_runner=_qwen_vlm_runner)

    r = client.get(f"/api/v1/cases/{case_id}/processing-timings")
    assert r.status_code == 200, r.text
    body = r.json()
    # Top-level fingerprint.
    assert body["case_id"] == case_id
    assert body["source"] == "vlm_runs.input_manifest"
    assert body["provider"] == "qwen3_vl"
    assert body["status"] == "SUCCEEDED"
    # Final timings include package_write_ms and post-package total_ms.
    pt = body["processing_timings_ms"]
    assert isinstance(pt, dict), pt
    assert "package_write_ms" in pt, (
        "endpoint must include final package_write_ms — that is the "
        "whole reason this endpoint exists separately from the "
        "immutable package file")
    assert isinstance(pt["package_write_ms"], int)
    # post-package total_ms is at least the package_write_ms.
    assert pt["total_ms"] >= pt["package_write_ms"], (
        "post-package total_ms must include the package-write phase: "
        f"total={pt['total_ms']} pkg={pt['package_write_ms']}")


def test_processing_timings_endpoint_omits_internal_fields(client, fresh_db):
    """The endpoint is timing-focused. It must NOT leak
    ``input_manifest``, ``provider_metadata``, ``usage``, or base64
    image data — those belong to the manifest, not to the legend."""
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)
    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_real_perception,
                     vlm_runner=_qwen_vlm_runner)

    r = client.get(f"/api/v1/cases/{case_id}/processing-timings")
    body = r.json()
    assert "input_manifest" not in body
    assert "provider_metadata" not in body
    assert "usage" not in body
    blob = r.text
    assert "data:image/jpeg;base64," not in blob
    assert "data:image/png;base64," not in blob


def test_processing_timings_endpoint_404_for_unknown_case(client):
    r = client.get("/api/v1/cases/does-not-exist/processing-timings")
    assert r.status_code == 404


def test_processing_timings_endpoint_404_when_no_vlm_run_yet(client, fresh_db):
    """An open case with no VlmRun yet returns 404 (the legend renders
    empty stage rows; provider fallback is handled by the package
    reasoning rows when present)."""
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    base = datetime(2026, 6, 17, 14, 0, 0)
    # Seed a case but DO NOT run analyze_case — no VlmRun row exists.
    case_id = _seed_segment_and_case(SM, tmp_path / "storage",
                                       pos_event_at=base)
    r = client.get(f"/api/v1/cases/{case_id}/processing-timings")
    assert r.status_code == 404


# ---------------------------------------------------------------------
# UI uses the new endpoint + fallback rules
# ---------------------------------------------------------------------

def test_review_ui_fetches_processing_timings_endpoint():
    src = (ROOT / "static" / "review.html").read_text()
    assert "/cases/${caseId}/processing-timings" in src
    # openCase awaits the timings response in parallel with the others.
    assert "timingsR" in src
    assert "timingsPayload" in src


def test_review_ui_legend_uses_endpoint_for_package_write_ms():
    """The package_write_ms row must come from the DB endpoint payload
    (``timingsPayload.processing_timings_ms.package_write_ms``), NOT
    from the immutable package's ``pkg.reasoning[].processing_timings_ms``.
    We pin this by inspecting which variable feeds the Package write
    row inside the renderer."""
    src = (ROOT / "static" / "review.html").read_text()
    start = src.index("function renderTimingsLegend")
    end = src.index("function kvSet", start)
    legend = src[start:end]
    # ``t`` is the local that the stage rows (including
    # "Package write") read from — and ``t`` is derived from
    # timingsPayload, not from the package reasoning entry.
    assert "t = (timingsPayload" in legend
    assert '["Package write",        t ? t.package_write_ms : null]' in legend
    # The renderer must NOT read package_write_ms from the package's
    # reasoning entry directly.
    assert "lastReasoning.package_write_ms" not in legend
    assert "lastReasoning.processing_timings_ms" not in legend


def test_review_ui_legend_falls_back_to_reasoning_when_no_timings():
    """When the timings endpoint is unavailable, provider / model /
    status / latency must still populate from the package's most
    recent reasoning row — they no longer require
    processing_timings_ms to exist."""
    src = (ROOT / "static" / "review.html").read_text()
    start = src.index("function renderTimingsLegend")
    end = src.index("function kvSet", start)
    legend = src[start:end]
    # The fallback initialises provider/model/status/latency from
    # lastReasoning unconditionally — BEFORE any check on timings.
    init_block = legend[:legend.index("if (timingsPayload")]
    assert "lastReasoning.provider" in init_block
    assert "lastReasoning.model_name" in init_block
    assert "lastReasoning.status" in init_block
    assert "lastReasoning.latency_ms" in init_block


def test_review_ui_timings_request_failure_does_not_break_open_case():
    """openCase must keep working when the timings request fails —
    only the case fetch is allowed to abort the operation.

    The timings fetch is wrapped in ``.catch(() => null)`` BEFORE the
    ``Promise.all`` join so a network-level rejection (DNS error,
    aborted connection, etc.) cannot reject the join and abort
    openCase. Without the wrapper, Promise.all would reject and the
    case view would silently fail to render."""
    src = (ROOT / "static" / "review.html").read_text()
    open_start = src.index("async function openCase")
    open_end = src.index("document.getElementById(\"close-detail\").onclick",
                         open_start)
    body = src[open_start:open_end]
    # Only !caseR.ok aborts — pkg + timings degrade gracefully.
    assert "if (!caseR.ok)" in body
    assert "if (!pkgR.ok)" not in body
    assert "if (!timingsR.ok)" not in body
    # The timings parse is wrapped in try so a malformed response
    # cannot throw an uncaught error.
    assert "timingsPayload = await timingsR.json()" in body
    # Network-level safety: the fetch is wrapped with ``.catch(() => null)``
    # (or an equivalent) BEFORE Promise.all sees it, AND the variable
    # used inside Promise.all is the already-protected handle.
    assert "/processing-timings`).catch(() => null)" in body, (
        "timings fetch must be wrapped with `.catch(() => null)` before "
        "Promise.all so a network-level rejection cannot abort openCase")
    # Confirm Promise.all uses the protected handle, not a raw fetch.
    promise_all_start = body.index("await Promise.all([")
    promise_all_end = body.index("])", promise_all_start)
    promise_all_args = body[promise_all_start:promise_all_end]
    assert "timingsReq" in promise_all_args
    assert "/processing-timings`)" not in promise_all_args, (
        "the raw timings fetch must not appear inside Promise.all — "
        "use the pre-wrapped handle so its rejection is swallowed")
    # The post-await null guard catches both network rejections (timingsR
    # is then null) AND any HTTP error responses.
    assert "if (timingsR && timingsR.ok)" in body
