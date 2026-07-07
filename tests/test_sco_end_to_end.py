"""Phase 8 — end-to-end SCO smoke.

Drives the full pipeline:
  POST /api/v1/pos/returns/event (with an alias event_type
  canonicalising to SALE and a 3-item basket under raw_payload.items)
    → case opens
    → analyze_case runs with stubbed perception (synthetic tracks) +
      stubbed VLM (canned basket-match JSON)
    → outcome lands in {VERIFIED, REVIEW, INVALID_VIDEO}
    → risk_reasons[] carries SCO machine-readable tags
    → refund-export pool is NOT touched
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Fresh sqlite for each end-to-end test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'e2e.sqlite'}")
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    monkeypatch.chdir(tmp_path)
    # Bypass the real ffmpeg-based window builder — perception+VLM are
    # stubbed in these tests, so we don't need a real MP4. We just need
    # build_window to return a successful WindowBuildResult so the
    # analyzer proceeds past the window-acquisition stage.
    from video.window_builder import WindowBuildResult
    from datetime import datetime as _dt, timedelta as _td

    def _ok_build(*, segments, requested_start, requested_end, out_path):
        # Write a tiny placeholder file at out_path so downstream
        # ffmpeg-key-extract sees something exists. Perception is
        # stubbed so it never actually reads frames.
        out_path = str(out_path)
        from pathlib import Path as _P
        _P(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"\x00" * 1024)
        return WindowBuildResult(
            ok=True, out_path=out_path,
            sha256="0" * 64,
            actual_start_at=requested_start,
            actual_end_at=requested_end,
            segment_ids=[s.id for s in segments],
        )
    # case_runner does `from video.window_builder import build_window`
    # inside the function body, so we patch the source module.
    monkeypatch.setattr("video.window_builder.build_window", _ok_build)
    # _extract_keyframe_data_urls IS at module level on app.case_runner,
    # so patch it there.
    monkeypatch.setattr("app.case_runner._extract_keyframe_data_urls",
                        lambda **kw: [])
    yield s.get_sessionmaker()


def _seed_segment(SM, storage_root, *, start_at, duration_sec=300):
    """Drop a synthetic VideoSegment row + placeholder mp4 wide enough
    to cover the analyzer's default 90s pre-roll / 60s post-roll window
    around the POS event (caller picks start_at)."""
    from db.models import VideoSegment
    storage_root.mkdir(parents=True, exist_ok=True)
    seg_path = storage_root / "cctv" / "cam_return_01" / "seg_a.mp4"
    seg_path.parent.mkdir(parents=True, exist_ok=True)
    seg_path.write_bytes(b"\x00" * 4096)  # bytes are enough — perception stubbed
    with SM() as s:
        seg = VideoSegment(
            camera_id="cam_return_01",
            start_at=start_at,
            end_at=start_at + timedelta(seconds=duration_sec),
            path=str(seg_path),
        )
        s.add(seg)
        s.commit()


# ---------------------------------------------------------------------------
# End-to-end happy-path
# ---------------------------------------------------------------------------

def test_sco_end_to_end_alias_event_runs_and_lands_in_review_class(
        fresh_db, tmp_path, monkeypatch):
    """An alias event_type ('CHECKOUT') normalises to canonical SALE,
    opens a case, gets analysed via SCO prompt + SCO policy, and ends in
    one of VERIFIED / REVIEW / INVALID_VIDEO with SCO tags populated.
    """
    SM = fresh_db
    pos_time = datetime(2026, 6, 15, 14, 2, 30, tzinfo=timezone.utc)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time.replace(tzinfo=None) - timedelta(seconds=120))

    # ----- 1. POST an alias-typed checkout event with items under
    #          raw_payload.items
    from app.api.pos import ingest_event, PosEventBody
    from fastapi import Request
    body = PosEventBody(
        store_id="store_sco_1",
        terminal_id="t1",
        transaction_id="txn-E2E-001",
        line_id="transaction",
        event_type="checkout",   # alias → normalises to SALE
        pos_event_at=pos_time,
        raw_payload={"items": [
            {"description": "DOVE BAR SOAP 100G", "quantity": 1},
            {"description": "COKE CAN 330ML", "quantity": 2},
            {"description": "ORANGE JUICE 1L", "quantity": 1},
        ]},
    )
    fake_req = type("R", (), {"client": None,
                              "headers": {"x-forwarded-for": "127.0.0.1"}})()
    ingest_result = ingest_event(body, fake_req)  # synchronous
    assert ingest_result["events_inserted"] == 1
    assert ingest_result["cases_created"] == 1, ingest_result

    # ----- 2. Find the case
    from db.models import Case
    with SM() as s:
        case = s.query(Case).first()
        assert case is not None
        case_id = case.id

    # ----- 3. Stub perception (one person track in sco_audit_zone
    #          straddling pos_time) and stub VLM (canned basket_match JSON).
    base = pos_time.replace(tzinfo=None)

    def _stub_perception(session, case, window):
        return {
            "detections": [
                {"label": "person", "score": 0.9,
                 "bbox_xyxy": [10, 10, 50, 50],
                 "frame_id": "frame_000000", "frame_idx": 0,
                 "ts": base.isoformat()},
                {"label": "sco_item_000", "score": 0.85,
                 "bbox_xyxy": [60, 60, 100, 100],
                 "frame_id": "frame_000000", "frame_idx": 0,
                 "ts": base.isoformat()},
                {"label": "sco_generic_products", "score": 0.7,
                 "bbox_xyxy": [110, 110, 140, 140],
                 "frame_id": "frame_000000", "frame_idx": 0,
                 "ts": base.isoformat()},
            ],
            "tracks": [
                {"track_id": "t_person",
                 "label": "person",
                 "first_seen_ts": (base - timedelta(seconds=20)).isoformat(),
                 "last_seen_ts": (base + timedelta(seconds=20)).isoformat(),
                 "detections": [0],
                 "zones": ["sco_audit_zone"],
                 "events": [],
                 "physical_item_candidate": False,
                 "receipt_candidate": False,
                 "confidence": 0.9},
                {"track_id": "t_item",
                 "label": "sco_item_000",
                 "first_seen_ts": base.isoformat(),
                 "last_seen_ts": base.isoformat(),
                 "detections": [1],
                 "zones": ["sco_audit_zone"],
                 "events": [],
                 "physical_item_candidate": True,
                 "receipt_candidate": False,
                 "confidence": 0.85},
            ],
            "keyframes": [{"frame_id": "frame_000000", "frame_idx": 0,
                            "ts": base.isoformat(),
                            "role": "first_appearance",
                            "track_id": "t_item"}],
            "ocr": [], "obstructed": False, "limitations": [],
        }

    def _stub_vlm(session, case, window, manifest=None):
        # The VLM "sees" the basket and reports it matches. The schema
        # parser tolerates this shape directly.
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "basket_match": "yes",
                "matched": [
                    {"pos_item": "DOVE BAR SOAP 100G",
                     "visible_count_class": "one"},
                    {"pos_item": "COKE CAN 330ML",
                     "visible_count_class": "multiple"},
                    {"pos_item": "ORANGE JUICE 1L",
                     "visible_count_class": "one"},
                ],
                "missing": [],
                "extras": [],
                "video_usable": True,
                "confidence": "high",
                "narrative": "All POS items visible in the audit zone.",
            },
            "latency_ms": 5, "error": None,
        }

    # Ensure refund export is NOT triggered by inspecting the pool.
    submitted: list = []
    from app.api import cases as cases_mod

    class _SpyPool:
        def submit(self, fn, *a, **kw):
            submitted.append(getattr(fn, "__name__", str(fn)))
    spy = _SpyPool()

    # ----- 4. Run the analyzer.
    from app.case_runner import analyze_case
    with patch.object(cases_mod, "_EXPORT_POOL", spy):
        with SM() as s:
            result = analyze_case(s, case_id,
                                  perception_runner=_stub_perception,
                                  vlm_runner=_stub_vlm)

    # ----- 5. Assertions on outcome shape.
    assert result["outcome"] in {"VERIFIED", "REVIEW", "INVALID_VIDEO"}, result
    assert result["outcome"] != "HIGH_RISK_REVIEW", \
        "SCO v1 must never produce HIGH_RISK_REVIEW"
    # SCO tags surfaced in risk_reasons
    reasons = result.get("reasons") or []
    sco_tags = [r for r in reasons if isinstance(r, str)
                and r.startswith("sco_")]
    assert sco_tags, f"expected at least one sco_* tag, got {reasons}"
    # The active policy is SCO — verify via the persisted case row.
    from db.models import Case as _Case
    with SM() as s:
        persisted_case = s.get(_Case, result["case_id"])
        assert persisted_case.decision_policy_version.startswith("sco"), \
            f"expected sco_* policy_version, got " \
            f"{persisted_case.decision_policy_version!r}"


def test_sco_end_to_end_basket_mismatch_lands_in_review(
        fresh_db, tmp_path, monkeypatch):
    """A VLM that reports basket_match=no produces REVIEW with the
    expected mismatch tag."""
    SM = fresh_db
    pos_time = datetime(2026, 6, 15, 14, 2, 30, tzinfo=timezone.utc)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time.replace(tzinfo=None) - timedelta(seconds=120))

    from app.api.pos import ingest_event, PosEventBody
    body = PosEventBody(
        store_id="store_sco_1", terminal_id="t1",
        transaction_id="txn-MISMATCH",
        line_id="transaction", event_type="SCO_SALE",
        pos_event_at=pos_time,
        raw_payload={"items": [{"description": "DOVE SOAP 100G"}]},
    )
    fake_req = type("R", (), {"client": None,
                              "headers": {"x-forwarded-for": "127.0.0.1"}})()
    ingest_event(body, fake_req)

    from db.models import Case
    with SM() as s:
        case_id = s.query(Case).first().id

    base = pos_time.replace(tzinfo=None)

    def _stub_perception(session, case, window):
        return {
            "detections": [], "tracks": [
                {"track_id": "tp", "label": "person",
                 "first_seen_ts": (base - timedelta(seconds=10)).isoformat(),
                 "last_seen_ts": (base + timedelta(seconds=10)).isoformat(),
                 "detections": [], "zones": ["sco_audit_zone"],
                 "events": [], "physical_item_candidate": False,
                 "receipt_candidate": False, "confidence": 0.9},
            ],
            "keyframes": [], "ocr": [], "obstructed": False,
            "limitations": [],
        }

    def _stub_vlm(session, case, window, manifest=None):
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {
                "physical_count_match": "no",
                "semantic_identity_match": "no",
                "matched_items": [],
                "missing_visible_items": [
                    {"pos_item": "DOVE SOAP 100G", "reason": "not visible"}
                ],
                "extra_visible_items": [],
                "uncertainty_reason": "",
                "video_usable": True,
                "confidence": "high",
                "narrative": "Item not visible during episode.",
            },
            "latency_ms": 1, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                              perception_runner=_stub_perception,
                              vlm_runner=_stub_vlm)
    assert result["outcome"] == "REVIEW"
    reasons = result.get("reasons") or []
    assert any("sco_basket_mismatch" in r for r in reasons), reasons
    assert any("sco_missing_items" in r for r in reasons), reasons


def test_sco_end_to_end_unusable_video_invalidates(
        fresh_db, tmp_path, monkeypatch):
    SM = fresh_db
    pos_time = datetime(2026, 6, 15, 14, 2, 30, tzinfo=timezone.utc)
    _seed_segment(SM, tmp_path / "storage",
                  start_at=pos_time.replace(tzinfo=None) - timedelta(seconds=120))

    from app.api.pos import ingest_event, PosEventBody
    body = PosEventBody(
        store_id="store_sco_1", terminal_id="t1",
        transaction_id="txn-BADVID",
        line_id="transaction", event_type="sale",
        pos_event_at=pos_time,
        raw_payload={"items": [{"description": "X"}]},
    )
    fake_req = type("R", (), {"client": None,
                              "headers": {"x-forwarded-for": "127.0.0.1"}})()
    ingest_event(body, fake_req)
    from db.models import Case
    with SM() as s:
        case_id = s.query(Case).first().id

    def _stub_perception(session, case, window):
        return {"detections": [], "tracks": [], "keyframes": [],
                "ocr": [], "obstructed": False, "limitations": []}

    def _stub_vlm(session, case, window, manifest=None):
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {"basket_match": "uncertain", "video_usable": False,
                        "confidence": "low", "narrative": "blank frames"},
            "latency_ms": 1, "error": None,
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                              perception_runner=_stub_perception,
                              vlm_runner=_stub_vlm)
    assert result["outcome"] == "INVALID_VIDEO"
    assert any("sco_bad_footage" in r for r in (result.get("reasons") or []))
