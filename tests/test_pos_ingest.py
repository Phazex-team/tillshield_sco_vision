"""Idempotency tests for POS ingest + correlation window planning."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    return s.get_sessionmaker()


def _sample_batch():
    from pos.schemas import PosBatchIn, PosEventIn
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    return PosBatchIn(
        source_system="pos_v1",
        store_id="store_1",
        received_at=ts,
        batch_start_at=ts,
        batch_end_at=ts,
        events=[
            PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-A", line_id="L1",
                event_type="SALE", pos_event_at=ts,
                sku="SKU-1", amount=49.99, currency="AED",
            ),
            PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-A", line_id="L2",
                event_type="REPLACEMENT", pos_event_at=ts,
                sku="SKU-2", amount=0.0, currency="AED",
            ),
        ],
    )


def test_ingest_creates_case_only_for_return(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.ingest import ingest_batch
    with SM() as s:
        result = ingest_batch(s, _sample_batch())
        s.commit()
    assert result["events_inserted"] == 2
    assert result["cases_created"] == 1  # only RETURN opens a case


def test_replaying_same_batch_is_idempotent(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.ingest import ingest_batch
    with SM() as s:
        ingest_batch(s, _sample_batch())
        s.commit()
    with SM() as s:
        result = ingest_batch(s, _sample_batch())
        s.commit()
    assert result["duplicate_batch"] is True
    assert result["events_inserted"] == 0
    assert result["cases_created"] == 0


def test_partial_overlap_with_different_batch_does_not_dup_events(tmp_path,
                                                                  monkeypatch):
    """A second batch carrying the same event natural key must not insert
    a duplicate ``PosEvent``, even though the batch payload differs."""
    from pos.schemas import PosBatchIn, PosEventIn
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.ingest import ingest_batch
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    with SM() as s:
        ingest_batch(s, _sample_batch())
        s.commit()
    overlap = PosBatchIn(
        source_system="pos_v1",
        store_id="store_1",
        received_at=ts + timedelta(minutes=30),
        events=[
            PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-A", line_id="L1",
                event_type="SALE", pos_event_at=ts,
            ),
            PosEventIn(
                store_id="store_1", terminal_id="t1",
                transaction_id="txn-B", line_id="L1",
                event_type="SALE", pos_event_at=ts + timedelta(minutes=5),
                amount=10.0,
            ),
        ],
    )
    with SM() as s:
        result = ingest_batch(s, overlap)
        s.commit()
    assert result["events_inserted"] == 1   # only txn-B is new
    assert result["events_already_present"] == 1
    assert result["cases_created"] == 1     # txn-B REFUND opens a case


def test_unknown_event_type_does_not_open_a_case(tmp_path, monkeypatch):
    """Phase 1 / SCO: event_type acceptance is config-driven and lives at
    the boundary (API endpoint + TillShield adapter). For direct
    ingest_batch callers, an unknown type is still persisted (so the
    audit trail captures what came in) but does NOT open a case because
    case_opening_types() only includes the canonical SCO type."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    bad = PosBatchIn(
        source_system="pos_v1", store_id="store_1", received_at=ts,
        events=[PosEventIn(
            store_id="store_1", terminal_id="t1",
            transaction_id="txn-X", line_id="L1",
            event_type="SOMETHING_ELSE",
            pos_event_at=ts,
        )],
    )
    with SM() as s:
        result = ingest_batch(s, bad)
        s.commit()
    assert result["events_inserted"] == 1, \
        "the event is still persisted for audit even when unknown"
    assert result["cases_created"] == 0, \
        "unknown types must not open a case"


def test_window_plan_invalid_when_no_segments(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.correlation import plan_window
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    with SM() as s:
        plan = plan_window(s, "cam_01", ts)
    assert plan.is_valid is False
    assert "no overlapping" in plan.invalid_reason


def test_window_plan_full_coverage(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import VideoSegment
    from pos.correlation import plan_window
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    with SM() as s:
        s.add(VideoSegment(
            camera_id="cam_01",
            start_at=ts - timedelta(seconds=200),
            end_at=ts + timedelta(seconds=200),
            path="storage/cctv/cam_01/seg_a.mp4",
        ))
        s.commit()
    with SM() as s:
        plan = plan_window(s, "cam_01", ts)
    assert plan.is_valid is True
    assert plan.coverage_ratio == 1.0
    assert len(plan.matched_segment_ids) == 1


def test_window_spans_full_transaction(tmp_path, monkeypatch):
    """The window must cover [tx_start, tx_end] (+ pre/post roll), not just
    a fixed window around the start — so long transactions aren't cut off."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.correlation import plan_window
    start = datetime(2026, 6, 15, 14, 0, 0)  # naive UTC (plan_window normalises)
    end = start + timedelta(seconds=150)  # a 2.5-min transaction
    with SM() as s:
        point = plan_window(s, "cam_01", start)                        # legacy: no end
        span = plan_window(s, "cam_01", start, pos_event_end_at=end)   # new: full span
    # Both start at tx_start - PRE_ROLL (90s).
    assert span.requested_start == start - timedelta(seconds=90)
    # Point window ends at start + POST (60s) and would cut off the txn.
    assert point.requested_end == start + timedelta(seconds=60)
    # Span window ends at tx_END + POST (60s) — covers the whole txn.
    assert span.requested_end == end + timedelta(seconds=60)
    assert span.requested_end > point.requested_end


def test_window_falls_back_to_point_when_no_end(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.correlation import plan_window
    start = datetime(2026, 6, 15, 14, 0, 0)  # naive UTC (plan_window normalises)
    with SM() as s:
        p = plan_window(s, "cam_01", start, pos_event_end_at=None)
    assert p.requested_end == start + timedelta(seconds=60)


def test_window_guards_inverted_span(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from pos.correlation import plan_window
    start = datetime(2026, 6, 15, 14, 0, 0)  # naive UTC (plan_window normalises)
    bad_end = start - timedelta(seconds=30)  # end before start (bad data)
    with SM() as s:
        p = plan_window(s, "cam_01", start, pos_event_end_at=bad_end)
    # Falls back to the start anchor rather than producing an inverted window.
    assert p.requested_end == start + timedelta(seconds=60)


def test_window_plan_low_coverage_invalid(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import VideoSegment
    from pos.correlation import plan_window
    ts = datetime(2026, 6, 15, 14, 0, 0, tzinfo=timezone.utc)
    with SM() as s:
        # 60s segment in the middle of the 300s window -> coverage = 0.20
        s.add(VideoSegment(
            camera_id="cam_01",
            start_at=ts - timedelta(seconds=30),
            end_at=ts + timedelta(seconds=30),
            path="storage/cctv/cam_01/seg_b.mp4",
        ))
        s.commit()
    with SM() as s:
        plan = plan_window(s, "cam_01", ts)
    assert plan.is_valid is False
    assert "coverage" in plan.invalid_reason
