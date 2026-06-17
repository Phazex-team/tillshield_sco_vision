"""Schema invariants + concurrent-write contracts.

The repo persists structured payloads in JSON columns (input_manifest,
output_json, raw_payload, segment_ids, nvr_metadata, etc.) and accepts
concurrent writes from the POS push surface AND the TillShield poller.
This file pins:

  * Every JSON column round-trips lossless ASCII + nested structure.
  * The natural-key uniqueness contract on ``pos_events`` survives a
    duplicate-batch race.
  * Storage cleanup execute is safe against an interleaved case that
    references a segment AFTER cleanup decides it is unlinked
    (linked-from-window invariant verified by repeated execution).
  * The deterministic decision policy never escalates without
    track-level evidence — repeated random VLM payloads stay below
    VERIFIED.

Tests are deliberately small + side-effect-isolated.
"""
from __future__ import annotations

import os
import shutil
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_EDIT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as ds
    ds._ENGINE = None
    ds._SESSION_FACTORY = None
    ds.init_schema()
    try:
        from app.memory_guard import get_policy
        get_policy().reset_for_test()
    except Exception:
        pass
    return ds, tmp_path


# ---------------------------------------------------------------------
# JSON column round-trip contracts
# ---------------------------------------------------------------------

def test_vlm_run_input_manifest_round_trips_nested_dict(fresh_db):
    """``VlmRun.input_manifest`` carries the processing_timings_ms
    payload — its nested dict shape must survive the SQLite JSON
    column in both directions."""
    from db.models import Case, PosBatch, PosEvent, VlmRun
    ds, _ = fresh_db
    SM = ds.get_sessionmaker()
    payload = {
        "window_id": "w1", "frame_count": 6,
        "perception": {"tracks": 2, "ocr": 0, "limitations": []},
        "processing_timings_ms": {
            "total_ms": 18420,
            "perception": {"sample_frames_ms": 3, "falcon_ms": 11,
                            "total_ms": 15},
            "vlm": {"provider": "qwen3_vl", "status": "SUCCEEDED",
                     "latency_ms": 17},
            "decision_ms": 0,
            "package_write_ms": 4,
        },
    }
    with SM() as s:
        batch = PosBatch(source_system="t", store_id="s"); s.add(batch); s.flush()
        pe = PosEvent(batch_id=batch.id, store_id="s", terminal_id="t1",
                       transaction_id="tx1", line_id="L1",
                       event_type="RETURN",
                       pos_event_at=datetime(2026, 6, 17))
        s.add(pe); s.flush()
        case = Case(pos_event_id=pe.id, camera_id="cam_01",
                     status="OPEN")
        s.add(case); s.flush()
        run = VlmRun(case_id=case.id, provider="qwen3_vl",
                      model_name="m", prompt_version="v",
                      input_manifest=payload, output_json={},
                      status="SUCCEEDED", latency_ms=1)
        s.add(run); s.commit()
        run_id = run.id
    with SM() as s:
        loaded = s.get(VlmRun, run_id)
        assert loaded.input_manifest == payload
        assert loaded.input_manifest["processing_timings_ms"]["perception"][
            "falcon_ms"] == 11


def test_video_window_segment_ids_round_trips_list(fresh_db):
    """``VideoWindow.segment_ids`` is a JSON list — order + identity
    must round-trip."""
    from db.models import Case, PosBatch, PosEvent, VideoWindow
    ds, _ = fresh_db
    SM = ds.get_sessionmaker()
    seg_ids = ["seg-a", "seg-b", "seg-c"]
    with SM() as s:
        batch = PosBatch(source_system="t", store_id="s"); s.add(batch); s.flush()
        pe = PosEvent(batch_id=batch.id, store_id="s", terminal_id="t1",
                       transaction_id="tx2", line_id="L1",
                       event_type="RETURN",
                       pos_event_at=datetime(2026, 6, 17))
        s.add(pe); s.flush()
        case = Case(pos_event_id=pe.id, camera_id="cam_01",
                     status="OPEN")
        s.add(case); s.flush()
        w = VideoWindow(case_id=case.id, camera_id="cam_01",
                         requested_start_at=datetime(2026, 6, 17),
                         requested_end_at=datetime(2026, 6, 17, 0, 1),
                         status="SUCCEEDED",
                         segment_ids=seg_ids)
        s.add(w); s.commit()
        wid = w.id
    with SM() as s:
        loaded = s.get(VideoWindow, wid)
        assert loaded.segment_ids == seg_ids


def test_audit_log_json_columns_preserve_unicode(fresh_db):
    """``AuditLog.before_json`` / ``after_json`` must preserve unicode
    string content (Arabic / Hebrew / German umlaut) since the POS
    feed and reviewer notes can carry it."""
    from app import audit
    from db.models import AuditLog
    ds, _ = fresh_db
    SM = ds.get_sessionmaker()
    payload = {"name": "Qusais — قصيص", "ümlaut": "Schöne"}
    with SM() as s:
        audit.record(s, action="test.unicode",
                     entity_type="t", entity_id="x",
                     actor_type="test",
                     before=payload,
                     after={"k": payload})
        s.commit()
    with SM() as s:
        row = s.query(AuditLog).filter(
            AuditLog.action == "test.unicode").first()
    assert row is not None
    assert row.before_json == payload
    assert row.after_json["k"]["name"] == "Qusais — قصيص"


# ---------------------------------------------------------------------
# POS batch idempotency under repeated ingest
# ---------------------------------------------------------------------

def test_pos_batch_replay_does_not_create_duplicate_events(fresh_db):
    """The natural key (store_id, terminal_id, transaction_id,
    line_id) is unique on pos_events. Replaying the same batch must
    therefore never insert a second event row."""
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
    from db.models import PosEvent
    ds, _ = fresh_db
    SM = ds.get_sessionmaker()
    batch = PosBatchIn(
        source_system="test", store_id="s1",
        received_at=datetime(2026, 6, 17, 14, 0, 0),
        events=[PosEventIn(
            store_id="s1", terminal_id="t1",
            transaction_id="txn-A", line_id="L1",
            event_type="RETURN",
            pos_event_at=datetime(2026, 6, 17, 14, 0, 0),
        )],
    )
    with SM() as s:
        r1 = ingest_batch(s, batch)
        s.commit()
    with SM() as s:
        r2 = ingest_batch(s, batch)
        s.commit()
    with SM() as s:
        count = s.query(PosEvent).count()
    assert count == 1
    assert r1["events_inserted"] == 1
    assert r2["events_inserted"] == 0
    assert r2.get("duplicate_batch") is True


def test_pos_concurrent_batches_with_same_event_only_persist_once(
        fresh_db):
    """Two threads racing to ingest the SAME batch must result in a
    single pos_events row. Both callers should get structured ingest
    summaries: one insert, one idempotent duplicate response."""
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
    from db.models import PosEvent
    ds, _ = fresh_db
    SM = ds.get_sessionmaker()
    batch = PosBatchIn(
        source_system="test", store_id="s2",
        received_at=datetime(2026, 6, 17, 14, 1, 0),
        events=[PosEventIn(
            store_id="s2", terminal_id="t1",
            transaction_id="txn-RACE", line_id="L1",
            event_type="RETURN",
            pos_event_at=datetime(2026, 6, 17, 14, 1, 0),
        )],
    )
    results: list = []
    errors: list = []
    barrier = threading.Barrier(2)

    def _go():
        try:
            barrier.wait()
            with SM() as s:
                results.append(ingest_batch(s, batch))
                s.commit()
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=_go)
    t2 = threading.Thread(target=_go)
    t1.start(); t2.start(); t1.join(); t2.join()
    with SM() as s:
        assert s.query(PosEvent).count() == 1
    assert errors == []
    assert len(results) == 2
    assert sorted(r["events_inserted"] for r in results) == [0, 1]
    assert any(r.get("duplicate_batch") is True for r in results)


# ---------------------------------------------------------------------
# Storage cleanup safety contract under repeated runs
# ---------------------------------------------------------------------

def _seed_two_segments(SM, storage_root, *, linked_to_window: bool):
    """Two segments — one free, one linked to a Window. Returns (free_id, linked_id)."""
    from db.models import (
        Case, PosBatch, PosEvent, VideoSegment, VideoWindow,
    )
    storage_root.mkdir(parents=True, exist_ok=True)
    seg_dir = storage_root / "segs"
    seg_dir.mkdir(exist_ok=True)
    free = seg_dir / "free.mp4"
    linked = seg_dir / "linked.mp4"
    free.write_bytes(b"\x00")
    linked.write_bytes(b"\x00")
    old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=48)
    with SM() as s:
        f = VideoSegment(camera_id="cam_01", start_at=old,
                          end_at=old + timedelta(seconds=10),
                          path=str(free), sha256="a"*64,
                          fps=25, width=160, height=120,
                          frame_count=250, duration_sec=10.0)
        l = VideoSegment(camera_id="cam_01", start_at=old + timedelta(seconds=20),
                          end_at=old + timedelta(seconds=30),
                          path=str(linked), sha256="b"*64,
                          fps=25, width=160, height=120,
                          frame_count=250, duration_sec=10.0)
        s.add_all([f, l]); s.flush()
        if linked_to_window:
            batch = PosBatch(source_system="t", store_id="s"); s.add(batch); s.flush()
            pe = PosEvent(batch_id=batch.id, store_id="s", terminal_id="t1",
                           transaction_id="tx", line_id="L1",
                           event_type="RETURN",
                           pos_event_at=old + timedelta(seconds=25))
            s.add(pe); s.flush()
            case = Case(pos_event_id=pe.id, camera_id="cam_01",
                         status="CLOSED", outcome="REVIEW")
            s.add(case); s.flush()
            w = VideoWindow(case_id=case.id, camera_id="cam_01",
                             requested_start_at=old + timedelta(seconds=22),
                             requested_end_at=old + timedelta(seconds=32),
                             status="SUCCEEDED",
                             segment_ids=[l.id])
            s.add(w)
        s.commit()
        return f.id, l.id


def test_cleanup_execute_is_idempotent(fresh_db):
    """Calling execute twice in a row should be a no-op the second
    time — the linked invariant holds and unlinked candidates are
    already gone."""
    from app.storage_guard import run_cleanup
    ds, tmp_path = fresh_db
    SM = ds.get_sessionmaker()
    free_id, linked_id = _seed_two_segments(
        SM, tmp_path / "storage", linked_to_window=True)
    with SM() as s:
        first = run_cleanup(s, execute=True); s.commit()
    assert first["deleted_rows"] == 1
    with SM() as s:
        second = run_cleanup(s, execute=True); s.commit()
    assert second["deleted_rows"] == 0
    # Linked seg never touched in either run.
    from db.models import VideoSegment
    with SM() as s:
        assert s.get(VideoSegment, linked_id) is not None
        assert s.get(VideoSegment, free_id) is None


def test_cleanup_dry_run_never_mutates_disk(fresh_db):
    from app.storage_guard import run_cleanup
    ds, tmp_path = fresh_db
    free_id, linked_id = _seed_two_segments(
        ds.get_sessionmaker(), tmp_path / "storage",
        linked_to_window=False)
    SM = ds.get_sessionmaker()
    with SM() as s:
        result = run_cleanup(s, execute=False); s.rollback()
    assert result["dry_run"] is True
    assert result["deleted_files"] == []
    # Both files still on disk.
    free_path = tmp_path / "storage" / "segs" / "free.mp4"
    linked_path = tmp_path / "storage" / "segs" / "linked.mp4"
    assert free_path.is_file()
    assert linked_path.is_file()


# ---------------------------------------------------------------------
# Decision policy hardening — VLM cannot escalate alone
# ---------------------------------------------------------------------

def test_decision_policy_resists_high_confidence_no_track_storm():
    """100 randomly-shaped high-confidence VLM payloads with no
    track evidence must all degrade to REVIEW. This is the K-series
    track-gating invariant under load."""
    import random
    from reasoning.decision_policy import decide, summary_from_vlm
    rng = random.Random(42)
    for _ in range(100):
        vlm_parsed = {
            "handover_occurred": rng.choice([True, False]),
            "physical_item_presented": True,
            "receipt_visible": rng.choice([True, False]),
            "narrative": "x" * rng.randint(1, 200),
            "confidence": rng.choice(["high", "medium"]),
            "obstructed": False,
            "camera_view_clear": True,
            "items_observed": ["bag", "receipt"],
        }
        summary = summary_from_vlm(vlm_parsed, footage_valid=True,
                                    obstructed=False, camera_gap=False,
                                    perception_result={"tracks": []})
        assert decide(summary).outcome != "VERIFIED"
