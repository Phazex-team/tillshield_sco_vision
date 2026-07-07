"""Real plumbing tests (no stubbed analyze_case path).

These pin the contract that previously hid behind stubs:

* ``_extract_keyframe_data_urls`` actually opens a real MP4 and emits
  base64 data URLs.
* ``analyze_case`` calls ``build_window`` after ``plan_window`` and
  sets ``VideoWindow.path`` to an on-disk MP4.
* The reasoning chain receives an ``EvidenceManifest`` whose ``frames``
  list is non-empty for a valid window — never ``frames=[]``.
* Perception output is persisted as Detection / Track /
  TrackObservation / Keyframe / OcrResult rows so the package + graph
  see real evidence.
* When the recorded segment file is missing on disk, the case is
  closed with ``INVALID_VIDEO`` and the reason is structured.
* ``ChainProvider.health()`` actually reports the inner providers'
  health rather than the base class "not implemented" stub.
* ``ChainProvider`` registers each provider's ``unload`` callback with
  the memory guard so the hard limit really drops weights.
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg required for window builder",
)


def _synthetic_frames(n: int, *, fps: int = 25,
                      start: datetime | None = None,
                      size: tuple[int, int] = (160, 120)):
    start = start or datetime(2026, 6, 15, 14, 0, 0)
    out = []
    dt = timedelta(seconds=1.0 / fps)
    for i in range(n):
        ts = start + dt * i
        frame = np.full((size[1], size[0], 3),
                        fill_value=(i * 5) % 255, dtype=np.uint8)
        out.append((ts, frame))
    return out


def _fresh_session(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    # Shrink the correlation window so a few-second synthetic segment is
    # enough to reach the 80% coverage threshold without forcing tests
    # to record minutes of synthetic video.
    import pos.correlation as pc
    monkeypatch.setattr(pc, "PRE_ROLL_SEC", 2)
    monkeypatch.setattr(pc, "POST_ROLL_SEC", 2)
    return s.get_sessionmaker()


def _seed_segment(SM, storage_root: Path, *, start_at: datetime,
                  duration_sec: int = 4) -> str:
    """Record a real synthetic MP4 + index it. Returns the path."""
    from video.segment_recorder import RecorderConfig, SegmentRecorder

    cfg = RecorderConfig(
        camera_id="cam_return_01",
        storage_root=storage_root,
        fps=25,
        width=160,
        height=120,
        segment_duration_sec=duration_sec,
    )
    rec = SegmentRecorder(cfg, session_factory=SM)
    rec.record_one_segment(_synthetic_frames(25 * duration_sec, start=start_at),
                           start_at=start_at)
    files = list((storage_root).rglob("*.mp4"))
    assert files, "recorder should have written at least one mp4"
    return str(files[0])


def _seed_pos_event(SM, *, pos_event_at: datetime) -> str:
    from db.models import Case
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn
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


# ---------------------------------------------------------------------------
# 1. Real frame extraction
# ---------------------------------------------------------------------------

def test_extract_keyframe_data_urls_returns_frames_for_real_mp4(tmp_path):
    from app.case_runner import _extract_keyframe_data_urls
    from video.segment_recorder import RecorderConfig, SegmentRecorder

    cfg = RecorderConfig(camera_id="cam", storage_root=tmp_path / "s",
                         fps=25, width=160, height=120,
                         segment_duration_sec=2)
    rec = SegmentRecorder(cfg, session_factory=None)
    rec.record_one_segment(_synthetic_frames(50))
    mp4s = list((tmp_path / "s").rglob("*.mp4"))
    assert mp4s
    window_start = datetime(2026, 6, 15, 14, 0, 0)
    frames = _extract_keyframe_data_urls(
        window_path=str(mp4s[0]),
        window_start_ts=window_start,
        keyframes=[], max_frames=4)
    assert len(frames) >= 1
    for f in frames:
        assert f["image_url"].startswith("data:image/jpeg;base64,")
        assert "frame_id" in f
        assert isinstance(f["frame_idx"], int)
        ts = datetime.fromisoformat(f["ts"])
        assert ts.year == 2026 and ts.month == 6 and ts.day == 15
        assert ts >= window_start


def test_extract_keyframe_data_urls_returns_empty_for_missing_file():
    from app.case_runner import _extract_keyframe_data_urls
    assert _extract_keyframe_data_urls(
        window_path="/nope/nada.mp4",
        window_start_ts=datetime(2026, 6, 15, 14, 0, 0),
        keyframes=[], max_frames=4) == []


# ---------------------------------------------------------------------------
# 2. analyze_case real path — VLM stubbed but window built for real
# ---------------------------------------------------------------------------

def test_analyze_case_builds_real_window_and_sends_frames_to_vlm(
        tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    storage = tmp_path / "storage"
    base = datetime(2026, 6, 15, 14, 0, 0)
    _seed_segment(SM, storage, start_at=base - timedelta(seconds=5),
                  duration_sec=10)
    case_id = _seed_pos_event(SM, pos_event_at=base)

    seen_manifests: list = []

    def _capture_vlm(session, case, window, manifest=None):
        seen_manifests.append(manifest)
        return {
            "provider": "qwen3_vl", "model_name": "stub",
            "parsed": {"handover_occurred": True,
                       "physical_item_presented": True,
                       "receipt_visible": True,
                       "confidence": "high",
                       "obstructed": False,
                       "camera_view_clear": True},
            "latency_ms": 5, "error": None,
        }

    def _real_perception(session, case, window):
        # Real run_perception requires falcon weights; this test only
        # cares that the window is built + frames extracted. Use a
        # minimal stub that returns ONE keyframe so the persistence
        # path is exercised too.
        return {
            "detections": [{
                "label": "shopping bag", "score": 0.9,
                "bbox_xyxy": [10, 10, 50, 50],
                "frame_id": "frame_000000", "frame_idx": 0,
                "ts": base.isoformat(),
            }],
            "tracks": [{
                "track_id": "track_0001",
                "label": "shopping bag",
                "first_seen_ts": base.isoformat(),
                "last_seen_ts": base.isoformat(),
                "detections": [0],
                "physical_item_candidate": True,
                "zones": ["counter_zone"],
                "events": ["entered_counter_zone", "handover_candidate"],
                "confidence": 0.9,
            }],
            "keyframes": [{
                "frame_id": "frame_000000", "frame_idx": 0,
                "ts": base.isoformat(),
                "role": "first_appearance",
                "track_id": "track_0001",
            }],
            "ocr": [], "obstructed": False, "limitations": [],
        }

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                              perception_runner=_real_perception,
                              vlm_runner=_capture_vlm)

    # The window MP4 must exist on disk and be readable.
    from db.models import VideoWindow
    with SM() as s:
        win = s.query(VideoWindow).filter(VideoWindow.case_id == case_id).first()
    assert win.path and Path(win.path).is_file()
    assert win.status == "SUCCEEDED"
    assert win.sha256 and len(win.sha256) == 64

    # The VLM saw a manifest with NON-EMPTY frames.
    assert seen_manifests, "vlm_runner not called"
    manifest = seen_manifests[0]
    assert manifest is not None
    assert manifest.frames, "manifest.frames must not be empty"
    for f in manifest.frames:
        assert f["image_url"].startswith("data:image/jpeg;base64,")
    assert manifest.tracks, "tracks must be in manifest"

    # Perception evidence persisted to dedicated tables.
    from db.models import Detection, Keyframe, Track, TrackObservation
    with SM() as s:
        assert s.query(Detection).count() >= 1
        assert s.query(Track).count() >= 1
        assert s.query(Keyframe).count() >= 1
        assert s.query(TrackObservation).count() >= 1

    # Evidence package payload exposes them too.
    from evidence.package import latest_package_for_case
    with SM() as s:
        pkg = latest_package_for_case(s, case_id)
    assert pkg["perception"]["detections"], \
        "package perception.detections must be non-empty"
    assert pkg["perception"]["tracks"], \
        "package perception.tracks must be non-empty"
    assert pkg["perception"]["keyframes"], \
        "package perception.keyframes must be non-empty"


# ---------------------------------------------------------------------------
# 3. Segment row but file missing -> INVALID_VIDEO
# ---------------------------------------------------------------------------

def test_analyze_case_missing_segment_file_marks_invalid_video(
        tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    from db.models import VideoSegment
    base = datetime(2026, 6, 15, 14, 0, 0)
    with SM() as s:
        s.add(VideoSegment(
            camera_id="cam_return_01",
            start_at=base - timedelta(seconds=200),
            end_at=base + timedelta(seconds=200),
            path=str(tmp_path / "nope.mp4"),
            sha256="a" * 64, fps=25,
            width=160, height=120,
            frame_count=1500, duration_sec=60,
            corrupt=False, has_gap=False,
        ))
        s.commit()
    case_id = _seed_pos_event(SM, pos_event_at=base)

    from app.case_runner import analyze_case
    with SM() as s:
        result = analyze_case(s, case_id,
                              perception_runner=lambda *a: None,
                              vlm_runner=lambda *a: {})
    assert result["outcome"] == "INVALID_VIDEO"
    assert "missing on disk" in (result.get("invalid_reason") or "")


# ---------------------------------------------------------------------------
# 4. Evidence graph contains DETECTION / TRACK / KEYFRAME nodes
# ---------------------------------------------------------------------------

def test_evidence_graph_includes_perception_node_types(tmp_path, monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)
    storage = tmp_path / "storage"
    base = datetime(2026, 6, 15, 14, 0, 0)
    _seed_segment(SM, storage, start_at=base - timedelta(seconds=5),
                  duration_sec=10)
    case_id = _seed_pos_event(SM, pos_event_at=base)

    def _stub_perception(*_a, **_k):
        return {
            "detections": [{"label": "bag", "score": 0.9,
                            "bbox_xyxy": [10, 10, 50, 50],
                            "frame_id": "f0", "frame_idx": 0,
                            "ts": base.isoformat()}],
            "tracks": [{"track_id": "track_0001", "label": "bag",
                        "first_seen_ts": base.isoformat(),
                        "last_seen_ts": base.isoformat(),
                        "detections": [0],
                        "physical_item_candidate": True,
                        "zones": ["counter_zone"],
                        "events": ["handover_candidate"],
                        "confidence": 0.9}],
            "keyframes": [{"frame_id": "f0", "frame_idx": 0,
                            "ts": base.isoformat(),
                            "role": "handover_candidate",
                            "track_id": "track_0001"}],
            "ocr": [], "obstructed": False, "limitations": [],
        }

    def _stub_vlm(*_a, **_k):
        return {"provider": "x", "model_name": "x",
                "parsed": {"handover_occurred": True,
                           "physical_item_presented": True,
                           "confidence": "high",
                           "obstructed": False,
                           "camera_view_clear": True},
                "latency_ms": 1, "error": None}

    from app.case_runner import analyze_case
    with SM() as s:
        analyze_case(s, case_id,
                     perception_runner=_stub_perception,
                     vlm_runner=_stub_vlm)

    from evidence.graph import graph_for_case
    with SM() as s:
        g = graph_for_case(s, case_id)
    node_types = {n["node_type"] for n in g["nodes"]}
    for nt in ("CASE", "POS_EVENT", "VIDEO_WINDOW", "VIDEO_SEGMENT",
               "DETECTION", "TRACK", "KEYFRAME", "VLM_CLAIM"):
        assert nt in node_types, f"missing graph node type {nt!r}"


# ---------------------------------------------------------------------------
# 5. ChainProvider health + memory-guard wiring
# ---------------------------------------------------------------------------

def test_chain_provider_health_returns_member_status(tmp_path, monkeypatch):
    """When the operator opts in to the Gemma fallback, the chain
    health line surfaces BOTH members. Production default is Qwen-only
    so this test explicitly enables fallback."""
    import app.config as ac
    fake_root = tmp_path / "models" / "hf"
    (fake_root / "Qwen/Qwen3-VL-30B-A3B-Instruct" / "snap").mkdir(parents=True)
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)
    from reasoning.providers import ChainProvider, build_active_provider

    cfg = ac.load_config()
    cfg.raw.setdefault("reasoning", {})["fallback_provider"] = "gemma"
    chain = build_active_provider(cfg)
    assert isinstance(chain, ChainProvider)
    h = chain.health()
    assert h.provider == "chain"
    assert "qwen3_vl" in h.detail and "gemma" in h.detail


def test_chain_provider_registers_unload_callbacks_with_memory_guard(
        tmp_path, monkeypatch):
    import app.config as ac
    fake_root = tmp_path / "models" / "hf"
    (fake_root / "Qwen/Qwen3-VL-30B-A3B-Instruct" / "snap").mkdir(parents=True)
    monkeypatch.setattr(ac, "BUNDLE_ROOT", fake_root)

    from app.memory_guard import (
        MemoryPolicy, MemoryPolicyConfig, set_policy_for_test,
    )
    policy = MemoryPolicy(MemoryPolicyConfig(
        soft_gb=90, hard_gb=100, emergency_gb=110, poll_interval_sec=999),
        probe=lambda: (120.0, 30.0))
    set_policy_for_test(policy)
    try:
        from reasoning.providers import build_active_provider
        cfg = ac.load_config()
        cfg.raw.setdefault("reasoning", {})["fallback_provider"] = "gemma"
        chain = build_active_provider(cfg)
        # Once built, the policy should know about each member.
        for member in chain.providers:
            assert member.name in policy._unload_callbacks
        # Simulate hard memory pressure — every registered callback fires.
        unload_calls: list[str] = []
        for name in policy._unload_callbacks:
            policy._unload_callbacks[name] = (
                lambda n=name: unload_calls.append(n))
        policy._probe = lambda: (120.0, 101.0)
        policy.poll()
        assert "qwen3_vl" in unload_calls
        assert "gemma" in unload_calls
    finally:
        policy.reset_for_test()
