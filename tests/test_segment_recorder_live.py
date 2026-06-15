"""Recorder live-loop tests.

Mocks ``rtsp_reader.RTSPReader.frames`` so the recorder runs its real
``_run_from_rtsp`` path without an RTSP server. Pins:

  * ``start()`` without an injected ``frame_source`` actually opens
    the RTSP reader and writes segments.
  * Multiple segments are written and indexed as the live loop runs.
  * Segments are immutable: the same canonical path is never written
    twice.
  * Existing synthetic-frame tests still pass.
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

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


def _fake_frame(seed: int = 0):
    return np.full((120, 160, 3), fill_value=(seed * 5) % 255,
                   dtype=np.uint8)


def test_run_from_rtsp_writes_and_indexes_real_segments(tmp_path,
                                                        monkeypatch):
    SM = _fresh_session(tmp_path, monkeypatch)

    # Stub RTSPReader so .frames(stop_evt) yields a finite stream.
    class _FakeReader:
        def __init__(self, *_a, **_k):
            self.closed = False

        def frames(self, stop_evt):
            # Emit ~50 frames quickly, then exit so the test doesn't
            # hang. The recorder slices by wall-clock duration, so we
            # also slow each yield slightly to let segment boundaries
            # fire.
            n = 0
            while not stop_evt.is_set() and n < 80:
                yield _fake_frame(n)
                n += 1
                time.sleep(0.05)

        def close(self):
            self.closed = True

    monkeypatch.setattr("rtsp_reader.RTSPReader", _FakeReader)

    from video.segment_recorder import RecorderConfig, SegmentRecorder
    cfg = RecorderConfig(
        camera_id="cam_live",
        storage_root=tmp_path / "storage",
        rtsp_url="rtsp://test/stream",
        segment_duration_sec=1,  # short so multiple segments land
        fps=25, width=160, height=120,
    )
    rec = SegmentRecorder(cfg, session_factory=SM)
    rec.start()
    # The fake reader yields ~80 frames at 50ms = ~4s wall time.
    time.sleep(4.5)
    rec.stop()

    from db.models import VideoSegment
    with SM() as s:
        rows = s.query(VideoSegment).all()
    assert len(rows) >= 2, \
        f"expected multiple segments, got {len(rows)}"
    for r in rows:
        assert r.path and Path(r.path).is_file()
        assert r.sha256 and len(r.sha256) == 64


def test_run_from_rtsp_refuses_overwrite_on_repeat_start_time(
        tmp_path, monkeypatch):
    """Two record_one_segment calls with the same start_at must not
    overwrite. The live loop uses datetime.now() so this is unlikely in
    practice, but the canonical-path immutability rule still holds."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from video.segment_recorder import RecorderConfig, SegmentRecorder
    cfg = RecorderConfig(camera_id="cam_x", storage_root=tmp_path / "s",
                         fps=25, width=160, height=120,
                         segment_duration_sec=2)
    rec = SegmentRecorder(cfg, session_factory=SM)
    frames = [(datetime(2026, 6, 15, 14, 0, 0),
               _fake_frame(i)) for i in range(25)]
    first = rec.record_one_segment(frames,
                                   start_at=datetime(2026, 6, 15, 14, 0, 0))
    second = rec.record_one_segment(frames,
                                    start_at=datetime(2026, 6, 15, 14, 0, 0))
    assert first
    assert second is None


def test_run_without_rtsp_or_frame_source_is_a_noop(tmp_path, monkeypatch,
                                                    caplog):
    """If neither rtsp_url nor frame_source is set, the recorder must
    log an error and exit cleanly (rather than block forever)."""
    SM = _fresh_session(tmp_path, monkeypatch)
    from video.segment_recorder import RecorderConfig, SegmentRecorder
    cfg = RecorderConfig(camera_id="cam_noop", storage_root=tmp_path / "s",
                         rtsp_url=None, fps=25, width=160, height=120,
                         segment_duration_sec=1)
    rec = SegmentRecorder(cfg, session_factory=SM)
    rec.start()
    # Background thread exits immediately when neither input is set.
    rec._thread.join(timeout=2)
    assert not rec._thread.is_alive()
