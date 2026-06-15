"""Continuous CCTV segment recorder.

Reads an RTSP stream (via the existing ``rtsp_reader.RTSPReader``) and
writes immutable fixed-duration MP4 chunks under
``storage/cctv/camera_id=<id>/date=YYYY-MM-DD/hour=HH/segment_*.mp4``.

After each chunk closes, the recorder:

  * computes its sha256,
  * probes it with ``video.integrity.probe_segment``,
  * inserts a ``video_segments`` row.

The recorder is intended to run as its own process so it stays alive
under inference-degradation (PRODUCTION_SPEC §7 / §9). Tests exercise
the offline path: writing one short clip into a temp dir, indexing it,
and confirming the row + on-disk file are produced.

Optional: full live recording requires the RTSP stream to be reachable.
The class supports ``frame_source`` injection so tests can hand it
synthetic frames.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional


log = logging.getLogger(__name__)


def _disk_too_low() -> bool:
    """Low-disk guard hook. Defined here (instead of imported at module
    top) so callers can monkeypatch it in tests."""
    try:
        from app.storage_guard import low_disk_state
        return low_disk_state()
    except Exception:
        return False


@dataclass
class RecorderConfig:
    camera_id: str
    storage_root: Path
    rtsp_url: Optional[str] = None
    segment_duration_sec: int = 60
    fps: int = 25
    width: int = 1920
    height: int = 1080
    codec: str = "mp4v"


def segment_path_for(storage_root: Path,
                     camera_id: str,
                     start_at: datetime) -> Path:
    """Compute the canonical immutable path for a segment.

    Layout matches PRODUCTION_SPEC §6:
        cctv/camera_id=<id>/date=YYYY-MM-DD/hour=HH/segment_<start>_<end>.mp4
    """
    if start_at.tzinfo is not None:
        start_at = start_at.astimezone(timezone.utc).replace(tzinfo=None)
    yyyy_mm_dd = start_at.strftime("%Y-%m-%d")
    hh = start_at.strftime("%H")
    return (storage_root
            / "cctv"
            / f"camera_id={camera_id}"
            / f"date={yyyy_mm_dd}"
            / f"hour={hh}"
            / f"segment_{start_at.strftime('%Y%m%dT%H%M%SZ')}.mp4")


class SegmentRecorder:
    """Records immutable CCTV segments and indexes them.

    Usage::

        rec = SegmentRecorder(cfg, session_factory=get_sessionmaker())
        rec.start()
        ...
        rec.stop()

    Tests inject ``frame_source`` (an iterable of (ts, np.ndarray)) so
    the recorder does not need an RTSP server to exercise its writer +
    index path.
    """

    def __init__(self,
                 cfg: RecorderConfig,
                 *,
                 session_factory=None,
                 frame_source: Optional[Callable[[],
                                                 Iterable[tuple[datetime,
                                                                "np.ndarray"]]]] = None
                 ):
        self.cfg = cfg
        self.session_factory = session_factory
        self.frame_source = frame_source
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=f"recorder-{self.cfg.camera_id}")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=10)

    # ------------------------------------------------------------------
    # Core record-and-index loop (test entry point too)
    # ------------------------------------------------------------------

    def record_one_segment(self,
                           frames: Iterable[tuple[datetime, "np.ndarray"]],
                           *,
                           start_at: Optional[datetime] = None) -> Optional[str]:
        """Write one segment + index it. Returns the inserted row id or
        ``None`` if the file is corrupt or the row collides."""
        import cv2  # local import keeps module light
        import numpy as np  # noqa: F401

        frames = list(frames)
        if not frames:
            log.warning("recorder %s: no frames received", self.cfg.camera_id)
            return None

        if start_at is None:
            start_at = frames[0][0]
        if start_at.tzinfo is not None:
            start_at_naive = start_at.astimezone(timezone.utc).replace(
                tzinfo=None)
        else:
            start_at_naive = start_at

        out_path = segment_path_for(self.cfg.storage_root,
                                    self.cfg.camera_id, start_at_naive)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            log.warning("recorder %s: refusing to overwrite %s",
                        self.cfg.camera_id, out_path)
            return None

        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        h, w = frames[0][1].shape[:2]
        writer = cv2.VideoWriter(str(out_path), fourcc,
                                 self.cfg.fps, (w, h))
        if not writer.isOpened():
            log.error("recorder %s: VideoWriter failed to open %s",
                      self.cfg.camera_id, out_path)
            return None
        try:
            for _ts, frame in frames:
                writer.write(frame)
        finally:
            writer.release()

        end_at = frames[-1][0]
        if end_at.tzinfo is not None:
            end_at_naive = end_at.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            end_at_naive = end_at

        from .integrity import probe_segment, sha256_file
        from .segment_index import insert_segment
        probe = probe_segment(
            out_path,
            expected_duration_sec=(end_at_naive - start_at_naive)
                .total_seconds() or self.cfg.segment_duration_sec,
        )
        if not probe.ok:
            log.error("recorder %s: probe failed for %s: %s",
                      self.cfg.camera_id, out_path, probe.error)
        sha = sha256_file(out_path)

        if self.session_factory is None:
            return out_path.name  # offline mode: index step skipped

        with self.session_factory() as session:
            row_id = insert_segment(
                session,
                camera_id=self.cfg.camera_id,
                start_at=start_at_naive,
                end_at=end_at_naive,
                path=str(out_path),
                sha256=sha,
                fps=probe.fps or self.cfg.fps,
                width=probe.width or self.cfg.width,
                height=probe.height or self.cfg.height,
                frame_count=probe.frame_count or len(frames),
                duration_sec=probe.duration_sec or 0.0,
                has_gap=probe.has_gap,
                corrupt=probe.corrupt,
            )
            session.commit()
            return row_id

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Live RTSP loop.

        Opens the configured RTSP URL via ``rtsp_reader.RTSPReader``,
        reads BGR frames continuously, and slices them into immutable
        fixed-duration chunks. Reconnects on failure and continues
        recording past inference-degraded states because the recorder
        consults nothing in the memory guard.

        Tests inject ``frame_source`` to bypass the network and prove
        the slicing + indexing path without an RTSP server.
        """
        if self.frame_source is not None:
            self._run_from_source()
            return
        if not self.cfg.rtsp_url:
            log.error("recorder %s: no rtsp_url and no frame_source — "
                      "nothing to record", self.cfg.camera_id)
            return
        self._run_from_rtsp()

    def _run_from_rtsp(self) -> None:
        from rtsp_reader import RTSPReader
        reader = RTSPReader(self.cfg.rtsp_url, name=self.cfg.camera_id)
        try:
            target = self.cfg.segment_duration_sec
            buf: list[tuple[datetime, "np.ndarray"]] = []
            seg_start: Optional[datetime] = None
            low_disk_logged = False
            for frame in reader.frames(self._stop_event):
                # Low-disk guard: when the storage root has dropped
                # below the configured threshold, drop the in-progress
                # buffer and wait. The API/reviewer UI stay alive.
                if _disk_too_low():
                    if not low_disk_logged:
                        log.warning("recorder %s: low disk state — "
                                    "pausing segment writes",
                                    self.cfg.camera_id)
                        low_disk_logged = True
                    buf = []
                    seg_start = None
                    self._stop_event.wait(5)
                    continue
                if low_disk_logged:
                    log.info("recorder %s: disk recovered — resuming",
                             self.cfg.camera_id)
                    low_disk_logged = False
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                if seg_start is None:
                    seg_start = now
                buf.append((now, frame))
                elapsed = (now - seg_start).total_seconds()
                if elapsed >= target:
                    try:
                        self.record_one_segment(buf, start_at=seg_start)
                    except Exception:
                        log.exception("recorder %s: segment write failed",
                                      self.cfg.camera_id)
                    buf = []
                    seg_start = None
                if self._stop_event.is_set():
                    break
            if buf and seg_start is not None:
                try:
                    self.record_one_segment(buf, start_at=seg_start)
                except Exception:
                    log.exception("recorder %s: trailing segment failed",
                                  self.cfg.camera_id)
        finally:
            reader.close()

    def _run_from_source(self) -> None:
        while not self._stop_event.is_set():
            try:
                start_at = datetime.now(timezone.utc).replace(tzinfo=None)
                frames = self.frame_source()
                self.record_one_segment(frames, start_at=start_at)
            except Exception:
                log.exception("recorder %s: segment cycle failed",
                              self.cfg.camera_id)
            self._stop_event.wait(self.cfg.segment_duration_sec)
