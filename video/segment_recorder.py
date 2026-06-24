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

        return self._index_segment_file(out_path, start_at_naive,
                                        end_at_naive, len(frames))

    def _index_segment_file(self, out_path: Path,
                            start_at_naive: datetime,
                            end_at_naive: datetime,
                            frame_count_fallback: int) -> Optional[str]:
        """Probe + hash + index an already-written segment file. Safe to
        call from a background finalizer thread so the capture loop never
        stalls (which is what created gaps between segments)."""
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
                frame_count=probe.frame_count or frame_count_fallback,
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
        """Gapless live RTSP loop.

        Frames are written to the segment's MP4 AS THEY ARRIVE, and when a
        segment reaches ``segment_duration_sec`` the file is closed and its
        finalization (ffprobe + sha256 + DB index) is handed to a
        background thread while the NEXT segment starts capturing
        immediately. The previous implementation buffered a whole segment
        in RAM and then ran the write+probe+hash+index inline, during which
        no frames were captured — that pause is what dropped local
        coverage below the window-builder threshold and forced the slow
        NVR fallback.
        """
        import cv2  # local import keeps module light
        from concurrent.futures import ThreadPoolExecutor

        from rtsp_reader import RTSPReader

        reader = RTSPReader(self.cfg.rtsp_url, name=self.cfg.camera_id)
        finalizer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"seg-final-{self.cfg.camera_id}")
        fourcc = cv2.VideoWriter_fourcc(*self.cfg.codec)
        target = self.cfg.segment_duration_sec
        # Write at most ``cfg.fps`` frames per REAL second so the segment
        # plays back in real time. The RTSP stream delivers far more frames
        # than cfg.fps; writing every one at cfg.fps produced ~5x slow
        # motion, which broke the window<->POS-time alignment (the analysed
        # clip didn't actually cover the transaction). ``interval`` is the
        # minimum real gap between written frames.
        interval = 1.0 / float(self.cfg.fps or 5)

        writer = None
        out_path: Optional[Path] = None
        seg_start: Optional[datetime] = None
        last_ts: Optional[datetime] = None
        last_write_ts: Optional[datetime] = None
        frame_count = 0
        low_disk_logged = False

        def _close_and_index() -> None:
            """Close the current writer and finalize it off-thread."""
            nonlocal writer, out_path, seg_start, last_ts, last_write_ts
            nonlocal frame_count
            if writer is None:
                return
            writer.release()
            if out_path is not None and seg_start is not None:
                finalizer.submit(self._safe_index, out_path, seg_start,
                                 last_ts or seg_start, frame_count)
            writer = None
            out_path = None
            seg_start = None
            last_ts = None
            last_write_ts = None
            frame_count = 0

        try:
            for frame in reader.frames(self._stop_event):
                # Low-disk guard: close any in-progress segment cleanly
                # (so it is still indexed) and pause. API/UI stay alive.
                if _disk_too_low():
                    if not low_disk_logged:
                        log.warning("recorder %s: low disk state — "
                                    "pausing segment writes",
                                    self.cfg.camera_id)
                        low_disk_logged = True
                    _close_and_index()
                    self._stop_event.wait(5)
                    continue
                if low_disk_logged:
                    log.info("recorder %s: disk recovered — resuming",
                             self.cfg.camera_id)
                    low_disk_logged = False

                now = datetime.now(timezone.utc).replace(tzinfo=None)
                if writer is None:
                    seg_start = now
                    out_path = segment_path_for(self.cfg.storage_root,
                                                self.cfg.camera_id, now)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(out_path), fourcc,
                                             self.cfg.fps, (w, h))
                    if not writer.isOpened():
                        log.error("recorder %s: VideoWriter failed to open %s",
                                  self.cfg.camera_id, out_path)
                        writer = None
                        out_path = None
                        seg_start = None
                        continue
                    frame_count = 0

                # Throttle to cfg.fps in real time. The first frame of a
                # segment is always written; subsequent frames only when at
                # least ``interval`` real seconds have passed. Skipped
                # frames still count toward segment-rotation timing below.
                if last_write_ts is None or \
                        (now - last_write_ts).total_seconds() >= interval * 0.9:
                    writer.write(frame)
                    frame_count += 1
                    last_write_ts = now
                    last_ts = now
                if (now - seg_start).total_seconds() >= target:
                    _close_and_index()  # starts next segment on next frame
                if self._stop_event.is_set():
                    break
            # Flush the trailing partial segment.
            _close_and_index()
        finally:
            reader.close()
            finalizer.shutdown(wait=True)

    def _safe_index(self, out_path: Path, start_at_naive: datetime,
                    end_at_naive: datetime, frame_count: int) -> None:
        """Background finalizer wrapper — never lets an indexing error
        kill the recorder."""
        try:
            self._index_segment_file(out_path, start_at_naive,
                                     end_at_naive, frame_count)
        except Exception:
            log.exception("recorder %s: segment finalize failed for %s",
                          self.cfg.camera_id, out_path)

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
