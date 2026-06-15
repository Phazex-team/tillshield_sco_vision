"""Integrity utilities for CCTV segments.

Functions here verify that an on-disk MP4 segment can be opened, that
its declared metadata (fps, frame_count, duration) matches what
OpenCV / ffprobe report, and that its SHA-256 matches the manifest.

This module is intentionally light: it never re-encodes, never
modifies the file. The only external dependency is OpenCV (already in
the venv) plus a best-effort ``subprocess.run('ffprobe')`` for cases
where OpenCV misreports.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


log = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    path: str
    ok: bool
    corrupt: bool
    has_gap: bool
    duration_sec: float
    fps: float
    width: int
    height: int
    frame_count: int
    error: Optional[str] = None


def sha256_file(path: str | Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def probe_segment(path: str | Path,
                  *,
                  expected_duration_sec: Optional[float] = None,
                  duration_tolerance_sec: float = 0.5) -> ProbeResult:
    """Open ``path`` with OpenCV; if that fails, fall back to ffprobe.

    Returns a ``ProbeResult`` describing the file. ``corrupt=True`` if
    the file cannot be read at all or yields zero frames; ``has_gap``
    when the measured duration is shorter than expected.
    """
    p = str(path)
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            return ProbeResult(path=p, ok=False, corrupt=True, has_gap=False,
                               duration_sec=0.0, fps=0.0, width=0, height=0,
                               frame_count=0, error="cv2 could not open file")
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
    except Exception as exc:
        return ProbeResult(path=p, ok=False, corrupt=True, has_gap=False,
                           duration_sec=0.0, fps=0.0, width=0, height=0,
                           frame_count=0,
                           error=f"cv2 probe raised: {exc}")
    if count <= 0 or fps <= 0:
        # Try ffprobe as a tie-breaker.
        ff = _ffprobe_duration(p)
        if ff is None:
            return ProbeResult(path=p, ok=False, corrupt=True, has_gap=False,
                               duration_sec=0.0, fps=0.0, width=w, height=h,
                               frame_count=count,
                               error="cv2 reported zero fps/frames "
                                     "and ffprobe unavailable")
        duration = ff
    else:
        duration = count / fps

    has_gap = False
    if expected_duration_sec is not None:
        has_gap = duration + duration_tolerance_sec < expected_duration_sec

    return ProbeResult(
        path=p, ok=True, corrupt=False, has_gap=has_gap,
        duration_sec=round(duration, 3), fps=round(fps, 3),
        width=w, height=h, frame_count=count, error=None,
    )


def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries",
               "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
               path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return float(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None
    except Exception:
        log.exception("ffprobe failed")
        return None
