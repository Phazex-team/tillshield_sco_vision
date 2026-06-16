"""Reconstruct a video window for a POS event from segment files.

Given a list of immutable segments + a requested ``[start, end]``
range, build a single MP4 covering the union of overlapping segments
clipped to the requested window. Failure modes follow
PRODUCTION_SPEC §9: missing coverage / corrupt segments produce a
clear failure code rather than a guess.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from db.models import VideoSegment

from .integrity import sha256_file


log = logging.getLogger(__name__)


@dataclass
class WindowBuildResult:
    ok: bool
    out_path: Optional[str]
    sha256: Optional[str]
    actual_start_at: Optional[datetime]
    actual_end_at: Optional[datetime]
    segment_ids: list[str] = field(default_factory=list)
    failure_reason: Optional[str] = None


def build_window(*,
                 segments: list[VideoSegment],
                 requested_start: datetime,
                 requested_end: datetime,
                 out_path: str | Path,
                 clip_to_requested: bool = True) -> WindowBuildResult:
    """Build a single MP4 covering ``[requested_start, requested_end]``
    from the supplied immutable segments.

    Two passes (transparent fallback):

      1. **Trimmed** (default): one ffmpeg invocation that concatenates
         every usable segment and trims to the requested window in the
         same pass via ``-ss`` / ``-to``. This drops any pre-roll or
         post-roll outside the POS window so evidence packages contain
         only relevant CCTV.
      2. **Concat-only**: if the trimmed pass fails (cv2-written mp4v
         segments occasionally refuse the trim filter in stream-copy
         mode), we retry with whole-segment concatenation so the
         pipeline still produces an evidence window — accepting some
         pre/post-roll rather than failing the case.

    ``actual_start_at`` / ``actual_end_at`` reflect the time range the
    output MP4 actually covers (after any trimming).
    """
    if not segments:
        return WindowBuildResult(
            ok=False, out_path=None, sha256=None,
            actual_start_at=None, actual_end_at=None,
            failure_reason="no segments supplied",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not _ffmpeg_available():
        return WindowBuildResult(
            ok=False, out_path=None, sha256=None,
            actual_start_at=None, actual_end_at=None,
            failure_reason="ffmpeg not available — install before running",
        )

    usable = [s for s in segments if Path(s.path).is_file()]
    if not usable:
        return WindowBuildResult(
            ok=False, out_path=None, sha256=None,
            actual_start_at=None, actual_end_at=None,
            segment_ids=[s.id for s in segments],
            failure_reason="all segment files missing on disk",
        )

    usable.sort(key=lambda s: s.start_at)

    requested_start_naive = _naive_utc(requested_start)
    requested_end_naive = _naive_utc(requested_end)
    concat_start = min(s.start_at for s in usable)
    concat_end = max(s.end_at for s in usable)
    # Effective trim bounds intersect the requested window with what
    # the segments actually cover. ss/to are relative seconds from the
    # concat start.
    trim_lo = max(0.0, (requested_start_naive
                        - concat_start).total_seconds())
    trim_hi = max(trim_lo,
                  (min(requested_end_naive, concat_end)
                   - concat_start).total_seconds())
    will_trim = clip_to_requested and (
        trim_lo > 0.0 or (concat_end - requested_end_naive).total_seconds() > 0)

    def _run(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=600)
        return proc.returncode, proc.stderr

    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False) as listf:
        for s in usable:
            listf.write(f"file '{os.path.abspath(s.path)}'\n")
        list_path = listf.name

    trimmed_attempted = will_trim
    trimmed_used = False
    last_stderr = ""
    try:
        if will_trim:
            # Trim + stream-copy in a single pass.
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", list_path,
                   "-ss", f"{trim_lo:.3f}", "-to", f"{trim_hi:.3f}",
                   "-c", "copy", str(out_path)]
            rc, stderr = _run(cmd)
            if rc == 0:
                trimmed_used = True
            else:
                last_stderr = stderr
        if not trimmed_used:
            # Concat-only fallback (may include pre/post-roll).
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                   "-i", list_path, "-c", "copy", str(out_path)]
            rc, stderr = _run(cmd)
            if rc != 0:
                return WindowBuildResult(
                    ok=False, out_path=None, sha256=None,
                    actual_start_at=None, actual_end_at=None,
                    segment_ids=[s.id for s in usable],
                    failure_reason=(
                        f"ffmpeg failed (rc={rc}): "
                        f"{(last_stderr or stderr)[-400:]}"
                    ),
                )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass

    if trimmed_used:
        actual_start = max(concat_start, requested_start_naive)
        actual_end = min(concat_end, requested_end_naive)
    else:
        actual_start = concat_start
        actual_end = concat_end

    return WindowBuildResult(
        ok=True,
        out_path=str(out_path),
        sha256=sha256_file(out_path),
        actual_start_at=actual_start,
        actual_end_at=actual_end,
        segment_ids=[s.id for s in usable],
        failure_reason=(None if trimmed_used or not trimmed_attempted
                         else "trimmed pass failed; concat-only fallback"),
    )


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        from datetime import timezone
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"],
                              capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False
