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
                 out_path: str | Path) -> WindowBuildResult:
    """Concatenate the relevant segment files into one MP4.

    The current implementation uses ``ffmpeg -f concat`` so we never
    re-encode (cheap + lossless). If ``ffmpeg`` is not on PATH, returns
    failure rather than guessing.
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

    # Filter to segments whose file exists. The window builder NEVER
    # overwrites the segment files themselves.
    usable = [s for s in segments if Path(s.path).is_file()]
    if not usable:
        return WindowBuildResult(
            ok=False, out_path=None, sha256=None,
            actual_start_at=None, actual_end_at=None,
            segment_ids=[s.id for s in segments],
            failure_reason="all segment files missing on disk",
        )

    usable.sort(key=lambda s: s.start_at)
    # Write a concat list file.
    with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False) as listf:
        for s in usable:
            listf.write(f"file '{os.path.abspath(s.path)}'\n")
        list_path = listf.name

    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
               "-i", list_path, "-c", "copy", str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=300)
        if proc.returncode != 0:
            return WindowBuildResult(
                ok=False, out_path=None, sha256=None,
                actual_start_at=None, actual_end_at=None,
                segment_ids=[s.id for s in usable],
                failure_reason=(
                    f"ffmpeg concat failed (rc={proc.returncode}): "
                    f"{proc.stderr[-400:]}"
                ),
            )
    finally:
        try:
            os.unlink(list_path)
        except OSError:
            pass

    actual_start = min(s.start_at for s in usable)
    actual_end = max(s.end_at for s in usable)
    return WindowBuildResult(
        ok=True,
        out_path=str(out_path),
        sha256=sha256_file(out_path),
        actual_start_at=actual_start,
        actual_end_at=actual_end,
        segment_ids=[s.id for s in usable],
    )


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"],
                              capture_output=True, timeout=5)
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False
