"""Segment index — read/write operations over ``video_segments``.

The recorder writes immutable segments to disk and inserts a row here.
The window builder, evidence package, and case workers query it.
Returning ORM rows everywhere keeps callers small.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import VideoSegment


def _naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def insert_segment(session: Session,
                   *,
                   camera_id: str,
                   start_at: datetime,
                   end_at: datetime,
                   path: str,
                   sha256: str,
                   fps: float,
                   width: int,
                   height: int,
                   frame_count: int,
                   duration_sec: float,
                   has_gap: bool = False,
                   corrupt: bool = False) -> Optional[str]:
    """Insert a segment row. Returns the new row id, or None when a
    segment with the same (camera_id, start_at) already exists. Segments
    are immutable; we never overwrite."""
    start_at = _naive_utc(start_at)
    end_at = _naive_utc(end_at)
    seg = VideoSegment(
        camera_id=camera_id,
        start_at=start_at,
        end_at=end_at,
        path=str(path),
        sha256=sha256,
        fps=fps,
        width=width,
        height=height,
        frame_count=frame_count,
        duration_sec=duration_sec,
        has_gap=has_gap,
        corrupt=corrupt,
    )
    session.add(seg)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        return None
    return seg.id


def segments_overlapping(session: Session,
                         camera_id: str,
                         start: datetime,
                         end: datetime,
                         *,
                         drift_margin_sec: int = 60) -> list[VideoSegment]:
    """Return segments overlapping ``[start, end]`` within the drift
    margin, ordered chronologically. Excludes corrupt segments."""
    start = _naive_utc(start)
    end = _naive_utc(end)
    margin = timedelta(seconds=drift_margin_sec)
    rows = session.execute(
        select(VideoSegment).where(
            VideoSegment.camera_id == camera_id,
            VideoSegment.corrupt.is_(False),
            VideoSegment.end_at >= start - margin,
            VideoSegment.start_at <= end + margin,
        ).order_by(VideoSegment.start_at.asc())
    ).scalars().all()
    return list(rows)


def coverage(session: Session,
             camera_id: str,
             start: datetime,
             end: datetime) -> dict:
    """Return a coverage summary for the given window."""
    start = _naive_utc(start)
    end = _naive_utc(end)
    rows = segments_overlapping(session, camera_id, start, end,
                                drift_margin_sec=0)
    if not rows:
        return {
            "camera_id": camera_id,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "segments": 0,
            "coverage_ratio": 0.0,
            "first_at": None,
            "last_at": None,
        }
    intervals = sorted((max(r.start_at, start), min(r.end_at, end))
                       for r in rows)
    merged = []
    cur_a, cur_b = intervals[0]
    for a, b in intervals[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            merged.append((cur_a, cur_b))
            cur_a, cur_b = a, b
    merged.append((cur_a, cur_b))
    total = (end - start).total_seconds()
    covered = sum((b - a).total_seconds() for a, b in merged)
    return {
        "camera_id": camera_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "segments": len(rows),
        "coverage_ratio": round(covered / total, 4) if total > 0 else 0.0,
        "first_at": min(r.start_at for r in rows).isoformat(),
        "last_at": max(r.end_at for r in rows).isoformat(),
    }
