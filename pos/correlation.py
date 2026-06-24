"""POS-to-video window correlation.

This is the boundary between a POS event and the visual evidence
window the perception pipeline will inspect. The algorithm matches
the design in the refactor plan:

1.  start = pos_event_at - 120s
2.  end   = pos_event_at + 180s
3.  Expand by ``drift_margin_sec`` (default 60s; 600s if clock drift
    is suspected) when querying ``video_segments``.
4.  Require coverage_ratio >= 0.8 within [start, end] before allowing
    the case to enter perception. Otherwise mark INVALID_VIDEO with a
    structured reason.

This module is purely deterministic — no model calls, no I/O beyond the
DB. The recorder, perception worker, and reasoning pipeline orchestrate
it from above.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import VideoSegment


PRE_ROLL_SEC = 90    # 90s before the POS event (customer presents item)
POST_ROLL_SEC = 60   # 1 min after (customer leaving)
DEFAULT_DRIFT_MARGIN_SEC = 60
EXPANDED_DRIFT_MARGIN_SEC = 600  # 10 minutes
COVERAGE_RATIO_THRESHOLD = 0.8


def _naive_utc(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC so it compares correctly with
    SQLite-roundtripped values (SQLite strips tzinfo). Tz-aware inputs
    are converted to UTC; naive inputs are assumed UTC already."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class WindowPlan:
    requested_start: datetime
    requested_end: datetime
    drift_margin_sec: int
    matched_segment_ids: list[str]
    actual_start: Optional[datetime]
    actual_end: Optional[datetime]
    coverage_ratio: float
    invalid_reason: Optional[str]

    @property
    def is_valid(self) -> bool:
        return self.invalid_reason is None


def plan_window(session: Session,
                camera_id: str,
                pos_event_at: datetime,
                *,
                drift_margin_sec: int = DEFAULT_DRIFT_MARGIN_SEC,
                pre_roll_sec: Optional[float] = None,
                post_roll_sec: Optional[float] = None) -> WindowPlan:
    """Compute a video window for ``pos_event_at`` on ``camera_id``.

    ``pre_roll_sec`` / ``post_roll_sec`` override the module defaults
    (``PRE_ROLL_SEC`` / ``POST_ROLL_SEC``) when an operator widens the
    window for a retime+reprocess. ``None`` falls back to the defaults.

    Caller decides what to do with the result:
      - ``plan.is_valid`` → submit to perception
      - otherwise         → mark case INVALID_VIDEO with ``invalid_reason``
    """
    pre = PRE_ROLL_SEC if pre_roll_sec is None else max(0.0, float(pre_roll_sec))
    post = POST_ROLL_SEC if post_roll_sec is None else max(0.0, float(post_roll_sec))
    pos_event_at = _naive_utc(pos_event_at)
    requested_start = pos_event_at - timedelta(seconds=pre)
    requested_end = pos_event_at + timedelta(seconds=post)
    margin = timedelta(seconds=drift_margin_sec)

    rows = session.execute(
        select(VideoSegment)
        .where(
            VideoSegment.camera_id == camera_id,
            VideoSegment.end_at >= requested_start - margin,
            VideoSegment.start_at <= requested_end + margin,
        )
        .order_by(VideoSegment.start_at.asc())
    ).scalars().all()

    if not rows:
        return WindowPlan(
            requested_start=requested_start,
            requested_end=requested_end,
            drift_margin_sec=drift_margin_sec,
            matched_segment_ids=[],
            actual_start=None,
            actual_end=None,
            coverage_ratio=0.0,
            invalid_reason="no overlapping CCTV segments",
        )

    usable = [s for s in rows if not s.corrupt]
    if not usable:
        return WindowPlan(
            requested_start=requested_start,
            requested_end=requested_end,
            drift_margin_sec=drift_margin_sec,
            matched_segment_ids=[s.id for s in rows],
            actual_start=None,
            actual_end=None,
            coverage_ratio=0.0,
            invalid_reason="all overlapping segments marked corrupt",
        )

    actual_start = min(s.start_at for s in usable)
    actual_end = max(s.end_at for s in usable)

    coverage = _coverage_ratio(usable, requested_start, requested_end)
    invalid_reason = None
    if coverage < COVERAGE_RATIO_THRESHOLD:
        invalid_reason = (
            f"coverage {coverage:.2f} below threshold "
            f"{COVERAGE_RATIO_THRESHOLD:.2f}"
        )

    return WindowPlan(
        requested_start=requested_start,
        requested_end=requested_end,
        drift_margin_sec=drift_margin_sec,
        matched_segment_ids=[s.id for s in usable],
        actual_start=actual_start,
        actual_end=actual_end,
        coverage_ratio=coverage,
        invalid_reason=invalid_reason,
    )


def _coverage_ratio(segments: list,
                    requested_start: datetime,
                    requested_end: datetime) -> float:
    """Fraction of ``[requested_start, requested_end]`` covered by the
    union of segment intervals."""
    if requested_end <= requested_start:
        return 0.0
    total = (requested_end - requested_start).total_seconds()

    # Clip segments to the requested window, then union them.
    clipped = []
    for s in segments:
        a = max(s.start_at, requested_start)
        b = min(s.end_at, requested_end)
        if b > a:
            clipped.append((a, b))
    if not clipped:
        return 0.0
    clipped.sort()
    covered = 0.0
    cur_a, cur_b = clipped[0]
    for a, b in clipped[1:]:
        if a <= cur_b:
            cur_b = max(cur_b, b)
        else:
            covered += (cur_b - cur_a).total_seconds()
            cur_a, cur_b = a, b
    covered += (cur_b - cur_a).total_seconds()
    return covered / total
