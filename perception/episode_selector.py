"""SCO episode selector (Phase 4).

Inputs from Falcon's single wide-window pass:
  * tracks: list[Track] with zones[] and first/last seen timestamps
  * pos_time: POS transaction timestamp (the anchor)
  * roi_name: the single SCO audit-zone name (config-driven)
  * window_start / window_end: the wide POS window we ran Falcon on

Output:
  EpisodeWindow(start, end, ambiguous, reason, coverage_ratio)

Semantics (deliberately conservative — see SCO design notes):
  * The "episode" is the contiguous block of person activity inside the
    SCO audit zone that overlaps pos_time.
  * If no person activity in zone → fall back to the full POS window
    with reason="no_activity". Not ambiguous — there's nothing to
    confuse us, just bad luck on detector coverage.
  * If multiple distinct activity groups overlap pos_time (gap between
    groups > MERGE_GAP_SEC) → ambiguous=True with reason="multiple_groups".
    Caller falls back to the wide window and tells the VLM episode
    confidence is low.
  * If a single group is "abnormally long" (longer than
    MAX_EPISODE_SEC) → ambiguous=True with reason="long_continuous"
    because that likely means two customers ran together without a
    detected gap. Same fallback.
  * If exactly one clean group is found that overlaps pos_time → return
    its time range, ambiguous=False, with a coverage_ratio relative to
    the POS window so the policy can demand minimum coverage to reach
    VERIFIED.

This module does NOT reuse the refund customer_present gate. The
refund gate matches on zone-name substring "customer" — SCO must
not depend on that string being present, since the configured zone
name is ``sco_audit_zone``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional


log = logging.getLogger(__name__)


# Tracks separated by more than this gap count as different episodes
# (different customers / different visits).
MERGE_GAP_SEC: float = 5.0

# An "episode" longer than this is suspicious — likely two customers
# overlapping or a person who lingered well past their transaction.
# Tuned conservatively; operators can override via config later.
MAX_EPISODE_SEC: float = 90.0

# When evaluating which episode is "the customer's", accept any episode
# whose [start, end] falls within ±POS_ANCHOR_TOLERANCE_SEC of pos_time.
POS_ANCHOR_TOLERANCE_SEC: float = 30.0


@dataclass
class EpisodeWindow:
    start: datetime
    end: datetime
    ambiguous: bool
    reason: str            # short machine-readable tag
    coverage_ratio: float  # episode duration / pos-window duration

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "ambiguous": self.ambiguous,
            "reason": self.reason,
            "coverage_ratio": round(self.coverage_ratio, 4),
        }


def select_sco_episode(tracks: Iterable[dict],
                       *,
                       pos_time: datetime,
                       roi_name: str,
                       window_start: datetime,
                       window_end: datetime,
                       merge_gap_sec: float = MERGE_GAP_SEC,
                       max_episode_sec: float = MAX_EPISODE_SEC,
                       anchor_tolerance_sec: float = POS_ANCHOR_TOLERANCE_SEC,
                       ) -> EpisodeWindow:
    """Pick the customer episode inside ``roi_name`` for the POS event
    at ``pos_time``. Defensive: never raises, returns a usable window
    even when input is sparse.

    ``tracks`` may be a list of Track dataclasses or pre-serialised
    track dicts (case_runner persistence emits dicts). Each entry needs
    at least ``label``, ``zones`` (list[str]), ``first_seen_ts``,
    ``last_seen_ts``.
    """
    window_secs = max((window_end - window_start).total_seconds(), 1e-6)
    person_in_zone = _select_person_tracks_in_zone(tracks, roi_name)
    if not person_in_zone:
        return EpisodeWindow(
            start=window_start, end=window_end,
            ambiguous=False, reason="no_activity",
            coverage_ratio=0.0,
        )

    # Build (start, end) intervals from the matching tracks and merge
    # those within merge_gap_sec into contiguous groups.
    intervals = sorted(
        ((t["first"], t["last"]) for t in person_in_zone),
        key=lambda p: p[0],
    )
    groups = _merge_intervals(intervals, gap_sec=merge_gap_sec)

    # Find groups whose [start - tol, end + tol] contain pos_time.
    tol = timedelta(seconds=anchor_tolerance_sec)
    overlapping = [
        (s, e) for (s, e) in groups
        if (s - tol) <= pos_time <= (e + tol)
    ]
    if not overlapping:
        # Person activity exists in zone but doesn't span pos_time.
        # Fall back to the wide window and mark non-ambiguous low-coverage.
        return EpisodeWindow(
            start=window_start, end=window_end,
            ambiguous=False, reason="anchor_outside_groups",
            coverage_ratio=0.0,
        )
    if len(overlapping) > 1:
        # Two distinct groups both pass the POS anchor test → can't tell
        # which one is the customer.
        return EpisodeWindow(
            start=window_start, end=window_end,
            ambiguous=True, reason="multiple_groups",
            coverage_ratio=_total_coverage(overlapping, window_secs),
        )

    # Single clean group.
    s, e = overlapping[0]
    duration = (e - s).total_seconds()
    if duration > max_episode_sec:
        # Group is implausibly long → likely two customers merged.
        return EpisodeWindow(
            start=window_start, end=window_end,
            ambiguous=True, reason="long_continuous",
            coverage_ratio=min(1.0, duration / window_secs),
        )
    return EpisodeWindow(
        start=s, end=e,
        ambiguous=False, reason="clean_episode",
        coverage_ratio=min(1.0, duration / window_secs),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_person_tracks_in_zone(tracks: Iterable, roi_name: str
                                  ) -> list[dict]:
    out: list[dict] = []
    for t in (tracks or []):
        first, last, label, zones = _track_fields(t)
        if first is None or last is None:
            continue
        if not _is_person_label(label):
            continue
        if roi_name not in zones:
            continue
        out.append({"first": first, "last": last,
                    "label": label, "zones": zones})
    return out


def _track_fields(t) -> tuple[Optional[datetime], Optional[datetime],
                              str, list[str]]:
    """Tolerantly extract (first, last, label, zones) from either a
    Track dataclass or a pre-serialised dict."""
    if isinstance(t, dict):
        first = _coerce_dt(t.get("first_seen_ts"))
        last = _coerce_dt(t.get("last_seen_ts"))
        label = str(t.get("label") or "")
        zones = list(t.get("zones") or [])
    else:
        first = _coerce_dt(getattr(t, "first_seen_ts", None))
        last = _coerce_dt(getattr(t, "last_seen_ts", None))
        label = str(getattr(t, "label", "") or "")
        zones = list(getattr(t, "zones", None) or [])
    return first, last, label, zones


def _coerce_dt(v) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            return None
    return None


def _is_person_label(label: str) -> bool:
    """Falcon emits category keys like 'person' for the default person
    detector. A POS-derived category key (sco_item_*, sco_generic_*) is
    NOT a person."""
    lower = label.lower()
    return "person" in lower and "sco_" not in lower


def _merge_intervals(intervals: list[tuple[datetime, datetime]],
                     *, gap_sec: float
                     ) -> list[tuple[datetime, datetime]]:
    if not intervals:
        return []
    gap = timedelta(seconds=gap_sec)
    merged: list[tuple[datetime, datetime]] = [intervals[0]]
    for s, e in intervals[1:]:
        prev_s, prev_e = merged[-1]
        if s - prev_e <= gap:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged


def _total_coverage(intervals: list[tuple[datetime, datetime]],
                    window_secs: float) -> float:
    total = sum((e - s).total_seconds() for s, e in intervals)
    return min(1.0, total / max(window_secs, 1e-6))
