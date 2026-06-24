"""Frame sampling policy (PRODUCTION_SPEC §10).

The sampler returns ``(frame_idx, ts)`` tuples telling the perception
pipeline which video positions to decode. Three tiers:

* base FPS across the full window (default 1 fps)
* motion bursts: higher FPS around regions tagged by zone_trigger
* handover bursts: highest FPS in [t-3s, t+5s] around handover candidates

The actual frame decoding happens in ``pipeline.py`` (cv2.VideoCapture)
so this module stays pure-Python and trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable


@dataclass
class SamplingPolicy:
    # base_fps=5 matches the recorder's 5 fps, so Falcon processes EVERY
    # recorded frame across the whole window (no quiet-region gaps), not
    # just 1 fps. Bursts stay at/above this around motion/handover.
    base_fps: float = 5.0
    burst_fps: float = 5.0
    handover_fps: float = 10.0
    burst_pre_sec: float = 3.0
    burst_post_sec: float = 5.0


def plan_indices(*,
                 fps: float,
                 frame_count: int,
                 base_start_ts: datetime,
                 policy: SamplingPolicy,
                 motion_bursts: Iterable[datetime] = (),
                 handover_candidates: Iterable[datetime] = ()
                 ) -> list[tuple[int, datetime]]:
    """Return (frame_idx, ts) pairs the pipeline should decode.

    ``motion_bursts`` and ``handover_candidates`` are timestamps; the
    sampler densifies sampling around each. Duplicates are removed so
    every chosen frame is decoded at most once.
    """
    if fps <= 0 or frame_count <= 0:
        return []

    def _idx_for(ts: datetime) -> int:
        delta = (ts - base_start_ts).total_seconds()
        return max(0, min(frame_count - 1, int(delta * fps)))

    indices: dict[int, datetime] = {}

    # Base sampling
    step = max(1, int(round(fps / max(policy.base_fps, 1e-3))))
    for i in range(0, frame_count, step):
        ts = base_start_ts + timedelta(seconds=i / fps)
        indices[i] = ts

    # Motion bursts
    for centre in motion_bursts:
        _densify(indices, fps, frame_count, base_start_ts, centre,
                 burst_fps=policy.burst_fps,
                 pre_sec=policy.burst_pre_sec,
                 post_sec=policy.burst_post_sec)

    # Handover candidates
    for centre in handover_candidates:
        _densify(indices, fps, frame_count, base_start_ts, centre,
                 burst_fps=policy.handover_fps,
                 pre_sec=policy.burst_pre_sec,
                 post_sec=policy.burst_post_sec)

    return sorted(indices.items(), key=lambda kv: kv[0])


def _densify(indices: dict, fps: float, frame_count: int,
             base_start_ts: datetime, centre: datetime,
             burst_fps: float, pre_sec: float, post_sec: float) -> None:
    if burst_fps <= 0:
        return
    start = centre - timedelta(seconds=pre_sec)
    end = centre + timedelta(seconds=post_sec)
    step_sec = 1.0 / burst_fps
    cur = start
    while cur <= end:
        delta = (cur - base_start_ts).total_seconds()
        idx = max(0, min(frame_count - 1, int(delta * fps)))
        indices[idx] = base_start_ts + timedelta(seconds=idx / fps)
        cur += timedelta(seconds=step_sec)
