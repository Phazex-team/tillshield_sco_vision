"""Lightweight IoU + label-compatible tracker (PRODUCTION_SPEC §10).

Traditional tracker. VLM does NOT create or merge IDs. Implementation
is intentionally simple: greedy nearest-neighbour by IoU with an
optional label-compatibility check, plus tentative/confirmed/lost
lifecycle. Sufficient for the slow movements at a return counter and
small enough to test offline with synthetic detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .schemas import Detection, Track


@dataclass
class TrackState:
    track_id: str
    label: str
    bbox_xyxy: list[float]
    first_seen_ts: datetime
    last_seen_ts: datetime
    detection_indices: list[int] = field(default_factory=list)
    misses: int = 0
    hits: int = 0
    status: str = "tentative"  # tentative -> confirmed -> lost -> closed
    confidence: float = 0.0


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _label_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    return a.lower() == b.lower()


class Tracker:
    """Greedy IoU tracker.

    Lifecycle:
      * Tentative: < ``confirm_hits`` observations.
      * Confirmed: >= ``confirm_hits`` observations.
      * Lost: >= ``lost_miss`` consecutive misses but still re-coverable.
      * Closed: >= ``close_miss`` consecutive misses.
    """

    def __init__(self,
                 *,
                 iou_threshold: float = 0.3,
                 confirm_hits: int = 2,
                 lost_miss: int = 3,
                 close_miss: int = 8,
                 require_label_match: bool = True):
        self.iou_threshold = iou_threshold
        self.confirm_hits = confirm_hits
        self.lost_miss = lost_miss
        self.close_miss = close_miss
        self.require_label_match = require_label_match
        self.tracks: list[TrackState] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> None:
        # Group detections by frame index so each frame is one pass.
        by_frame: dict[int, list[tuple[int, Detection]]] = {}
        for idx, det in enumerate(detections):
            by_frame.setdefault(det.frame_idx, []).append((idx, det))

        for _frame_idx in sorted(by_frame):
            frame_dets = by_frame[_frame_idx]
            self._step(frame_dets)

    def _step(self, frame_dets: list[tuple[int, "Detection"]]) -> None:
        used_track = [False] * len(self.tracks)
        used_det: set[int] = set()

        # Greedy match: descending IoU.
        candidates: list[tuple[float, int, int, int]] = []
        for det_idx, det in frame_dets:
            for ti, t in enumerate(self.tracks):
                if t.status == "closed":
                    continue
                if self.require_label_match and \
                        not _label_compatible(t.label, det.label):
                    continue
                score = iou(t.bbox_xyxy, det.bbox_xyxy)
                if score >= self.iou_threshold:
                    candidates.append((score, ti, det_idx,
                                       frame_dets.index((det_idx, det))))
        candidates.sort(reverse=True)
        for score, ti, det_idx, fi in candidates:
            if used_track[ti] or det_idx in used_det:
                continue
            t = self.tracks[ti]
            det = frame_dets[fi][1]
            t.bbox_xyxy = det.bbox_xyxy
            t.last_seen_ts = det.ts
            t.detection_indices.append(det_idx)
            t.hits += 1
            t.misses = 0
            t.confidence = max(t.confidence, det.score)
            if t.hits >= self.confirm_hits and t.status == "tentative":
                t.status = "confirmed"
            used_track[ti] = True
            used_det.add(det_idx)

        # Unmatched detections -> new tentative tracks.
        for det_idx, det in frame_dets:
            if det_idx in used_det:
                continue
            new = TrackState(
                track_id=f"track_{self._next_id:04d}",
                label=det.label,
                bbox_xyxy=list(det.bbox_xyxy),
                first_seen_ts=det.ts,
                last_seen_ts=det.ts,
                detection_indices=[det_idx],
                hits=1,
                misses=0,
                status="tentative",
                confidence=det.score,
            )
            self._next_id += 1
            self.tracks.append(new)
            used_track.append(True)

        # Unmatched tracks -> miss
        for ti, used in enumerate(used_track[:len(self.tracks)]):
            if used:
                continue
            t = self.tracks[ti]
            if t.status == "closed":
                continue
            t.misses += 1
            if t.misses >= self.close_miss:
                t.status = "closed"
            elif t.misses >= self.lost_miss:
                t.status = "lost"

    def export(self) -> list[Track]:
        return [
            Track(
                track_id=t.track_id,
                label=t.label,
                first_seen_ts=t.first_seen_ts,
                last_seen_ts=t.last_seen_ts,
                detections=list(t.detection_indices),
                confidence=t.confidence,
                events=[t.status],
            )
            for t in self.tracks
            if t.status != "closed" or t.hits >= self.confirm_hits
        ]
