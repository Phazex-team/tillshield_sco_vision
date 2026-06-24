"""Temporal object memory (PRODUCTION_SPEC §10).

Annotates each track with which zone it occupies over time, whether it
became a handover candidate, and which events it accumulated. Pure
functions over Track + Detection lists so it's trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .schemas import Detection, Track


@dataclass
class Zone:
    name: str
    x: int
    y: int
    w: int
    h: int
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    # Optional polygon vertices ([[x, y], ...], source-frame px). When
    # set, ``contains`` uses point-in-polygon; otherwise the rectangle.
    points: Optional[list] = None

    def contains(self, bbox_xyxy: list[float]) -> bool:
        cx = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2])
        cy = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3])
        if isinstance(self.points, list) and len(self.points) >= 3:
            from app.camera_rois import point_in_polygon
            return point_in_polygon(cx, cy, self.points)
        return (self.x <= cx <= self.x + self.w
                and self.y <= cy <= self.y + self.h)


def annotate_tracks(tracks: list[Track],
                    detections: list[Detection],
                    *,
                    zones: list[Zone],
                    physical_item_labels: tuple[str, ...] = (
                        "bag", "shopping bag", "item", "product",
                        "box", "package", "clothing"),
                    receipt_labels: tuple[str, ...] = (
                        "receipt", "document", "paper"),
                    ) -> list[Track]:
    """Return a NEW list of tracks with zones/events/candidate flags
    populated. Detections are indexed by ``detection_indices`` on each
    track."""
    out: list[Track] = []
    for t in tracks:
        zones_visited: list[str] = []
        events: list[str] = []
        for det_idx in t.detections:
            det = detections[det_idx]
            for zone in zones:
                if zone.contains(det.bbox_xyxy):
                    if zone.name not in zones_visited:
                        zones_visited.append(zone.name)
                        events.append(f"entered_{zone.name}")
        if t.events:
            events.extend(t.events)
        physical = any(label in t.label.lower()
                       for label in physical_item_labels)
        receipt = any(label in t.label.lower()
                       for label in receipt_labels)
        handover_candidate = (
            physical and
            any(z in zones_visited for z in
                ("counter_zone", "staff_zone"))
        )
        if handover_candidate:
            events.append("handover_candidate")
        out.append(Track(
            track_id=t.track_id,
            label=t.label,
            first_seen_ts=t.first_seen_ts,
            last_seen_ts=t.last_seen_ts,
            detections=list(t.detections),
            zones=zones_visited,
            events=events,
            physical_item_candidate=physical,
            receipt_candidate=receipt,
            confidence=t.confidence,
        ))
    return out
