"""Wire-light dataclasses for perception output (PRODUCTION_SPEC §10).

These objects are passed between sampling / falcon_client / sam2_client
/ tracker / temporal_memory / ocr / keyframes and consumed by
``perception.pipeline.run_perception``. The pipeline produces a single
``PerceptionResult`` dict that ``app.case_runner.analyze_case`` feeds
into the reasoning chain + decision policy.

Everything here is plain dataclasses so test code can build mock
results without touching real models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class FrameMeta:
    frame_id: str
    frame_idx: int
    ts: datetime


@dataclass
class Detection:
    label: str
    score: float
    bbox_xyxy: list[float]   # [x1, y1, x2, y2]
    frame_id: str
    frame_idx: int
    ts: datetime
    query: Optional[str] = None


@dataclass
class Mask:
    detection_idx: int  # index back into the detections list
    mask_uri: Optional[str] = None  # path on disk if persisted
    score: float = 0.0
    bbox_xyxy: Optional[list[float]] = None


@dataclass
class Track:
    track_id: str
    label: str
    first_seen_ts: datetime
    last_seen_ts: datetime
    detections: list[int] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    physical_item_candidate: bool = False
    receipt_candidate: bool = False
    confidence: float = 0.0


@dataclass
class OcrResult:
    frame_id: str
    bbox_xyxy: list[float]
    text: str
    confidence: float
    crop_uri: Optional[str] = None


@dataclass
class Keyframe:
    frame_id: str
    frame_idx: int
    ts: datetime
    role: str  # "first_appearance", "counter_placement", "handover",
                # "receipt", "occlusion", "final_state", ...
    uri: Optional[str] = None
    track_id: Optional[str] = None
