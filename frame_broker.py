"""Thread-safe shared state for live video + recent detections.

v3: a ``BrokerRegistry`` owns one ``FrameBroker`` per camera id; every
read/write goes through ``registry.get(camera_id)``. The dashboard's MJPEG
generator and snapshot endpoint pick the camera by query param.

Writers (per camera): CameraWorker (raw frames), the analyzer hook
(annotated dets, results).
Readers: MJPEG generator, snapshot endpoint, stats endpoint.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BBoxPx:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str


@dataclass
class LastResult:
    session_id: str
    camera: str
    timestamp: float
    item_presented: Optional[bool]
    confidence: str
    objects_detected: list = field(default_factory=list)
    notes: str = ""
    flag_for_review: bool = False
    item_count: Optional[int] = None
    classifier: str = ""
    scenario_label: str = ""


class FrameBroker:
    """Single-camera live state store."""

    def __init__(self, camera_id: str = "", overlay_ttl_sec: float = 8.0):
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._raw: Optional[np.ndarray] = None
        self._raw_ts: float = 0.0
        self._bboxes: list[BBoxPx] = []
        self._bboxes_ts: float = 0.0
        self._last_result: Optional[LastResult] = None
        self._overlay_ttl = overlay_ttl_sec

    def set_raw(self, bgr: np.ndarray) -> None:
        with self._lock:
            self._raw = bgr
            self._raw_ts = time.time()

    def set_detection(self, bboxes_px, last_result: LastResult) -> None:
        with self._lock:
            self._bboxes = [
                BBoxPx(int(x1), int(y1), int(x2), int(y2), label)
                for (x1, y1, x2, y2, label) in bboxes_px
            ]
            self._bboxes_ts = time.time()
            self._last_result = last_result

    def snapshot(self):
        """Return (raw_bgr_copy, bboxes, last_result, bbox_age_sec, raw_age_sec).
        Bboxes older than overlay_ttl are dropped from the returned list."""
        with self._lock:
            raw = None if self._raw is None else self._raw.copy()
            raw_ts = self._raw_ts
            now = time.time()
            bboxes = list(self._bboxes) if (now - self._bboxes_ts) <= self._overlay_ttl else []
            last = self._last_result
        return raw, bboxes, last, (now - self._bboxes_ts), (now - raw_ts)


class BrokerRegistry:
    """Per-camera registry. ``get(cam_id)`` lazily creates a FrameBroker."""

    def __init__(self, overlay_ttl_sec: float = 8.0):
        self._lock = threading.Lock()
        self._brokers: dict[str, FrameBroker] = {}
        self._overlay_ttl = overlay_ttl_sec

    def get(self, camera_id: str) -> FrameBroker:
        with self._lock:
            br = self._brokers.get(camera_id)
            if br is None:
                br = FrameBroker(camera_id=camera_id,
                                 overlay_ttl_sec=self._overlay_ttl)
                self._brokers[camera_id] = br
            return br

    def has(self, camera_id: str) -> bool:
        with self._lock:
            return camera_id in self._brokers

    def ids(self) -> list[str]:
        with self._lock:
            return list(self._brokers.keys())

    def all(self) -> dict[str, FrameBroker]:
        with self._lock:
            return dict(self._brokers)
