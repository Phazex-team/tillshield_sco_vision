"""Zone-restricted motion trigger.

Ignores everything in staff_zone. Fires only when motion inside
customer_zone exceeds thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Zone:
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_cfg(cls, d: dict) -> "Zone":
        return cls(d["x"], d["y"], d["w"], d["h"])

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]


class ZoneTrigger:
    def __init__(self, customer_zone: Zone, staff_zone: Zone,
                 motion_threshold: int = 25, area_ratio: float = 0.02):
        self.customer = customer_zone
        self.staff = staff_zone
        self.motion_threshold = motion_threshold
        self.area_ratio = area_ratio
        self.prev_gray: Optional[np.ndarray] = None

    def check(self, frame_bgr: np.ndarray) -> bool:
        patch = self.customer.crop(frame_bgr)
        if patch.size == 0:
            return False
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray
        _, thr = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        changed = float(np.count_nonzero(thr)) / float(thr.size)
        return changed >= self.area_ratio
