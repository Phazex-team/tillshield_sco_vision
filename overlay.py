"""Shared OpenCV overlay renderer used by both the live MJPEG feed and the
snapshot saved to disk per session, so both show the same annotations:
staff/customer zones, Falcon bounding boxes with labels, and a
session/timestamp/item-count banner in the top-left.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import cv2
import numpy as np

from zone_trigger import Zone


STAFF_COLOR_BGR = (0, 0, 255)       # red
CUSTOMER_COLOR_BGR = (0, 255, 0)    # green
BBOX_COLOR_BGR = (0, 255, 255)      # yellow

_FONT = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class OverlayBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: Optional[float] = None  # Falcon does not emit per-box scores;
                                        # populated only if a future backend does


@dataclass
class OverlaySession:
    session_id: str = ""
    timestamp: Optional[float] = None
    item_presented: Optional[bool] = None
    confidence: str = ""
    item_count: Optional[int] = None


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def draw_zones(frame: np.ndarray, *, staff: Optional[Zone] = None,
               customer: Optional[Zone] = None) -> None:
    if staff is not None:
        cv2.rectangle(frame, (staff.x, staff.y),
                      (staff.x + staff.w, staff.y + staff.h),
                      STAFF_COLOR_BGR, 2)
        cv2.putText(frame, "staff (ignored)", (staff.x + 4, staff.y + 20),
                    _FONT, 0.6, STAFF_COLOR_BGR, 2)
    if customer is not None:
        cv2.rectangle(frame, (customer.x, customer.y),
                      (customer.x + customer.w, customer.y + customer.h),
                      CUSTOMER_COLOR_BGR, 2)
        cv2.putText(frame, "customer (watched)",
                    (customer.x + 4, customer.y + 20),
                    _FONT, 0.6, CUSTOMER_COLOR_BGR, 2)


def draw_bboxes(frame: np.ndarray, bboxes: Iterable) -> None:
    for b in bboxes:
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLOR_BGR, 2)
        label = getattr(b, "label", "") or ""
        conf = getattr(b, "confidence", None)
        text = f"{label} {conf:.2f}" if isinstance(conf, (int, float)) else label
        if text:
            cv2.putText(frame, text, (x1 + 2, max(12, y1 - 6)),
                        _FONT, 0.55, BBOX_COLOR_BGR, 2)


def draw_session_banner(frame: np.ndarray, session: Optional[OverlaySession],
                        raw_age: Optional[float]) -> None:
    if session is None:
        cv2.putText(frame, "last analysed: —", (10, 28),
                    _FONT, 0.6, (180, 180, 180), 1)
    else:
        parts = [session.session_id] if session.session_id else []
        if session.item_presented is not None:
            parts.append(f"item={session.item_presented}")
        if session.item_count is not None:
            parts.append(f"count={session.item_count}")
        if session.confidence:
            parts.append(f"conf={session.confidence}")
        txt = "  ".join(parts) or "session"
        color = (0, 200, 0) if session.item_presented else (0, 165, 255)
        cv2.putText(frame, txt, (10, 28), _FONT, 0.7, color, 2)
        if session.timestamp is not None:
            ts_str = datetime.fromtimestamp(session.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S")
            ago = _fmt_age(time.time() - session.timestamp)
            cv2.putText(frame, f"analysed: {ts_str}  ({ago} ago)",
                        (10, 54), _FONT, 0.55, (220, 220, 220), 1)

    if raw_age is not None:
        cv2.putText(frame, f"frame age {raw_age:.1f}s",
                    (10, frame.shape[0] - 10),
                    _FONT, 0.5, (200, 200, 200), 1)


def render_overlay(
    frame: np.ndarray,
    *,
    staff_zone: Optional[Zone] = None,
    customer_zone: Optional[Zone] = None,
    bboxes: Optional[Iterable] = None,
    session: Optional[OverlaySession] = None,
    raw_age: Optional[float] = None,
    copy: bool = False,
) -> np.ndarray:
    """Draw zones, bboxes, and the session banner on ``frame`` in place
    (or on a copy when ``copy=True``). Returns the drawn-on frame."""
    out = frame.copy() if copy else frame
    draw_zones(out, staff=staff_zone, customer=customer_zone)
    if bboxes is not None:
        draw_bboxes(out, bboxes)
    draw_session_banner(out, session, raw_age)
    return out
