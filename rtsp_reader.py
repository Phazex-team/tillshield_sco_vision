"""RTSP capture with auto-reconnect. Yields PIL Images."""
from __future__ import annotations

import logging
import time
from typing import Iterator, Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class RTSPReader:
    def __init__(self, url: str, name: str, reconnect_sec: int = 5):
        self.url = url
        self.name = name
        self.reconnect_sec = reconnect_sec
        self.cap: Optional[cv2.VideoCapture] = None

    def _open(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return self.cap.isOpened()

    def read_bgr(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            if not self._open():
                return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def frames(self, stop_evt) -> Iterator[np.ndarray]:
        """Yield BGR frames; reconnect on failure.

        A single failed read (brief jitter / one dropped packet) must NOT
        cost a fixed multi-second blackout — a 10s sleep permanently loses
        ~50 frames from the current 60s segment (the root of the big
        recorder gaps). So we reopen IMMEDIATELY on the first failures and
        only fall back to a short, capped backoff if it keeps failing.
        """
        disconnected_since: Optional[float] = None
        fails = 0
        while not stop_evt.is_set():
            frame = self.read_bgr()
            if frame is None:
                fails += 1
                if disconnected_since is None:
                    disconnected_since = time.time()
                    log.warning("[%s] RTSP read failed; reconnecting", self.name)
                # Immediate reopen for the first couple of failures; escalate
                # to a short capped backoff only on a sustained outage.
                if fails >= 3:
                    time.sleep(min(self.reconnect_sec, 0.5 * fails))
                self._open()
                continue
            if disconnected_since is not None:
                gap = time.time() - disconnected_since
                log.warning("[%s] RTSP recovered after %.1fs", self.name, gap)
                disconnected_since = None
                fails = 0
            yield frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
