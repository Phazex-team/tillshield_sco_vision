"""Live zone overlay. Run: python monitor.py --calibrate
Shows each configured camera with staff_zone (red) + customer_zone (green)
overlaid. Press Q to exit.
"""
from __future__ import annotations

import cv2

from rtsp_reader import RTSPReader
from zone_trigger import Zone


def _draw_zone(frame, zone: Zone, color, label):
    cv2.rectangle(frame, (zone.x, zone.y),
                  (zone.x + zone.w, zone.y + zone.h), color, 2)
    cv2.putText(frame, label, (zone.x + 4, zone.y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def calibrate(cfg: dict) -> None:
    staff = Zone.from_cfg(cfg["zones"]["staff_zone"])
    customer = Zone.from_cfg(cfg["zones"]["customer_zone"])
    cams = cfg["cameras"]

    readers = [RTSPReader(c["rtsp_url"], c["name"],
                          cfg["settings"]["rtsp_reconnect_sec"]) for c in cams]

    # Pre-create windows on the main thread (macOS Cocoa backend requires this).
    win_names = {r.name: f"calibrate - {r.name}" for r in readers}
    for name in win_names.values():
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        try:
            cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

    print("calibration: press Q in any window to exit")
    printed_meta: set[str] = set()
    try:
        while True:
            for r in readers:
                frame = r.read_bgr()
                if frame is None:
                    print(f"[{r.name}] no frame yet (waiting for RTSP)...", flush=True)
                    continue

                if r.name not in printed_meta:
                    h, w = frame.shape[:2]
                    ch = 1 if frame.ndim == 2 else frame.shape[2]
                    px = frame[h // 2, w // 2].tolist() if frame.ndim == 3 else int(frame[h // 2, w // 2])
                    print(f"[{r.name}] stream OK  resolution={w}x{h}  channels={ch}  "
                          f"center_pixel_BGR={px}  dtype={frame.dtype}", flush=True)
                    # sample a few pixels across the frame so user can sanity-check content
                    samples = [(0, 0), (w // 2, h // 2), (w - 1, h - 1),
                               (w // 4, h // 4), (3 * w // 4, 3 * h // 4)]
                    print(f"[{r.name}] pixel samples (x,y)->BGR: " +
                          ", ".join(f"({x},{y})->{frame[y, x].tolist()}" for x, y in samples),
                          flush=True)
                    printed_meta.add(r.name)

                _draw_zone(frame, staff, (0, 0, 255), "staff_zone (ignored)")
                _draw_zone(frame, customer, (0, 255, 0), "customer_zone (watched)")

                cv2.imshow(win_names[r.name], frame)
                cv2.waitKey(1)  # force window to render on macOS

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        for r in readers:
            r.close()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
