"""Launch one segment recorder per configured camera.

Reads ``config.yaml`` for cameras, opens each RTSP stream via
``rtsp_reader.RTSPReader``, and writes immutable segments under
``storage/cctv/camera_id=<id>/...``. Runs until SIGINT/SIGTERM; the
recorder NEVER consults the memory guard — it stays alive when
inference is degraded.
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segment-duration-sec", type=int, default=60)
    args = ap.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("recorder.launcher")

    from app.config import load_config
    from db.session import get_sessionmaker, init_schema
    from video.segment_recorder import RecorderConfig, SegmentRecorder

    init_schema()
    SM = get_sessionmaker()
    cfg = load_config()
    storage = cfg.storage_root

    recorders: list[SegmentRecorder] = []
    for cam in cfg.cameras:
        cam_id = cam.get("id")
        rtsp = cam.get("rtsp_url")
        if not cam_id or not rtsp:
            log.warning("skipping camera %r: missing id/rtsp_url", cam)
            continue
        rec = SegmentRecorder(
            RecorderConfig(
                camera_id=cam_id,
                storage_root=storage,
                rtsp_url=rtsp,
                segment_duration_sec=args.segment_duration_sec,
                fps=int(cfg.settings.get("gemma_video_fps_source", 25)),
                width=int(cfg.settings.get("mp4_evidence_width", 1920)),
                height=int(cfg.settings.get("mp4_evidence_height", 1080)),
            ),
            session_factory=SM,
        )
        recorders.append(rec)
        rec.start()
        log.info("recorder %s: started", cam_id)

    if not recorders:
        log.error("no recorders started; exiting")
        return 2

    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        log.info("got signal %d; stopping recorders", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    while not stop:
        time.sleep(1)
    for rec in recorders:
        rec.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
