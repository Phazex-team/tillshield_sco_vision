"""Launch and hot-reconcile one segment recorder per configured camera.

Reads ``config.yaml`` for cameras, opens each RTSP stream via
``rtsp_reader.RTSPReader``, and writes immutable segments under
``storage/cctv/camera_id=<id>/...``.

Camera changes hot-apply with NO restart: the launcher watches
``config.yaml``'s mtime and, on any change, re-derives the desired camera
set and reconciles the live workers (start new, stop removed, recreate
changed, leave unchanged) via ``RecorderSupervisor``. It records every
configured camera and NEVER consults the memory guard — it stays alive
when inference is degraded. Runs until SIGINT/SIGTERM.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _config_mtime(path: Path) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--segment-duration-sec", type=int, default=60)
    ap.add_argument("--reconcile-interval-sec", type=float, default=3.0,
                    help="How often to check config.yaml for camera "
                         "changes and hot-apply them. Default 3s.")
    ap.add_argument("--camera-id", action="append", default=None,
                    help="Only record these camera id(s). Repeatable. "
                         "Default: every configured camera.")
    args = ap.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("recorder.launcher")

    from app.config import DEFAULT_CONFIG_PATH, load_config
    from db.session import get_sessionmaker, init_schema
    from video.recorder_supervisor import (RecorderSupervisor,
                                            build_recorder_configs,
                                            default_state_path)
    from video.segment_recorder import SegmentRecorder

    init_schema()
    SM = get_sessionmaker()

    wanted = set(args.camera_id) if args.camera_id else None
    cfg_path = Path(DEFAULT_CONFIG_PATH)

    def _desired():
        return build_recorder_configs(load_config(),
                                      args.segment_duration_sec, wanted)

    supervisor = RecorderSupervisor(
        worker_factory=lambda rc: SegmentRecorder(rc, session_factory=SM),
        state_path=default_state_path(),
    )

    last_mtime = _config_mtime(cfg_path)
    result = supervisor.reconcile(_desired(), config_mtime=last_mtime)
    log.info("recorder initial reconcile: %s", result.as_dict())
    if not supervisor.active_camera_ids():
        # No cameras yet is not fatal — the operator may add one via the
        # admin UI and it will hot-apply. Warn and keep watching.
        log.warning("no cameras configured to record; watching config.yaml "
                    "for additions")

    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        log.info("got signal %d; stopping recorders", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    while not stop:
        time.sleep(args.reconcile_interval_sec)
        mtime = _config_mtime(cfg_path)
        if mtime == last_mtime:
            continue
        last_mtime = mtime
        try:
            result = supervisor.reconcile(_desired(), config_mtime=mtime)
        except Exception:
            log.exception("recorder: reconcile after config change failed")
            continue
        if result.changed:
            log.info("recorder reconcile (config changed): %s",
                     result.as_dict())

    supervisor.stop_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
