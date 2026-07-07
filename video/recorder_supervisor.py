"""Runtime reconcile for the segment recorder.

The recorder used to build a fixed list of ``SegmentRecorder`` workers at
startup and never look at ``config.yaml`` again, so any camera add / edit /
remove needed a manual container restart. ``RecorderSupervisor`` replaces
that with a reconcile step: given the *desired* set of cameras (derived
from the current ``config.yaml``) it starts workers for new cameras, stops
workers for removed cameras, recreates workers whose RTSP/encode settings
changed, and leaves unchanged workers running untouched.

The launcher (`scripts/run_segment_recorder.py`) watches ``config.yaml``'s
mtime and calls :meth:`RecorderSupervisor.reconcile` whenever it changes,
so admin-API edits hot-apply with no restart. ``config.yaml`` stays the
single source of truth — the supervisor holds no state the file doesn't.

After every reconcile the supervisor writes a small JSON heartbeat
(``run/recorder_state.json``, shared with the app container via the bind
mount) recording the active cameras and the config mtime it last applied.
The admin API reads it to report — factually, cross-process — whether a
just-written change has been picked up by the recorder yet.

All worker map mutations are guarded by a reentrant lock so a reconcile
triggered by a config change can never race another.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .segment_recorder import RecorderConfig, SegmentRecorder


log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_state_path() -> Path:
    """Location of the recorder heartbeat JSON. Shared between the
    recorder (writer) and the app (reader) via the ``./run`` bind mount.
    Override with ``RECORDER_STATE_PATH``."""
    env = os.environ.get("RECORDER_STATE_PATH")
    return Path(env) if env else PROJECT_ROOT / "run" / "recorder_state.json"


def _worker_signature(cfg: RecorderConfig) -> tuple:
    """The RecorderConfig fields that, if changed, require recreating the
    worker (a new capture loop). ``camera_id`` is the identity key, not
    part of the signature. Everything here changes what/how we capture."""
    return (cfg.rtsp_url, cfg.segment_duration_sec, cfg.fps,
            cfg.width, cfg.height, cfg.codec)


def build_recorder_configs(cfg,
                           segment_duration_sec: int,
                           wanted: Optional[set[str]] = None
                           ) -> list[RecorderConfig]:
    """Translate an ``AppConfig`` into the desired ``RecorderConfig`` list.

    Shared by the launcher's initial start and every reconcile so the two
    can never drift. Cameras missing an ``id`` or ``rtsp_url`` are skipped
    (they cannot be recorded); ``wanted`` optionally restricts to a subset
    of camera ids (the launcher's ``--camera-id`` filter).
    """
    storage = cfg.storage_root
    out: list[RecorderConfig] = []
    for cam in (cfg.cameras or []):
        cam_id = cam.get("id")
        rtsp = cam.get("rtsp_url")
        if not cam_id or not rtsp:
            log.warning("recorder: skipping camera %r (missing id/rtsp_url)",
                        cam_id or cam)
            continue
        if wanted is not None and cam_id not in wanted:
            continue
        out.append(RecorderConfig(
            camera_id=cam_id,
            storage_root=storage,
            rtsp_url=rtsp,
            segment_duration_sec=segment_duration_sec,
            fps=int(cfg.settings.get("gemma_video_fps_source", 25)),
            width=int(cfg.settings.get("mp4_evidence_width", 1920)),
            height=int(cfg.settings.get("mp4_evidence_height", 1080)),
        ))
    return out


@dataclass
class ReconcileResult:
    """What a single reconcile changed. Logged + written to the heartbeat
    so every live camera change is auditable."""
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed or self.updated or self.failed)

    def as_dict(self) -> dict:
        return asdict(self)


class RecorderSupervisor:
    """Owns the live set of ``SegmentRecorder`` workers and reconciles it
    against a desired camera list. Thread-safe.

    ``worker_factory`` builds (but does not start) a worker for a
    ``RecorderConfig`` — injected so tests can supply fakes that never
    touch RTSP or spawn threads.
    """

    def __init__(self,
                 *,
                 worker_factory: Callable[[RecorderConfig], SegmentRecorder],
                 state_path: Optional[Path] = None):
        self._factory = worker_factory
        self._state_path = Path(state_path) if state_path else None
        self._lock = threading.RLock()
        self._workers: dict[str, SegmentRecorder] = {}
        self._specs: dict[str, tuple] = {}

    # ------------------------------------------------------------------
    # Reconcile
    # ------------------------------------------------------------------

    def reconcile(self,
                  desired: list[RecorderConfig],
                  *,
                  config_mtime: Optional[float] = None) -> ReconcileResult:
        """Drive the live worker set toward ``desired``.

        * camera present in desired but not running  -> start (added)
        * camera running but not in desired          -> stop  (removed)
        * camera in both, signature changed          -> restart (updated)
        * camera in both, signature identical        -> leave  (unchanged)

        A failure on any single camera is captured in ``result.failed``
        and never aborts the rest of the reconcile.
        """
        with self._lock:
            desired_by_id: dict[str, RecorderConfig] = {}
            for c in desired:
                if not c.camera_id or not c.rtsp_url:
                    continue
                desired_by_id[c.camera_id] = c

            result = ReconcileResult()

            # Removals first so a rename (old id gone, new id added) frees
            # the old worker before the new one starts.
            for cam_id in sorted(set(self._workers) - set(desired_by_id)):
                try:
                    self._stop_worker(cam_id)
                    result.removed.append(cam_id)
                except Exception as exc:  # pragma: no cover - defensive
                    log.exception("recorder: failed to stop %s", cam_id)
                    result.failed[cam_id] = f"stop failed: {exc}"

            for cam_id in sorted(desired_by_id):
                cfg = desired_by_id[cam_id]
                sig = _worker_signature(cfg)
                if cam_id not in self._workers:
                    try:
                        self._start_worker(cam_id, cfg, sig)
                        result.added.append(cam_id)
                    except Exception as exc:
                        log.exception("recorder: failed to start %s", cam_id)
                        result.failed[cam_id] = f"start failed: {exc}"
                elif self._specs.get(cam_id) != sig:
                    try:
                        self._stop_worker(cam_id)
                        self._start_worker(cam_id, cfg, sig)
                        result.updated.append(cam_id)
                    except Exception as exc:
                        log.exception("recorder: failed to update %s", cam_id)
                        result.failed[cam_id] = f"update failed: {exc}"
                else:
                    result.unchanged.append(cam_id)

            self._write_state(config_mtime, result)
            return result

    # ------------------------------------------------------------------
    # Worker map (must hold the lock)
    # ------------------------------------------------------------------

    def _start_worker(self, cam_id: str, cfg: RecorderConfig,
                      sig: tuple) -> None:
        worker = self._factory(cfg)
        worker.start()
        self._workers[cam_id] = worker
        self._specs[cam_id] = sig

    def _stop_worker(self, cam_id: str) -> None:
        worker = self._workers.pop(cam_id, None)
        self._specs.pop(cam_id, None)
        if worker is not None:
            worker.stop()

    # ------------------------------------------------------------------
    # Introspection / shutdown
    # ------------------------------------------------------------------

    def active_camera_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._workers)

    def stop_all(self) -> None:
        with self._lock:
            for cam_id in sorted(self._workers):
                try:
                    self._stop_worker(cam_id)
                except Exception:  # pragma: no cover - defensive
                    log.exception("recorder: failed to stop %s on shutdown",
                                  cam_id)
            self._write_state(None, ReconcileResult())

    # ------------------------------------------------------------------
    # Heartbeat (atomic write; best-effort — never breaks a reconcile)
    # ------------------------------------------------------------------

    def _write_state(self, config_mtime: Optional[float],
                     result: Optional[ReconcileResult]) -> None:
        if self._state_path is None:
            return
        state = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "config_mtime": config_mtime,
            "active_cameras": sorted(self._workers),
            "last_reconcile": result.as_dict() if result else None,
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(
                self._state_path.suffix + ".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, self._state_path)
        except Exception:  # pragma: no cover - defensive
            log.exception("recorder: failed to write heartbeat %s",
                          self._state_path)
