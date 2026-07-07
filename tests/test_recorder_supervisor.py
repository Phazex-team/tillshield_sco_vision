"""RecorderSupervisor reconcile logic — the hot-apply core.

Pins:
* reconcile computes add / update / remove / unchanged correctly against
  the live worker set, using fake workers (no RTSP, no real threads).
* an rtsp/encode change recreates the worker; a no-op reconcile touches
  nothing; a removed camera's worker is stopped.
* a per-camera start/stop failure is captured, not fatal, and does not
  abort the rest of the reconcile.
* the heartbeat JSON is written atomically with the active cameras + the
  config mtime applied.
* build_recorder_configs skips cameras missing id/rtsp and honours the
  camera-id filter.
* concurrent reconciles keep the internal maps consistent (thread-safe).
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from video.recorder_supervisor import (RecorderSupervisor,  # noqa: E402
                                       build_recorder_configs)
from video.segment_recorder import RecorderConfig  # noqa: E402


class FakeWorker:
    """Stand-in for SegmentRecorder: records start/stop without threads."""
    def __init__(self, cfg, *, fail_start=False, fail_stop=False):
        self.cfg = cfg
        self.started = False
        self.stopped = False
        self._fail_start = fail_start
        self._fail_stop = fail_stop

    def start(self):
        if self._fail_start:
            raise RuntimeError("boom start")
        self.started = True

    def stop(self):
        if self._fail_stop:
            raise RuntimeError("boom stop")
        self.stopped = True


def _cfg(cam_id, rtsp="rtsp://h/s", **kw):
    return RecorderConfig(camera_id=cam_id, storage_root=Path("/tmp"),
                          rtsp_url=rtsp, **kw)


def _sup(tmp_path, created: list | None = None):
    made = created if created is not None else []

    def factory(cfg):
        w = FakeWorker(cfg)
        made.append(w)
        return w
    return RecorderSupervisor(worker_factory=factory,
                              state_path=tmp_path / "state.json"), made


def test_reconcile_adds_new_cameras(tmp_path):
    sup, made = _sup(tmp_path)
    r = sup.reconcile([_cfg("a"), _cfg("b")])
    assert r.added == ["a", "b"]
    assert r.removed == [] and r.updated == [] and r.unchanged == []
    assert sup.active_camera_ids() == ["a", "b"]
    assert all(w.started for w in made)


def test_reconcile_unchanged_touches_nothing(tmp_path):
    sup, made = _sup(tmp_path)
    sup.reconcile([_cfg("a")])
    n_after_first = len(made)
    r = sup.reconcile([_cfg("a")])
    assert r.unchanged == ["a"]
    assert r.added == [] and r.updated == [] and r.removed == []
    # No new worker created, old one not stopped.
    assert len(made) == n_after_first
    assert made[0].started and not made[0].stopped


def test_reconcile_removes_dropped_camera(tmp_path):
    sup, made = _sup(tmp_path)
    sup.reconcile([_cfg("a"), _cfg("b")])
    r = sup.reconcile([_cfg("a")])
    assert r.removed == ["b"] and r.unchanged == ["a"]
    assert sup.active_camera_ids() == ["a"]
    b_worker = next(w for w in made if w.cfg.camera_id == "b")
    assert b_worker.stopped


def test_reconcile_recreates_on_rtsp_change(tmp_path):
    sup, made = _sup(tmp_path)
    sup.reconcile([_cfg("a", rtsp="rtsp://old")])
    r = sup.reconcile([_cfg("a", rtsp="rtsp://new")])
    assert r.updated == ["a"]
    old = made[0]
    new = made[-1]
    assert old.stopped and new.started
    assert new.cfg.rtsp_url == "rtsp://new"
    assert sup.active_camera_ids() == ["a"]


def test_reconcile_recreates_on_encode_change(tmp_path):
    sup, made = _sup(tmp_path)
    sup.reconcile([_cfg("a", fps=5)])
    r = sup.reconcile([_cfg("a", fps=10)])
    assert r.updated == ["a"]


def test_reconcile_start_failure_is_captured_not_fatal(tmp_path):
    def factory(cfg):
        return FakeWorker(cfg, fail_start=(cfg.camera_id == "bad"))
    sup = RecorderSupervisor(worker_factory=factory,
                             state_path=tmp_path / "s.json")
    r = sup.reconcile([_cfg("good"), _cfg("bad")])
    assert r.added == ["good"]
    assert "bad" in r.failed
    # The good camera still started despite the bad one failing.
    assert sup.active_camera_ids() == ["good"]


def test_heartbeat_written_atomically_with_state(tmp_path):
    sup, _ = _sup(tmp_path)
    sup.reconcile([_cfg("a"), _cfg("b")], config_mtime=123.5)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["active_cameras"] == ["a", "b"]
    assert state["config_mtime"] == 123.5
    assert state["last_reconcile"]["added"] == ["a", "b"]
    assert "updated_at" in state
    # No leftover temp file.
    assert not list(tmp_path.glob("*.tmp"))


def test_stop_all_stops_every_worker(tmp_path):
    sup, made = _sup(tmp_path)
    sup.reconcile([_cfg("a"), _cfg("b")])
    sup.stop_all()
    assert all(w.stopped for w in made)
    assert sup.active_camera_ids() == []


def test_reconcile_skips_camera_without_rtsp(tmp_path):
    sup, _ = _sup(tmp_path)
    r = sup.reconcile([_cfg("a"), RecorderConfig(camera_id="b",
                                                 storage_root=Path("/tmp"),
                                                 rtsp_url=None)])
    assert r.added == ["a"]
    assert sup.active_camera_ids() == ["a"]


def test_concurrent_reconciles_keep_maps_consistent(tmp_path):
    sup, _ = _sup(tmp_path)
    barrier = threading.Barrier(4)

    def worker(ids):
        barrier.wait()
        for _ in range(20):
            sup.reconcile([_cfg(i) for i in ids])

    threads = [threading.Thread(target=worker, args=(ids,)) for ids in
               (["a", "b"], ["a", "c"], ["a", "b", "c"], ["a"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # Whatever the final desired set won, the internal worker/spec maps
    # must agree (no half-updated state) and include the always-present "a".
    active = sup.active_camera_ids()
    assert set(active) == set(sup._specs)  # maps consistent
    assert "a" in active


# --- build_recorder_configs -------------------------------------------------

class _FakeCfg:
    def __init__(self, cameras):
        self.cameras = cameras
        self.storage_root = Path("/tmp/storage")
        self.settings = {"gemma_video_fps_source": 5,
                         "mp4_evidence_width": 640, "mp4_evidence_height": 360}


def test_build_recorder_configs_skips_incomplete_and_filters():
    cfg = _FakeCfg([
        {"id": "a", "rtsp_url": "rtsp://a"},
        {"id": "b", "rtsp_url": ""},          # no rtsp -> skipped
        {"id": None, "rtsp_url": "rtsp://x"},  # no id -> skipped
        {"id": "c", "rtsp_url": "rtsp://c"},
    ])
    all_cams = build_recorder_configs(cfg, 60)
    assert sorted(c.camera_id for c in all_cams) == ["a", "c"]
    assert all_cams[0].fps == 5 and all_cams[0].segment_duration_sec == 60

    only_c = build_recorder_configs(cfg, 60, wanted={"c"})
    assert [c.camera_id for c in only_c] == ["c"]
