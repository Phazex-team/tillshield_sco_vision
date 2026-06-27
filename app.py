"""FastAPI dashboard + embedded SCO Vision pipeline (multi-camera).

v3 changes from v2:
  * Multi-camera. Every endpoint that takes / returns frames or stats
    accepts ``?camera=<id>`` (defaults to the first configured camera if
    omitted, for back-compat).
  * Per-camera classifier (see ``classifiers.py``). Default port is 3902.
  * Gemma is no longer in-process — it runs on a vLLM server (config:
    ``models.gemma.vllm_url``).
  * No hardcoded absolute paths; ``FRAUD_CONFIG`` env var picks the
    config file (defaults to ``./config.yaml`` resolved relative to this
    script).
  * Prompts default-source moved into ``classifiers.py`` (single source
    of truth). ``/prompts`` GET returns the EFFECTIVE prompts for the
    selected camera, with the classifier defaults as the fallback.

Run:
  python app.py
Open: http://localhost:3902
"""
from __future__ import annotations

import csv
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import tracer
from classifiers import (CLASSIFIERS, get_classifier, list_classifiers,
                         resolve_prompts)
from frame_broker import BrokerRegistry
from monitor import (CameraWorker, InferenceWorker, SessionDispatcher,
                     build_analyzer, cameras_by_id, load_config)
from overlay import OverlaySession, render_overlay
from rtsp_reader import RTSPReader
from session_logger import RetentionJanitor, SessionLogger
from zone_trigger import Zone

log = logging.getLogger("app")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("FRAUD_CONFIG",
                                  str(PROJECT_ROOT / "config.yaml")))
STATIC_DIR = PROJECT_ROOT / "static"


def _resolve_dir(rel_or_abs: str) -> str:
    """Resolve a possibly-relative path against the project root so a v3
    process started from any cwd writes into the same place."""
    p = Path(rel_or_abs)
    return str(p if p.is_absolute() else (PROJECT_ROOT / p).resolve())


class AppState:
    def __init__(self):
        self.cfg: dict = load_config(str(CONFIG_PATH))
        # Normalize storage paths to absolute under project root.
        s = self.cfg.setdefault("settings", {})
        for k in ("log_dir", "video_dir", "snapshot_dir"):
            if s.get(k):
                s[k] = _resolve_dir(s[k])
        self.stop_evt = threading.Event()
        self.brokers = BrokerRegistry(overlay_ttl_sec=8.0)
        self.slogger: Optional[SessionLogger] = None
        self.workers: list[CameraWorker] = []
        self.inference: Optional[InferenceWorker] = None
        self.dispatcher: Optional[SessionDispatcher] = None
        self.janitor: Optional[RetentionJanitor] = None
        self.pipeline_ready = threading.Event()
        self.pipeline_error: Optional[str] = None

    # ---- helpers -----------------------------------------------------

    def cameras(self) -> list[dict]:
        return list(self.cfg.get("cameras") or [])

    def cam(self, camera_id: Optional[str]) -> dict:
        cams = self.cameras()
        if not cams:
            raise HTTPException(404, "no cameras configured")
        if not camera_id:
            return cams[0]
        for c in cams:
            if (c.get("id") or c.get("name")) == camera_id:
                return c
        raise HTTPException(404, f"unknown camera id: {camera_id}")

    def cam_id(self, camera_id: Optional[str]) -> str:
        return self.cam(camera_id).get("id") or self.cam(camera_id).get("name") or ""

    # ---- lifecycle ---------------------------------------------------

    def start_pipeline(self):
        def runner():
            try:
                tracer.init(self.cfg)
                s = self.cfg["settings"]
                self.slogger = SessionLogger(
                    log_dir=s["log_dir"],
                    video_dir=s.get("video_dir"),
                    snapshot_dir=s.get("snapshot_dir"),
                )
                analyze, probe = build_analyzer(self.cfg, self.slogger,
                                                broker_registry=self.brokers)
                self.inference = InferenceWorker(
                    analyze, self.stop_evt,
                    max_queue=int(s.get("inference_queue_max", 10)),
                )
                self.inference.start()

                self.dispatcher = SessionDispatcher(
                    inference_enqueue_fn=self.inference.enqueue,
                    probe_fn=probe,
                    merge_window_sec=float(s.get("session_merge_window_sec", 45)),
                    similarity_threshold=float(s.get("session_merge_similarity_threshold", 0.6)),
                    stop_evt=self.stop_evt,
                )
                self.dispatcher.start()

                self.janitor = RetentionJanitor(
                    video_dir=self.slogger.video_dir,
                    snapshot_dir=self.slogger.snapshots_dir,
                    log_dir=self.slogger.dir,
                    retention_days=int(s.get("retention_days", 7)),
                    interval_hours=float(s.get("retention_check_interval_hours", 6)),
                    stop_evt=self.stop_evt,
                )
                self.janitor.start()

                def on_frame(camera_id: str, bgr: np.ndarray):
                    self.brokers.get(camera_id).set_raw(bgr)

                self.workers = [
                    CameraWorker(c, self.cfg, self.dispatcher.submit,
                                 self.stop_evt, on_frame=on_frame)
                    for c in self.cameras()
                ]
                for w in self.workers:
                    w.start()
                self.pipeline_ready.set()
                log.info("pipeline ready (v3) — %d camera(s)", len(self.workers))
            except Exception as e:
                self.pipeline_error = str(e)
                log.exception("pipeline startup failed: %s", e)
                self.pipeline_ready.set()

        threading.Thread(target=runner, daemon=True, name="pipeline-boot").start()

    def stop_pipeline(self):
        self.stop_evt.set()
        for w in self.workers:
            w.join(timeout=5)
        if self.dispatcher is not None:
            self.dispatcher.join(timeout=30)
        if self.inference is not None:
            self.inference.join(timeout=30)
        if self.slogger is not None:
            try:
                report = self.slogger.write_daily_report()
                log.info("\n%s", report)
            finally:
                self.slogger.close()


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.start_pipeline()
    yield
    log.info("shutdown: stopping pipeline")
    state.stop_pipeline()


app = FastAPI(lifespan=lifespan, title="SCO Vision Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- frame rendering ----------

def render_live_frame(camera_id: str) -> Optional[bytes]:
    broker = state.brokers.get(camera_id)
    raw, bboxes, last, _bbox_age, raw_age = broker.snapshot()
    if raw is None:
        return None
    frame = raw
    cam_cfg = state.cam(camera_id)
    zones = cam_cfg.get("zones") or {}
    session = None
    if last is not None:
        session = OverlaySession(
            session_id=last.session_id,
            timestamp=last.timestamp,
            item_presented=last.item_presented,
            confidence=last.confidence,
            item_count=getattr(last, "item_count", None),
        )
    render_overlay(
        frame,
        staff_zone=Zone.from_cfg(zones["staff_zone"]) if "staff_zone" in zones else None,
        customer_zone=Zone.from_cfg(zones["customer_zone"]) if "customer_zone" in zones else None,
        bboxes=bboxes, session=session, raw_age=raw_age,
    )
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        return None
    return buf.tobytes()


def mjpeg_generator(camera_id: str):
    boundary = b"--frame"
    while not state.stop_evt.is_set():
        jpeg = render_live_frame(camera_id)
        if jpeg is None:
            time.sleep(0.1)
            continue
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 12)


# ---------- endpoints ----------

@app.get("/", response_class=HTMLResponse)
def root():
    idx = STATIC_DIR / "index.html"
    return HTMLResponse(idx.read_text())


@app.get("/video_feed")
def video_feed(camera: Optional[str] = Query(default=None)):
    cam_id = state.cam_id(camera)
    return StreamingResponse(
        mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
def snapshot(camera: Optional[str] = Query(default=None)):
    """Single JPEG — used by the config tab's zone-drawing canvas."""
    cam = state.cam(camera)
    cam_id = cam.get("id") or cam.get("name") or ""
    broker = state.brokers.get(cam_id)
    raw, _b, _l, _ba, _ra = broker.snapshot()
    if raw is None:
        try:
            r = RTSPReader(cam["rtsp_url"], "snapshot", 3)
            frame = r.read_bgr()
            r.close()
            if frame is None:
                raise HTTPException(503, "no frame available yet")
            raw = frame
        except Exception as e:
            raise HTTPException(503, f"snapshot failed: {e}")
    ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "jpeg encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/cameras")
def get_cameras():
    out = []
    for c in state.cameras():
        cid = c.get("id") or c.get("name") or ""
        cls = (c.get("classifier") or "fraud").lower()
        meta = get_classifier(cls)
        resolved = resolve_prompts(c)
        out.append({
            "id": cid,
            "name": c.get("name", cid),
            "rtsp_url": c.get("rtsp_url", ""),
            "classifier": cls,
            "scenario_label": meta["display_label"],
            "color": meta["color"],
            "token_budget": c.get("token_budget") or meta["token_budget"],
            "cooldown_sec": c.get("cooldown_sec", 0),
            "has_zones": bool(c.get("zones")),
            # Effective values (camera override > classifier default).
            "enable_thinking": resolved["enable_thinking"],
            "max_frames": resolved["max_frames"],
            # Raw per-camera overrides (None when no override is set).
            "enable_thinking_override": c.get("enable_thinking")
                if isinstance(c.get("enable_thinking"), bool) else None,
            "max_frames_override": c.get("max_frames")
                if isinstance(c.get("max_frames"), int) else None,
        })
    return {"cameras": out}


@app.get("/classifiers")
def get_classifiers():
    return {"classifiers": list_classifiers()}


@app.get("/config")
def get_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---- per-camera config update --------------------------------------

class ZoneBody(BaseModel):
    x: int
    y: int
    w: int
    h: int


class CameraUpdate(BaseModel):
    rtsp_url: Optional[str] = None
    classifier: Optional[str] = None
    token_budget: Optional[int] = None
    cooldown_sec: Optional[int] = None
    enable_thinking: Optional[bool] = None  # null/missing -> use classifier default
    max_frames: Optional[int] = None        # null/missing -> use classifier default
    name: Optional[str] = None
    staff_zone: Optional[ZoneBody] = None
    customer_zone: Optional[ZoneBody] = None


def _persist_config():
    CONFIG_PATH.write_text(yaml.safe_dump(state.cfg, sort_keys=False))


@app.post("/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate):
    cam = state.cam(camera_id)
    if body.rtsp_url is not None:
        cam["rtsp_url"] = body.rtsp_url
    if body.name is not None:
        cam["name"] = body.name
    if body.classifier is not None:
        if body.classifier not in CLASSIFIERS:
            raise HTTPException(400, f"unknown classifier: {body.classifier}")
        cam["classifier"] = body.classifier
        # Reset prompt overrides when classifier changes so the new
        # classifier defaults take effect (operator can re-customise).
        cam["prompts"] = {"falcon": "", "gemma_system": "", "gemma_user": ""}
    if body.token_budget is not None:
        cam["token_budget"] = body.token_budget
    if body.cooldown_sec is not None:
        cam["cooldown_sec"] = body.cooldown_sec
    if "enable_thinking" in body.model_fields_set:
        if body.enable_thinking is None:
            cam.pop("enable_thinking", None)
        else:
            cam["enable_thinking"] = bool(body.enable_thinking)
    if "max_frames" in body.model_fields_set:
        if body.max_frames is None:
            cam.pop("max_frames", None)
        else:
            cam["max_frames"] = max(1, int(body.max_frames))
    if body.staff_zone is not None or body.customer_zone is not None:
        zones = cam.setdefault("zones", {})
        if body.staff_zone is not None:
            zones["staff_zone"] = body.staff_zone.model_dump()
        if body.customer_zone is not None:
            zones["customer_zone"] = body.customer_zone.model_dump()
    _persist_config()
    return {"ok": True, "note": "saved (RTSP / structural changes need a restart)"}


# ---- prompts: per-camera, classifier-aware -------------------------

class PromptsBody(BaseModel):
    falcon: str
    gemma_system: str
    gemma_user: str


@app.get("/prompts")
def get_prompts(camera: Optional[str] = Query(default=None)):
    cam = state.cam(camera)
    cls = (cam.get("classifier") or "fraud").lower()
    meta = get_classifier(cls)
    overrides = cam.get("prompts") or {}
    defaults = {
        "falcon": meta["falcon_prompt"],
        "gemma_system": meta["gemma_system"],
        "gemma_user": meta["gemma_user"],
    }
    effective = {
        k: (overrides.get(k) if (overrides.get(k) or "").strip() else defaults[k])
        for k in defaults
    }
    return {
        "camera_id": cam.get("id") or cam.get("name"),
        "classifier": cls,
        "scenario_label": meta["display_label"],
        "color": meta["color"],
        "prompts": effective,
        "overrides": overrides,
        "defaults": defaults,
        "is_default": {k: (overrides.get(k, "") or "") == "" for k in defaults},
    }


@app.post("/prompts")
def post_prompts(body: PromptsBody, camera: Optional[str] = Query(default=None)):
    cam = state.cam(camera)
    cam["prompts"] = {
        "falcon": body.falcon,
        "gemma_system": body.gemma_system,
        "gemma_user": body.gemma_user,
    }
    _persist_config()
    return {"ok": True, "note": "prompts saved — applied on the next session"}


@app.post("/test_connection")
def test_connection(body: dict):
    cam_id = body.get("camera")
    url = body.get("rtsp_url") or state.cam(cam_id)["rtsp_url"]
    r = RTSPReader(url, "test", 3)
    t0 = time.time()
    frame = r.read_bgr()
    dur = time.time() - t0
    r.close()
    if frame is None:
        return {"ok": False, "ms": int(dur * 1000)}
    h, w = frame.shape[:2]
    return {"ok": True, "ms": int(dur * 1000), "resolution": f"{w}x{h}"}


@app.get("/stats")
def stats(camera: Optional[str] = Query(default=None)):
    if state.slogger is None:
        return {
            "version": "v3", "total": 0, "presented": 0, "none": 0,
            "flagged": 0, "low_conf": 0,
            "pipeline_ready": state.pipeline_ready.is_set(),
            "pipeline_error": state.pipeline_error,
        }
    c = state.slogger.counts
    by_cls = {
        k: {"total": v.total, "events": v.events, "flagged": v.flagged}
        for k, v in state.slogger.by_classifier.items()
    }
    base = {
        "version": "v3",
        "total": c.total,
        "presented": c.handovers,  # alias for UI continuity
        "handovers": c.handovers,
        "none": c.none,
        "flagged": c.flagged,
        "low_conf": c.low_conf,
        "pipeline_ready": state.pipeline_ready.is_set(),
        "pipeline_error": state.pipeline_error,
        "date": state.slogger.date,
        "by_classifier": by_cls,
    }
    return base


def _today_csv_path() -> Path:
    if state.slogger is not None:
        return state.slogger.csv_path
    return Path(state.cfg["settings"]["log_dir"]) / f"sessions_{datetime.now():%Y-%m-%d}.csv"


@app.get("/logs")
def logs(camera: Optional[str] = Query(default=None),
         classifier: Optional[str] = Query(default=None),
         limit: Optional[int] = Query(default=None)):
    p = _today_csv_path()
    if not p.exists():
        return {"rows": [], "path": str(p)}
    rows = []
    cam_filter = camera.strip() if camera else None
    cls_filter = classifier.strip().lower() if classifier else None
    with p.open("r", newline="") as f:
        for row in csv.DictReader(f):
            if cam_filter and (row.get("camera_id") or row.get("camera")) != cam_filter:
                continue
            if cls_filter and (row.get("classifier") or "").lower() != cls_filter:
                continue
            row["flag_for_review"] = str(row.get("flag_for_review", "")).lower() == "true"
            for key in ("handover_occurred", "item_presented"):
                val = str(row.get(key, "")).lower()
                row[key] = True if val == "true" else False if val == "false" else None
            rows.append(row)
    if limit and limit > 0:
        rows = rows[-limit:]
    return {"rows": rows, "path": str(p)}


@app.get("/snapshot/{session_id}")
def snapshot_by_session(session_id: str):
    if "/" in session_id or ".." in session_id:
        raise HTTPException(400, "bad session id")
    snaps = Path(state.slogger.snapshots_dir) if state.slogger else \
        Path(state.cfg["settings"].get("snapshot_dir",
             Path(state.cfg["settings"]["log_dir"]) / "snapshots"))
    matches = sorted(snaps.glob(f"{session_id}_*.png")) + \
              sorted(snaps.glob(f"{session_id}_*.jpg"))
    if not matches:
        raise HTTPException(404, "no snapshot for that session")
    p = matches[-1]
    mime = "image/png" if p.suffix == ".png" else "image/jpeg"
    return Response(content=p.read_bytes(), media_type=mime)


@app.get("/videos/{session_id}")
def video_by_session(session_id: str):
    if "/" in session_id or ".." in session_id:
        raise HTTPException(400, "bad session id")
    vdir = Path(state.slogger.video_dir) if state.slogger else \
        Path(state.cfg["settings"].get("video_dir",
             Path(state.cfg["settings"]["log_dir"]) / "videos"))
    matches = sorted(vdir.glob(f"{session_id}_*.mp4"))
    if not matches:
        raise HTTPException(404, "no video for that session")
    p = matches[-1]
    return Response(content=p.read_bytes(), media_type="video/mp4",
                    headers={"Accept-Ranges": "bytes",
                             "Content-Disposition": f'inline; filename="{p.name}"'})


@app.get("/logs.csv")
def logs_csv():
    p = _today_csv_path()
    if not p.exists():
        raise HTTPException(404, "no log file for today")
    return Response(
        content=p.read_bytes(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{p.name}"'},
    )


@app.get("/health")
def health():
    return {"ok": True, "pipeline_ready": state.pipeline_ready.is_set(),
            "pipeline_error": state.pipeline_error,
            "cameras": [c.get("id") for c in state.cameras()]}


# ---------- entry ----------

def main():
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    s = load_config(str(CONFIG_PATH))["settings"]
    host = s.get("http_host", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", s.get("http_port", 3902)))
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower(),
                access_log=False)


if __name__ == "__main__":
    main()
