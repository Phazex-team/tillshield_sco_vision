"""v3 session recorder + inference pipeline (multi-camera, per-camera config).

Pipeline shape (per camera, fanned out from N camera entries in config.yaml):

    RTSP --> CameraWorker
              | rolling pre-roll (5s)
              | motion in customer_zone -> session
              | end on N seconds idle OR session_max_sec
              v
            SessionDispatcher (one pending slot PER camera; never merges
              clips from different cameras)
              | within merge_window + same camera -> probe + similarity
              v
            InferenceWorker (queue=10, FIFO, drops counted PER camera)
              | analyze(clip):
              |   - resolve per-camera classifier + token_budget + prompts
              |   - Falcon detect(start) + detect(middle) (ROI-cropped)
              |   - Gemma reason(video) via vLLM HTTP
              |   - PNG snapshot (full-res) + MP4 evidence (downscaled)
              v
            CSV log + per-camera FrameBroker update
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Callable, Optional

import cv2
import numpy as np
import yaml

from rtsp_reader import RTSPReader, bgr_to_pil
from zone_trigger import Zone, ZoneTrigger
from session_logger import SessionLogger
from classifiers import resolve_prompts


log = logging.getLogger("monitor")


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def cameras_by_id(cfg: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in cfg.get("cameras", []) or []:
        cid = c.get("id") or c.get("name") or "cam_unknown"
        out[cid] = c
    return out


# ----------------------------------------------------------------------
# Session data structures
# ----------------------------------------------------------------------

_DOWNSCALE_W = 640
_DOWNSCALE_H = 360


@dataclass
class SessionClip:
    camera: str            # legacy, == camera display name
    camera_id: str         # NEW in v3 — short stable id from config
    start_time: datetime
    end_time: datetime
    frames: list[np.ndarray] = field(default_factory=list)
    sample_fps: int = 5
    merged_from: list[str] = field(default_factory=list)
    # Frames in ``frames`` are downscaled to 640x360 to keep memory
    # bounded. ``last_full_res`` is one full-res copy kept for the
    # snapshot PNG. ``native_w`` / ``native_h`` let analyze() scale
    # zones / bboxes between the two coord systems.
    last_full_res: Optional[np.ndarray] = None
    native_w: int = 0
    native_h: int = 0


# ----------------------------------------------------------------------
# Camera worker
# ----------------------------------------------------------------------

class CameraWorker(threading.Thread):
    """One per camera. Owns the RTSPReader, runs the session state machine,
    pushes completed ``SessionClip`` objects into the dispatcher.
    """

    def __init__(self, cam_cfg: dict, cfg: dict,
                 on_session: Callable[[SessionClip], None],
                 stop_evt: threading.Event,
                 on_frame: Optional[Callable[[str, np.ndarray], None]] = None):
        cam_id = cam_cfg.get("id") or cam_cfg.get("name") or "cam_unknown"
        super().__init__(daemon=True, name=f"cam-{cam_id}")
        self.cam_cfg = cam_cfg
        self.cam_id = cam_id
        self.cfg = cfg
        self.on_session = on_session
        self.stop_evt = stop_evt
        self.on_frame = on_frame

    def run(self) -> None:
        s = self.cfg["settings"]
        sample_fps = int(s.get("gemma_video_fps_source", 5))
        sample_interval = 1.0 / max(1, sample_fps)
        pre_roll_sec = float(s.get("pre_roll_sec", 5))
        end_empty_sec = float(s.get("session_end_empty_sec", 5))
        max_session_sec = float(s.get("session_max_sec", 60))
        min_session_sec = float(s.get("min_session_sec", 2))
        cooldown_sec = float(self.cam_cfg.get("cooldown_sec", 0))

        pre_roll_frames = max(1, int(pre_roll_sec * sample_fps))

        zones = self.cam_cfg.get("zones") or self.cfg.get("zones") or {}
        if "customer_zone" not in zones or "staff_zone" not in zones:
            log.error("[%s] no zones configured (need cameras[].zones)",
                      self.cam_id)
            return
        customer_zone = Zone.from_cfg(zones["customer_zone"])
        staff_zone = Zone.from_cfg(zones["staff_zone"])
        trigger = ZoneTrigger(
            customer_zone=customer_zone, staff_zone=staff_zone,
            motion_threshold=s.get("motion_threshold", 25),
            area_ratio=s.get("motion_area_ratio", 0.02),
        )

        reader = RTSPReader(self.cam_cfg["rtsp_url"], self.cam_id,
                            s.get("rtsp_reconnect_sec", 10))

        pre_roll: deque = deque(maxlen=pre_roll_frames)
        in_session = False
        session_frames: list[np.ndarray] = []
        session_start_wall: Optional[datetime] = None
        last_motion_t = 0.0
        last_sample_t = 0.0
        last_session_end_t = 0.0
        native_h = native_w = 0
        last_full_res: Optional[np.ndarray] = None

        log.info("[%s] recorder armed (sample_fps=%d, pre_roll=%.1fs, "
                 "end_empty=%.1fs, max=%.1fs, cooldown=%.1fs)",
                 self.cam_id, sample_fps, pre_roll_sec, end_empty_sec,
                 max_session_sec, cooldown_sec)

        for frame in reader.frames(self.stop_evt):
            now = time.time()

            if self.on_frame is not None:
                try:
                    self.on_frame(self.cam_id, frame)
                except Exception:
                    log.exception("on_frame hook failed")

            if now - last_sample_t < sample_interval:
                continue
            last_sample_t = now

            if native_h == 0:
                native_h, native_w = frame.shape[:2]
            last_full_res = frame
            # cv2.resize returns a new array — no .copy() needed.
            frame_small = cv2.resize(frame, (_DOWNSCALE_W, _DOWNSCALE_H),
                                     interpolation=cv2.INTER_AREA)

            motion = trigger.check(frame)
            if motion:
                last_motion_t = now

            if not in_session:
                pre_roll.append(frame_small)
                if motion:
                    if cooldown_sec > 0 and (now - last_session_end_t) < cooldown_sec:
                        continue
                    session_frames = list(pre_roll)
                    pre_roll.clear()
                    session_start_wall = datetime.now()
                    in_session = True
                    log.info("[%s] session START", self.cam_id)
            else:
                session_frames.append(frame_small)

                elapsed = (datetime.now() - session_start_wall).total_seconds() \
                    if session_start_wall else 0.0
                silent_for = now - last_motion_t
                should_end = (silent_for >= end_empty_sec
                              or elapsed >= max_session_sec)
                if should_end:
                    session_end_wall = datetime.now()
                    duration = (session_end_wall - session_start_wall).total_seconds() \
                        if session_start_wall else 0.0
                    if duration >= min_session_sec:
                        clip = SessionClip(
                            camera=self.cam_cfg.get("name", self.cam_id),
                            camera_id=self.cam_id,
                            start_time=session_start_wall,
                            end_time=session_end_wall,
                            frames=session_frames,
                            sample_fps=sample_fps,
                            last_full_res=(last_full_res.copy()
                                           if last_full_res is not None
                                           else None),
                            native_w=native_w,
                            native_h=native_h,
                        )
                        log.info("[%s] session END (%.1fs, frames=%d, reason=%s)",
                                 self.cam_id, duration, len(session_frames),
                                 "max" if elapsed >= max_session_sec else "empty")
                        try:
                            self.on_session(clip)
                        except Exception:
                            log.exception("on_session dispatch failed")
                    else:
                        log.info("[%s] session discarded (too short: %.1fs)",
                                 self.cam_id, duration)
                    in_session = False
                    session_frames = []
                    session_start_wall = None
                    pre_roll.clear()
                    last_session_end_t = now

        reader.close()


# ----------------------------------------------------------------------
# Inference worker
# ----------------------------------------------------------------------

class InferenceWorker(threading.Thread):
    """Consumes ``SessionClip`` jobs and runs Falcon + Gemma + MP4 encode.

    v3: bigger queue, per-camera drop counter so noisy cameras don't
    obscure quiet ones.
    """

    def __init__(self, analyze_fn: Callable[[SessionClip], None],
                 stop_evt: threading.Event, max_queue: int = 10):
        super().__init__(daemon=True, name="inference")
        self.analyze_fn = analyze_fn
        self.stop_evt = stop_evt
        self.queue: Queue = Queue(maxsize=max_queue)
        self.dropped: dict[str, int] = {}
        self.processed: dict[str, int] = {}

    def enqueue(self, clip: SessionClip) -> None:
        try:
            self.queue.put_nowait(clip)
        except Full:
            self.dropped[clip.camera_id] = self.dropped.get(clip.camera_id, 0) + 1
            log.info("[inference] queue full, dropped clip for %s "
                     "(dropped=%d)", clip.camera_id,
                     self.dropped[clip.camera_id])

    def run(self) -> None:
        while not (self.stop_evt.is_set() and self.queue.empty()):
            try:
                clip = self.queue.get(timeout=0.5)
            except Empty:
                continue
            t0 = time.time()
            try:
                self.analyze_fn(clip)
                self.processed[clip.camera_id] = self.processed.get(clip.camera_id, 0) + 1
                log.info("[inference] %s done in %.1fs "
                         "(processed=%d, dropped=%d)",
                         clip.camera_id, time.time() - t0,
                         self.processed.get(clip.camera_id, 0),
                         self.dropped.get(clip.camera_id, 0))
            except Exception:
                log.exception("[inference] failed")
            finally:
                # Release downscaled frame buffers + the full-res snapshot
                # before the next iteration so memory doesn't accumulate.
                try:
                    clip.frames.clear()
                    clip.last_full_res = None
                except Exception:
                    pass
                gc.collect()


# ----------------------------------------------------------------------
# Analyzer factory
# ----------------------------------------------------------------------

def build_analyzer(cfg: dict, slogger: SessionLogger,
                   broker_registry=None):
    """Return ``(analyze, probe_description)``.

    Heavy models load here so ``--calibrate`` never touches them. Gemma is
    a thin HTTP client to vLLM; Falcon is still in-process and serialised
    by ``falcon_lock``.
    """
    from falcon_detector import FalconDetector, bbox_summary
    from gemma_reasoner import GemmaVideoReasoner
    from video_encoder import encode_evidence_mp4

    falcon = FalconDetector(cfg["models"]["falcon"]["name"])
    gcfg = cfg["models"]["gemma"]
    gemma = GemmaVideoReasoner(
        model_name=gcfg["name"],
        max_tokens=int(gcfg.get("max_tokens", 768)),
        temperature=float(gcfg.get("temperature", 0.1)),
        max_video_frames=int(cfg["settings"].get("gemma_video_max_seconds", 60))
            * int(cfg["settings"].get("gemma_video_fps", 1)),
        video_fps=int(cfg["settings"].get("gemma_video_fps", 1)),
        vllm_url=gcfg.get("vllm_url"),
        request_timeout_sec=float(gcfg.get("request_timeout_sec", 120)),
        request_retries=int(gcfg.get("request_retries", 3)),
        request_retry_backoff_sec=float(gcfg.get("request_retry_backoff_sec", 5)),
    )
    gemma_video_fps = int(cfg["settings"].get("gemma_video_fps", 1))
    gemma_max_seconds = int(cfg["settings"].get("gemma_video_max_seconds", 60))
    global_max_frames = max(1, gemma_max_seconds * max(1, gemma_video_fps))
    mp4_w = int(cfg["settings"].get("mp4_evidence_width", 640))
    mp4_h = int(cfg["settings"].get("mp4_evidence_height", 360))
    mp4_fps = int(cfg["settings"].get("mp4_evidence_fps", 5))
    mp4_crf = int(cfg["settings"].get("mp4_evidence_crf", 28))
    roi_enabled_global = bool(cfg["settings"].get("roi_crop_enabled", True))
    roi_margin = float(cfg["settings"].get("roi_crop_margin_pct", 0.10))

    cams = cameras_by_id(cfg)
    falcon_lock = threading.Lock()

    def _cam_cfg(camera_id: str) -> dict:
        return cams.get(camera_id) or {}

    def _zones_for(camera_id: str):
        cc = _cam_cfg(camera_id)
        return cc.get("zones") or cfg.get("zones") or {}

    def _roi(camera_id: str, frame_bgr, native_wh=None) -> tuple | None:
        if not roi_enabled_global:
            return None
        z = _zones_for(camera_id)
        if "customer_zone" not in z:
            return None
        h, w = frame_bgr.shape[:2]
        cz = Zone.from_cfg(z["customer_zone"])
        # If the frame is at a different size than the camera's native
        # capture (we now downscale to 640x360), scale native zone coords
        # into the current frame's coord space.
        if native_wh and native_wh[0] and native_wh[1]:
            sx = w / float(native_wh[0])
            sy = h / float(native_wh[1])
            cx, cy = int(cz.x * sx), int(cz.y * sy)
            cw, ch = int(cz.w * sx), int(cz.h * sy)
        else:
            cx, cy, cw, ch = cz.x, cz.y, cz.w, cz.h
        mx = int(round(cw * roi_margin))
        my = int(round(ch * roi_margin))
        return (
            max(0, cx - mx),
            max(0, cy - my),
            min(w, cx + cw + mx),
            min(h, cy + ch + my),
        )

    def probe_description(clip_or_frames, max_frames: int = 3) -> str:
        if isinstance(clip_or_frames, SessionClip):
            frames_bgr = clip_or_frames.frames
            camera_id = clip_or_frames.camera_id
            native_wh = (clip_or_frames.native_w, clip_or_frames.native_h)
        else:
            frames_bgr = clip_or_frames
            camera_id = ""
            native_wh = None
        if not frames_bgr:
            return ""
        n = len(frames_bgr)
        if n <= max_frames:
            picks = list(range(n))
        else:
            mid = n // 2
            span = max_frames // 2
            picks = sorted({max(0, mid - span), mid, min(n - 1, mid + span)})
        pils = [bgr_to_pil(frames_bgr[i]) for i in picks]
        roi = _roi(camera_id, frames_bgr[0], native_wh) if camera_id else None
        if roi is not None:
            pils = [p.crop(roi) for p in pils]
        return gemma.quick_describe(pils, max_frames=max_frames)

    def analyze(clip: SessionClip) -> None:
        from frame_broker import BBoxPx, LastResult
        from overlay import OverlaySession, render_overlay

        cc = _cam_cfg(clip.camera_id)
        resolved = resolve_prompts(cc)
        falcon_query = resolved["falcon"]
        sys_prompt = resolved["gemma_system"]
        usr_prompt = resolved["gemma_user"]
        token_budget = resolved["token_budget"]
        classifier_key = resolved["classifier"]
        scenario_label = resolved["display_label"]
        enable_thinking = resolved["enable_thinking"]
        max_frames = resolved["max_frames"] or global_max_frames

        sid = slogger.next_session_id(clip.camera_id)
        t_session = time.time()
        native_wh = (clip.native_w, clip.native_h)
        # Scale factor to translate downscaled (640x360) coords -> native.
        if clip.native_w and clip.native_h:
            up_sx = clip.native_w / float(_DOWNSCALE_W)
            up_sy = clip.native_h / float(_DOWNSCALE_H)
        else:
            up_sx = up_sy = 1.0

        n = len(clip.frames)
        if n == 0:
            log.warning("analyze: empty clip")
            return
        mid_idx = n // 2
        mid_full = clip.frames[mid_idx]
        mid_pil = bgr_to_pil(mid_full)
        frame_h, frame_w = mid_full.shape[:2]

        stride = max(1, int(round(clip.sample_fps / max(1, gemma_video_fps))))
        gemma_bgr = clip.frames[::stride]
        gemma_pil = [bgr_to_pil(f) for f in gemma_bgr]

        roi = _roi(clip.camera_id, mid_full, native_wh)
        if roi is not None:
            mid_pil_for_model = mid_pil.crop(roi)
            gemma_pil_for_model = [p.crop(roi) for p in gemma_pil]
        else:
            mid_pil_for_model = mid_pil
            gemma_pil_for_model = gemma_pil

        start_pil = bgr_to_pil(clip.frames[0])
        if roi is not None:
            start_pil = start_pil.crop(roi)

        with falcon_lock:
            _, start_dets = falcon.detect(start_pil, query=falcon_query)
            annotated_pil, dets = falcon.detect(mid_pil_for_model, query=falcon_query)
        try:
            from tracer import trace_falcon
            trace_falcon(clip.camera_id, sid, "start",  start_dets)
            trace_falcon(clip.camera_id, sid, "action", dets)
        except Exception:
            log.exception("trace_falcon hook failed (non-fatal)")
        start_objects = bbox_summary(start_dets)
        action_objects = bbox_summary(dets)

        log.info("[%s] inference: classifier=%s thinking=%s max_frames=%d "
                 "token_budget=%d", clip.camera_id, classifier_key,
                 enable_thinking, max_frames, token_budget)
        result = gemma.reason(
            gemma_pil_for_model,
            start_objects=start_objects,
            action_objects=action_objects,
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            token_budget=token_budget,
            classifier=classifier_key,
            enable_thinking=enable_thinking,
            max_frames=max_frames,
            camera_id=clip.camera_id,
            session_id=sid,
        )

        if roi is not None:
            ox, oy = roi[0], roi[1]
            for d in dets:
                x1, y1, x2, y2 = d.bbox_px
                d.bbox_px = (x1 + ox, y1 + oy, x2 + ox, y2 + oy)

        if clip.merged_from:
            returns = len(clip.merged_from)
            prefix = (f"[Subject re-engaged {returns} more time"
                      f"{'s' if returns != 1 else ''} during this session] ")
            result["narrative"] = prefix + str(result.get("narrative", ""))

        snapshot_path = ""
        try:
            ts_str = clip.start_time.strftime("%Y-%m-%d_%H-%M-%S")
            snap_out = slogger.snapshots_dir / f"{sid}_{ts_str}.png"
            if clip.last_full_res is not None:
                # Snapshot canvas is full-res; zones are already in native
                # coords; scale Falcon bboxes UP from downscaled to native.
                frame_bgr = clip.last_full_res.copy()
                bboxes = [BBoxPx(int(d.bbox_px[0] * up_sx),
                                 int(d.bbox_px[1] * up_sy),
                                 int(d.bbox_px[2] * up_sx),
                                 int(d.bbox_px[3] * up_sy),
                                 d.label) for d in dets]
            else:
                frame_bgr = cv2.cvtColor(np.array(mid_pil.convert("RGB")),
                                         cv2.COLOR_RGB2BGR)
                bboxes = [BBoxPx(int(d.bbox_px[0]), int(d.bbox_px[1]),
                                 int(d.bbox_px[2]), int(d.bbox_px[3]),
                                 d.label) for d in dets]
            session_overlay = OverlaySession(
                session_id=sid, timestamp=clip.end_time.timestamp(),
                item_presented=bool(result.get("handover_occurred", False)),
                confidence=str(result.get("confidence", "")),
                item_count=int(result.get("item_count") or 0),
            )
            zones = _zones_for(clip.camera_id)
            render_overlay(
                frame_bgr,
                staff_zone=Zone.from_cfg(zones["staff_zone"]) if "staff_zone" in zones else None,
                customer_zone=Zone.from_cfg(zones["customer_zone"]) if "customer_zone" in zones else None,
                bboxes=bboxes, session=session_overlay,
            )
            cv2.imwrite(str(snap_out), frame_bgr)
            snapshot_path = str(snap_out)
        except Exception:
            log.exception("snapshot save failed")

        mp4_path = ""
        try:
            ts_str = clip.start_time.strftime("%Y-%m-%d_%H-%M-%S")
            mp4_out = slogger.video_dir / f"{sid}_{ts_str}.mp4"
            mp4_stride = max(1, int(round(clip.sample_fps / max(1, mp4_fps))))
            mp4_frames = [cv2.resize(f, (mp4_w, mp4_h),
                                     interpolation=cv2.INTER_AREA)
                          for f in clip.frames[::mp4_stride]]
            encode_evidence_mp4(mp4_frames, mp4_out,
                                width=mp4_w, height=mp4_h,
                                fps=mp4_fps, crf=mp4_crf)
            mp4_path = str(mp4_out)
        except Exception:
            log.exception("mp4 encode failed")

        slogger.log(
            camera=clip.camera, camera_id=clip.camera_id,
            session_id=sid,
            classifier=classifier_key, scenario_label=scenario_label,
            start_time=clip.start_time, end_time=clip.end_time,
            result=result, snapshot_path=snapshot_path, mp4_path=mp4_path,
            merged_from=list(clip.merged_from),
        )

        if broker_registry is not None:
            broker = broker_registry.get(clip.camera_id)
            # Live MJPEG draws on full-res raw frames; scale bboxes UP.
            bboxes_px = [(int(d.bbox_px[0] * up_sx),
                          int(d.bbox_px[1] * up_sy),
                          int(d.bbox_px[2] * up_sx),
                          int(d.bbox_px[3] * up_sy),
                          d.label) for d in dets]
            broker.set_detection(
                bboxes_px,
                LastResult(
                    session_id=sid, camera=clip.camera,
                    timestamp=clip.end_time.timestamp(),
                    item_presented=bool(result.get("handover_occurred", False)),
                    confidence=result.get("confidence", "low"),
                    objects_detected=list(result.get("objects_detected", [])),
                    notes=result.get("narrative", "")[:240],
                    flag_for_review=bool(result.get("flag_for_review", False)),
                    item_count=int(result.get("item_count") or 0),
                    classifier=classifier_key,
                    scenario_label=scenario_label,
                ),
            )

        try:
            from tracer import trace_session
            trace_session(
                camera_id=clip.camera_id, session_id=sid,
                classifier=classifier_key,
                duration_sec=(clip.end_time - clip.start_time).total_seconds(),
                num_frames=int(result.get("_num_frames", 0)),
                result=("event" if result.get("handover_occurred") else "no-event"),
                confidence=str(result.get("confidence", "")),
            )
        except Exception:
            log.exception("trace_session hook failed (non-fatal)")
        log.debug("[%s] analyze done in %.1fs",
                  clip.camera_id, time.time() - t_session)

    return analyze, probe_description


# ----------------------------------------------------------------------
# Session dispatcher — per-camera pending slot
# ----------------------------------------------------------------------

class SessionDispatcher(threading.Thread):
    """Per-camera ``pending`` slot. Only merges clips from the SAME
    ``camera_id``. Different cameras flow through independently and never
    interact.
    """

    def __init__(self, *, inference_enqueue_fn: Callable[[SessionClip], None],
                 probe_fn: Callable,
                 merge_window_sec: float,
                 similarity_threshold: float,
                 stop_evt: threading.Event):
        super().__init__(daemon=True, name="session-dispatcher")
        self.inference_enqueue_fn = inference_enqueue_fn
        self.probe_fn = probe_fn
        self.merge_window_sec = float(merge_window_sec)
        self.similarity_threshold = float(similarity_threshold)
        self.stop_evt = stop_evt
        self.queue: Queue = Queue()
        # camera_id -> (pending_clip, pending_desc_or_None)
        self.pending: dict[str, tuple[SessionClip, Optional[str]]] = {}

    def submit(self, clip: SessionClip) -> None:
        self.queue.put(clip)

    def _safe_probe(self, clip: SessionClip) -> str:
        try:
            s = self.probe_fn(clip) or ""
        except Exception:
            log.exception("[merge:%s] probe failed", clip.camera_id)
            return ""
        return s.strip().lower()

    def _finalize(self, camera_id: str) -> None:
        entry = self.pending.pop(camera_id, None)
        if entry is None:
            return
        pending_clip, _ = entry
        log.info("[merge:%s] finalizing (merged_from=%d)",
                 camera_id, len(pending_clip.merged_from))
        self.inference_enqueue_fn(pending_clip)

    def _stale(self, camera_id: str, stale_after_sec: float) -> bool:
        entry = self.pending.get(camera_id)
        if entry is None:
            return False
        elapsed = (datetime.now() - entry[0].end_time).total_seconds()
        return elapsed >= stale_after_sec

    def run(self) -> None:
        from difflib import SequenceMatcher

        stale_after_sec = self.merge_window_sec + 120.0

        while not (self.stop_evt.is_set() and self.queue.empty() and not self.pending):
            try:
                clip = self.queue.get(timeout=1.0)
            except Empty:
                for cid in list(self.pending.keys()):
                    if self._stale(cid, stale_after_sec):
                        self._finalize(cid)
                continue

            cid = clip.camera_id
            entry = self.pending.get(cid)
            if entry is None:
                self.pending[cid] = (clip, None)
                continue

            pending_clip, pending_desc = entry
            gap = (clip.start_time - pending_clip.end_time).total_seconds()
            if gap > self.merge_window_sec or gap < 0:
                self._finalize(cid)
                self.pending[cid] = (clip, None)
                continue

            if pending_desc is None:
                pending_desc = self._safe_probe(pending_clip)
            new_desc = self._safe_probe(clip)
            ratio = SequenceMatcher(None, pending_desc, new_desc).ratio() \
                if (pending_desc or new_desc) else 0.0
            log.info("[merge:%s] gap=%.1fs similarity=%.2f", cid, gap, ratio)

            if ratio >= self.similarity_threshold:
                pending_clip.frames.extend(clip.frames)
                pending_clip.end_time = clip.end_time
                pending_clip.merged_from.append(
                    clip.start_time.isoformat(timespec="seconds")
                )
                if clip.last_full_res is not None:
                    pending_clip.last_full_res = clip.last_full_res
                self.pending[cid] = (pending_clip, pending_desc)
            else:
                self._finalize(cid)
                self.pending[cid] = (clip, new_desc)


# ----------------------------------------------------------------------
# Standalone entry — monitor.py without the dashboard.
# ----------------------------------------------------------------------

def run_monitor(cfg: dict) -> int:
    from session_logger import RetentionJanitor

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = cfg["settings"]
    slogger = SessionLogger(
        log_dir=s["log_dir"],
        video_dir=s.get("video_dir"),
        snapshot_dir=s.get("snapshot_dir"),
    )
    stop_evt = threading.Event()

    def handle_sig(signum, frame):
        print("\nshutting down...", flush=True)
        stop_evt.set()
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    try:
        analyze, probe = build_analyzer(cfg, slogger)
    except KeyboardInterrupt:
        print("\ninterrupted during model load -- writing summary...", flush=True)
        print("\n" + slogger.write_daily_report())
        slogger.close()
        return 130

    inference = InferenceWorker(
        analyze, stop_evt,
        max_queue=int(s.get("inference_queue_max", 10)),
    )
    inference.start()

    dispatcher = SessionDispatcher(
        inference_enqueue_fn=inference.enqueue,
        probe_fn=probe,
        merge_window_sec=float(s.get("session_merge_window_sec", 45)),
        similarity_threshold=float(s.get("session_merge_similarity_threshold", 0.6)),
        stop_evt=stop_evt,
    )
    dispatcher.start()

    janitor = RetentionJanitor(
        video_dir=slogger.video_dir, snapshot_dir=slogger.snapshots_dir,
        log_dir=slogger.dir,
        retention_days=int(s.get("retention_days", 7)),
        interval_hours=float(s.get("retention_check_interval_hours", 6)),
        stop_evt=stop_evt,
    )
    janitor.start()

    workers = [CameraWorker(c, cfg, dispatcher.submit, stop_evt)
               for c in cfg["cameras"]]
    for w in workers:
        w.start()

    try:
        while not stop_evt.is_set():
            time.sleep(0.5)
    finally:
        stop_evt.set()
        for w in workers:
            w.join(timeout=5)
        inference.join(timeout=30)
        print("\n" + slogger.write_daily_report())
        slogger.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.environ.get("FRAUD_CONFIG", "config.yaml"))
    ap.add_argument("--calibrate", action="store_true",
                    help="show live camera feed with zone overlay")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"config not found: {cfg_path}", file=sys.stderr)
        return 2
    cfg = load_config(str(cfg_path))

    if args.calibrate:
        from calibrate import calibrate
        calibrate(cfg)
        return 0
    return run_monitor(cfg)


if __name__ == "__main__":
    sys.exit(main())
