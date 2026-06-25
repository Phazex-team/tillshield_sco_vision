"""One-way export of analysis RESULTS to the TillShield Refund Agent.

This is purely ADDITIVE and outbound. It reads a completed case and pushes
the result — plus a boxed + lightly face-masked copy of the evidence clip —
to the edge agent (default ``http://localhost:8081``). It NEVER mutates the
case, the stored evidence/window video, the analysis, or any config: a
failure here is logged and swallowed so it can never affect the pipeline.

Per REFUND-AGENT-CLIENT.md:
  1. POST /events/refund            (JSON result)        -> refund_id
  2. POST /events/refund/evidence   (multipart video)    -> evidence_id

Update-on-reprocess: the agent rewrites the record when the same
``pos_transaction_id`` is sent again (no separate update endpoint in the
doc). If an explicit ``update_path`` is configured later, it's used instead.
We remember case -> refund_id in the audit_log to avoid blind duplicates.

Face masking (Option 2, no ML model): the top ~25% (head/face region) of
Falcon ``person`` boxes whose centre falls inside the customer zone is
lightly blurred — applied ONLY to this export copy, never to our evidence.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import urllib.request
import uuid
from typing import Optional

log = logging.getLogger(__name__)

_AUDIT_ACTION = "case.exported_to_refund_agent"


def _conf(cfg) -> dict:
    raw = ((cfg.raw.get("integrations") or {}).get("refund_agent") or {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "base_url": str(raw.get("base_url", "http://localhost:8081")).rstrip("/"),
        "create_path": raw.get("create_path", "/events/refund"),
        "evidence_path": raw.get("evidence_path", "/events/refund/evidence"),
        # Empty => re-POST create with same pos_transaction_id (agent upserts).
        "update_path": raw.get("update_path", ""),
        "timeout": int(raw.get("timeout_sec", 30)),
        "send_video": bool(raw.get("send_video", True)),
        "mask_faces": bool(raw.get("mask_customer_faces", True)),
        "blur_strength": str(raw.get("face_blur_strength", "light")),
    }


def maybe_export_case(case_id: str, cfg=None) -> None:
    """Background entry point. Guarded + never raises."""
    try:
        from app.config import load_config
        cfg = cfg or load_config()
        conf = _conf(cfg)
        if not conf["enabled"]:
            return
        _export(case_id, cfg, conf)
    except Exception:
        log.exception("refund-agent export failed for case %s (non-fatal)",
                      case_id)


# ---------------------------------------------------------------------------

def _iso_z(dt) -> Optional[str]:
    if dt is None:
        return None
    s = dt.isoformat()
    return s if s.endswith("Z") else s + "Z"


def _build_payload(case, pos, vlm_out: dict, window) -> dict:
    """Map our OUTPUT to the agent's /events/refund fields. Output data
    only — no store_id (agent derives it), no config, no model names."""
    reasons = " · ".join(case.risk_reasons or [])
    narrative = (vlm_out.get("narrative") or "").strip()
    summary = " — ".join(p for p in (case.outcome, reasons) if p)
    if narrative:
        summary = f"{summary} | {narrative}" if summary else narrative
    # Client-facing wording: never expose the internal "VLM" term in the
    # EXPORTED summary (our own app's stored reasons are left unchanged).
    summary = summary.replace("VLM", "Vision AI")
    payload = {
        "pos_transaction_id": getattr(pos, "transaction_id", None),
        "refund_amount": abs(float(pos.amount)) if pos and pos.amount is not None else None,
        "currency": getattr(pos, "currency", None),
        "refund_time": _iso_z(getattr(pos, "pos_event_at", None)),
        "cashier_id": getattr(pos, "staff_id", None),
        "item_present": vlm_out.get("item_presented"),
        "customer_present": vlm_out.get("customer_present"),
        "handover": vlm_out.get("handover_occurred"),
        "agent_summary": summary or None,
    }
    if window is not None:
        payload["video_start_time"] = _iso_z(getattr(window, "actual_start_at", None))
        payload["video_end_time"] = _iso_z(getattr(window, "actual_end_at", None))
    # Drop nulls so we only send what we actually have.
    return {k: v for k, v in payload.items() if v is not None}


def _post_json(url: str, body: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _post_multipart(url: str, fields: dict, file_path: str,
                    file_field: str, timeout: int) -> dict:
    boundary = "----refundagent" + uuid.uuid4().hex
    parts: list[bytes] = []
    for k, v in fields.items():
        parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                      f'name="{k}"\r\n\r\n{v}\r\n').encode())
    fname = os.path.basename(file_path)
    parts.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                  f'name="{file_field}"; filename="{fname}"\r\n'
                  "Content-Type: video/mp4\r\n\r\n").encode())
    with open(file_path, "rb") as f:
        parts.append(f.read())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    data = b"".join(parts)
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _prior_refund_id(session, case_id: str) -> Optional[str]:
    from db.models import AuditLog
    row = (session.query(AuditLog)
           .filter(AuditLog.action == _AUDIT_ACTION,
                   AuditLog.entity_id == case_id)
           .order_by(AuditLog.id.desc()).first())
    if row and isinstance(row.after_json, dict):
        return row.after_json.get("refund_id")
    return None


def _export(case_id: str, cfg, conf: dict) -> None:
    from db.models import Case, PosEvent, VlmRun, VideoWindow
    from db.session import get_sessionmaker
    from app import audit

    SM = get_sessionmaker()
    with SM() as s:
        case = s.get(Case, case_id)
        if case is None:
            return
        pos = s.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
        run = (s.query(VlmRun).filter_by(case_id=case_id)
               .order_by(VlmRun.started_at.desc()).first())
        vlm_out = (run.output_json or {}) if run else {}
        wid = ((run.input_manifest or {}).get("window_id") if run else None)
        window = s.get(VideoWindow, wid) if wid else None

        payload = _build_payload(case, pos, vlm_out, window)
        prior = _prior_refund_id(s, case_id)

        # ---- step 1: create (or update by re-sending same txn) ----
        if prior and conf["update_path"]:
            url = conf["base_url"] + conf["update_path"].replace(
                "{refund_id}", prior).replace("{id}", prior)
        else:
            url = conf["base_url"] + conf["create_path"]
        resp = _post_json(url, payload, conf["timeout"])
        status = resp.get("status")
        result = resp.get("result") or {}
        refund_id = result.get("refund_id") or prior
        log.info("refund export case=%s status=%s refund_id=%s (prior=%s)",
                 case_id[:8], status, refund_id, prior)

        # 202 queued => no refund_id yet; record intent, skip video for now.
        if status == "queued" or not refund_id:
            audit.record(s, action=_AUDIT_ACTION, entity_type="case",
                         entity_id=case_id, actor_type="exporter",
                         after={"status": status, "refund_id": refund_id,
                                "payload": payload})
            s.commit()
            return

        # ---- step 2: boxed + face-masked evidence video (optional) ----
        evidence_id = None
        if conf["send_video"] and window is not None and getattr(window, "path", None):
            tmp = None
            try:
                tmp = _render_export_video(s, case_id, window, conf)
                if tmp:
                    ev_url = conf["base_url"] + conf["evidence_path"]
                    fields = {"refund_incident_id": refund_id}
                    if payload.get("video_start_time"):
                        fields["video_start_time"] = payload["video_start_time"]
                    if payload.get("video_end_time"):
                        fields["video_end_time"] = payload["video_end_time"]
                    ev = _post_multipart(ev_url, fields, tmp, "file", conf["timeout"])
                    evidence_id = (ev.get("result") or {}).get("evidence_id")
                    log.info("refund export case=%s evidence_id=%s",
                             case_id[:8], evidence_id)
            except Exception:
                log.exception("refund export: evidence step failed case=%s", case_id)
            finally:
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        audit.record(s, action=_AUDIT_ACTION, entity_type="case",
                     entity_id=case_id, actor_type="exporter",
                     after={"status": status, "refund_id": refund_id,
                            "evidence_id": evidence_id, "payload": payload})
        s.commit()


def _render_export_video(session, case_id: str, window, conf: dict) -> Optional[str]:
    """Render an EXPORT-ONLY copy of the window clip with Falcon boxes drawn
    and the customer's head/face (top of customer-zone person boxes) lightly
    blurred. Reads the original window video read-only; writes a temp file.
    Returns the temp mp4 path (caller deletes it). Never touches our evidence."""
    import cv2
    import numpy as np
    from db.models import Detection

    src = window.path
    if not src or not os.path.exists(src):
        return None
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # customer-zone rectangle, scaled from config source dims to video dims.
    cz = _customer_zone(case_id, w, h, session)

    # detections grouped by frame_idx (carry-forward to keep boxes continuous).
    dets = session.query(Detection).filter(Detection.case_id == case_id).all()
    by_idx: dict[int, list] = {}
    for d in dets:
        bb = getattr(d, "bbox_xyxy", None)
        idx = getattr(d, "frame_idx", None)
        if bb and idx is not None:
            by_idx.setdefault(int(idx), []).append(
                (str(getattr(d, "label", "") or ""), [float(x) for x in bb]))
    sampled = sorted(by_idx)

    blur_k = {"light": 0.18, "medium": 0.32}.get(conf["blur_strength"], 0.18)

    tmp_raw = os.path.join(tempfile.gettempdir(),
                           f"rfexport_{case_id[:8]}_{uuid.uuid4().hex[:6]}.mp4")
    writer = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    last_boxes: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            # nearest sampled frame_idx <= i (carry-forward)
            cur = [s for s in sampled if s <= i]
            if cur:
                last_boxes = by_idx.get(cur[-1], last_boxes)
            for label, (x1, y1, x2, y2) in last_boxes:
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                is_item = "item" in label.lower()
                cv2.rectangle(frame, p1, p2,
                              (132, 220, 61) if is_item else (239, 141, 91), 2)
                if label:
                    cv2.putText(frame, label, (int(x1), max(0, int(y1) - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (132, 220, 61) if is_item else (239, 141, 91), 1)
                # face mask: person box whose centre is in the customer zone
                if conf["mask_faces"] and cz and "person" in label.lower():
                    cxp, cyp = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    if cz[0] <= cxp <= cz[2] and cz[1] <= cyp <= cz[3]:
                        _blur_head(frame, x1, y1, x2, y2, blur_k)
            writer.write(frame)
            i += 1
    finally:
        cap.release()
        writer.release()

    # transcode to browser/cloud-friendly H.264 (separate file).
    out = tmp_raw.replace(".mp4", "_h264.mp4")
    import subprocess
    rc = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_raw, "-c:v", "libx264", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", out],
        capture_output=True).returncode
    try:
        os.remove(tmp_raw)
    except OSError:
        pass
    return out if rc == 0 and os.path.exists(out) else None


def _blur_head(frame, x1, y1, x2, y2, frac: float) -> None:
    import cv2
    bh = (y2 - y1) * 0.25  # top quarter = head/face region
    hx1, hy1 = max(0, int(x1)), max(0, int(y1))
    hx2, hy2 = int(x2), int(y1 + bh)
    if hx2 <= hx1 or hy2 <= hy1:
        return
    roi = frame[hy1:hy2, hx1:hx2]
    if roi.size == 0:
        return
    # light blur: kernel scaled to region (odd), small => subtle.
    k = max(3, int(min(roi.shape[0], roi.shape[1]) * frac) | 1)
    frame[hy1:hy2, hx1:hx2] = cv2.GaussianBlur(roi, (k, k), 0)


def _customer_zone(case_id: str, vw: int, vh: int, session) -> Optional[tuple]:
    """Return (x1,y1,x2,y2) of the customer zone scaled to the video, or None."""
    try:
        from db.models import Case
        from app.config import load_config
        cam_id = session.get(Case, case_id).camera_id
        cfg = load_config()
        cam = next((c for c in cfg.raw.get("cameras", [])
                    if c.get("id") == cam_id), None)
        if not cam:
            return None
        zones = cam.get("zones") or {}
        z = next((v for k, v in zones.items() if "customer" in k.lower()), None)
        if not z:
            return None
        sw = float(z.get("source_width") or vw)
        sh = float(z.get("source_height") or vh)
        sx, sy = vw / sw, vh / sh
        x1 = float(z["x"]) * sx
        y1 = float(z["y"]) * sy
        x2 = (float(z["x"]) + float(z["w"])) * sx
        y2 = (float(z["y"]) + float(z["h"])) * sy
        return (x1, y1, x2, y2)
    except Exception:
        return None
