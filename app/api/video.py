"""Video segment coverage + window stream endpoints."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse


router = APIRouter(prefix="/video", tags=["video"])


@router.get("/segments/coverage")
def segments_coverage(camera_id: str, start_at: datetime,
                      end_at: datetime) -> dict:
    from db.session import get_sessionmaker
    from video.segment_index import coverage

    if end_at <= start_at:
        raise HTTPException(status_code=400,
                            detail="end_at must be after start_at")
    SM = get_sessionmaker()
    with SM() as s:
        return coverage(s, camera_id, start_at, end_at)


@router.get("/windows/{window_id}/stream")
def stream_window(window_id: str):
    from db.models import VideoWindow
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        win = s.get(VideoWindow, window_id)
        if win is None or not win.path:
            raise HTTPException(status_code=404, detail="window not found")
        path = Path(win.path)
        if not path.is_file():
            raise HTTPException(status_code=410,
                                detail="window file no longer on disk")
        return FileResponse(str(path), media_type="video/mp4",
                            filename=path.name)


def _no_store_error(status_code: int, detail: str) -> JSONResponse:
    """Return a safe JSON error with ``Cache-Control: no-store``.

    Raising ``HTTPException`` would lose the cache header on the error
    response, so we return a ``JSONResponse`` directly here. The detail
    is operator-readable text — never a raw path or rtsp_url.
    """
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/cameras/{camera_id}/preview-frame")
def camera_preview_frame(camera_id: str, response: Response):
    """Return ONE representative JPEG frame from the camera's newest
    locally-recorded segment.

    This is a read-only, on-demand peek for the reviewer UI's Pipeline
    tab. It is intentionally NOT:

      * a live RTSP stream (no per-viewer RTSP open, no WebSocket /
        MJPEG / HLS),
      * an inference pass (Falcon / SAM 2 / OCR / Qwen3-VL / Gemma are
        not consulted),
      * an admin / ROI calibration surface (that lives at
        ``GET /admin/camera-rois/{camera_id}/snapshot`` and IS admin-
        token gated),
      * a process control (it never starts/stops the recorder).

    Input is restricted to ``camera_id``. The endpoint NEVER returns
    ``rtsp_url`` or the segment's on-disk path. ``Cache-Control:
    no-store`` is stamped on both success and error responses so a
    browser-cached frame can't outlive the segment it came from.
    """
    response.headers["Cache-Control"] = "no-store"

    from app.config import load_config
    cfg = load_config()
    if not any(c.get("id") == camera_id for c in cfg.cameras):
        return _no_store_error(
            404, f"camera {camera_id!r} not configured")

    from sqlalchemy import select
    from db.models import VideoSegment
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        seg = s.execute(
            select(VideoSegment)
            .where(VideoSegment.camera_id == camera_id)
            .order_by(VideoSegment.start_at.desc())
        ).scalars().first()
    if seg is None or not seg.path:
        return _no_store_error(
            404,
            f"no local segment available for camera {camera_id!r}")

    import os
    if not os.path.exists(seg.path):
        return _no_store_error(
            410,
            f"latest segment for {camera_id!r} no longer on disk; "
            "retention may have cleared it")

    try:
        import cv2  # type: ignore
    except Exception as exc:
        return _no_store_error(
            503, f"cv2 unavailable: {type(exc).__name__}")

    cap = cv2.VideoCapture(seg.path)
    if not cap.isOpened():
        return _no_store_error(
            503, "decoder could not open the latest segment")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            return _no_store_error(
                503, "latest segment reports zero frames")
        # Middle frame is the most operator-useful representative for a
        # short segment. Fall back to the first frame when the codec
        # misreports total frames and the seek misses.
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            return _no_store_error(
                503, "decoder returned no frame from the latest segment")
        height, width = frame_bgr.shape[:2]
        ok, buf = cv2.imencode(
            ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return _no_store_error(
                503, "jpeg encode failed for preview frame")
    finally:
        cap.release()

    import base64
    b64 = base64.b64encode(bytes(buf)).decode("ascii")
    return {
        "camera_id": camera_id,
        "source": "latest_segment",
        "image_url": f"data:image/jpeg;base64,{b64}",
        "width": int(width),
        "height": int(height),
        # Stable identifiers only — no on-disk path, no rtsp_url.
        "segment_id": seg.id,
        "captured_at": seg.start_at.isoformat() if seg.start_at else None,
        "segment_start_at": (seg.start_at.isoformat()
                              if seg.start_at else None),
    }
