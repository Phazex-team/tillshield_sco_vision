"""Video segment coverage + window stream endpoints."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse


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
