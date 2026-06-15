"""Disk / storage status endpoint."""
from __future__ import annotations

from fastapi import APIRouter


router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/disk")
def disk() -> dict:
    from app.storage_guard import disk_status
    from db.session import get_sessionmaker
    SM = get_sessionmaker()
    with SM() as s:
        return disk_status(s)
