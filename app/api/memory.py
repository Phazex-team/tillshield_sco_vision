"""Memory status endpoint (PRODUCTION_SPEC §7)."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter


router = APIRouter(tags=["memory"])


@router.get("/memory")
def memory() -> dict:
    from app.memory_guard import get_policy
    status = get_policy().poll()
    return asdict(status)
