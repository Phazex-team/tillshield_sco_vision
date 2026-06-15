"""FastAPI application factory.

Mounts the v1 API routers and the legacy static dashboard. The factory
is intentionally side-effect-light: importing it should not load any
model or start the segment recorder. Wire-up of long-running services
happens in ``scripts/run_app.py``.
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Return / Refund Visual Review",
        version="3.0.0",
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
    )

    # Mount API v1 routers.
    from app.api import (
        health as health_router,
        memory as memory_router,
        pos as pos_router,
        cases as cases_router,
        evidence as evidence_router,
        review as review_router,
        video as video_router,
    )
    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(memory_router.router, prefix="/api/v1")
    app.include_router(pos_router.router, prefix="/api/v1")
    app.include_router(cases_router.router, prefix="/api/v1")
    app.include_router(evidence_router.router, prefix="/api/v1")
    app.include_router(review_router.router, prefix="/api/v1")
    app.include_router(video_router.router, prefix="/api/v1")

    # Legacy dashboard static files (review-safe; see static/index.html).
    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True),
                  name="static")

    return app


# Convenience for ``uvicorn app.main:app``
app = create_app()
