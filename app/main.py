"""FastAPI application factory.

Mounts the v1 API routers and the legacy static dashboard. The factory
is intentionally side-effect-light: importing it should not load any
model or start the segment recorder. Wire-up of long-running services
happens in ``scripts/run_app.py``.

The factory does ensure the SQLite/Postgres schema is initialised
before requests are accepted — a fresh-repo deployment must never 500
on the first read of ``/api/v1/storage/disk`` or ``/api/v1/cases``.
"""
from __future__ import annotations

import contextlib
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


log = logging.getLogger(__name__)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    """Start/stop the TillShield POS-agent poller alongside the app.

    The poller only starts when ``integrations.tillshield.poll_enabled``
    is true. It runs in a daemon thread so a slow/unavailable POS agent
    never blocks boot, and is stopped cleanly on shutdown. Failure to
    start the poller is isolated — the API/UI still serve.
    """
    worker = None
    try:
        from app.config import load_config
        from pos.tillshield_poll import PollWorker, load_poll_config
        pc = load_poll_config(load_config())
        if pc.enabled:
            worker = PollWorker(interval=pc.poll_every_seconds)
            worker.start()
        else:
            log.info("tillshield poller disabled (poll_enabled=false)")
    except Exception:
        log.exception("tillshield poller failed to start; API still serving")
    try:
        yield
    finally:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Return / Refund Visual Review",
        version="3.0.0",
        docs_url="/api/v1/docs",
        openapi_url="/api/v1/openapi.json",
        lifespan=_lifespan,
    )

    # Idempotent schema init so a fresh repo serves correctly even
    # before scripts/run_app.py runs its own init step.
    try:
        from db.session import init_schema
        init_schema()
    except Exception:
        log.exception("db schema init failed at app construction")

    # Mount API v1 routers.
    from app.api import (
        admin as admin_router,
        cases as cases_router,
        evidence as evidence_router,
        health as health_router,
        integrations_tillshield as tillshield_router,
        memory as memory_router,
        ops as ops_router,
        pos as pos_router,
        review as review_router,
        storage as storage_router,
        video as video_router,
    )
    app.include_router(health_router.router, prefix="/api/v1")
    app.include_router(memory_router.router, prefix="/api/v1")
    app.include_router(pos_router.router, prefix="/api/v1")
    app.include_router(cases_router.router, prefix="/api/v1")
    app.include_router(evidence_router.router, prefix="/api/v1")
    app.include_router(review_router.router, prefix="/api/v1")
    app.include_router(video_router.router, prefix="/api/v1")
    app.include_router(admin_router.router, prefix="/api/v1")
    app.include_router(storage_router.router, prefix="/api/v1")
    app.include_router(ops_router.router, prefix="/api/v1")
    app.include_router(tillshield_router.router, prefix="/api/v1")

    # Legacy dashboard static files (review-safe; see static/index.html).
    static_dir = Path(__file__).resolve().parents[1] / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True),
                  name="static")

    return app


# Convenience for ``uvicorn app.main:app``
app = create_app()
