"""TillShield POS ingest endpoints.

Accepts the TillShield agent's table-shaped transaction objects on
``POST /api/v1/integrations/tillshield/transactions/{event,batch}`` and
hands them to ``pos.tillshield.ingest_tillshield_batch``, which
normalises and forwards them to the existing idempotent POS ingest.

The endpoint:
  * NEVER fails on duplicates — replaying the same transaction_id is a
    200 with ``duplicate_batch=true`` semantics in the payload.
  * Accepts a configurable shared secret via the
    ``X-PhazeX-Ingest-Token`` header. When the token is unset, the
    endpoint is unauthenticated (dev mode).
  * Audits every ingest with the source IP + user-agent so an operator
    can prove who sent what.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request


log = logging.getLogger(__name__)


router = APIRouter(prefix="/integrations/tillshield", tags=["tillshield"])


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


def _expected_token() -> Optional[str]:
    import os

    from app.config import load_config
    env = os.environ.get("TILLSHIELD_INGEST_TOKEN")
    if env:
        return env.strip() or None
    cfg = load_config()
    integrations = (cfg.raw.get("integrations") or {})
    ts = integrations.get("tillshield") or {}
    tok = ts.get("ingest_token")
    return str(tok).strip() if tok else None


def _check_auth(token_header: Optional[str]) -> None:
    expected = _expected_token()
    if not expected:
        return
    if not token_header or token_header.strip() != expected:
        raise HTTPException(status_code=401,
                            detail="invalid or missing ingest token")


@router.post("/transactions/event", status_code=200)
def ingest_event(request: Request,
                 x_phazex_ingest_token: Optional[str] = Header(default=None),
                 ) -> dict:
    """Accept a single TillShield transaction."""
    from app import audit
    from app.config import load_config
    from db.session import get_sessionmaker
    from pos.tillshield import ingest_tillshield_batch
    from pos.tillshield_schemas import (
        TillShieldBatch, TillShieldTransaction,
    )

    _check_auth(x_phazex_ingest_token)

    body = _read_json(request)
    try:
        txn = TillShieldTransaction(**body)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid TillShield transaction: {exc}")
    batch = TillShieldBatch(source_system="tillshield_agent",
                            events=[txn])
    cfg = load_config()
    SM = get_sessionmaker()
    with SM() as s:
        summary = ingest_tillshield_batch(
            s, cfg=cfg, batch=batch, source_ip=_client_ip(request))
        audit.record(
            s,
            action="tillshield.ingest_event",
            entity_type="tillshield_transaction",
            entity_id=txn.transaction_id,
            actor_type="tillshield_agent",
            after={"summary": summary,
                   "normalised_type": (txn.transaction_type or "").upper()},
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
    return summary


@router.post("/transactions/batch", status_code=200)
def ingest_batch_endpoint(
        request: Request,
        x_phazex_ingest_token: Optional[str] = Header(default=None),
        ) -> dict:
    """Accept a batch of TillShield transactions."""
    from app import audit
    from app.config import load_config
    from db.session import get_sessionmaker
    from pos.tillshield import ingest_tillshield_batch
    from pos.tillshield_schemas import TillShieldBatch

    _check_auth(x_phazex_ingest_token)

    body = _read_json(request)
    try:
        batch = TillShieldBatch(**body)
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid TillShield batch: {exc}")
    cfg = load_config()
    SM = get_sessionmaker()
    with SM() as s:
        summary = ingest_tillshield_batch(
            s, cfg=cfg, batch=batch, source_ip=_client_ip(request))
        audit.record(
            s,
            action="tillshield.ingest_batch",
            entity_type="tillshield_batch",
            actor_type="tillshield_agent",
            after={"summary": summary,
                   "events": len(batch.events)},
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
    return summary


def _read_json(request: Request) -> dict:
    """Synchronous read of the request body — keeps the handlers
    non-async because the downstream session work is synchronous too."""
    import asyncio
    import json
    loop = asyncio.new_event_loop()
    try:
        raw = loop.run_until_complete(request.body())
    finally:
        loop.close()
    if not raw:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400,
                            detail=f"invalid JSON body: {exc}")
