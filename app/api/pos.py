"""POS ingest endpoints.

Single-event (`POST /pos/returns/event`) and batch (`POST /pos/returns/batch`)
ingest. Both are idempotent via the natural key
``(store_id, terminal_id, transaction_id, line_id)`` (per PRODUCTION_SPEC §8).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field


router = APIRouter(prefix="/pos", tags=["pos"])


class PosEventBody(BaseModel):
    store_id: str
    terminal_id: str
    transaction_id: str
    line_id: str
    event_type: str = Field(description="RETURN | REFUND | REPLACEMENT")
    pos_event_at: datetime
    staff_id: Optional[str] = None
    sku: Optional[str] = None
    item_description: Optional[str] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    raw_payload: Optional[dict] = None


class PosBatchBody(BaseModel):
    source_system: str
    store_id: str
    received_at: Optional[datetime] = None
    batch_start_at: Optional[datetime] = None
    batch_end_at: Optional[datetime] = None
    events: list[PosEventBody]
    raw_payload: Optional[dict] = None


@router.post("/returns/event", status_code=200)
def ingest_event(body: PosEventBody, request: Request) -> dict:
    from app import audit
    from db.session import get_sessionmaker
    from pos.event_normalizer import normalize_event_type
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn

    received = datetime.now(timezone.utc)
    # Canonicalise event_type at the boundary. SCO mode rejects anything
    # that isn't in sco_checkout.accept_event_types.
    canonical = normalize_event_type(body.event_type)
    if canonical is None:
        raise HTTPException(
            status_code=400,
            detail=(f"event_type {body.event_type!r} is not accepted "
                    "(check sco_checkout.accept_event_types in config)"),
        )
    event_payload = body.model_dump()
    event_payload["event_type"] = canonical
    batch = PosBatchIn(
        source_system="api_single",
        store_id=body.store_id,
        received_at=received,
        events=[PosEventIn(**event_payload)],
    )
    SM = get_sessionmaker()
    with SM() as s:
        try:
            result = ingest_batch(s, batch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        audit.record(
            s,
            action="pos.ingest_event",
            entity_type="pos_batch",
            entity_id=result.get("batch_id"),
            actor_type="pos_api",
            after={"ingest_result": result},
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
    return result


@router.post("/returns/batch", status_code=200)
def ingest_batch_endpoint(body: PosBatchBody, request: Request) -> dict:
    from app import audit
    from db.session import get_sessionmaker
    from pos.event_normalizer import normalize_event_type
    from pos.ingest import ingest_batch
    from pos.schemas import PosBatchIn, PosEventIn

    received = body.received_at or datetime.now(timezone.utc)
    events = []
    for e in body.events:
        canonical = normalize_event_type(e.event_type)
        if canonical is None:
            raise HTTPException(
                status_code=400,
                detail=(f"event_type {e.event_type!r} is not accepted "
                        "(check sco_checkout.accept_event_types in config)"),
            )
        payload = e.model_dump()
        payload["event_type"] = canonical
        events.append(PosEventIn(**payload))
    batch = PosBatchIn(
        source_system=body.source_system,
        store_id=body.store_id,
        received_at=received,
        batch_start_at=body.batch_start_at,
        batch_end_at=body.batch_end_at,
        events=events,
        raw_payload=body.raw_payload,
    )
    SM = get_sessionmaker()
    with SM() as s:
        try:
            result = ingest_batch(s, batch)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        audit.record(
            s,
            action="pos.ingest_batch",
            entity_type="pos_batch",
            entity_id=result.get("batch_id"),
            actor_type="pos_api",
            after={"ingest_result": result,
                   "events": len(body.events)},
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
    return result


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None
