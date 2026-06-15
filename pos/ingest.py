"""Idempotent POS batch ingest + case creation.

Contract:
  * Replaying the same batch (same payload hash OR same natural key
    tuple) does NOT create duplicate ``PosEvent`` or ``Case`` rows.
  * A ``RETURN`` / ``REFUND`` event opens (at most) one ``Case`` with
    status=OPEN and outcome=NULL.
  * ``REPLACEMENT`` events insert the event but do not open a case;
    they are correlated by the perception pipeline as context.
  * Camera assignment uses the store->camera mapping in ``config.yaml``;
    no mapping yet means a single default camera id is used (so the MVP
    keeps working until multi-store config lands).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import Case, PosBatch, PosEvent
from .schemas import PosBatchIn, PosEventIn


CASE_OPENING_TYPES = {"RETURN", "REFUND"}


def _payload_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def _jsonable(value):
    """Recursively coerce datetimes to ISO strings so SQLAlchemy's JSON
    column can hold the raw payload. SQLite's JSON encoder refuses
    ``datetime`` natively."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def _default_camera_for_store(store_id: str) -> str:
    # Hook for the store->camera mapping. The MVP has a single camera
    # configured in ``config.yaml``; until pos<->camera routing lands we
    # ship the events to that camera.
    from app.config import load_config
    cfg = load_config()
    if cfg.cameras:
        return cfg.cameras[0].get("id") or "cam_01"
    return "cam_01"


def ingest_batch(session: Session, batch: PosBatchIn) -> dict:
    """Persist a batch idempotently. Returns a small summary dict.

    Result shape:
      {
        "batch_id": "<uuid>" or None if already-seen,
        "events_inserted": int,
        "events_already_present": int,
        "cases_created": int,
      }
    """
    batch.validate()

    raw = _jsonable(batch.raw_payload or {
        "source_system": batch.source_system,
        "store_id": batch.store_id,
        "events": [vars(e) for e in batch.events],
    })
    payload_hash = _payload_hash(raw)

    # Idempotency check on batch level: same payload hash short-circuits.
    existing_batch = session.execute(
        select(PosBatch).where(PosBatch.payload_hash == payload_hash)
    ).scalar_one_or_none()
    if existing_batch is not None:
        return {
            "batch_id": existing_batch.id,
            "events_inserted": 0,
            "events_already_present": len(batch.events),
            "cases_created": 0,
            "duplicate_batch": True,
        }

    pb = PosBatch(
        source_system=batch.source_system,
        store_id=batch.store_id,
        received_at=batch.received_at,
        batch_start_at=batch.batch_start_at,
        batch_end_at=batch.batch_end_at,
        payload_hash=payload_hash,
        raw_payload=raw,
    )
    session.add(pb)
    session.flush()

    inserted = 0
    already = 0
    cases_created = 0
    for ev in batch.events:
        pos_event_id = _upsert_event(session, pb.id, ev)
        if pos_event_id is None:
            already += 1
            continue
        inserted += 1
        if ev.event_type in CASE_OPENING_TYPES:
            if _open_case_for_event(session, pos_event_id, ev.store_id):
                cases_created += 1

    return {
        "batch_id": pb.id,
        "events_inserted": inserted,
        "events_already_present": already,
        "cases_created": cases_created,
        "duplicate_batch": False,
    }


def _upsert_event(session: Session,
                  batch_id: str,
                  ev: PosEventIn) -> Optional[str]:
    """Insert one event by natural key. Return the row id, or None
    if a row with that natural key already existed."""
    existing = session.execute(
        select(PosEvent).where(
            PosEvent.store_id == ev.store_id,
            PosEvent.terminal_id == ev.terminal_id,
            PosEvent.transaction_id == ev.transaction_id,
            PosEvent.line_id == ev.line_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return None
    row = PosEvent(
        batch_id=batch_id,
        store_id=ev.store_id,
        terminal_id=ev.terminal_id,
        transaction_id=ev.transaction_id,
        line_id=ev.line_id,
        event_type=ev.event_type,
        pos_event_at=ev.pos_event_at,
        ingested_at=datetime.now(timezone.utc),
        staff_id=ev.staff_id,
        sku=ev.sku,
        item_description=ev.item_description,
        quantity=ev.quantity,
        amount=ev.amount,
        currency=ev.currency,
        raw_payload=ev.raw_payload,
    )
    session.add(row)
    try:
        session.flush()
    except IntegrityError:
        # Race lost; the row exists, so it's not new.
        session.rollback()
        return None
    return row.id


def _open_case_for_event(session: Session,
                         pos_event_id: str,
                         store_id: str) -> bool:
    existing = session.execute(
        select(Case).where(Case.pos_event_id == pos_event_id)
    ).scalar_one_or_none()
    if existing is not None:
        return False
    camera_id = _default_camera_for_store(store_id)
    session.add(Case(
        pos_event_id=pos_event_id,
        camera_id=camera_id,
        status="OPEN",
    ))
    session.flush()
    return True
