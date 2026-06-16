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
from typing import Callable, Optional

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from db.models import Case, PosBatch, PosEvent
from .schemas import PosBatchIn, PosEventIn


CASE_OPENING_TYPES = {"RETURN", "REFUND"}

# A camera resolver maps (store_id, terminal_id) -> camera_id or None.
CameraResolver = Callable[[str, str], Optional[str]]


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


def _configured_camera_ids(cfg) -> set[str]:
    return {c.get("id") for c in (cfg.cameras or []) if c.get("id")}


def _workstation_camera_map(cfg) -> dict[str, str]:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    ts = integrations.get("tillshield") or {}
    raw = ts.get("workstation_camera_map") or {}
    return {str(k): str(v) for k, v in raw.items() if k is not None and v}


def resolve_camera_for_pos_event(store_id: str,
                                 terminal_id: str,
                                 cfg=None) -> Optional[str]:
    """Resolve the camera for a POS event by canonical ``terminal_id``.

    Workstation-aware routing:
      * If ``integrations.tillshield.workstation_camera_map`` is set, the
        terminal MUST be mapped to a camera that exists under
        ``cameras:`` — otherwise return ``None`` (the caller persists the
        event but opens no case; it is never sent to a default camera).
      * If no map is configured at all, fall back to the legacy
        store-default camera so existing single-camera deploys keep
        working.
    """
    if cfg is None:
        from app.config import load_config
        cfg = load_config()
    wsmap = _workstation_camera_map(cfg)
    if not wsmap:
        return _default_camera_for_store(store_id)
    camera_id = wsmap.get(str(terminal_id))
    if not camera_id:
        return None  # workstation not mapped to any camera
    if camera_id not in _configured_camera_ids(cfg):
        return None  # mapped camera id is not defined under cameras:
    return camera_id


def ingest_batch(session: Session, batch: PosBatchIn,
                 *, resolve_camera: Optional[CameraResolver] = None) -> dict:
    """Persist a batch idempotently. Returns a small summary dict.

    ``resolve_camera`` (optional) maps (store_id, terminal_id) -> camera
    id for case routing. When supplied and it returns ``None`` for an
    event, the event is still persisted but NO case is opened (counted in
    ``ignored_unmapped_workstation_events``). When omitted, the legacy
    store-default camera is used so existing callers are unchanged.

    Result shape:
      {
        "batch_id": "<uuid>" or None if already-seen,
        "events_inserted": int,
        "events_already_present": int,
        "cases_created": int,
        "ignored_unmapped_workstation_events": int,
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
            "ignored_unmapped_workstation_events": 0,
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
    unmapped = 0
    for ev in batch.events:
        pos_event_id = _upsert_event(session, pb.id, ev)
        if pos_event_id is None:
            already += 1
            continue
        inserted += 1
        if ev.event_type in CASE_OPENING_TYPES:
            opened, reason = _open_case_for_event(
                session, pos_event_id, ev.store_id, ev.terminal_id,
                resolve_camera)
            if opened:
                cases_created += 1
            elif reason == "unmapped":
                unmapped += 1

    return {
        "batch_id": pb.id,
        "events_inserted": inserted,
        "events_already_present": already,
        "cases_created": cases_created,
        "ignored_unmapped_workstation_events": unmapped,
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
                         store_id: str,
                         terminal_id: str,
                         resolve_camera: Optional[CameraResolver] = None,
                         ) -> tuple[bool, Optional[str]]:
    """Open one case for ``pos_event_id``. Returns ``(opened, reason)``.

    ``reason`` is ``"exists"`` (case already opened), ``"unmapped"`` (no
    valid camera route — event kept, case skipped), or ``None`` on open.
    """
    existing = session.execute(
        select(Case).where(Case.pos_event_id == pos_event_id)
    ).scalar_one_or_none()
    if existing is not None:
        return (False, "exists")
    if resolve_camera is not None:
        camera_id = resolve_camera(store_id, terminal_id)
    else:
        camera_id = _default_camera_for_store(store_id)
    if not camera_id:
        # No valid workstation->camera route: never fall back to the
        # first/default camera. Persist the event, open no case.
        return (False, "unmapped")
    session.add(Case(
        pos_event_id=pos_event_id,
        camera_id=camera_id,
        status="OPEN",
    ))
    session.flush()
    return (True, None)
