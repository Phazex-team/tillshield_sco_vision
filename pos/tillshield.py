"""TillShield POS normaliser + ingest helper.

The TillShield agent stores rows in ``tillshield_agent.pos_api_transactions``.
This module converts each such row into the app's canonical
``pos.schemas.PosEventIn`` and ingests the resulting batch through the
existing idempotent ``pos.ingest.ingest_batch`` — so the downstream
case flow (window resolve, perception, decision policy, evidence
package, reviewer workflow) does not need any TillShield-specific
branching.

Hard rules (PRODUCTION_SPEC):
  * Never download anything at runtime.
  * Never accuse fraud.
  * Idempotent: replaying the same transaction_id never creates a
    second case.
  * Preserve the full raw TillShield payload for audit.

Configuration knobs live under ``config.yaml.integrations.tillshield``:
``accepted_event_types``, ``line_level_cases``, ``ingest_token``,
``ignore_unknown_types``.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional

from sqlalchemy.orm import Session

from pos.ingest import ingest_batch
from pos.schemas import PosBatchIn, PosEventIn

from .tillshield_schemas import TillShieldBatch, TillShieldTransaction


log = logging.getLogger(__name__)


DEFAULT_ACCEPTED_EVENT_TYPES: tuple[str, ...] = (
    "SALE", "SCO_SALE", "CHECKOUT",
)


def _normalise_event_type(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return raw.strip().upper().replace("-", "_").replace(" ", "_")


def _accepted_event_types(cfg) -> set[str]:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    ts = integrations.get("tillshield") or {}
    custom = ts.get("accepted_event_types")
    if custom:
        return {_normalise_event_type(x) for x in custom}
    return set(DEFAULT_ACCEPTED_EVENT_TYPES)


def _accepted_return_types(cfg) -> set[str]:
    """Compatibility alias for older imports; returns SCO accepted types."""
    return _accepted_event_types(cfg)


def _line_level_enabled(cfg) -> bool:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    ts = integrations.get("tillshield") or {}
    return bool(ts.get("line_level_cases", False))


def _ignore_unknown(cfg) -> bool:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    ts = integrations.get("tillshield") or {}
    # Defaults to True: unknown transaction types are silently skipped
    # (counted in ``ignored_non_return_events``) so a noisy POS feed
    # never causes an HTTP error.
    return bool(ts.get("ignore_unknown_types", True))


def _parse_timezone(value: Optional[str]) -> Optional[tzinfo]:
    """Parse a POS timezone config value into a ``tzinfo``.

    Accepts an IANA name (``Asia/Dubai``) or a fixed offset
    (``+04:00`` / ``-05:30``). Returns ``None`` if unset or unparseable
    (callers then treat naive timestamps as UTC — the legacy behavior)."""
    if not value:
        return None
    v = str(value).strip()
    if v[0] in "+-" and ":" in v:
        try:
            sign = 1 if v[0] == "+" else -1
            hh, mm = v[1:].split(":")
            return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
        except Exception:
            log.warning("tillshield: bad pos_timezone offset %r; ignoring", v)
            return None
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(v)
    except Exception:
        log.warning("tillshield: unknown pos_timezone %r; ignoring", v)
        return None


def _resolve_pos_timezone(cfg) -> Optional[tzinfo]:
    integrations = (cfg.raw.get("integrations") if cfg else None) or {}
    ts = integrations.get("tillshield") or {}
    return _parse_timezone(ts.get("pos_timezone"))


def _to_naive_utc(dt: datetime, pos_tz: Optional[tzinfo]) -> datetime:
    """Normalise a POS ``transaction_date`` to naive UTC so it lines up
    with the recorder's UTC segment timestamps during correlation.

      * tz-aware input  -> converted to UTC, tzinfo stripped.
      * naive + pos_tz  -> interpreted in ``pos_tz``, converted to UTC.
      * naive, no pos_tz -> returned unchanged (assumed already UTC).
    """
    if not isinstance(dt, datetime):
        return dt
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    if pos_tz is not None:
        return dt.replace(tzinfo=pos_tz).astimezone(
            timezone.utc).replace(tzinfo=None)
    return dt


def _line_id_for(txn: TillShieldTransaction, item: Optional[dict],
                 fallback_index: int) -> str:
    if item:
        for key in ("line_id", "lineId", "id", "sequence", "seq",
                    "transaction_line_id"):
            v = item.get(key)
            if v is not None and str(v).strip():
                return str(v)
    if fallback_index <= 0:
        return "transaction"
    return f"transaction_line_{fallback_index}"


def normalise_to_pos_events(
        txn: TillShieldTransaction,
        *,
        line_level: bool = False,
        source_ip: Optional[str] = None,
        pos_tz: Optional[tzinfo] = None) -> list[PosEventIn]:
    """Build one (or more) ``PosEventIn`` rows from a TillShield row.

    Returns ``[]`` if the row's transaction_type is not an accepted SCO
    checkout event; callers should count these in
    ``ignored_non_return_events`` for backwards-compatible metrics.

    ``pos_tz`` (when set) is the timezone the POS agent reports
    ``transaction_date`` in; the stored ``pos_event_at`` is normalised to
    naive UTC so it correlates with UTC CCTV segments.
    """
    # In SCO mode, canonicalise via pos.event_normalizer so the stored
    # event_type matches what ingest.case_opening_types() expects. In
    # legacy mode, fall back to the local normaliser which preserves the
    # raw uppercased form.
    from pos.event_normalizer import normalize_event_type
    event_type = (normalize_event_type(txn.transaction_type)
                  or _normalise_event_type(txn.transaction_type))
    event_at = _to_naive_utc(txn.transaction_date, pos_tz)

    raw_payload = {
        "source_system": "tillshield_agent",
        "source_event_id": txn.transaction_id,
        "reference_id": txn.reference_id,
        "cashier_name": txn.cashier_name,
        "transaction_end_at": (txn.transaction_end_date.isoformat()
                               if txn.transaction_end_date else None),
        "source_ip": source_ip or txn.source_ip,
        "received_at": (txn.received_at.isoformat()
                        if txn.received_at else None),
        "total_items": txn.total_items,
        "items": txn.items or [],
        "raw_payload": txn.payload or {},
    }

    items = txn.items or []
    if not line_level or not items:
        return [PosEventIn(
            store_id=txn.store_id,
            terminal_id=txn.workstation_id,
            transaction_id=txn.transaction_id,
            line_id="transaction",
            event_type=event_type,
            pos_event_at=event_at,
            staff_id=txn.operator_id,
            sku=None,
            item_description=None,
            quantity=txn.total_items,
            amount=float(txn.total_amount) if txn.total_amount is not None
                else None,
            currency=txn.currency,
            raw_payload=raw_payload,
        )]

    out: list[PosEventIn] = []
    seen: set[str] = set()
    for i, item in enumerate(items, start=1):
        line_id = _line_id_for(txn, item, i)
        if line_id in seen:
            # If the upstream reuses IDs across lines, fall back to
            # transaction-level so we don't fabricate uniqueness.
            return [PosEventIn(
                store_id=txn.store_id,
                terminal_id=txn.workstation_id,
                transaction_id=txn.transaction_id,
                line_id="transaction",
                event_type=event_type,
                pos_event_at=event_at,
                staff_id=txn.operator_id,
                amount=float(txn.total_amount)
                    if txn.total_amount is not None else None,
                currency=txn.currency,
                raw_payload=raw_payload,
            )]
        seen.add(line_id)
        try:
            qty = float(item.get("quantity") or item.get("qty") or 0) \
                or None
        except (TypeError, ValueError):
            qty = None
        try:
            amount = float(item.get("amount")
                            or item.get("line_total")
                            or item.get("price") or 0) or None
        except (TypeError, ValueError):
            amount = None
        out.append(PosEventIn(
            store_id=txn.store_id,
            terminal_id=txn.workstation_id,
            transaction_id=txn.transaction_id,
            line_id=line_id,
            event_type=event_type,
            pos_event_at=event_at,
            staff_id=txn.operator_id,
            sku=str(item.get("sku") or item.get("plu") or
                    item.get("product_code") or "") or None,
            item_description=str(item.get("description")
                                 or item.get("name") or "") or None,
            quantity=qty,
            amount=amount,
            currency=txn.currency,
            raw_payload={**raw_payload, "item": item},
        ))
    return out


def ingest_tillshield_batch(session: Session,
                            *,
                            cfg,
                            batch: TillShieldBatch,
                            source_ip: Optional[str] = None,
                            resolve_camera=None) -> dict:
    """Translate + ingest. Returns an HTTP-friendly summary.

    ``resolve_camera`` (optional) is forwarded to ``ingest_batch`` for
    workstation-aware case routing. When omitted (the push endpoints'
    default) the legacy store-default camera is used.
    """
    accepted = _accepted_event_types(cfg)
    line_level = _line_level_enabled(cfg)
    ignore_unknown = _ignore_unknown(cfg)
    pos_tz = _resolve_pos_timezone(cfg)

    summary = {
        "events_inserted": 0,
        "events_already_present": 0,
        "cases_created": 0,
        "case_ids": [],
        "ignored_non_return_events": 0,
        "ignored_unmapped_workstation_events": 0,
        "errors": [],
    }

    pos_events: list[PosEventIn] = []
    raw_for_audit: list[dict] = []
    for idx, txn in enumerate(batch.events):
        normalised_type = _normalise_event_type(txn.transaction_type)
        if normalised_type not in accepted:
            summary["ignored_non_return_events"] += 1
            if not ignore_unknown:
                summary["errors"].append({
                    "index": idx,
                    "transaction_id": txn.transaction_id,
                    "error": (f"transaction_type {normalised_type!r} not "
                              f"in accepted set {sorted(accepted)}"),
                })
            continue
        try:
            events = normalise_to_pos_events(
                txn, line_level=line_level, source_ip=source_ip,
                pos_tz=pos_tz)
        except Exception as exc:
            summary["errors"].append({
                "index": idx,
                "transaction_id": txn.transaction_id,
                "error": f"normalise failed: {exc}",
            })
            continue
        pos_events.extend(events)
        raw_for_audit.append({
            "transaction_id": txn.transaction_id,
            "normalised_event_type": normalised_type,
            "produced_pos_events": len(events),
        })

    if not pos_events:
        return summary

    pos_batch = PosBatchIn(
        source_system="tillshield_agent",
        store_id=pos_events[0].store_id,
        received_at=datetime.now(timezone.utc),
        events=pos_events,
        raw_payload={"tillshield_batch": raw_for_audit},
    )
    try:
        result = ingest_batch(session, pos_batch,
                              resolve_camera=resolve_camera)
    except ValueError as exc:
        summary["errors"].append({"error": str(exc)})
        return summary

    summary["events_inserted"] = result.get("events_inserted", 0)
    summary["events_already_present"] = result.get(
        "events_already_present", 0)
    summary["cases_created"] = result.get("cases_created", 0)
    summary["ignored_unmapped_workstation_events"] = result.get(
        "ignored_unmapped_workstation_events", 0)

    # Surface the case IDs we just opened so the calling agent can wire
    # them into its own audit trail.
    if summary["cases_created"]:
        from sqlalchemy import select
        from db.models import Case, PosEvent
        natural_keys = [(e.store_id, e.terminal_id,
                         e.transaction_id, e.line_id)
                        for e in pos_events]
        for store_id, term, txn_id, line in natural_keys:
            ev = session.execute(
                select(PosEvent).where(
                    PosEvent.store_id == store_id,
                    PosEvent.terminal_id == term,
                    PosEvent.transaction_id == txn_id,
                    PosEvent.line_id == line,
                )).scalar_one_or_none()
            if ev is None:
                continue
            case = session.execute(
                select(Case).where(Case.pos_event_id == ev.id)
            ).scalar_one_or_none()
            if case is not None:
                summary["case_ids"].append(case.id)
    return summary
