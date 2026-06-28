"""Wire-level schemas for POS ingest.

Kept dataclass-light so the ingest path doesn't depend on Pydantic v1/v2
specifics (the legacy MVP doesn't currently use Pydantic). The FastAPI
router converts incoming JSON into these and then calls
``pos.ingest.ingest_batch``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# Compatibility constant for callers that still import it. Active validation
# is config-driven via pos.event_normalizer.normalize_event_type, but in this
# SCO-only copy the public accepted family is checkout aliases.
VALID_EVENT_TYPES = {"SALE", "SCO_SALE", "CHECKOUT"}


@dataclass
class PosEventIn:
    store_id: str
    terminal_id: str
    transaction_id: str
    line_id: str
    event_type: str
    pos_event_at: datetime
    staff_id: Optional[str] = None
    sku: Optional[str] = None
    item_description: Optional[str] = None
    quantity: Optional[float] = None
    amount: Optional[float] = None
    currency: Optional[str] = None
    raw_payload: Optional[dict] = None

    def validate(self) -> None:
        if not self.store_id or not self.terminal_id:
            raise ValueError("store_id and terminal_id are required")
        if not self.transaction_id or not self.line_id:
            raise ValueError("transaction_id and line_id are required")
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("event_type is required and must be a string")
        # Active acceptance is config-driven and happens at the boundary
        # (pos.event_normalizer.normalize_event_type in app/api/pos.py and
        # in the TillShield adapter). Dataclass-level acceptance was
        # dropped because it required loading app.config inside this
        # data carrier — too coupled for unit tests that construct
        # PosEventIn directly. Bad types still cannot open a case
        # because pos.ingest gates case creation on
        # pos.event_normalizer.case_opening_types().
        if not isinstance(self.pos_event_at, datetime):
            raise ValueError("pos_event_at must be a datetime")


@dataclass
class PosBatchIn:
    source_system: str
    store_id: str
    received_at: datetime
    batch_start_at: Optional[datetime] = None
    batch_end_at: Optional[datetime] = None
    events: list[PosEventIn] = field(default_factory=list)
    raw_payload: Optional[dict] = None

    def validate(self) -> None:
        if not self.source_system:
            raise ValueError("source_system is required")
        if not self.store_id:
            raise ValueError("store_id is required")
        if not self.events:
            raise ValueError("batch must contain at least one event")
        for e in self.events:
            e.validate()
