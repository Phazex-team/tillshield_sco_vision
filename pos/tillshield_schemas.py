"""Wire-level schemas for the TillShield POS integration.

Mirrors the ``tillshield_agent.pos_api_transactions`` row shape so the
ingest endpoint can accept the table object verbatim. Field names match
the dump exactly; the normaliser maps them into the app's
``pos.schemas.PosEventIn``.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TillShieldTransaction(BaseModel):
    """One row from ``tillshield_agent.pos_api_transactions``."""
    transaction_id: str
    transaction_date: datetime
    transaction_end_date: Optional[datetime] = None
    transaction_type: str
    reference_id: Optional[str] = None
    store_id: str
    workstation_id: str
    operator_id: Optional[str] = None
    cashier_name: Optional[str] = None
    currency: Optional[str] = None
    total_items: Optional[int] = None
    total_amount: Optional[float] = None
    items: Optional[list[dict]] = Field(default_factory=list)
    payload: Optional[dict] = None
    source_ip: Optional[str] = None
    received_at: Optional[datetime] = None


class TillShieldBatch(BaseModel):
    source_system: str = "tillshield_agent"
    events: list[TillShieldTransaction]
