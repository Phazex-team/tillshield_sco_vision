"""Audit-log writer.

Centralises every audit_log row insertion so the API + workers don't
each grow their own boilerplate. ``record`` is a single function the
rest of the codebase calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.orm import Session

from db.models import AuditLog


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def record(session: Session,
           *,
           action: str,
           entity_type: Optional[str] = None,
           entity_id: Optional[str] = None,
           actor_id: Optional[str] = None,
           actor_type: Optional[str] = "system",
           before: Optional[dict] = None,
           after: Optional[dict] = None,
           ip: Optional[str] = None,
           user_agent: Optional[str] = None) -> str:
    """Insert one ``audit_log`` row and return its id.

    The caller is responsible for committing the session — audit writes
    should land in the same transaction as the action they describe.
    """
    row = AuditLog(
        actor_id=actor_id,
        actor_type=actor_type,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        before_json=_jsonable(before) if before is not None else None,
        after_json=_jsonable(after) if after is not None else None,
        ip=ip,
        user_agent=user_agent,
    )
    session.add(row)
    session.flush()
    return row.id
