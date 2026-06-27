"""SQLite-backed smoke test for the SQLAlchemy schema.

Verifies models import, ``create_all`` works against a temporary SQLite
file, and the natural keys behave as expected.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_create_all_and_insert_pos_event(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path/'t.sqlite'}")
    # Force singletons to rebuild against the temp URL.
    import db.session as s
    s._ENGINE = None
    s._SESSION_FACTORY = None
    s.init_schema()
    from db.models import Case, PosEvent
    from datetime import datetime
    with s.get_sessionmaker()() as session:
        ev = PosEvent(
            store_id="store_1",
            terminal_id="t1",
            transaction_id="txn-100",
            line_id="L1",
            event_type="SALE",
            pos_event_at=datetime(2026, 6, 15, 14, 0, 0),
        )
        session.add(ev)
        session.flush()
        case = Case(pos_event_id=ev.id, camera_id="cam_01", status="OPEN")
        session.add(case)
        session.commit()
        # Natural-key duplicate must violate the unique constraint.
        dup = PosEvent(
            store_id="store_1",
            terminal_id="t1",
            transaction_id="txn-100",
            line_id="L1",
            event_type="SALE",
            pos_event_at=datetime(2026, 6, 15, 14, 0, 0),
        )
        session.add(dup)
        import pytest
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            session.commit()
