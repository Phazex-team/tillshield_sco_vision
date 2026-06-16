"""Parity check between SQLAlchemy ORM tables and Postgres migration SQL.

Every ``Base.metadata.tables`` entry must have a matching
``CREATE TABLE`` in ``db/migrations/0001_core.sql``. Catches the
specific gap the previous checkpoint missed: new ORM tables landing
without a Postgres migration update.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _migration_text() -> str:
    return (ROOT / "db" / "migrations" / "0001_core.sql").read_text()


def _migration_table_names() -> set[str]:
    src = _migration_text()
    return set(re.findall(
        r"CREATE TABLE IF NOT EXISTS\s+(\w+)", src, re.IGNORECASE))


def test_every_orm_table_has_a_migration_create():
    from db.models import Base
    orm_tables = set(Base.metadata.tables.keys())
    mig_tables = _migration_table_names()
    missing = orm_tables - mig_tables
    assert not missing, (
        "Postgres migration is missing CREATE TABLE for the following "
        f"ORM tables: {sorted(missing)}"
    )


def test_perception_tables_present_in_migration():
    """Pin the specific gap the reviewer caught."""
    mig = _migration_table_names()
    for required in ("detections", "tracks", "track_observations",
                     "keyframes", "ocr_results"):
        assert required in mig, \
            f"migration is missing {required!r}"
