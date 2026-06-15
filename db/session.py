"""SQLAlchemy 2.x engine + session factory.

Defaults to SQLite (file in the project root) for fast local iteration;
flip ``DATABASE_URL`` to a Postgres URL to switch with no code change.

Heavy production hardening (pgcrypto UUIDs, JSONB indexes) lives in the
SQL migration files under ``db/migrations/`` — the SQLAlchemy models in
``db/models.py`` are designed to be schema-compatible with that target.
"""
from __future__ import annotations

from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import load_config


_ENGINE: Engine | None = None
_SESSION_FACTORY: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    global _ENGINE, _SESSION_FACTORY
    if _ENGINE is not None:
        return _ENGINE
    cfg = load_config()
    url = cfg.database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    _ENGINE = create_engine(url, future=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        @event.listens_for(_ENGINE, "connect")
        def _enable_fk(dbapi_conn, _conn_record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, autoflush=False,
                                    autocommit=False, future=True)
    return _ENGINE


def get_sessionmaker() -> sessionmaker[Session]:
    get_engine()
    assert _SESSION_FACTORY is not None
    return _SESSION_FACTORY


def session_scope() -> Iterator[Session]:
    sm = get_sessionmaker()
    s = sm()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init_schema() -> None:
    """Create tables from the SQLAlchemy models. Idempotent.

    Production deploys should use ``db/migrations/*.sql`` instead; this
    helper exists so SQLite-backed dev/test runs don't need Alembic.
    """
    from db import models  # noqa: F401  (registers tables on Base)
    from db.models import Base
    Base.metadata.create_all(bind=get_engine())
