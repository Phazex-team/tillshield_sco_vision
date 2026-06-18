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
    connect_args = (
        # ``timeout`` is the sqlite3 driver-level busy timeout (seconds):
        # a writer waits up to this long for a competing lock to clear
        # instead of failing immediately with "database is locked". The
        # app runs a background poller thread (TillShield) writing
        # concurrently with request handlers, so this is required.
        {"check_same_thread": False, "timeout": 30}
        if url.startswith("sqlite") else {}
    )
    _ENGINE = create_engine(url, future=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        @event.listens_for(_ENGINE, "connect")
        def _sqlite_pragmas(dbapi_conn, _conn_record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            # WAL lets readers and a single writer proceed concurrently
            # and is the single biggest fix for "database is locked"
            # under poller-thread + request-handler write contention.
            cur.execute("PRAGMA journal_mode=WAL")
            # Belt-and-braces busy timeout at the SQLite level (ms), in
            # addition to the driver ``timeout`` above.
            cur.execute("PRAGMA busy_timeout=30000")
            # Safe to relax durability under WAL; big throughput win for
            # the poller's many small commits.
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    # ``expire_on_commit=False`` keeps ORM instances usable after a
    # commit. ``analyze_case`` commits at several checkpoints mid-run so
    # the SQLite write lock is not held across slow NVR/ffmpeg/model
    # work; without this, every attribute access after such a commit
    # would re-query and the long analysis would still serialise reads.
    _SESSION_FACTORY = sessionmaker(bind=_ENGINE, autoflush=False,
                                    autocommit=False, future=True,
                                    expire_on_commit=False)
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
