"""Disk-status + retention policy helpers.

* ``disk_status()`` reports total / used / free / storage-root-size /
  oldest-raw-segment / candidate-deletable-segments.
* ``identify_expired_unlinked_segments()`` returns the list of raw
  ``video_segments`` rows that are older than the retention window AND
  not referenced by any case / video_window / artifact. Linked segments
  are NEVER returned.
* ``run_cleanup(execute=False)`` orchestrates the cleanup; dry-run by
  default.
* ``low_disk_state()`` returns True when the storage root has less than
  ``storage.min_free_disk_gb`` available; the recorder consults this
  but the API and reviewer UI do not.
"""
from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Artifact, VideoSegment, VideoWindow


log = logging.getLogger(__name__)


def _storage_root() -> Path:
    from app.config import load_config
    return load_config().storage_root


def _retention_hours() -> int:
    from app.config import load_config
    return int(load_config().raw.get("storage", {}).get(
        "raw_segment_retention_hours", 10))


def _min_free_disk_gb() -> int:
    from app.config import load_config
    return int(load_config().raw.get("storage", {}).get(
        "min_free_disk_gb", 25))


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total


# ---------------------------------------------------------------------------
# Disk status
# ---------------------------------------------------------------------------

def disk_status(session: Optional[Session] = None) -> dict:
    root = _storage_root()
    root.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(str(root))

    oldest_raw_segment_at = None
    expired_count = 0
    if session is not None:
        oldest = (session.query(VideoSegment)
                  .order_by(VideoSegment.start_at.asc()).first())
        if oldest:
            oldest_raw_segment_at = oldest.start_at
        expired_count = len(identify_expired_unlinked_segments(session))

    min_free_gb = _min_free_disk_gb()
    free_gb = free / (1024 ** 3)
    return {
        "storage_root": str(root),
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "free_gb": round(free_gb, 2),
        "storage_root_size_bytes": _dir_size(root),
        "min_free_gb": min_free_gb,
        "low_disk_state": free_gb < min_free_gb,
        "retention_hours": _retention_hours(),
        "oldest_raw_segment_at": (oldest_raw_segment_at.isoformat()
                                   if oldest_raw_segment_at else None),
        "expired_unlinked_segments": expired_count,
    }


# ---------------------------------------------------------------------------
# Retention policy
# ---------------------------------------------------------------------------

def identify_expired_unlinked_segments(session: Session) -> list[VideoSegment]:
    """Return raw segments that are eligible for deletion.

    A segment is eligible iff:
      * its ``end_at`` is older than ``raw_segment_retention_hours``,
        AND
      * no ``video_windows.segment_ids`` references it (i.e. no case
        ever resolved against it), AND
      * no ``artifacts.uri`` references its on-disk path (defence
        against operator-attached evidence).

    Linked segments are NEVER returned — the cleanup function relies on
    this invariant.
    """
    cutoff = _utc_now_naive() - timedelta(hours=_retention_hours())
    candidates = session.execute(
        select(VideoSegment).where(VideoSegment.end_at < cutoff)
    ).scalars().all()
    if not candidates:
        return []

    cand_ids = {c.id for c in candidates}
    cand_paths = {c.path for c in candidates if c.path}

    linked_ids: set[str] = set()
    windows = session.execute(select(VideoWindow)).scalars().all()
    for w in windows:
        for sid in (w.segment_ids or []):
            if sid in cand_ids:
                linked_ids.add(sid)

    artifacts = session.execute(select(Artifact)).scalars().all()
    linked_by_path: set[str] = set()
    for a in artifacts:
        if a.uri and a.uri in cand_paths:
            linked_by_path.add(a.uri)

    return [c for c in candidates
            if c.id not in linked_ids
            and (c.path or "") not in linked_by_path]


def run_cleanup(session: Session, *, execute: bool = False) -> dict:
    """Delete expired unlinked segments (files + DB rows).

    When ``execute=False`` (default) the function returns the list of
    segments that WOULD be deleted but does nothing on disk. When
    ``execute=True`` the files are unlinked and the DB rows are
    deleted in the same transaction; the caller commits.
    """
    expired = identify_expired_unlinked_segments(session)
    deleted_files: list[str] = []
    failed: list[dict] = []
    if not execute:
        return {
            "dry_run": True,
            "candidates": [{"id": s.id, "path": s.path,
                            "start_at": s.start_at.isoformat()
                                if s.start_at else None}
                            for s in expired],
            "deleted_files": [],
            "deleted_rows": 0,
            "failed": failed,
        }
    for seg in expired:
        try:
            if seg.path and os.path.exists(seg.path):
                os.remove(seg.path)
                deleted_files.append(seg.path)
        except OSError as exc:
            failed.append({"id": seg.id, "path": seg.path,
                           "error": str(exc)})
            continue
        session.delete(seg)
    return {
        "dry_run": False,
        "candidates": [{"id": s.id, "path": s.path} for s in expired],
        "deleted_files": deleted_files,
        "deleted_rows": len(deleted_files),
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Low-disk state
# ---------------------------------------------------------------------------

def low_disk_state() -> bool:
    """Recorder consults this before opening a new chunk. The API and
    reviewer UI do not call it — they stay alive regardless."""
    root = _storage_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        _total, _used, free = shutil.disk_usage(str(root))
    except OSError:
        return False
    return (free / (1024 ** 3)) < _min_free_disk_gb()
