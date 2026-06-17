"""Disk / storage status endpoint + retention cleanup endpoints.

The cleanup endpoints reuse ``app.storage_guard.run_cleanup`` so the
"never delete linked segments" invariant is enforced by the same code
that the recorder/retention test suite already exercises. The execute
route is gated by the existing admin-token mechanism.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request


log = logging.getLogger(__name__)


router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/disk")
def disk() -> dict:
    from app.storage_guard import disk_status
    from db.session import get_sessionmaker
    SM = get_sessionmaker()
    with SM() as s:
        return disk_status(s)


@router.post("/cleanup/dry-run")
def cleanup_dry_run() -> dict:
    """Compute candidate expired-unlinked segments WITHOUT touching disk.

    Always safe — never accepts an admin token, never writes, never
    deletes. Returns the same shape the execute call would, with
    ``dry_run: true``."""
    from app.storage_guard import run_cleanup
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        result = run_cleanup(s, execute=False)
        # No DB writes happen in dry-run, but rollback to be explicit.
        s.rollback()
        return result


@router.post("/cleanup/execute")
def cleanup_execute(request: Request,
                    x_phazex_admin_token: Optional[str] = Header(default=None),
                    ) -> dict:
    """Delete expired unlinked raw segments. Requires the admin token
    when one is configured (``ADMIN_EDIT_TOKEN`` env or
    ``config.yaml.admin.edit_token``). Linked-to-case segments are
    NEVER touched — the guarantee comes from
    ``identify_expired_unlinked_segments``. Every execute is audited."""
    from app.api.admin import _check_admin_token
    from app.storage_guard import run_cleanup
    from db.session import get_sessionmaker
    from app import audit

    _check_admin_token(x_phazex_admin_token)

    SM = get_sessionmaker()
    with SM() as s:
        result = run_cleanup(s, execute=True)
        # Audit BEFORE commit so a failure to write the audit row aborts
        # the deletion via the same transaction.
        audit.record(
            s,
            action="storage.cleanup_executed",
            entity_type="storage", entity_id="cleanup",
            actor_type="admin_api",
            after={
                "deleted_rows": result.get("deleted_rows"),
                "deleted_files_count": len(result.get("deleted_files") or []),
                "failed_count": len(result.get("failed") or []),
            },
            ip=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
        return result
