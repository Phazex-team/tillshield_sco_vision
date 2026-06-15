"""Artifact write helpers — small wrappers used by perception + reasoning."""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from db.models import Artifact


def register_artifact(session: Session,
                      *,
                      case_id: str,
                      artifact_type: str,
                      uri: str,
                      sha256: Optional[str] = None,
                      mime_type: Optional[str] = None,
                      frame_ts=None,
                      frame_idx: Optional[int] = None,
                      metadata: Optional[dict] = None) -> str:
    row = Artifact(
        case_id=case_id,
        artifact_type=artifact_type,
        uri=uri,
        sha256=sha256,
        mime_type=mime_type,
        frame_ts=frame_ts,
        frame_idx=frame_idx,
        artifact_metadata=metadata,
    )
    session.add(row)
    session.flush()
    return row.id
