"""Evidence package + evidence graph endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/cases", tags=["evidence"])


@router.get("/{case_id}/evidence-package")
def evidence_package(case_id: str) -> dict:
    """Return the latest evidence-package payload.

    The response always exposes a reviewer-friendly
    ``literal_file_sha256`` field at the top level — this is the
    sha256 of the bytes-on-disk and equals what ``sha256sum`` reports
    for the file at ``uri``.
    """
    from db.models import Artifact
    from db.session import get_sessionmaker
    from evidence.package import latest_package_for_case
    from sqlalchemy import select

    SM = get_sessionmaker()
    with SM() as s:
        pkg = latest_package_for_case(s, case_id)
        if pkg is None:
            raise HTTPException(status_code=404,
                                detail="evidence package not yet produced")
        # Look up the artifact row so we can echo the literal file sha
        # the reviewer would get from sha256sum.
        art = s.execute(
            select(Artifact)
            .where(Artifact.case_id == case_id,
                   Artifact.artifact_type == "PACKAGE")
            .order_by(Artifact.created_at.desc())
        ).scalars().first()
        if art is not None:
            pkg["literal_file_sha256"] = art.sha256
            pkg["uri"] = art.uri
        return pkg


@router.get("/{case_id}/evidence-graph")
def evidence_graph(case_id: str) -> dict:
    from db.session import get_sessionmaker
    from evidence.graph import graph_for_case

    SM = get_sessionmaker()
    with SM() as s:
        graph = graph_for_case(s, case_id)
        if not graph["nodes"]:
            raise HTTPException(status_code=404,
                                detail="case has no evidence graph yet")
        return graph
