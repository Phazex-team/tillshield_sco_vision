"""Evidence package + evidence graph endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/cases", tags=["evidence"])


@router.get("/{case_id}/evidence-package")
def evidence_package(case_id: str) -> dict:
    from db.session import get_sessionmaker
    from evidence.package import latest_package_for_case

    SM = get_sessionmaker()
    with SM() as s:
        pkg = latest_package_for_case(s, case_id)
        if pkg is None:
            raise HTTPException(status_code=404,
                                detail="evidence package not yet produced")
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
