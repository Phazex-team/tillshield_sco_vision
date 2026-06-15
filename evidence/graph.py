"""Minimal evidence-graph adapter.

The production graph model uses ``evidence_nodes`` + ``evidence_edges``
(PRODUCTION_SPEC §13). For phase-1 we project the existing relational
tables (POS event, case, video window, segments, artifacts, VLM runs,
review actions) into the node/edge view callers expect, so the API
returns a useful graph without requiring a separate write path.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import (
    Artifact,
    Case,
    PosEvent,
    ReviewAction,
    VideoSegment,
    VideoWindow,
    VlmRun,
)


def graph_for_case(session: Session, case_id: str) -> dict:
    """Return a ``{nodes, edges}`` projection for ``case_id``."""
    case = session.get(Case, case_id)
    if case is None:
        return {"nodes": [], "edges": []}

    nodes = []
    edges = []

    def add_node(node_type: str, ref_id: str, payload: Optional[dict] = None,
                 label: Optional[str] = None):
        nodes.append({
            "id": f"{node_type}:{ref_id}",
            "node_type": node_type,
            "ref_id": ref_id,
            "label": label or node_type,
            "payload": payload or {},
        })

    def add_edge(src: str, dst: str, edge_type: str,
                 payload: Optional[dict] = None):
        edges.append({
            "src": src, "dst": dst,
            "edge_type": edge_type,
            "payload": payload or {},
        })

    case_node = f"CASE:{case.id}"
    add_node("CASE", case.id, payload={
        "camera_id": case.camera_id, "status": case.status,
        "outcome": case.outcome, "risk_score": case.risk_score,
    })

    if case.pos_event_id:
        pos = session.get(PosEvent, case.pos_event_id)
        if pos:
            add_node("POS_EVENT", pos.id, payload={
                "transaction_id": pos.transaction_id,
                "event_type": pos.event_type,
            })
            add_edge(f"POS_EVENT:{pos.id}", case_node, "LINKED_TO_TRANSACTION")

    windows = session.execute(
        select(VideoWindow).where(VideoWindow.case_id == case.id)
    ).scalars().all()
    for w in windows:
        add_node("VIDEO_WINDOW", w.id, payload={
            "status": w.status,
            "requested_start_at": w.requested_start_at.isoformat()
                if w.requested_start_at else None,
            "requested_end_at": w.requested_end_at.isoformat()
                if w.requested_end_at else None,
        })
        add_edge(case_node, f"VIDEO_WINDOW:{w.id}", "HAS_WINDOW")
        for seg_id in (w.segment_ids or []):
            seg = session.get(VideoSegment, seg_id)
            if seg:
                add_node("VIDEO_SEGMENT", seg.id, payload={
                    "path": seg.path, "sha256": seg.sha256,
                })
                add_edge(f"VIDEO_WINDOW:{w.id}",
                         f"VIDEO_SEGMENT:{seg.id}", "COVERS_SEGMENT")

    arts = session.execute(
        select(Artifact).where(Artifact.case_id == case.id)
    ).scalars().all()
    for a in arts:
        add_node("ARTIFACT", a.id, payload={
            "artifact_type": a.artifact_type, "uri": a.uri,
            "sha256": a.sha256,
        }, label=a.artifact_type)
        add_edge(case_node, f"ARTIFACT:{a.id}", "HAS_ARTIFACT")

    runs = session.execute(
        select(VlmRun).where(VlmRun.case_id == case.id)
    ).scalars().all()
    for r in runs:
        add_node("VLM_CLAIM", r.id, payload={
            "provider": r.provider, "status": r.status,
            "prompt_version": r.prompt_version,
        })
        add_edge(case_node, f"VLM_CLAIM:{r.id}", "SUPPORTS_CLAIM")

    revs = session.execute(
        select(ReviewAction).where(ReviewAction.case_id == case.id)
    ).scalars().all()
    for r in revs:
        add_node("REVIEW_ACTION", r.id, payload={
            "action": r.action, "outcome": r.outcome,
        })
        add_edge(case_node, f"REVIEW_ACTION:{r.id}", "REVIEWED_AS")

    return {"nodes": nodes, "edges": edges}
