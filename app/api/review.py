"""Reviewer action endpoint."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel


router = APIRouter(prefix="/cases", tags=["review"])


VALID_REVIEWER_ACTIONS = {
    "verified_basket_match",
    # Legacy alias accepted for historical rows / old clients.
    "verified_physical_return",
    "needs_review",
    "high_risk_review",
    "invalid_video",
    "camera_blind_spot",
    "pos_camera_mismatch",
    "reprocess_requested",
}


VALID_OUTCOMES = {
    "VERIFIED", "REVIEW", "HIGH_RISK_REVIEW", "INVALID_VIDEO",
}


class ReviewActionBody(BaseModel):
    reviewer_id: Optional[str] = None
    action: str
    outcome: Optional[str] = None
    notes: Optional[str] = None
    labels: Optional[dict] = None


@router.post("/{case_id}/review-actions", status_code=201)
def submit_action(case_id: str, body: ReviewActionBody,
                  request: Request) -> dict:
    if body.action not in VALID_REVIEWER_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"action must be one of {sorted(VALID_REVIEWER_ACTIONS)}",
        )
    if body.outcome and body.outcome not in VALID_OUTCOMES:
        raise HTTPException(
            status_code=400,
            detail=f"outcome must be one of {sorted(VALID_OUTCOMES)}",
        )

    from app import audit
    from db.models import Case, ReviewAction
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        case = s.get(Case, case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")

        before = {"status": case.status, "outcome": case.outcome}
        ra = ReviewAction(
            case_id=case.id,
            reviewer_id=body.reviewer_id,
            action=body.action,
            outcome=body.outcome,
            notes=body.notes,
            labels=body.labels,
        )
        s.add(ra)

        # The reviewer's action moves the case to IN_REVIEW or CLOSED
        # depending on the action. Closure requires an outcome.
        if body.action == "reprocess_requested":
            case.status = "REPROCESSING"
        elif body.outcome:
            case.outcome = body.outcome
            case.status = "CLOSED"
        else:
            case.status = "IN_REVIEW"

        s.flush()

        audit.record(
            s,
            action="case.review_action",
            entity_type="case", entity_id=case.id,
            actor_id=body.reviewer_id,
            actor_type="reviewer",
            before=before,
            after={"status": case.status, "outcome": case.outcome,
                   "review_action_id": ra.id, "action": body.action,
                   "labels": body.labels},
            ip=_client_ip(request),
            user_agent=request.headers.get("user-agent"),
        )
        s.commit()
        return {
            "case_id": case.id,
            "status": case.status,
            "outcome": case.outcome,
            "review_action_id": ra.id,
        }


def _client_ip(request: Request) -> Optional[str]:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None
