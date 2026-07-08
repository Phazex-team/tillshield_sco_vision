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
        # Surface the INDEPENDENT Falcon audit-zone item count (computed at
        # analysis time, stored on the latest vlm_run manifest; never a VLM
        # output) so the reviewer UI can show FL vs VLM vs POS side by side.
        from db.models import VlmRun
        vr = s.execute(
            select(VlmRun).where(VlmRun.case_id == case_id)
            .order_by(VlmRun.started_at.desc())
        ).scalars().first()
        if vr is not None and isinstance(vr.input_manifest, dict):
            flc = vr.input_manifest.get("fl_audit_zone_count")
            if flc is not None and isinstance(pkg.get("perception"), dict):
                pkg["perception"]["audit_zone_item_count"] = flc
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


@router.get("/{case_id}/processing-timings")
def case_processing_timings(case_id: str) -> dict:
    """Return the *final* processing timings for ``case_id`` straight
    from the latest ``VlmRun.input_manifest`` row.

    Why a dedicated endpoint exists:
      The immutable evidence package on disk is hashed once, BEFORE
      ``package_write_ms`` could be measured. The same VlmRun row in
      the DB is then re-assigned with the final timings (including
      ``package_write_ms`` and a post-package ``total_ms``). Reading
      the package file would therefore show stale pre-package totals.
      This endpoint reads the DB so the reviewer UI can render the
      *real* end-to-end totals without rewriting (and re-hashing) the
      immutable package.

    Timing-focused response shape (intentionally minimal — no
    ``input_manifest``, no ``provider_metadata``, no ``usage``, no
    base64 frames)::

        {
          "case_id":               str,
          "provider":              str | null,
          "model_name":            str | null,
          "model_snapshot":        str | null,
          "status":                str | null,
          "latency_ms":            int | null,
          "error":                 str | null,
          "processing_timings_ms": dict | null,
          "source":                "vlm_runs.input_manifest"
        }
    """
    from sqlalchemy import select

    from db.models import Case, VlmRun
    from db.session import get_sessionmaker

    SM = get_sessionmaker()
    with SM() as s:
        if s.get(Case, case_id) is None:
            raise HTTPException(status_code=404,
                                detail=f"case {case_id!r} not found")
        run = s.execute(
            select(VlmRun)
            .where(VlmRun.case_id == case_id)
            # Newest first by finished_at then started_at; either may
            # be NULL on a still-running/old row, so we fall back to
            # the auto-generated ``id`` last for a deterministic order.
            .order_by(VlmRun.finished_at.desc().nullslast(),
                      VlmRun.started_at.desc().nullslast(),
                      VlmRun.id.desc())
            .limit(1)
        ).scalars().first()
        if run is None:
            raise HTTPException(
                status_code=404,
                detail="no vlm_run yet for case")

        manifest = run.input_manifest if isinstance(
            run.input_manifest, dict) else {}
        timings = manifest.get("processing_timings_ms")
        if not isinstance(timings, dict):
            timings = None

        return {
            "case_id": case_id,
            "provider": run.provider,
            "model_name": run.model_name,
            "model_snapshot": run.model_snapshot,
            "status": run.status,
            "latency_ms": run.latency_ms,
            "error": run.error,
            "processing_timings_ms": timings,
            "source": "vlm_runs.input_manifest",
        }
