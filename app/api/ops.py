"""Operations / pipeline-status aggregator endpoint.

``GET /api/v1/ops/status`` collects READ-ONLY status from existing
modules so an operator can see where the pipeline is failing without
inventing new state. Every field is sourced from a real backend
function — when the value cannot be derived (e.g. no segments yet for
a camera), the response says ``"unknown"`` with a short reason rather
than faking ``OK``.

Status pill rules per panel (kept stable so the UI can render
consistently):

  * ``OK``       — the subsystem reports healthy.
  * ``WARNING``  — degraded but still serving (e.g. low disk, memory
                   soft limit, TillShield validation issue,
                   stale camera segment).
  * ``ERROR``    — clearly broken (vLLM unreachable, hard memory
                   limit, provider chain build failure, low_disk_state).
  * ``UNKNOWN``  — the subsystem state can't be derived from the
                   running code (no segments yet, polling never ran,
                   provider missing health, etc.).
"""
from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional


from fastapi import APIRouter


log = logging.getLogger(__name__)


router = APIRouter(prefix="/ops", tags=["ops"])


OK = "OK"
WARNING = "WARNING"
ERROR = "ERROR"
UNKNOWN = "UNKNOWN"

# Cameras whose newest VideoSegment is older than this are flagged as
# stale (the recorder is the only thing that lands segments).
SEGMENT_STALE_AFTER_SEC = 600

# Counts beyond which the cases pill is degraded.
OPEN_CASES_WARN = 25
REPROCESSING_CASES_WARN = 5


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso_or_none(dt) -> Optional[str]:
    if dt is None:
        return None
    return dt.isoformat() if isinstance(dt, datetime) else str(dt)


# ---------------------------------------------------------------------
# Sub-collectors. Each must NEVER raise — they catch and degrade.
# ---------------------------------------------------------------------

def _memory_panel() -> dict:
    try:
        from app.memory_guard import (
            STATE_EMERGENCY, STATE_HARD, STATE_NORMAL, STATE_SOFT,
            get_policy,
        )
        status = get_policy().poll()
        snap = asdict(status)
        st = snap.get("state")
        if st == STATE_NORMAL:
            pill = OK
        elif st == STATE_SOFT:
            pill = WARNING
        elif st in (STATE_HARD, STATE_EMERGENCY):
            pill = ERROR
        else:
            pill = UNKNOWN
        snap["pill"] = pill
        return snap
    except Exception as exc:
        return {"pill": UNKNOWN, "error": f"{type(exc).__name__}: {exc}",
                "detail": "memory guard unavailable"}


def _storage_panel(session) -> dict:
    try:
        from app.storage_guard import disk_status
        d = disk_status(session)
        if d.get("low_disk_state"):
            pill = ERROR
        elif d.get("expired_unlinked_segments", 0) > 0:
            pill = WARNING
        else:
            pill = OK
        d["pill"] = pill
        return d
    except Exception as exc:
        return {"pill": UNKNOWN, "error": f"{type(exc).__name__}: {exc}",
                "detail": "disk status unavailable"}


def _tillshield_panel(session, cfg) -> dict:
    try:
        from pos.tillshield_poll import (
            load_poll_config, read_status, validate_poll_config,
        )
        pc = load_poll_config(cfg)
        out: dict = {
            "enabled": bool(pc.enabled),
            "base_url": pc.base_url,
            "poll_every_seconds": pc.poll_every_seconds,
            "allowed_workstation_ids": list(pc.allowed_workstation_ids),
            "workstation_camera_map": dict(pc.workstation_camera_map),
            "validation_issues": list(validate_poll_config(cfg)),
        }
        if pc.enabled:
            try:
                out["status"] = read_status(session)
            except Exception as exc:
                out["status"] = None
                out["status_error"] = f"{type(exc).__name__}: {exc}"
        else:
            out["status"] = None

        if not pc.enabled:
            out["pill"] = UNKNOWN
            out["detail"] = "polling disabled"
            return out
        if out["validation_issues"]:
            out["pill"] = ERROR
            out["detail"] = (f"{len(out['validation_issues'])} "
                             "validation issue(s)")
            return out
        st = out.get("status") or {}
        last_poll = st.get("last_poll_at")
        last_err = st.get("last_error")
        last_ok = st.get("last_successful_poll_at")
        if last_err:
            out["pill"] = WARNING
            out["detail"] = f"last_error={last_err!r}"
        elif not last_poll:
            out["pill"] = UNKNOWN
            out["detail"] = "polling enabled but no cycle has run yet"
        elif not last_ok:
            out["pill"] = WARNING
            out["detail"] = "poll attempted but no successful cycle yet"
        else:
            out["pill"] = OK
            out["detail"] = f"last_successful_poll_at={last_ok}"
        return out
    except Exception as exc:
        return {"pill": UNKNOWN, "error": f"{type(exc).__name__}: {exc}",
                "detail": "tillshield panel unavailable"}


def _provider_chain_panel(cfg) -> dict:
    try:
        from reasoning.providers import build_active_provider
        provider = build_active_provider(cfg)
        if provider.name == "chain":
            members = [p.name for p in provider.providers]
        else:
            members = [provider.name]
        return {
            "pill": OK if members else ERROR,
            "members": members,
            "active_primary": members[0] if members else None,
            "detail": f"chain={members}",
        }
    except Exception as exc:
        return {"pill": ERROR, "members": [],
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "provider chain build failed"}


def _qwen_vllm_panel(cfg) -> dict:
    try:
        from app.startup import qwen_vllm_status
        s = qwen_vllm_status(cfg)
        s = dict(s)
        if s["backend"] != "vllm_openai":
            s["pill"] = UNKNOWN
            return s
        if not s["enabled"]:
            s["pill"] = UNKNOWN
            return s
        if s["healthy"] is True:
            s["pill"] = OK
        elif s["healthy"] is False:
            s["pill"] = ERROR
        else:
            s["pill"] = UNKNOWN
        return s
    except Exception as exc:
        return {"pill": UNKNOWN, "backend": "unknown",
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "qwen3_vl readiness check raised"}


def _gemma_panel(cfg) -> dict:
    """Reach the fallback Gemma provider's existing ``health()``."""
    try:
        from app.config import ModelConfig
        from reasoning.providers import get_provider
        gemma_cfg = cfg.models.get("gemma") if cfg else None
        if gemma_cfg is None or not gemma_cfg.enabled:
            return {"pill": UNKNOWN, "healthy": None,
                    "detail": "gemma not configured / disabled"}
        kwargs = {k: v for k, v in gemma_cfg.extra.items()}
        provider = get_provider("gemma", model_name=gemma_cfg.name,
                                enabled=True, **kwargs)
        h = provider.health()
        return {
            "pill": OK if h.healthy else WARNING,
            "healthy": bool(h.healthy),
            "detail": h.detail,
            "model": gemma_cfg.name,
        }
    except Exception as exc:
        return {"pill": UNKNOWN, "healthy": None,
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "gemma health probe raised"}


def _camera_segment_freshness(session, cfg) -> list[dict]:
    """For each configured camera, look up the newest VideoSegment row
    and report its age. No segments yet => ``UNKNOWN`` with reason."""
    out: list[dict] = []
    try:
        from db.models import VideoSegment
        from sqlalchemy import select
        for cam in (cfg.cameras or []):
            cid = cam.get("id")
            if not cid:
                continue
            latest = session.execute(
                select(VideoSegment)
                .where(VideoSegment.camera_id == cid)
                .order_by(VideoSegment.start_at.desc())
                .limit(1)
            ).scalar_one_or_none()
            rec: dict = {
                "id": cid,
                "name": cam.get("name") or cid,
                "classifier": cam.get("classifier"),
                "rtsp_configured": bool(cam.get("rtsp_url")),
            }
            if latest is None:
                rec["latest_segment_at"] = None
                rec["latest_segment_age_seconds"] = None
                rec["pill"] = UNKNOWN
                rec["detail"] = "no segments recorded yet"
                out.append(rec)
                continue
            age_seconds = (_utc_now_naive() - latest.start_at).total_seconds()
            rec["latest_segment_at"] = _iso_or_none(latest.start_at)
            rec["latest_segment_age_seconds"] = round(age_seconds, 1)
            rec["latest_segment_path_present"] = bool(latest.path)
            if age_seconds < SEGMENT_STALE_AFTER_SEC:
                rec["pill"] = OK
                rec["detail"] = "recent segment landed"
            else:
                rec["pill"] = WARNING
                rec["detail"] = (f"newest segment is "
                                 f"{int(age_seconds)}s old; recorder "
                                 "may be lagging or stopped")
            out.append(rec)
    except Exception as exc:
        out.append({"pill": UNKNOWN,
                    "error": f"{type(exc).__name__}: {exc}",
                    "detail": "segment lookup raised"})
    return out


def _cases_panel(session) -> dict:
    """Light per-status / per-outcome counts from the cases table."""
    try:
        from db.models import Case, CASE_OUTCOMES, CASE_STATUSES, VlmRun
        from sqlalchemy import func, select
        per_status: dict[str, int] = {s: 0 for s in CASE_STATUSES}
        per_outcome: dict[str, int] = {o: 0 for o in CASE_OUTCOMES}
        for st, count in session.execute(
                select(Case.status, func.count()).group_by(Case.status)).all():
            per_status[str(st)] = int(count)
        for oc, count in session.execute(
                select(Case.outcome, func.count())
                .where(Case.outcome.isnot(None))
                .group_by(Case.outcome)).all():
            per_outcome[str(oc)] = int(count)
        with_vlm_errors = int(session.execute(
            select(func.count()).select_from(VlmRun)
            .where(VlmRun.status == "FAILED")).scalar() or 0)
        opens = per_status.get("OPEN", 0)
        reprocessing = per_status.get("REPROCESSING", 0)
        if reprocessing >= REPROCESSING_CASES_WARN:
            pill = WARNING
        elif opens >= OPEN_CASES_WARN:
            pill = WARNING
        else:
            pill = OK
        return {
            "pill": pill,
            "per_status": per_status,
            "per_outcome": per_outcome,
            "vlm_runs_failed": with_vlm_errors,
        }
    except Exception as exc:
        return {"pill": UNKNOWN,
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "cases counts unavailable"}


def _collect_warnings(panels: dict) -> list[str]:
    """Flatten a small list of operator-actionable warning strings from
    the per-panel detail. Same style as startup warnings — short,
    non-blocking, one line each."""
    warnings: list[str] = []
    qwen = panels.get("qwen_vllm") or {}
    if qwen.get("pill") == ERROR:
        warnings.append(
            f"qwen3_vl vllm backend not ready: {qwen.get('detail', '')}")
    elif qwen.get("backend") == "vllm_openai" \
            and qwen.get("healthy") is False:
        warnings.append(
            f"qwen3_vl vllm backend degraded: {qwen.get('detail', '')}")
    storage = panels.get("storage") or {}
    if storage.get("low_disk_state"):
        warnings.append(
            f"low_disk_state: free {storage.get('free_gb')}G < threshold "
            f"{storage.get('min_free_gb')}G")
    if (storage.get("expired_unlinked_segments") or 0) > 0:
        warnings.append(
            f"{storage['expired_unlinked_segments']} expired unlinked "
            "raw segments eligible for cleanup")
    mem = panels.get("memory") or {}
    if mem.get("pill") in (WARNING, ERROR):
        warnings.append(f"memory state={mem.get('state')}; "
                        f"{mem.get('degraded_reason', '')}")
    ts = panels.get("tillshield") or {}
    for issue in (ts.get("validation_issues") or []):
        warnings.append(f"tillshield validation: {issue}")
    if ts.get("status") and ts["status"].get("last_error"):
        warnings.append(f"tillshield last_error: {ts['status']['last_error']}")
    for cam in (panels.get("cameras") or []):
        if cam.get("pill") == WARNING:
            warnings.append(
                f"camera {cam.get('id')}: {cam.get('detail')}")
    if (panels.get("provider_chain") or {}).get("pill") == ERROR:
        warnings.append(
            "provider chain build failed: "
            f"{panels['provider_chain'].get('detail', '')}")
    return warnings


@router.get("/status")
def ops_status() -> dict:
    """Aggregate read-only pipeline status. Never raises — every
    sub-collector degrades to ``UNKNOWN`` on failure."""
    from app.config import is_production_offline_mode, load_config
    from db.session import get_sessionmaker

    try:
        cfg = load_config()
    except Exception as exc:
        cfg = None
        cfg_err = f"{type(exc).__name__}: {exc}"
    else:
        cfg_err = None

    SM = get_sessionmaker()
    with SM() as session:
        memory = _memory_panel()
        storage = _storage_panel(session)
        tillshield = _tillshield_panel(session, cfg) if cfg else \
            {"pill": UNKNOWN, "detail": "config unavailable",
             "error": cfg_err}
        providers = _provider_chain_panel(cfg) if cfg else \
            {"pill": UNKNOWN, "detail": "config unavailable",
             "error": cfg_err}
        qwen_vllm = _qwen_vllm_panel(cfg) if cfg else \
            {"pill": UNKNOWN, "backend": "unknown",
             "detail": "config unavailable", "error": cfg_err}
        gemma = _gemma_panel(cfg) if cfg else \
            {"pill": UNKNOWN, "detail": "config unavailable",
             "error": cfg_err}
        cameras = _camera_segment_freshness(session, cfg) if cfg else []
        cases = _cases_panel(session)

        panels = {
            "memory": memory,
            "storage": storage,
            "tillshield": tillshield,
            "provider_chain": providers,
            "qwen_vllm": qwen_vllm,
            "gemma": gemma,
            "cameras": cameras,
            "cases": cases,
        }
        warnings = _collect_warnings(panels)

        return {
            "generated_at": _utc_now_naive().isoformat(),
            "production_offline_mode": is_production_offline_mode(),
            "health": {"status": "ok"},  # the API process itself is up
            "memory": memory,
            "storage": storage,
            "tillshield": tillshield,
            "provider_chain": providers,
            "qwen_vllm": qwen_vllm,
            "gemma": gemma,
            "cameras": cameras,
            "cases": cases,
            "warnings": warnings,
            "config_error": cfg_err,
        }
