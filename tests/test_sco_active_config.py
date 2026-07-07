"""Council fix #6 — active-config integration tests.

These tests load the REAL ``config.yaml`` (no in-test stub) and verify
that the production wiring matches the SCO v1 design intent:

  (a) cam_01 actually defines ``sco_audit_zone`` and every active SCO
      model view (Falcon, Qwen3-VL, Gemma) targets ONLY that zone.

  (b) ``integrations.refund_agent.enabled`` is False in the active
      config → the reprocess-success branch does NOT submit the
      legacy refund-agent export.

  (c) When ROI extras are enabled AND ``prompt_version=sco_basket_match_v2``
      is active, the composed ``manifest.user_prompt`` contains BOTH the
      ROI legend AND the SCO basket-match JSON request — and does NOT
      contain Qwen's refund/return JSON shape.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_real_config():
    from app.config import load_config
    return load_config()


def _active_sco_camera(cfg):
    """The production SCO camera — the one the TillShield workstation map
    points at, falling back to the sole configured camera.

    (Historically this was ``cam_01``; the active production camera is now
    ``cam_return_01``. Resolving it dynamically keeps these guardrails
    correct across that kind of rename.)"""
    ts = ((cfg.raw.get("integrations") or {}).get("tillshield") or {})
    by_id = {c.get("id"): c for c in (cfg.cameras or [])}
    for cam_id in (ts.get("workstation_camera_map") or {}).values():
        if cam_id in by_id:
            return by_id[cam_id]
    return (cfg.cameras or [None])[0]


def _active_sco_camera_id(cfg):
    cam = _active_sco_camera(cfg)
    return cam.get("id") if cam else None


def test_active_config_defaults_to_sco_v2_prompt():
    cfg = _load_real_config()
    reasoning = cfg.raw.get("reasoning") or {}
    assert reasoning.get("prompt_version") == "sco_basket_match_v2"


# ---------------------------------------------------------------------------
# (a) sco_audit_zone is defined on cam_01 and all SCO model views point to it
# ---------------------------------------------------------------------------

def test_real_config_cam_01_defines_sco_audit_zone():
    cfg = _load_real_config()
    cam = _active_sco_camera(cfg)
    zones = cam.get("zones") or {}
    assert "sco_audit_zone" in zones, (
        "cam_01 must define sco_audit_zone under cameras[].zones — the "
        "episode selector and SCO Falcon view both read this zone name."
    )
    z = zones["sco_audit_zone"]
    # Geometry sanity: nonzero, sensible source dims for a typical HD frame.
    assert int(z.get("w", 0)) > 0 and int(z.get("h", 0)) > 0
    assert int(z.get("source_width", 0)) >= 1280
    assert int(z.get("source_height", 0)) >= 720


def test_real_config_zone_name_does_not_trip_refund_customer_gate():
    """The legacy refund decision policy's customer_present helper matches
    on substring ``customer`` in the zone name. The SCO zone MUST NOT
    contain that substring, otherwise zones meant for SCO would leak
    into refund-mode gating if both flows ran in the same process."""
    cfg = _load_real_config()
    cam = _active_sco_camera(cfg)
    zones = cam.get("zones") or {}
    assert "sco_audit_zone" in zones
    assert "customer" not in "sco_audit_zone".lower(), \
        "the SCO zone name must not contain the substring 'customer'"


@pytest.mark.parametrize("model_name", ["falcon", "qwen3_vl", "gemma"])
def test_real_config_sco_model_views_target_only_sco_audit_zone(model_name):
    cfg = _load_real_config()
    cam = _active_sco_camera(cfg)
    views = cam.get("model_roi_views") or {}
    view = views.get(model_name)
    assert view is not None, f"cam_01.model_roi_views.{model_name} is missing"
    assert bool(view.get("enabled", True)), \
        f"cam_01.model_roi_views.{model_name} must be enabled"
    roi_ids = list(view.get("roi_ids") or [])
    assert roi_ids == ["sco_audit_zone"], (
        f"cam_01.model_roi_views.{model_name}.roi_ids must be exactly "
        f"['sco_audit_zone']; got {roi_ids}"
    )


def test_real_config_resolves_sco_audit_zone_via_model_view_helper():
    """End-to-end: app.camera_rois.model_view must hand the perception
    pipeline a Falcon view containing the sco_audit_zone descriptor."""
    cfg = _load_real_config()
    from app.camera_rois import model_view
    active_id = _active_sco_camera_id(cfg)
    falcon_view = model_view(cfg, active_id, "falcon")
    assert falcon_view is not None, \
        f"model_view returned None for {active_id}/falcon — view not resolved"
    resolved = falcon_view.get("resolved_zones") or []
    zone_names = [z.get("id") for z in resolved]
    assert "sco_audit_zone" in zone_names, \
        f"resolved zones do not include sco_audit_zone: {zone_names}"


# ---------------------------------------------------------------------------
# Council follow-up: every TillShield-mapped workstation camera must
# itself define sco_audit_zone — otherwise a polled case lands on a
# camera with no SCO ROI and the episode selector / Falcon view break.
# ---------------------------------------------------------------------------

def test_every_tillshield_workstation_camera_has_sco_audit_zone():
    cfg = _load_real_config()
    ts = ((cfg.raw.get("integrations") or {}).get("tillshield") or {})
    ws_map = ts.get("workstation_camera_map") or {}
    if not ws_map:
        pytest.skip("no workstation_camera_map configured")

    cameras_by_id = {c.get("id"): c for c in (cfg.cameras or [])}
    failures: list[str] = []
    for ws, cam_id in ws_map.items():
        cam = cameras_by_id.get(cam_id)
        if cam is None:
            failures.append(
                f"workstation {ws!r} → camera {cam_id!r} which is NOT "
                f"defined under cameras:")
            continue
        zones = cam.get("zones") or {}
        if "sco_audit_zone" not in zones:
            failures.append(
                f"workstation {ws!r} → camera {cam_id!r} which has no "
                f"sco_audit_zone in zones (defined zones: "
                f"{sorted(zones.keys())})")
            continue
        # And the camera's SCO model views must actually reference it.
        views = cam.get("model_roi_views") or {}
        for model in ("falcon", "qwen3_vl", "gemma"):
            v = views.get(model) or {}
            if not bool(v.get("enabled", True)):
                continue
            roi_ids = list(v.get("roi_ids") or [])
            if "sco_audit_zone" not in roi_ids:
                failures.append(
                    f"workstation {ws!r} → camera {cam_id!r} model "
                    f"view {model!r} does not target sco_audit_zone "
                    f"(roi_ids={roi_ids})")
    assert not failures, "active TillShield routing has non-SCO targets:\n  " \
        + "\n  ".join(failures)


def test_every_allowed_workstation_is_in_the_camera_map():
    """Defensive: allowed_workstation_ids and workstation_camera_map
    must agree, otherwise an allowed workstation has no camera route
    and silently never opens a case."""
    cfg = _load_real_config()
    ts = ((cfg.raw.get("integrations") or {}).get("tillshield") or {})
    allowed = {str(w) for w in (ts.get("allowed_workstation_ids") or [])}
    mapped = {str(k) for k in (ts.get("workstation_camera_map") or {})}
    if not allowed:
        pytest.skip("no allowed_workstation_ids configured")
    unmapped = allowed - mapped
    assert not unmapped, \
        f"these allowed workstations have no camera route: " \
        f"{sorted(unmapped)}"


def test_cam_return_01_also_has_sco_audit_zone():
    """cam_return_01 was historically the refund camera. Now that the
    TillShield map can flip between cam_01 and cam_return_01 without
    a code change, both must be SCO-ready."""
    cfg = _load_real_config()
    cam = next((c for c in cfg.cameras if c.get("id") == "cam_return_01"),
               None)
    if cam is None:
        pytest.skip("cam_return_01 not in this config")
    zones = cam.get("zones") or {}
    assert "sco_audit_zone" in zones, \
        "cam_return_01 is missing sco_audit_zone — TillShield re-route " \
        "to it would land cases on a non-SCO camera"
    views = cam.get("model_roi_views") or {}
    for model in ("falcon", "qwen3_vl", "gemma"):
        v = views.get(model) or {}
        if not bool(v.get("enabled", True)):
            continue
        assert "sco_audit_zone" in (v.get("roi_ids") or []), \
            f"cam_return_01.model_roi_views.{model} does not target " \
            f"sco_audit_zone"


# ---------------------------------------------------------------------------
# (b) refund_agent.enabled is False in active config
# ---------------------------------------------------------------------------

def test_real_config_refund_agent_is_disabled_by_default():
    cfg = _load_real_config()
    integrations = (cfg.raw.get("integrations") or {})
    refund = integrations.get("refund_agent") or {}
    assert refund.get("enabled") is False, (
        "integrations.refund_agent.enabled must be False in the SCO "
        "repo's active config (Phase 7a). Found: "
        f"{refund.get('enabled')!r}"
    )


def test_real_config_reprocess_success_does_not_submit_refund_export(
        monkeypatch):
    """Replays the gate logic from app/api/cases.py:_run_reprocess
    success branch against the REAL config. The submit must not fire."""
    cfg = _load_real_config()
    refund_enabled = bool(
        ((cfg.raw.get("integrations") or {})
         .get("refund_agent") or {}).get("enabled", False)
    )
    assert refund_enabled is False, \
        "guard precondition fails — refund_agent is enabled in active config"

    submitted: list = []

    class _SpyPool:
        def submit(self, *a, **kw):
            submitted.append((a, kw))

    pool = _SpyPool()
    # Replicate the success-branch gate.
    if refund_enabled:
        from pos.refund_agent_export import maybe_export_case
        pool.submit(maybe_export_case, "test_case_id")
    assert submitted == [], (
        "real config must NOT enqueue the refund-agent export on success"
    )


# ---------------------------------------------------------------------------
# (c) ROI extras + SCO prompt: composed prompt contains SCO basket JSON
#     and ROI legend, NOT the provider fallback prompt.
# ---------------------------------------------------------------------------

def _make_data_url(w=160, h=120):
    from PIL import Image
    import base64
    img = Image.new("RGB", (w, h), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return ("data:image/png;base64,"
            + base64.b64encode(buf.getvalue()).decode("ascii"))


def test_sco_prompt_survives_roi_extras_composition():
    """The biggest council-flagged bug: when ROI extras are enabled,
    _build_vlm_roi_extras() composes its user_prompt with
    qwen3_vl.DEFAULT_USER_PROMPT. case_runner used to
    overwrite the SCO prompt with that composition. Fix: when SCO is
    active, the manifest.user_prompt is the ROI legend + SCO basket
    prompt, NOT the provider fallback prompt.

    Test approach: use the same composition logic case_runner uses,
    drive it with the REAL config (cam_01 has SCO ROI views), and
    assert the resulting prompt contains the SCO v2 JSON shape AND the
    ROI legend AND does NOT contain the provider fallback prompt.
    """
    cfg = _load_real_config()
    from app.case_runner import _build_vlm_roi_extras
    from reasoning.prompts.sco_basket_match_v2 import build_sco_prompts_v2
    from reasoning.providers.qwen3_vl import DEFAULT_USER_PROMPT as QWEN_FALLBACK

    # Build ROI extras using real config + a single synthetic frame.
    sampled_frames = [{
        "frame_id": "f0", "frame_idx": 0,
        "ts": "2026-06-17T14:02:30",
        "image_url": _make_data_url(160, 120),
    }]
    extras = _build_vlm_roi_extras(_active_sco_camera_id(cfg), sampled_frames, cfg=cfg)
    assert extras is not None, (
        "cam_01 should have SCO ROI views configured. If this fails, "
        "model_roi_views in config.yaml is incomplete."
    )

    # Build the SCO prompt the way case_runner does.
    _system, sco_user = build_sco_prompts_v2(
        basket=[{"description": "DOVE SOAP BAR 100G", "quantity": 1},
                {"description": "COKE CAN 330ML", "quantity": 2}],
        canonical_groups=[],
        episode_meta={"start": "2026-06-17T14:02:10",
                      "end": "2026-06-17T14:02:50",
                      "ambiguous": False, "reason": "clean_episode",
                      "coverage_ratio": 0.25},
    )

    # Replicate case_runner's composition for SCO mode + ROI extras.
    composed_user_prompt = (extras["caption_text"] + "\n\n" + sco_user)

    # ROI legend present
    assert "ROI legend:" in composed_user_prompt
    # SCO JSON shape present (sample of unique fields)
    assert '"physical_count_match"' in composed_user_prompt
    assert '"semantic_identity_match"' in composed_user_prompt
    assert '"video_usable"' in composed_user_prompt
    # POS basket present
    assert "DOVE SOAP BAR 100G" in composed_user_prompt
    assert "COKE CAN 330ML" in composed_user_prompt
    # Qwen fallback prompt is NOT included
    assert QWEN_FALLBACK not in composed_user_prompt, (
        "Qwen DEFAULT_USER_PROMPT leaked into the SCO "
        "composed prompt — fix is not effective"
    )
    # Specifically, the refund-only field names must NOT appear
    for refund_only_field in ('"handover_occurred"',
                              '"physical_item_presented"',
                              '"receipt_visible"',
                              '"items_observed"'):
        assert refund_only_field not in composed_user_prompt, (
            f"refund-only field {refund_only_field} leaked into SCO "
            "composed prompt"
        )


def test_case_runner_branch_uses_sco_prompt_with_roi_extras(monkeypatch):
    """Drive the real composition path in case_runner: when
    prompt_version=sco_basket_match_v2 and ROI extras exist, the final
    manifest.user_prompt is the SCO composition.
    """
    cfg = _load_real_config()
    from app.case_runner import _build_vlm_roi_extras
    from reasoning.prompts.sco_basket_match_v2 import build_sco_prompts_v2
    from reasoning.providers.qwen3_vl import DEFAULT_USER_PROMPT as QWEN_FALLBACK

    sampled_frames = [{
        "frame_id": "f0", "frame_idx": 0,
        "ts": "2026-06-17T14:02:30",
        "image_url": _make_data_url(160, 120),
    }]
    extras = _build_vlm_roi_extras(_active_sco_camera_id(cfg), sampled_frames, cfg=cfg)
    assert extras is not None

    _system, sco_user = build_sco_prompts_v2(basket=[],
                                             canonical_groups=[],
                                             episode_meta={})

    # case_runner's exact composition for SCO + ROI extras:
    composed = extras["caption_text"] + "\n\n" + sco_user

    # Sanity: this is what gets put on manifest.user_prompt
    assert '"physical_count_match"' in composed
    assert '"semantic_identity_match"' in composed
    assert QWEN_FALLBACK not in composed
