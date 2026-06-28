"""UI rebrand + SAM3 admin-controls wiring.

Council scope:
  * Visible UI text reflects SCO checkout review, not refund / return /
    fraud / handover.
  * SAM3 appears in admin model-controls alongside Falcon, Qwen3-VL,
    Gemma, SAM2, OCR.
  * SAM3 toggle updates config correctly and must not remove
    Falcon / VLM options.
  * Review page renders SCO risk reasons cleanly.

These are UI/admin-only tests. They do not touch perception, VLM
prompts, policies, the container merger, the Qwen runtime, DB
migrations, or the exporter.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REVIEW_HTML = (ROOT / "static" / "review.html").read_text()


# ---------------------------------------------------------------------------
# 1. UI rebrand
# ---------------------------------------------------------------------------

def test_fastapi_title_is_sco_branded():
    from app.main import create_app
    app = create_app()
    assert "SCO Vision" in app.title
    # Reject the old refund title.
    for forbidden in ("Return / Refund Visual Review", "Refund Visual"):
        assert forbidden not in app.title, app.title


def test_review_page_title_and_h1_are_sco():
    assert "<title>SCO Vision" in REVIEW_HTML
    assert "<h1>SCO Vision" in REVIEW_HTML
    # No legacy refund title strings anywhere user-visible
    assert "Return / Refund Visual Review — Case Queue" not in REVIEW_HTML
    # The legacy h1 string is gone.
    assert "<h1>Return / Refund Visual Review</h1>" not in REVIEW_HTML


def test_verified_action_label_does_not_say_refund():
    """The verified-decision button used to read 'Verified physical
    return'. SCO operators are reviewing a CHECKOUT, not a refund."""
    # The data-action attribute kept its historical name so audit
    # rows persist — only the user-visible label changed.
    m = re.search(r'data-action="verified_physical_return"[^>]*>\s*([^<]+)',
                  REVIEW_HTML)
    assert m, "verified action button missing"
    label = m.group(1).strip()
    assert "basket" in label.lower(), label
    assert "physical return" not in label.lower()


def test_sco_reason_filter_present_in_case_queue():
    # Filter dropdown with all 10 SCO tag values + the basket_match
    # verified-class label.
    for tag in [
        "sco_basket_match", "sco_basket_mismatch", "sco_identity_uncertain",
        "sco_count_uncertain", "sco_extra_candidates", "sco_missing_items",
        "sco_episode_ambiguous", "sco_episode_short", "sco_low_confidence",
        "sco_bad_footage",
    ]:
        assert f'value="{tag}"' in REVIEW_HTML, \
            f"SCO reason filter missing tag {tag}"


def test_case_grid_has_sco_reasons_column():
    assert "<th>SCO reasons</th>" in REVIEW_HTML
    # And the row template calls the SCO pills renderer.
    assert "scoReasonPills(row.risk_reasons)" in REVIEW_HTML


def test_sco_tag_labels_map_is_complete():
    # The SCO_TAG_LABELS map in the JS must have a human-readable
    # entry for every tag the v2 policy can emit. Pulled from
    # reasoning.sco_policy_v2.
    from reasoning.sco_policy_v2 import (
        TAG_BASKET_MATCH, TAG_BASKET_MISMATCH, TAG_IDENTITY_UNCERTAIN,
        TAG_COUNT_UNCERTAIN, TAG_MISSING_ITEMS, TAG_EXTRA_CANDIDATES,
        TAG_EPISODE_AMBIGUOUS, TAG_EPISODE_SHORT, TAG_LOW_CONFIDENCE,
        TAG_BAD_FOOTAGE, TAG_NO_VLM,
    )
    js_block = re.search(r"const\s+SCO_TAG_LABELS\s*=\s*\{(.*?)\};",
                         REVIEW_HTML, re.DOTALL)
    assert js_block, "SCO_TAG_LABELS map missing from review.html"
    body = js_block.group(1)
    for tag in [TAG_BASKET_MATCH, TAG_BASKET_MISMATCH,
                TAG_IDENTITY_UNCERTAIN, TAG_COUNT_UNCERTAIN,
                TAG_MISSING_ITEMS, TAG_EXTRA_CANDIDATES,
                TAG_EPISODE_AMBIGUOUS, TAG_EPISODE_SHORT,
                TAG_LOW_CONFIDENCE, TAG_BAD_FOOTAGE, TAG_NO_VLM]:
        assert tag in body, f"SCO_TAG_LABELS missing entry for {tag}"


def test_sco_vlm_render_is_sco_shape_aware():
    """The 'Model claims' panel renders both v2 SCO output AND the
    legacy refund shape — pin both code paths so a regression on
    either side fails loudly."""
    # SCO shape detection
    assert "physical_count_match" in REVIEW_HTML
    assert "semantic_identity_match" in REVIEW_HTML
    # SCO-specific render fields
    assert "matched_items" in REVIEW_HTML
    assert "extra_visible_items" in REVIEW_HTML
    assert "missing_visible_items" in REVIEW_HTML
    assert "uncertainty_reason" in REVIEW_HTML
    # Legacy refund fallback still rendered for old cases
    assert "handover_occurred" in REVIEW_HTML
    assert "items_handed_over" in REVIEW_HTML


# ---------------------------------------------------------------------------
# 2. SAM3 admin model-controls
# ---------------------------------------------------------------------------

def test_admin_model_controls_includes_sam3():
    """GET /admin/model-controls must surface a 'sam3' entry alongside
    Falcon, Qwen3-VL, Gemma, SAM2, OCR. Independent + default off."""
    from app.api.admin import MODEL_CONTROL_SPECS, ALLOWED_MODEL_CONTROL_KEYS
    ids = {s["id"] for s in MODEL_CONTROL_SPECS}
    assert "sam3" in ids
    assert {"falcon", "sam2", "ocr", "qwen3_vl", "gemma", "sam3"} <= ids
    assert "sam3" in ALLOWED_MODEL_CONTROL_KEYS
    spec = next(s for s in MODEL_CONTROL_SPECS if s["id"] == "sam3")
    assert spec["independent"] is True
    assert spec["default_when_missing"] is False
    assert spec["dependencies"] == []
    assert spec["config_key"] == "sam3"


def test_admin_model_controls_get_returns_sam3_state():
    """The /admin/model-controls response shape includes sam3 with a
    sensible default (False — config.yaml ships with sam3 off)."""
    from app.api.admin import get_model_controls
    out = get_model_controls()
    state = out["state"]
    items = out["models"]
    assert "sam3" in state
    assert any(it["id"] == "sam3" for it in items)
    sam3_item = next(it for it in items if it["id"] == "sam3")
    # SAM 3 is OFF in active config.yaml; if an operator flipped it on
    # locally, that's also acceptable for this test — what we pin is
    # presence + correct independent/dep shape.
    assert sam3_item["independent"] is True
    assert sam3_item["dependencies"] == []
    assert "SAM 3" in sam3_item["label"]


def test_admin_validation_accepts_sam3_only_perception():
    """An operator running SCO with SAM 3 ON and Falcon OFF must be
    accepted — SAM 3 is an independent perception backend."""
    from app.api.admin import _validate_model_control_update
    current = {"falcon": True, "sam2": False, "ocr": False,
               "qwen3_vl": True, "gemma": True, "sam3": False}
    payload = {"falcon": False, "sam3": True}
    out = _validate_model_control_update(payload, current)
    assert out["falcon"] is False
    assert out["sam3"] is True
    assert out["qwen3_vl"] is True   # VLM untouched


def test_admin_validation_rejects_no_independent_source():
    """When EVERY independent backend (Falcon, SAM3, Qwen, Gemma) is
    off, the admin endpoint rejects — used to be Falcon/Q/G only,
    now SAM3 also counts."""
    from fastapi import HTTPException
    from app.api.admin import _validate_model_control_update
    current = {"falcon": True, "sam2": False, "ocr": False,
               "qwen3_vl": True, "gemma": True, "sam3": False}
    payload = {"falcon": False, "sam3": False,
               "qwen3_vl": False, "gemma": False}
    with pytest.raises(HTTPException) as exc:
        _validate_model_control_update(payload, current)
    detail = exc.value.detail
    assert "SAM 3" in detail["error"] or "sam3" in detail["error"].lower()


def test_admin_sam2_falcon_dependency_unchanged():
    """SAM 3 is independent; SAM 2 still requires Falcon. The new
    independence rule must not have weakened SAM 2's dependency."""
    from fastapi import HTTPException
    from app.api.admin import _validate_model_control_update
    current = {"falcon": True, "sam2": True, "ocr": False,
               "qwen3_vl": True, "gemma": True, "sam3": False}
    # Operator tries to disable Falcon while SAM 2 is still on → reject.
    with pytest.raises(HTTPException):
        _validate_model_control_update({"falcon": False}, current)


def test_admin_warnings_mention_sam3_when_falcon_off():
    """When Falcon is off and SAM3 is on, the operator gets a
    contextual warning explaining the SCO-only behaviour."""
    from app.api.admin import _model_control_warnings
    state = {"falcon": False, "sam2": False, "ocr": False,
             "qwen3_vl": True, "gemma": True, "sam3": True}
    warnings = _model_control_warnings(state)
    assert any("SAM 3" in w and "Perception (FL)" in w for w in warnings), \
        warnings


def test_admin_warnings_both_perception_off():
    """Both backends off = both perception sources gone, the operator
    is warned cases will fall to REVIEW."""
    from app.api.admin import _model_control_warnings
    state = {"falcon": False, "sam2": False, "ocr": False,
             "qwen3_vl": True, "gemma": True, "sam3": False}
    warnings = _model_control_warnings(state)
    assert any("Both perception backends disabled" in w
               or "perception backends disabled" in w for w in warnings), \
        warnings


# ---------------------------------------------------------------------------
# 3. UI JS payload includes SAM3 (the model-controls Save button)
# ---------------------------------------------------------------------------

def test_review_js_payload_includes_sam3_toggle():
    """The save-handler builds an explicit payload dict and POSTs it to
    /admin/model-controls. SAM 3 must be in that dict, otherwise the
    server never sees the toggle."""
    # Grep for the payload object literal — it's the only place in the
    # JS that hard-lists all six toggles.
    m = re.search(r"const\s+payload\s*=\s*\{([^}]+)\};", REVIEW_HTML)
    assert m, "model-controls payload object literal missing"
    body = m.group(1)
    for key in ("falcon:", "sam2:", "sam3:", "ocr:", "qwen3_vl:", "gemma:"):
        assert key in body, f"payload missing {key!r}: {body}"


def test_review_js_local_validation_treats_sam3_as_independent():
    """The client-side preview validator should mirror the backend
    rule: SAM 3 alone (with no Falcon and no VLM) is rejected; SAM 3
    + a VLM is accepted."""
    # We assert by string search since this is a JS file. The
    # `_modelControlsLocalValidation` JS function should include SAM 3
    # in the independence rule.
    assert "_modelControlsLocalValidation" in REVIEW_HTML
    block = re.search(
        r"function\s+_modelControlsLocalValidation\s*\([^)]*\)\s*\{(.*?)\n\}",
        REVIEW_HTML, re.DOTALL,
    )
    assert block, "_modelControlsLocalValidation missing"
    body = block.group(1)
    assert "state.sam3" in body, body
    assert "SAM 3" in body, body
