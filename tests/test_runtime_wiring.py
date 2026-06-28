"""Pin the runtime wiring + UI language closed-out at this checkpoint.

These tests verify the things the previous checkpoint accepted as
"working in code but disconnected from the live app":

  1. monitor.py builds the active provider via the chain abstraction —
     not by instantiating ``GemmaVideoReasoner`` and calling ``reason``
     directly on the main inference path.
  2. The active provider chain has Qwen3-VL first when the config has
     ``qwen3_vl.enabled: true`` AND repo-local weights resolve.
  3. The dashboard contains no user-visible "FLAGGED"/"Fraud Detection"
     language and no ``classifier || 'fraud'`` default.
  4. The legacy "fraud" classifier alias still resolves to the
     review-safe entry (no accusation prompts can be re-introduced via
     a stale per-camera config).
  5. The decision policy is still the final authority — outcomes
     constrained to the four review-safe enum values.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. monitor.py uses the provider chain, not a direct gemma.reason call,
#    on the analyze() inference path.
# ---------------------------------------------------------------------------

def test_monitor_uses_provider_chain_on_analyze_path():
    src = (ROOT / "monitor.py").read_text()
    # The chain abstraction must be wired.
    assert "from reasoning.providers import build_active_provider" in src
    assert "build_active_provider(" in src
    assert "provider.analyze_evidence(" in src
    # The legacy direct-Gemma call on the analyze() path must be gone.
    assert "gemma.reason(" not in src, (
        "monitor.py still calls gemma.reason() directly; the chain "
        "provider must own the main inference path."
    )
    # quick_describe (the cheap dedupe probe) MAY still use Gemma — it
    # never produces a case outcome — but we keep it explicit here so
    # the contract is obvious.
    assert "gemma.quick_describe(" in src


def test_monitor_normalises_provider_result_before_policy():
    src = (ROOT / "monitor.py").read_text()
    assert "_normalise_vlm_result(" in src
    # The decision policy still wraps the normalised result before any
    # downstream consumer reads ``flag_for_review``.
    assert "from reasoning.decision_policy import" in src
    assert "decision = decide(" in src
    assert 'result["flag_for_review"] = decision.outcome != OUTCOME_VERIFIED' in src


# ---------------------------------------------------------------------------
# 2. Qwen primary on the active config
# ---------------------------------------------------------------------------

def test_qwen_enabled_in_active_config():
    from app.config import load_config
    cfg = load_config()
    assert cfg.models["qwen3_vl"].enabled is True
    assert cfg.models["gemma"].enabled is True


def test_active_provider_is_chain_with_qwen_first():
    """Production default is Qwen-only (fallback_provider: null). To
    pin the Qwen-first ordering when the operator re-enables the
    Gemma fallback, we explicitly opt in here. The Qwen-only path is
    covered by tests/test_startup_qwen_default.py."""
    from app.config import load_config
    from reasoning.providers import ChainProvider, build_active_provider

    cfg = load_config()
    cfg.raw.setdefault("reasoning", {})["fallback_provider"] = "gemma"
    p = build_active_provider(cfg)
    assert isinstance(p, ChainProvider), \
        f"expected ChainProvider, got {type(p).__name__}"
    assert p.providers[0].name == "qwen3_vl"
    assert p.providers[1].name == "gemma"


# ---------------------------------------------------------------------------
# 3. Dashboard language audit
# ---------------------------------------------------------------------------

_VISIBLE_LABEL_PHRASES_BANNED = (
    ">flagged<",                # text inside any element
    ">flagged ",
    " flagged<",
    "fraud detection",
    "return fraud",
    "fraud indicator",
)


def test_static_dashboard_user_visible_text_is_review_safe():
    """The new production reviewer UI is static/review.html. The legacy
    static/index.html is now a thin redirect notice + can no longer
    contain accusation language either."""
    for fname in ("index.html", "review.html"):
        src = (ROOT / "static" / fname).read_text().lower()
        for phrase in _VISIBLE_LABEL_PHRASES_BANNED:
            assert phrase not in src, (
                f"static/{fname} still contains banned user-visible "
                f"phrase {phrase!r}"
            )
    review_src = (ROOT / "static" / "review.html").read_text().lower()
    assert "needs review" in review_src
    # UI rebrand: header reads "SCO Vision — Self-Checkout Reviewer".
    # The pre-rebrand "Return / Refund Visual Review" header is gone.
    assert "sco vision" in review_src
    assert "self-checkout reviewer" in review_src
    assert "return / refund visual review" not in review_src


def test_no_legacy_classifier_default_in_static_ui():
    """The legacy form fallback ``cam.classifier || 'fraud'`` must not
    ship. We tolerate the legacy index.html being a redirect notice."""
    for fname in ("index.html", "review.html"):
        src = (ROOT / "static" / fname).read_text()
        assert "classifier || 'fraud'" not in src


# ---------------------------------------------------------------------------
# 4. Legacy 'fraud' alias still resolves to review-safe entry
# ---------------------------------------------------------------------------

def test_legacy_fraud_alias_resolves_to_return_review():
    from classifiers import get_classifier, resolve_prompts
    legacy = get_classifier("fraud")
    safe = get_classifier("return_review")
    assert legacy is safe

    # A camera config that still says ``classifier: fraud`` must resolve
    # to the safe prompts, NOT to anything that mentions accusation.
    resolved = resolve_prompts({"classifier": "fraud"})
    assert resolved["classifier"] == "return_review"
    for needle in ("determine fraud", "is this fraud",
                   "loss-prevention analyst"):
        assert needle not in resolved["gemma_system"].lower()
        assert needle not in resolved["gemma_user"].lower()


# ---------------------------------------------------------------------------
# 5. Decision policy still final authority
# ---------------------------------------------------------------------------

def test_decision_policy_outcomes_constrained_after_wiring():
    from reasoning.decision_policy import (
        VALID_OUTCOMES, EvidenceSummary, decide,
    )
    cases = [
        EvidenceSummary(footage_valid=False),
        EvidenceSummary(footage_valid=True, vlm_confidence="low"),
        EvidenceSummary(footage_valid=True, receipt_visible=True,
                        physical_item_track=False, vlm_confidence="high"),
        EvidenceSummary(footage_valid=True, physical_item_track=True,
                        item_reaches_counter=True, vlm_confidence="high"),
        EvidenceSummary(footage_valid=True, obstructed=True),
    ]
    for s in cases:
        d = decide(s)
        assert d.outcome in VALID_OUTCOMES
        assert "FRAUD" not in d.outcome


# ---------------------------------------------------------------------------
# 6. Registry shape after correction
# ---------------------------------------------------------------------------

def test_registry_required_assets_are_deployable_today():
    import yaml
    registry = yaml.safe_load((ROOT / "offline_assets.yaml").read_text())
    required = {r["name"] for r in registry.get("required") or []}
    # SAM 2 is required (the deployable segmenter today). SAM 3 is
    # optional/preferred-upgrade — not required, so it cannot indefinitely
    # block the bundle.
    assert "sam2" in required
    assert "sam3" not in required

    optional = {r["name"] for r in registry.get("optional") or []}
    assert "sam3" in optional
    assert "falcon_ocr_specialized" in optional

    # Falcon-Perception covers OCR (per upstream README); a separate
    # Falcon-OCR is not required. The role tag spells this out.
    falcon = next(r for r in registry["required"]
                  if r["name"] == "falcon_perception")
    assert falcon.get("role") == "detector_and_ocr"


# ---------------------------------------------------------------------------
# 7. Provider normalisation produces all legacy keys + chain attempts
# ---------------------------------------------------------------------------

def test_normalise_vlm_result_fills_legacy_keys():
    import monitor
    from reasoning.providers.base import VLMResult

    vlm = VLMResult(
        provider="qwen3_vl",
        model_name="Qwen/Qwen3-VL-30B-A3B-Instruct",
        raw_text="...",
        parsed={
            "handover_occurred": True,
            "physical_item_presented": True,
            "receipt_visible": True,
            "items_observed": ["shirt"],
            "narrative": "the customer placed a shirt on the counter",
            "confidence": "high",
            "obstructed": False,
            "camera_view_clear": True,
            "limitations": [],
            "_chain_attempts": ["qwen3_vl=ok"],
        },
        latency_ms=1234,
    )
    out = monitor._normalise_vlm_result(
        vlm, num_frames=12, inference_started_at=0.0,
    )
    # Every legacy key downstream consumers read must be present.
    for k in ("handover_occurred", "item_count", "items_handed_over",
              "customer_description", "narrative", "confidence",
              "flag_for_review", "people", "item_presented",
              "objects_detected", "_num_frames", "_latency_ms",
              "_provider", "_provider_model",
              "physical_item_presented", "receipt_visible",
              "obstructed", "camera_view_clear", "limitations",
              "_chain_attempts"):
        assert k in out, f"missing key {k!r}"
    assert out["_provider"] == "qwen3_vl"
    assert out["_chain_attempts"] == ["qwen3_vl=ok"]
    assert out["item_count"] == 1
    assert out["flag_for_review"] is False  # default; policy overrides later
