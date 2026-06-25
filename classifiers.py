"""Single source of truth for classifier defaults.

Each entry in ``CLASSIFIERS`` defines a scenario the system can run on a
camera: what objects Falcon Perception should look for, what role Gemma
plays, what question is asked, how detailed the image tokenization should
be, and how the result is labelled in the dashboard / CSV.

A camera in ``config.yaml`` references a classifier by key (``classifier:
fraud``). Per-camera prompt overrides (under ``cameras[].prompts``) win
over the classifier defaults; an empty/missing override falls back to the
classifier's value.

Adding a new classifier:
  1. Add a new key in ``CLASSIFIERS`` below.
  2. That's it — the dashboard dropdown is built from this dict, so the
     new option appears automatically with its color / display label.
"""
from __future__ import annotations

from typing import Iterable

# Allowed Gemma 4 image-token budgets (per the model's vision config).
TOKEN_BUDGETS: tuple[int, ...] = (70, 140, 280, 560, 1120)


def coerce_token_budget(value, default: int = 560) -> int:
    """Snap arbitrary input to the nearest allowed Gemma image-token budget."""
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    if v in TOKEN_BUDGETS:
        return v
    # Snap to nearest allowed value.
    return min(TOKEN_BUDGETS, key=lambda b: abs(b - v))


# JSON schema all classifiers ask for. Kept identical to v2's schema so
# downstream CSV/dashboard code does not need migrations.
_BASE_JSON_SCHEMA = (
    "{\n"
    '  "handover_occurred": true|false,\n'
    '  "item_count": integer (0 if no event),\n'
    '  "items_handed_over": ["brief", "item", "descriptions"],\n'
    '  "customer_description": "one sentence describing the subject",\n'
    '  "narrative": "one sentence: what happened",\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "flag_for_review": true|false\n'
    "}"
)


# Review-safe schema for the return-review classifier. NO ``flag_for_review``
# field: the deterministic ``reasoning.decision_policy`` decides the
# outcome from these structured signals; the VLM only describes what it
# can/can't see.
_REVIEW_SAFE_JSON_SCHEMA = (
    "{\n"
    '  "handover_occurred": true|false,\n'
    '  "physical_item_presented": true|false,\n'
    '  "receipt_visible": true|false,\n'
    '  "items_observed": ["brief", "item", "descriptions"],\n'
    '  "customer_description": "one sentence describing the subject",\n'
    '  "narrative": "one sentence: what the camera shows",\n'
    '  "confidence": "high"|"medium"|"low",\n'
    '  "obstructed": true|false,\n'
    '  "camera_view_clear": true|false,\n'
    '  "limitations": ["list any blind spots, occlusions, or ambiguities"]\n'
    "}"
)


CLASSIFIERS: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Review-safe replacement for the old "fraud" classifier. The VLM
    # describes what the camera shows; it does not classify fraud. The
    # final outcome (VERIFIED / REVIEW / HIGH_RISK_REVIEW / INVALID_VIDEO)
    # is decided by ``reasoning.decision_policy.decide``. The "fraud" key
    # below is kept as an alias so any cached config that still says
    # ``classifier: fraud`` resolves to this review-safe entry, not to a
    # stale accusatory prompt.
    "return_review": {
        "display_label": "Return / Refund Visual Review",
        "color": "#5b8def",  # neutral blue
        "token_budget": 1120,
        "enable_thinking": True,
        "max_frames": 20,
        "falcon_prompt": (
            "bag, shopping bag, clothing, shirt, box, package, paper, "
            "document, receipt, phone, card, wallet, item, product"
        ),
        "gemma_system": (
            "You are an evidence describer reviewing a short video clip "
            "of a return / refund counter. You do NOT decide anything. "
            "A separate deterministic policy decides the case outcome "
            "from the structured signals you report.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}   <- objects present at clip start\n"
            "- action_objects: {action_objects} <- objects detected during the action\n\n"
            "FIXTURE RULE (apply first):\n"
            "Any object label present in start_objects is a PRE-EXISTING "
            "FIXTURE — counter items, display boards, stands, or "
            "leftovers. Exclude these from the handover judgement and do "
            "not mention them in your output.\n\n"
            "Reporting rules:\n"
            " * Describe only what you actually see; never infer intent.\n"
            " * Never use the words 'fraud', 'fraudulent', 'theft', or "
            "'suspect'.\n"
            " * Set ``handover_occurred`` to true when the customer's own "
            "physical product is visible at the counter during the clip "
            "— handed to staff, placed on the counter, or held out toward "
            "staff — as long as that item is NOT one of the start_objects "
            "fixtures. The exact instant of release may fall between the "
            "sampled frames, so base this on the visible evidence that "
            "such a product is present and associated with the customer, "
            "not on catching the hand motion itself. Set it false when no "
            "customer-brought product is visible (empty-handed, only a "
            "receipt or paper document, or the item is a pre-existing "
            "fixture).\n"
            " * Only describe a customer as present if a person is clearly "
            "visible on the customer side of the counter. If ONLY staff are "
            "visible and no customer is present, set handover_occurred=false "
            "and physical_item_presented=false, and say 'no customer "
            "present' in the narrative. Do NOT assume a customer exists just "
            "because this is a return counter.\n"
            " * Set ``physical_item_presented`` true if a tangible product "
            "(not just paper/documents) belonging to the customer is "
            "visible at the counter at any point — placed, held, or handed. "
            "Base this on the product being PRESENT, not on catching the "
            "hand-over motion.\n"
            " * Set ``receipt_visible`` if a receipt, document, or paper "
            "slip is visible.\n"
            " * Set ``obstructed`` if any part of the handover area is "
            "occluded by people, fixtures, or angle.\n"
            " * Set ``camera_view_clear`` only if the relevant area is "
            "unobstructed for the full clip.\n"
            " * Use ``limitations`` to list blind spots, glare, motion "
            "blur, or anything that limits your description.\n"
            " * If you are unsure of any field, lower ``confidence`` to "
            "'low'.\n\n"
            "Output ONLY the JSON object described in the user turn. No "
            "preamble, no markdown, no commentary."
        ),
        "gemma_user": (
            "Describe what the camera shows of the return-counter "
            "interaction for THE customer who triggered this session "
            "(ignore bystanders):\n"
            "1) Is the customer's own physical product visible at the "
            "counter at ANY point in the clip — placed on it, held, or "
            "handed to staff? Judge this by the product being PRESENT and "
            "associated with the customer, NOT by catching the exact "
            "hand-over motion (the moment of release may fall between "
            "frames). It need not be a fixture from start_objects.\n"
            "2) Is that product a tangible item (bag, clothing, box, "
            "package, etc.), not just a receipt or paper document?\n"
            "3) Was a receipt or document visible?\n"
            "4) Was your view obstructed at any point?\n"
            "5) One sentence: what the camera shows.\n\n"
            "Respond with this exact JSON schema only:\n"
            + _REVIEW_SAFE_JSON_SCHEMA
        ),
    },
    # ------------------------------------------------------------------
    "safety": {
        "display_label": "Safety / PPE Compliance",
        "color": "#ffb020",  # orange
        "token_budget": 560,
        "enable_thinking": True,   # PPE compliance needs careful analysis
        "max_frames": 10,          # scene snapshot is enough
        "falcon_prompt": (
            "person, worker, helmet, hardhat, hi-vis vest, safety vest, "
            "gloves, goggles, mask, boots, ladder, spill, wet floor sign, "
            "forklift, machine"
        ),
        "gemma_system": (
            "You are a workplace-safety analyst reviewing a short video "
            "clip of an industrial / retail floor area.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}\n"
            "- action_objects: {action_objects}\n\n"
            "Look for safety violations: missing PPE (helmet, hi-vis, "
            "gloves, goggles), unsafe behaviour (running, climbing on "
            "fixtures, blocked exits), or hazardous conditions (spills, "
            "obstructions).\n\n"
            "Treat ``handover_occurred=true`` as 'a safety event was "
            "observed', ``items_handed_over`` as the list of "
            "violations / hazards, ``item_count`` as the number of "
            "distinct violations.\n\n"
            "If unsure, set handover_occurred=false and "
            "flag_for_review=true.\n\n"
            "Output ONLY the JSON object described in the user turn."
        ),
        "gemma_user": (
            "Analyse this clip for safety violations or hazards:\n"
            "1) Did a safety event occur?\n"
            "2) How many distinct violations / hazards? List them.\n"
            "3) Describe the person of interest briefly (clothing, "
            "position).\n"
            "4) One sentence: what happened.\n\n"
            "Respond with this exact JSON schema only:\n" + _BASE_JSON_SCHEMA
        ),
    },
    # ------------------------------------------------------------------
    "manufacturing": {
        "display_label": "Manufacturing QC",
        "color": "#4ea1ff",  # blue
        "token_budget": 280,
        "enable_thinking": False,  # fast binary check
        "max_frames": 5,           # single moment check
        "falcon_prompt": (
            "widget, part, component, assembly, defect, scratch, dent, "
            "missing screw, label, barcode, package, conveyor, tray"
        ),
        "gemma_system": (
            "You are a quality-control inspector reviewing a short clip "
            "from a manufacturing line.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}\n"
            "- action_objects: {action_objects}\n\n"
            "Treat ``handover_occurred=true`` as 'a defect or anomaly "
            "was observed', ``items_handed_over`` as the list of "
            "defects (missing parts, damage, mis-labels), ``item_count`` "
            "as the number of distinct defects.\n\n"
            "If the part appears nominal, set handover_occurred=false. "
            "If unsure, also set flag_for_review=true.\n\n"
            "Output ONLY the JSON object described in the user turn."
        ),
        "gemma_user": (
            "Inspect this clip for defects or anomalies on the part / "
            "assembly visible in the customer zone:\n"
            "1) Was a defect observed?\n"
            "2) How many distinct defects? What are they?\n"
            "3) Describe the part briefly.\n"
            "4) One sentence: what was wrong (or 'part appears nominal').\n\n"
            "Respond with this exact JSON schema only:\n" + _BASE_JSON_SCHEMA
        ),
    },
    # ------------------------------------------------------------------
    "shelf": {
        "display_label": "Shelf Compliance",
        "color": "#3ddc84",  # green
        "token_budget": 560,
        "enable_thinking": False,  # fast count
        "max_frames": 8,           # product count
        "falcon_prompt": (
            "shelf, product, box, bottle, can, packet, gap, empty space, "
            "price tag, sign, planogram"
        ),
        "gemma_system": (
            "You are a retail merchandising auditor reviewing a short "
            "clip of a store shelf.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}\n"
            "- action_objects: {action_objects}\n\n"
            "Treat ``handover_occurred=true`` as 'a compliance issue "
            "was observed' (out-of-stock gap, mis-faced product, "
            "missing price tag), ``items_handed_over`` as the list of "
            "issues, ``item_count`` as the number of distinct issues.\n\n"
            "If the shelf appears properly stocked and faced, set "
            "handover_occurred=false. If unsure, also set "
            "flag_for_review=true.\n\n"
            "Output ONLY the JSON object described in the user turn."
        ),
        "gemma_user": (
            "Audit this clip for shelf compliance:\n"
            "1) Were any compliance issues observed (gaps, mis-facing, "
            "missing tags)?\n"
            "2) How many distinct issues? List them.\n"
            "3) Describe the shelf section briefly.\n"
            "4) One sentence: shelf condition summary.\n\n"
            "Respond with this exact JSON schema only:\n" + _BASE_JSON_SCHEMA
        ),
    },
    # ------------------------------------------------------------------
    "access": {
        "display_label": "Access Control",
        "color": "#a07cff",  # purple
        "token_budget": 280,
        "enable_thinking": False,  # person present or not, fast
        "max_frames": 5,
        "falcon_prompt": (
            "person, door, gate, badge, lanyard, uniform, restricted "
            "area sign, turnstile"
        ),
        "gemma_system": (
            "You are an access-control analyst reviewing a short clip "
            "of a restricted-access doorway or zone.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}\n"
            "- action_objects: {action_objects}\n\n"
            "Treat ``handover_occurred=true`` as 'an access event of "
            "interest occurred' (unauthorised entry, tailgating, "
            "person without visible badge), ``items_handed_over`` as "
            "the list of concerns, ``item_count`` as the number of "
            "people involved.\n\n"
            "Routine authorised entry by a single badged person is NOT "
            "an event. If unsure, set handover_occurred=false and "
            "flag_for_review=true.\n\n"
            "Output ONLY the JSON object described in the user turn."
        ),
        "gemma_user": (
            "Analyse this clip for access-control concerns:\n"
            "1) Did a concerning access event occur?\n"
            "2) How many people involved? What concerns?\n"
            "3) Describe the person of interest briefly.\n"
            "4) One sentence: what happened.\n\n"
            "Respond with this exact JSON schema only:\n" + _BASE_JSON_SCHEMA
        ),
    },
    # ------------------------------------------------------------------
    "custom": {
        "display_label": "Custom (operator-defined)",
        "color": "#8a93a6",  # grey
        "token_budget": 1120,
        "enable_thinking": True,   # unknown use case — be safe
        "max_frames": 20,
        "falcon_prompt": "object",
        "gemma_system": (
            "You are a generic video-clip analyst. The operator will "
            "supply the actual instructions via the per-camera prompt "
            "overrides. If you see this default text, the operator has "
            "not yet customised the prompt - emit a placeholder JSON "
            "with handover_occurred=false and flag_for_review=true.\n\n"
            "start_objects: {start_objects}\n"
            "action_objects: {action_objects}\n\n"
            "Output ONLY the JSON object described in the user turn."
        ),
        "gemma_user": (
            "Describe what is happening in this clip in one sentence "
            "and return:\n" + _BASE_JSON_SCHEMA
        ),
    },
}


def list_classifiers() -> list[dict]:
    """Return classifier metadata for the dashboard ``/classifiers``
    endpoint. Order matches insertion order in ``CLASSIFIERS``."""
    return [
        {
            "key": key,
            "display_label": entry["display_label"],
            "color": entry["color"],
            "token_budget": entry["token_budget"],
            "enable_thinking": bool(entry.get("enable_thinking", False)),
            "max_frames": int(entry.get("max_frames", 20)),
        }
        for key, entry in CLASSIFIERS.items()
    ]


def get_classifier(key: str) -> dict:
    """Return the classifier dict for ``key`` or fall back to ``custom``.

    The legacy key ``"fraud"`` is silently remapped to the review-safe
    ``"return_review"`` so old configs cannot resurrect accusatory prompt
    text by accident.
    """
    if key == "fraud":
        return CLASSIFIERS["return_review"]
    return CLASSIFIERS.get(key) or CLASSIFIERS["custom"]


def resolve_prompts(camera_cfg: dict) -> dict:
    """Resolve the effective prompts + token budget + scenario label for a
    camera, applying per-camera overrides over the classifier defaults.

    Returns a dict with keys:
        ``classifier``, ``display_label``, ``color``, ``token_budget``,
        ``falcon``, ``gemma_system``, ``gemma_user``.
    """
    classifier_key = (camera_cfg.get("classifier") or "return_review").strip().lower()
    if classifier_key == "fraud":
        classifier_key = "return_review"
    base = get_classifier(classifier_key)
    overrides = camera_cfg.get("prompts") or {}

    # Override-key -> classifier-default-key mapping.
    KEY_MAP = {
        "falcon": "falcon_prompt",
        "gemma_system": "gemma_system",
        "gemma_user": "gemma_user",
    }

    def _resolved(override_key: str) -> str:
        v = overrides.get(override_key)
        if v is not None and str(v).strip():
            return str(v)
        return base[KEY_MAP[override_key]]

    explicit_budget = camera_cfg.get("token_budget")
    token_budget = (coerce_token_budget(explicit_budget, base["token_budget"])
                    if explicit_budget is not None else base["token_budget"])

    return {
        "classifier": classifier_key,
        "display_label": base["display_label"],
        "color": base["color"],
        "token_budget": int(token_budget),
        "falcon": _resolved("falcon"),
        "gemma_system": _resolved("gemma_system"),
        "gemma_user": _resolved("gemma_user"),
        # Per-camera override > classifier default. ``max_frames`` may be
        # ``None`` if the classifier has no value either — caller falls
        # back to the global ``gemma_video_max_seconds * gemma_video_fps``.
        "enable_thinking": _resolved_bool(camera_cfg, "enable_thinking",
                                          base.get("enable_thinking")),
        "max_frames": _resolved_int(camera_cfg, "max_frames",
                                    base.get("max_frames")),
    }


def _resolved_bool(camera_cfg: dict, key: str, default):
    v = camera_cfg.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None if default is None else bool(default)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(default) if default is not None else None
    return bool(v)


def _resolved_int(camera_cfg: dict, key: str, default):
    v = camera_cfg.get(key)
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return default
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return default


def iter_classifier_keys() -> Iterable[str]:
    return CLASSIFIERS.keys()
