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


CLASSIFIERS: dict[str, dict] = {
    # ------------------------------------------------------------------
    "fraud": {
        "display_label": "Return Fraud Detection",
        "color": "#ff5566",  # red
        "token_budget": 1120,
        "enable_thinking": True,   # ambiguous handover scenes need reasoning
        "max_frames": 20,          # full temporal detail for hand movements
        "falcon_prompt": (
            "bag, shopping bag, clothing, shirt, box, package, paper, "
            "document, receipt, phone, card, wallet, item, product"
        ),
        "gemma_system": (
            "You are a retail loss-prevention analyst reviewing a short "
            "video clip of a return/refund counter, cropped to the "
            "customer side.\n\n"
            "Falcon Perception pre-analysed THREE frames of this clip:\n"
            "- start_objects: {start_objects}   <- objects present at clip start\n"
            "- action_objects: {action_objects} <- objects detected at handover moment\n\n"
            "FIXTURE RULE (apply first, before all other rules):\n"
            "Any object label present in start_objects is a PRE-EXISTING "
            "FIXTURE - counter items, display boards, stands, or "
            "leftovers. Exclude these entirely from handover judgement "
            "regardless of position or confidence score. Do not mention "
            "them in your output.\n\n"
            "Only evaluate objects that appear in action_objects but NOT "
            "in start_objects.\n\n"
            "Your job is to judge the ACTION of handover, not the "
            "presence of objects. Follow these rules strictly:\n"
            " * Only count items the customer ACTIVELY BROUGHT and "
            "PLACED on the counter or HANDED to the staff during this "
            "clip.\n"
            " * DO NOT count items already on the counter when the clip "
            "starts.\n"
            " * DO NOT count items the staff is holding or handling on "
            "their own, nor items the staff hands BACK to the customer.\n"
            " * Browsing, gesturing, asking questions, or signing "
            "paperwork alone are NOT handovers.\n"
            " * If unsure, set handover_occurred=false and "
            "flag_for_review=true.\n\n"
            "Output ONLY the JSON object described in the user turn. No "
            "preamble, no markdown, no commentary."
        ),
        "gemma_user": (
            "Analyse this clip and answer for THE customer who triggered "
            "this session (ignore bystanders):\n"
            "1) Did the customer hand any items to staff?\n"
            "2) How many distinct items? What are they?\n"
            "3) Describe the customer briefly (clothing, position).\n"
            "4) One sentence: what the customer did.\n\n"
            "Respond with this exact JSON schema only:\n" + _BASE_JSON_SCHEMA
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
    """Return the classifier dict for ``key`` or fall back to ``custom``."""
    return CLASSIFIERS.get(key) or CLASSIFIERS["custom"]


def resolve_prompts(camera_cfg: dict) -> dict:
    """Resolve the effective prompts + token budget + scenario label for a
    camera, applying per-camera overrides over the classifier defaults.

    Returns a dict with keys:
        ``classifier``, ``display_label``, ``color``, ``token_budget``,
        ``falcon``, ``gemma_system``, ``gemma_user``.
    """
    classifier_key = (camera_cfg.get("classifier") or "fraud").strip().lower()
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
