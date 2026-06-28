"""Deep static analysis of static/review.html — every button has a
handler, every fetch path maps to a routed API, every tab pane has
both an HTML node and an activation hook.

This is structural, not behavioural. We do NOT spin up a headless
browser (no JSDOM/Playwright in the offline wheelhouse). What we can
guarantee here is that the UI's promises match what the backend
actually serves.

Tests intentionally avoid heuristics that would silently rot — every
assertion either inspects a concrete identifier in the HTML/JS or
introspects the live FastAPI route table.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _html() -> str:
    return (ROOT / "static" / "review.html").read_text()


# ---------------------------------------------------------------------
# Buttons + handlers
# ---------------------------------------------------------------------

_BUTTON_ID_RE = re.compile(
    r'<button[^>]*\bid="([a-zA-Z][a-zA-Z0-9_-]*)"', re.IGNORECASE)

# IDs that are wired by event-delegation (rendered into a container
# at runtime and handled by closest()) rather than by a direct
# ``getElementById(...).onclick`` binding. Pinning the binding style
# means future refactors must explicitly document the delegation.
_DELEGATED_BUTTON_IDS: frozenset = frozenset({
    # No delegated buttons today — every static button in review.html
    # is bound by ``document.getElementById("<id>")`` in the script
    # block. This frozenset exists so the contract is explicit: when
    # a new delegated control is added, the test calls it out instead
    # of silently passing.
})


def test_every_static_button_has_a_handler_or_documented_delegation():
    src = _html()
    button_ids = set(_BUTTON_ID_RE.findall(src))
    assert button_ids, "expected at least one <button id=...> in the UI"
    missing: list[str] = []
    for bid in sorted(button_ids):
        if bid in _DELEGATED_BUTTON_IDS:
            continue
        pattern = f'getElementById("{bid}")'
        if pattern not in src:
            missing.append(bid)
    assert not missing, (
        f"buttons with no getElementById handler binding: "
        f"{missing}. Either bind one in the script block or add the id "
        "to _DELEGATED_BUTTON_IDS with a comment.")


# ---------------------------------------------------------------------
# Fetch URLs map to routed API endpoints
# ---------------------------------------------------------------------

_FETCH_PATH_RE = re.compile(
    r"\bfetch\s*\(\s*`\$\{API\}([^`\?]+?)(?:\?[^`]*)?`")


def _routed_paths() -> set[str]:
    from app.main import create_app
    app = create_app()
    out: set[str] = set()
    for r in app.routes:
        path = getattr(r, "path", "") or ""
        if path.startswith("/api/v1"):
            out.add(path[len("/api/v1"):])
    return out


def _normalise_template_path(p: str) -> str:
    # JS uses ``${caseId}`` etc. inside the path; map those onto the
    # FastAPI placeholder shape ``{caseId}`` for comparison.
    return re.sub(r"\$\{[^}]+\}", lambda m: "{" + m.group(0)[2:-1] + "}", p)


def test_every_fetch_path_resolves_to_a_real_route():
    src = _html()
    js_paths = {_normalise_template_path(p)
                for p in _FETCH_PATH_RE.findall(src)}
    assert js_paths, "expected at least one fetch path in the UI"
    routed = _routed_paths()
    # Build a placeholder-insensitive match: a JS path matches a routed
    # path when they have the same shape ignoring placeholder names.
    def _shape(p: str) -> str:
        return re.sub(r"\{[^}]+\}", "{x}", p)
    routed_shapes = {_shape(p) for p in routed}
    missing = [p for p in sorted(js_paths)
               if _shape(p) not in routed_shapes]
    assert not missing, (
        f"UI fetches paths that are not routed by FastAPI: {missing}\n"
        f"Routed shapes:\n  " + "\n  ".join(sorted(routed_shapes)))


def test_every_dynamic_path_segment_is_safely_handled():
    """Every ``${...}`` that lands inside a fetch URL must either:

    * be wrapped in ``encodeURIComponent(...)`` so a stray ``/`` or
      ``?`` can't smuggle a path or query, OR
    * be one of the documented server-supplied UUIDv4 identifiers
      that the JS receives back from the API (those are not
      user-typed and have a fixed shape).
    """
    src = _html()
    placeholders = set(re.findall(
        r"fetch\s*\(\s*`\$\{API\}[^`]*\$\{([^}]+)\}", src))
    # Server-supplied UUIDv4-shaped values from API responses.
    server_supplied = {"caseId", "currentCase"}
    # ``params`` is a query string built with URLSearchParams — already
    # escaped by the URLSearchParams encoder, so it's safe.
    safe_helpers = {"params"}
    for p in placeholders:
        if p in server_supplied or p in safe_helpers:
            continue
        if p.startswith("encodeURIComponent(") and p.endswith(")"):
            continue
        # Anything else is a regression we want to be loud about.
        raise AssertionError(
            f"dynamic path segment ${{{p}}} is neither wrapped in "
            "encodeURIComponent nor a whitelisted server-supplied id")


# ---------------------------------------------------------------------
# Tab plumbing
# ---------------------------------------------------------------------

def test_every_tab_button_has_a_matching_pane():
    src = _html()
    tabs = set(re.findall(r'data-tab="([a-z0-9_-]+)"', src))
    panes = set(re.findall(r'id="tab-([a-z0-9_-]+)"', src))
    assert tabs == panes, (
        f"tab buttons / pane ids drift:\n"
        f"  tabs only:  {sorted(tabs - panes)}\n"
        f"  panes only: {sorted(panes - tabs)}")


def test_activate_tab_dispatches_each_tab_to_a_refresh_function():
    """When a tab is activated, the JS must call a real refresh
    function. The Cases tab is special — it's already refreshed by
    the boot path — but every other tab must dispatch."""
    src = _html()
    activate = re.search(
        r"function activateTab\(name\)\s*\{(.+?)\n\}",
        src, flags=re.DOTALL)
    assert activate is not None, "activateTab function missing"
    body = activate.group(1)
    # Each tab name appears in an ``if (name === ...)`` branch.
    for name in ("pipeline", "prompts", "rois", "storage", "config"):
        assert re.search(rf'name\s*===\s*"{name}"', body), (
            f"activateTab missing dispatch for tab {name!r}")


# ---------------------------------------------------------------------
# Status pills + style coverage
# ---------------------------------------------------------------------

def test_every_runtime_status_pill_class_has_a_style_rule():
    src = _html()
    # Pills the JS renders dynamically (from backend payloads).
    runtime_pills = {"OK", "WARNING", "ERROR", "UNKNOWN",
                      "VERIFIED", "REVIEW", "HIGH_RISK_REVIEW",
                      "INVALID_VIDEO", "OPEN", "CLOSED",
                      "IN_REVIEW", "REPROCESSING"}
    css_pills = set(re.findall(r'\.pill\.([A-Z_]+)\s*\{', src))
    missing = runtime_pills - css_pills
    assert not missing, (
        f"runtime pills with no CSS rule (would render unstyled): "
        f"{sorted(missing)}")


# ---------------------------------------------------------------------
# Labels operators see — pin a few stable strings
# ---------------------------------------------------------------------

def test_review_safe_labels_are_intact():
    src = _html()
    for label in (
        # Header H1 rebranded with the SCO UI conversion.
        "SCO Vision — Self-Checkout Reviewer",
        "Case Queue",                     # cases panel title
        "Pipeline status",                # pipeline tab title
        "Prompt editor",                  # prompts tab title
        "Camera ROIs &amp; model views",  # ROI tab title (HTML-escaped)
        "Disk &amp; retention",            # storage tab title (HTML-escaped)
        "Effective configuration",        # config tab title
        "Model controls",                 # model-controls panel title
        "Processing timings",             # timing legend title
        "Reviewer Decision",              # reviewer actions section
    ):
        assert label in src, f"reviewer UI lost label: {label!r}"


def test_review_safe_reviewer_actions_all_have_human_text():
    src = _html()
    # Each ``data-action`` button has a visible label after the
    # checkbox-style emoji prefix. We pin the text strings the
    # operator clicks. The verified-action label was rebranded from
    # "Verified physical return" → "Verified — basket matches" with
    # the SCO UI rebrand; the data-action attribute is unchanged so
    # historical audit rows keep their meaning.
    for visible in (
        "Verified — basket matches",
        "Needs review",
        "High-risk review",
        "Invalid video",
        "Camera blind spot",
        "POS / camera mismatch",
    ):
        assert visible in src, f"reviewer action button lost text: {visible!r}"


def test_admin_token_input_present_for_every_write_panel():
    """Every panel that calls a token-gated PATCH must surface a
    password-style admin-token input. UI must not silently send saves
    without giving the operator a place to enter the token."""
    src = _html()
    for token_input_id in (
        "prompt-admin-token",          # PATCH /admin/prompts/{camera_id}
        "roi-admin-token",             # PATCH /admin/camera-rois/{camera_id}
        "model-controls-admin-token",  # PATCH /admin/model-controls
        "storage-admin-token",         # POST /storage/cleanup/execute
    ):
        pattern = (rf'id="{token_input_id}"[^>]*type="password"')
        assert re.search(pattern, src), (
            f"admin-token input missing or wrong type: {token_input_id!r}")
