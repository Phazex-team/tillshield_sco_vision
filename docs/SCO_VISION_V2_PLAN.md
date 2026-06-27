# SCO Vision v2 — Roadmap

Sibling to [`docs/SCO_VISION_V1.md`](SCO_VISION_V1.md). v1 shipped the
config-driven SCO basket-match pipeline; v2 is what we need to do before
this is safe to put in front of real customers and staff.

This is a working plan, not a marketing document. Every workstream is
tied to a concrete file in the repo and to an acceptance test we can
write before merging.

---

## 1. v2 goals

1. **Stronger item/count confidence.** Move the public claim from
   "basket composition appears consistent" to "POS line items are
   visibly accounted for with N visible instances each (or
   uncertain)". Counts come from multi-frame tracking, not from a
   single VLM glance.

2. **Reviewer/audit UX.** Matched-only bounding boxes by default in
   the evidence overlay. Toggle to reveal unmatched candidates.
   Full `audit.json` containing every detection (matched, unmatched,
   generic, ambiguous) so an investigator can re-derive any signal
   without re-running the pipeline.

3. **Deployment hardening.** Replace the placeholder
   `sco_audit_zone` geometry with per-camera calibrated zones,
   validate ROI scaling at boot, support multiple SCO terminals
   mapped to distinct cameras, and add operational telemetry
   (Phoenix spans + a small ops dashboard panel).

4. **Defensible language.** Continue to never say
   `theft`, `fraud`, `dishonest`, `scanned`, or `unscanned` in
   prompts, schema, policy reasons, evidence text, or UI labels.
   Add a CI check that fails if any of these tokens appears in
   user-visible output paths.

---

## 2. v2 non-goals

| Non-goal | Why |
|---|---|
| **No accusation automation.** | The pipeline reports observation, the human reviews. Even a "high-confidence mismatch" still routes to REVIEW. |
| **No automatic `HIGH_RISK_REVIEW`.** | Reserved for v2.x and only with explicit product approval. SCO v1 and v2 outcomes stay in `{VERIFIED, REVIEW, INVALID_VIDEO}`. |
| **No "scan event" claim from video alone.** | The camera cannot see a barcode scan or a weight reading. Even if a basket-match is perfect, we say "items visible match POS bill", never "all items were scanned". |
| **No uncontrolled network/LLM calls.** | The SKU translator stays local. Optional LLM fallback (v2.1) is gated by config, runs against the locally-served VLM, and is cached. No SaaS endpoints. |
| **No multi-zone architecture by default.** | Single `sco_audit_zone` per camera unless we collect operational evidence that one ROI cannot disambiguate scanner/bagging/customer behaviour. |
| **No re-ID promise.** | Episode selector improves person/group continuity but never claims to identify "the same customer across two visits". |

---

## 3. Workstreams

Each workstream lists the v1 file it builds on, what the v2 change is,
and what the failure mode is if we skip it.

### W1. SCO exporter

**Builds on:** the disabled refund exporter call site at
[`app/api/cases.py`](../app/api/cases.py) (success branch in
`_run_reprocess`, currently gated on
`integrations.refund_agent.enabled`). The legacy refund-side reference
implementation is `pos/refund_agent_export.py`.

**New files:**
- `pos/sco_agent_export.py` — symmetric to refund exporter. Exposes
  `maybe_export_sco_case(case_id, cfg=None)`. One-way, guarded, never
  raises out of the export pool.
- `pos/_export_common.py` — bbox renderer, clip encoder, face-mask
  helper. Extracted only after the SCO exporter is written and the
  duplication is real (avoid pre-factoring).

**Config:**
- New `integrations.sco_agent` block (enabled, base_url, create_path,
  evidence_path, timeout_sec, send_video, mask_customer_faces).
- New top-level boolean for which exporter runs (or both, or
  neither). Default: `sco_agent.enabled: true`,
  `refund_agent.enabled: false`.

**Outputs the exporter writes:**
- Matched-only overlay MP4 (clip + boxes only for detections tagged
  `matched`).
- `audit.json` containing the full per-detection tag list (matched,
  unmatched, generic, ambiguous), plus the SCO basket-match VLM
  output, plus the `sco_episode` block, plus
  `risk_reasons[]`.
- Optional unmatched-overlay MP4 (off by default; rendered on demand
  in the reviewer UI).

**Failure mode if skipped:** evidence package retains v1's frame-only
shape; reviewer has to mentally combine VLM JSON, episode dict, and
raw detections to understand why a case landed in REVIEW.

### W2. UI rebrand + reviewer workflow

**Builds on:** `static/review.html` and the case-grid JS. The FastAPI
title still says "Return / Refund Visual Review" at
`app/main.py:86`.

**Changes:**
- FastAPI title → `"SCO Vision — Self-Checkout Reviewer"`.
- Case grid columns: drop refund-specific columns (refund_amount,
  handover_occurred); add SCO-specific (basket_match,
  matched/missing/extra counts, episode ambiguity flag, coverage).
- Risk-reason filter chips for the SCO tag namespace: `sco_basket_match`,
  `sco_basket_mismatch`, `sco_episode_ambiguous`,
  `sco_episode_short`, `sco_low_confidence`, `sco_missing_items`,
  `sco_extra_candidates`, `sco_bad_footage`.
- Evidence viewer: matched-only overlay loads by default;
  "Show unmatched candidates" toggle re-loads the unmatched MP4 if
  the exporter produced one.
- Header text scrub: any remaining `Refund`, `Return`, `Handover`
  copy that's now misleading.

**Failure mode if skipped:** operators see refund-shaped UI for an
SCO product. False-flag risk: a reviewer used to refund semantics
will misread basket-match outcomes.

### W3. ROI calibration

**Builds on:** the placeholder geometry under
[`config.yaml`](../config.yaml) — both `cam_01` and `cam_return_01`
currently define `sco_audit_zone` as a default rectangle covering most
of a 1920×1080 frame. ROI scaling math lives in
`app/camera_rois.py` (`scale_zones_to_frame`).

**Changes:**
- Per-camera calibration captured via the existing admin UI
  (`PATCH /api/v1/admin/camera-rois/{cam_id}`) against a real frame
  snapshot.
- Document the calibration steps in
  [`docs/SCO_VISION_V1.md`](SCO_VISION_V1.md) → add a
  "Per-camera calibration" section.
- Add a boot-time check that warns (not crashes) when
  `sco_audit_zone.w * h` exceeds 90% of `source_width * source_height`
  (the placeholder heuristic) so operators can't accidentally ship
  the default.
- Frame-size scaling regression: extend
  [`tests/test_visual_roi_calibration.py`](../tests/test_visual_roi_calibration.py)
  with a `sco_audit_zone` case at a non-1080p frame size.

**Failure mode if skipped:** Falcon sees most of the frame including
the next terminal over → false unmatched extras → REVIEW spam.

### W4. Quantity / counting v2

**Builds on:** the perception tracker
([`perception/tracker.py`](../perception/tracker.py)), the episode
selector at
[`perception/episode_selector.py`](../perception/episode_selector.py),
and the basket-match schema at
[`reasoning/schemas/sco_basket_match.py`](../reasoning/schemas/sco_basket_match.py)
(currently presence-only: `visible_count_class ∈ {one, multiple, uncertain}`).

**Changes:**
- New `perception/item_counter.py` that consumes Falcon detections +
  tracker output across the selected episode and emits per-POS-line
  counts with confidence (`high`, `medium`, `low`,
  `not_countable`).
- Treat duplicates within a single track as 1; treat distinct tracks
  of the same label as separate instances.
- Schema additions in
  [`reasoning/schemas/sco_basket_match.py`](../reasoning/schemas/sco_basket_match.py):
  `MatchedItem.visible_count_class` widened to optionally include
  `visible_count_int: Optional[int]` and `count_confidence`.
  **Backward-compatible** — v1 callers and v1 evidence packages
  keep working.
- Policy update in
  [`reasoning/sco_policy.py`](../reasoning/sco_policy.py): a new
  optional gate `count_matches_pos_qty` that activates only when
  `count_confidence` is `high` or `medium`. Defaults to soft (a
  count mismatch is `REVIEW` with tag `sco_quantity_mismatch`,
  not `INVALID_VIDEO` and never auto-flag).
- Hard-coded limitations doc note: counting is unreliable for
  stacked items, items bagged before scanning, transparent
  packages, and bursts > 3 of the same class per frame.

**Failure mode if skipped:** v2 still says "one or multiple" for
quantities — fine for v1 but fails any operator question that
starts with "how many did the customer have?".

### W5. SKU translator v2

**Builds on:**
[`perception/sku_translator.py`](../perception/sku_translator.py) —
deterministic cleanup, brand-preserving, no LLM in hot path. Cache at
`storage/sku_translator/cache.json`. Overrides at
`config/sku_overrides.yaml`.

**Changes:**
- Overrides file is the primary tuning surface. Add a small admin
  UI for editing it (`/api/v1/admin/sku-overrides`) so retail ops
  can fix bad mappings without a code deploy.
- Optional local-LLM fallback gated by
  `sco_checkout.sku_translator.llm_fallback_enabled: false`
  (default off). When on: only invoke for SKUs that the
  deterministic rules + overrides didn't resolve; only call the
  locally-served VLM (no network); persist into the same JSON
  cache.
- Reviewer surface: in the case detail UI, show the
  `pos_item → falcon_query` mapping the case used, with a
  one-click "promote to override" action.
- Cache-coherence: bump cache schema to `v2`, drop legacy
  uppercase-only keys.

**Failure mode if skipped:** noisy retail SKU strings like
`"DOVE-WHITE 100G 6X"` produce Falcon queries that don't match the
visual product, the SCO basket-match goes `uncertain`, and the case
lands in REVIEW for no good reason.

### W6. Episode selector v2

**Builds on:**
[`perception/episode_selector.py`](../perception/episode_selector.py).
Current v1 outcomes: `clean_episode`, `multiple_groups` (ambiguous),
`long_continuous` (ambiguous), `no_activity`,
`anchor_outside_groups`.

**Changes:**
- Person/group continuity: keep the merge-by-gap rule but extend
  with a coarse motion-continuity heuristic so a track that briefly
  drops doesn't fracture the episode (operator-tunable `merge_gap_sec`
  per camera).
- Ambiguity metrics: instead of a binary flag, emit
  `episode.ambiguity_score ∈ [0.0, 1.0]` plus the existing
  `reason` tag. Policy in
  [`reasoning/sco_policy.py`](../reasoning/sco_policy.py)
  treats `ambiguity_score >= 0.5` as ambiguous; below 0.5
  downgrades VLM confidence one step instead of failing the gate.
- Customer leave-and-return: when a single group leaves and a new
  group enters during the POS window, emit
  `reason="customer_changeover"` with ambiguous=True. Do not
  attempt re-ID.
- Stays explicitly out of true person-identification scope. The
  selector still says "the group occupying `sco_audit_zone`
  around POS time", never "the same person".

**Failure mode if skipped:** busy SCO areas where customers
overlap rapidly produce ambiguous episodes → everything goes
REVIEW → the system is operationally useless during peak hours.

### W7. Falcon / VLM evidence alignment

**Builds on:** the Falcon category naming convention introduced in
[`app/case_runner.py`](../app/case_runner.py)
(`_summarise_falcon_for_sco`) — labels `sco_item_NNN` for POS-derived
queries, `sco_generic_*` for the catch-all, defaults for
`item/person/receipt`.

**Changes:**
- Persistence: every detection row gets a `match_tag` column / JSON
  field with one of `{matched, unmatched, generic, default,
  ambiguous}`. Migration adds a NULL-able column so v1 evidence
  packages still load.
- Evidence package writer ([`evidence/package.py`](../evidence/package.py))
  always serialises every detection; the matched-only overlay is a
  render-time choice in the exporter (W1), not a filtering
  decision at persist time.
- VLM prompt evidence section gets the same matched/unmatched
  split so the model isn't reasoning about a single opaque
  detection count.
- A reviewer-API endpoint returns the full audit detection list
  for the case (`GET /api/v1/cases/{id}/audit-detections`) so a
  data-team replay can re-derive any signal without re-running
  perception.

**Failure mode if skipped:** the reviewer cannot tell which
detections drove the outcome, can't promote a SKU override from
context, can't reproduce a disputed case.

### W8. Multi-camera / multi-terminal

**Builds on:** the TillShield poller's `workstation_camera_map` at
[`config.yaml`](../config.yaml) and the active-config tests in
[`tests/test_sco_active_config.py`](../tests/test_sco_active_config.py)
(`test_every_tillshield_workstation_camera_has_sco_audit_zone`).

**Changes:**
- Boot-time validator: refuse to start (or log a loud error and
  drain the poller) if any camera in `workstation_camera_map` is
  missing `sco_audit_zone` or has a misconfigured model view.
  Implementation in `app/startup.py` alongside the existing
  TillShield poll validation.
- Per-camera ROI calibration documented in W3 produces N distinct
  `sco_audit_zone` geometries.
- Per-camera Falcon weights are still loaded once per process —
  no architectural change needed unless we want to scale beyond
  ~4 SCO terminals per box, in which case W9 applies.

**Failure mode if skipped:** adding a new SCO terminal silently
ships with the default-coordinates `sco_audit_zone` placeholder
→ noisy detections → REVIEW spam.

### W9. Optional multi-scenario architecture

**Builds on:** the per-prompt-version branch in
[`app/case_runner.py`](../app/case_runner.py)
(`if prompt_version == "sco_basket_match_v1": ...`) and the
`prompt_version` thread that already exists end-to-end.

**Decision criteria for reintroducing routing:**
- Real ops asks for refund AND SCO in the same repo (running
  cost or org reason).
- Sharing the Falcon weight load is worth more than the per-repo
  isolation we have today (each repo can ship at its own pace).

**If reintroduced:**
- New `app/scenarios/__init__.py` dispatcher invoked at the top
  of `analyze_case()`. Routes by `case.pos_event.event_type`
  (NOT by a new DB column — same constraint as v1).
- Split `app/case_runner.py` into `_analyze_sco_case` and
  `_analyze_refund_case` for readability; the dispatcher is the
  only switch.
- Falcon as a shared service: lift `FalconClient` to a
  separate model server (e.g. a small FastAPI on `:8002`).
  Both repos call it over local HTTP. vLLM is already shared
  via `:8001` so this is a known pattern.
- Config separation: scenarios live in
  `config/scenarios/refund.yaml` and
  `config/scenarios/sco.yaml`, both merged into the root
  `config.yaml` at load time.

**Failure mode if skipped:** v1 stays as-is, refund and SCO live
in separate repos, double Falcon load. Acceptable for v2.

---

## 4. Acceptance criteria

| Workstream | Pass/fail | Tests |
|---|---|---|
| **W1 SCO exporter** | `pos/sco_agent_export.py` exists; for any case analysed in SCO mode with the exporter enabled, the export pool receives exactly one submit; the resulting evidence directory contains a matched-only MP4 + `audit.json` whose detection list length equals the perception result detection count. | `tests/test_sco_agent_export.py` — submit-once, on-success-only, never-raises; render test that verifies overlay frames contain boxes only for `match_tag=matched`. |
| **W2 UI / reviewer** | FastAPI title regex match for `^SCO Vision`. No `Refund` / `Return` / `Handover` tokens in `static/review.html` user-visible text (`<title>`, `<h1>`, button labels). Risk-reason filter chips include every `sco_*` tag emitted by [`reasoning/sco_policy.py`](../reasoning/sco_policy.py). | Extend `tests/test_ui_handlers.py` (after the pre-existing failure there is fixed): assert title, assert no refund tokens, assert chip set. |
| **W3 ROI calibration** | Boot-time warning fires on the placeholder geometry (`w*h >= 0.9 * source_w*source_h`). Frame-scaling regression covers a non-1080p decoded frame. | Extend [`tests/test_sco_active_config.py`](../tests/test_sco_active_config.py) with a `test_real_config_sco_audit_zone_not_placeholder` test that operators flip on once they've calibrated. |
| **W4 Counting** | `MatchedItem.visible_count_int` populates with `count_confidence ∈ {high, medium}` on a controlled test fixture (3-frame mp4 with 2 instances of one class). `tests/test_sco_policy.py` extended with quantity-mismatch matrix → REVIEW with `sco_quantity_mismatch` tag, never VERIFIED. | `tests/test_item_counter.py` (new); extend `tests/test_sco_policy.py`. |
| **W5 SKU translator v2** | `/api/v1/admin/sku-overrides` GET/PUT round-trips; promote-to-override action persists. Local LLM fallback gated off by default; flipping it on calls only the locally-served provider chain and writes the result to the cache. | `tests/test_sku_overrides_api.py` (new); extend `tests/test_sku_translator.py` with an LLM-fallback path test using a stub provider. |
| **W6 Episode selector v2** | `ambiguity_score ∈ [0, 1]` present on episode dict for clean / overlapping / customer-changeover synthetic tracks. SCO policy demotes confidence (not fails outright) when `0.2 <= ambiguity_score < 0.5`. | Extend `tests/test_episode_selector.py`; extend `tests/test_sco_policy.py` matrix. |
| **W7 Falcon/VLM alignment** | Every persisted detection has `match_tag != null`. Audit-detections endpoint returns the full list. The matched-only overlay test from W1 fails fast if `match_tag` is missing. | `tests/test_evidence_match_tag.py` (new). |
| **W8 Multi-camera** | Adding a new camera to `workstation_camera_map` without `sco_audit_zone` is rejected at boot (logged + poller drained). Adding it WITH `sco_audit_zone` passes. | Extend `tests/test_sco_active_config.py::test_every_tillshield_workstation_camera_has_sco_audit_zone` (already partially in place — extend with an explicit boot-time validator hook). |
| **W9 Multi-scenario (if invoked)** | Refund flow regression remains green with `prompt_version="return_review_v1"`. SCO and refund cases route to their adapters via the dispatcher with no false branch. | Existing refund regression tests + new `tests/test_scenario_dispatcher.py`. |

---

## 5. Risks and constraints

| Risk | Where it bites | Mitigation in v2 |
|---|---|---|
| **VLM quantity counting is weak past 3–4 similar items.** | W4 (counting). | Counts derived from Falcon tracker, not VLM. VLM only adjudicates ambiguous cases. Hard policy ceiling: counts always carry a confidence field; low-confidence counts NEVER block VERIFIED on their own. |
| **Busy SCO scenes cause customer mixing.** | W6 (episode selector). | Ambiguity score, not binary flag. Episode goes ambiguous-with-degraded-confidence rather than ambiguous-and-rejected. `customer_changeover` is a first-class reason. |
| **Bagged/stacked items reduce visual certainty.** | W4, W7. | Visible_count is `not_countable` when the tracker can't see distinct boxes. Documented limitation surfaced in the reviewer UI. |
| **Real POS SKU strings need override quality.** | W5. | Admin UI for overrides, promote-from-context action, JSON cache versioning. Local LLM fallback is gated off by default; deterministic rules first. |
| **Falcon double-load when refund and SCO run independently.** | W8, W9. | v2 stays per-repo. W9 lays the path to a shared Falcon service if memory pressure becomes a real ops complaint. |
| **DGX unified memory cap (121 GiB) under simultaneous SCO + refund + co-tenants.** | All workstreams that add a model. | Documented in auto-memory. Any v2 model addition must respect the existing `gpu.{soft,hard,emergency}_memory_limit_gb` gates. No new heavyweight loads without an explicit unload-on-pressure plan. |
| **Operator runs the placeholder ROI in production.** | W3. | Boot-time warning + active-config test that flips green once calibrated. |
| **Refund regression silently breaks because nobody runs it.** | W9 deferment. | CI runs the legacy refund path with `prompt_version="return_review_v1"` on every commit. Already true in v1 — keep it. |

---

## 6. Recommended order

| Release | Workstreams | Rationale |
|---|---|---|
| **v1.1** | W1 (SCO exporter) + W2 (UI rebrand) + part of W7 (`match_tag` on detections) | Highest reviewer-visible value. Closes the "this looks like the refund product" gap before adding any model complexity. No new model code. |
| **v1.2** | W3 (ROI calibration) + remainder of W8 (boot-time validator) + operational smoke under real footage | Pre-flight before any pilot. Cheap to do, expensive to skip. |
| **v2.0** | W4 (multi-frame item tracking and count confidence) + W6 (episode selector v2) + remainder of W7 (audit-detections endpoint) | The first release where we make a stronger public claim than "appears consistent". Needs all three together: tracker→count, episode→confidence floor, full audit so disputes are defensible. |
| **v2.1** | W5 (SKU translator v2: admin UI + optional local LLM fallback) | Address the "real POS strings are messy" problem after we have real ops data on which SKUs fail. |
| **v2.2** | W9 (optional multi-scenario / shared Falcon service) | Only triggered if ops demands one-repo refund+SCO OR if memory pressure from double Falcon load becomes a real complaint. |

Each release is a separate commit chain with its own regression baseline,
documented the same way [`docs/SCO_VISION_V1.md`](SCO_VISION_V1.md)
captures v1 design decisions.

---

## Cross-references

- v1 design: [`docs/SCO_VISION_V1.md`](SCO_VISION_V1.md)
- Case orchestrator: [`app/case_runner.py`](../app/case_runner.py)
- SKU translator v1: [`perception/sku_translator.py`](../perception/sku_translator.py)
- Episode selector v1: [`perception/episode_selector.py`](../perception/episode_selector.py)
- SCO prompt v1: [`reasoning/prompts/sco_basket_match.py`](../reasoning/prompts/sco_basket_match.py)
- SCO output schema v1: [`reasoning/schemas/sco_basket_match.py`](../reasoning/schemas/sco_basket_match.py)
- SCO decision policy v1: [`reasoning/sco_policy.py`](../reasoning/sco_policy.py)
- Active-config tests: [`tests/test_sco_active_config.py`](../tests/test_sco_active_config.py)
- Live config: [`config.yaml`](../config.yaml)
