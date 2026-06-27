# SCO Vision v1 — Design Notes

**What this repo is:** `sco_vision/` is a renamed copy of `fraud_detection_v3`
adapted into a Self-Checkout (SCO) basket-match verification app. The legacy
refund-review flow stays on disk but is **not** the active runtime path.

**Why a separate copy:** the user wanted a parallel SCO app without disturbing
the production refund-review repo. Multi-scenario routing inside one repo was
considered and explicitly deferred — see "Deferred to v1.1" below.

## What v1 does

```
POST /api/v1/pos/returns/event  (alias normalised to canonical SALE)
  ↓
case opens (sqlite/postgres: pos_events + cases)
  ↓
auto-analyser (or manual reprocess)
  ↓
plan_window  →  build_window  →  perception (Falcon ONCE on wide POS window)
                                    in sco_audit_zone ROI
                                    queries = defaults + per-line SCO + generic
  ↓
episode_selector  (derives customer/group episode from person tracks
                   anchored to POS time; flags ambiguous when overlapping)
  ↓
VLM (Qwen3-VL primary, Gemma fallback)
   prompt = sco_basket_match_v1
   system: forbid {fraud, theft, scanned, unscanned, ...}
   user: POS basket + Falcon summary + episode meta + JSON shape
  ↓
ScoBasketMatch (pydantic) → sco_policy.decide_sco
  ↓
outcome ∈ {VERIFIED, REVIEW, INVALID_VIDEO} + risk_reasons[] tags
```

## v1 design decisions (do not "fix" these)

| Decision | Why |
|---|---|
| **No DB migrations.** | Stay inside existing `case.outcome` enum: `{VERIFIED, REVIEW, HIGH_RISK_REVIEW, INVALID_VIDEO}`. SCO never emits `HIGH_RISK_REVIEW`. |
| **No new API routes.** | Reuse `POST /api/v1/pos/returns/event`. Basket goes in `raw_payload.items` — no schema extension. |
| **Config-driven event acceptance.** | `sco_checkout.accept_event_types` + `canonical_event_type` in `config.yaml`. Case-insensitive normaliser (strip, uppercase, `space|hyphen → _`). |
| **No scenario dispatcher in v1.** | `app/case_runner.py` branches inline on `prompt_version == "sco_basket_match_v1"`. Future re-routing is one `if` line away. |
| **Falcon merge fix is reserved-keys-safe.** | `perception/falcon_client.py` always merges DEFAULTS first; custom categories cannot overwrite `{item, person, receipt}`. Tests prove this. |
| **SKU translator is deterministic and local.** | `perception/sku_translator.py` strips size/UoM/noise. **Brand tokens preserved** (`coke can`, `dove soap` give Falcon better matches than `can`/`soap`). Cache at `storage/sku_translator/cache.json`. Overrides at `config/sku_overrides.yaml`. **No LLM in the hot path.** |
| **Single ROI: `sco_audit_zone`.** | Defined on every active SCO camera. Falcon, Qwen3-VL, and Gemma model views point to this ROI only. The zone name MUST NOT contain the substring `customer` (the refund `customer_present` gate matches on that substring). |
| **Episode selector is SCO-specific.** | `perception/episode_selector.py` derives the customer episode from person tracks + POS time anchor. It does NOT call the refund `customer_present` gate — that gate is refund-counter-zone semantics and shouldn't leak into SCO. |
| **SCO policy has 5 strict gates for VERIFIED.** | video_usable + episode not ambiguous + coverage ≥ `MIN_EPISODE_COVERAGE` + VLM confidence ≥ medium + basket_match=yes with no missing AND no extras. Anything else → REVIEW. Sub-outcomes live in `risk_reasons[]` as stable machine tags (`sco_basket_match`, `sco_basket_mismatch`, `sco_episode_ambiguous`, etc.). |
| **Refund export disabled.** | `app/api/cases.py` gates `_EXPORT_POOL.submit(maybe_export_case, ...)` behind `integrations.refund_agent.enabled` (default `false`). The exporter module stays on disk — re-enable is a config flip. |

## File map

| File | Phase | Purpose |
|---|---|---|
| `perception/falcon_client.py` | 0.5 | Reserved-keys-safe category merge. |
| `perception/pipeline.py` | 0.5, 3 | Plumb `falcon_categories` through `run_perception` / `run_perception_on_window`. |
| `pos/event_normalizer.py` | 1 | Single source of truth: case-insensitive normalisation, `accepted_aliases`, `canonical_event_type`, `case_opening_types`. |
| `pos/schemas.py` | 1 | `PosEventIn.validate()` no longer hard-codes a type set — config-driven at boundary. |
| `pos/ingest.py` | 1 | Canonicalises `event_type` per event via a copy (never mutates caller's object — preserves batch idempotency). |
| `app/api/pos.py` | 1 | Endpoint normalises + 400s on rejection. |
| `pos/tillshield.py` | 1 | Uses SCO normaliser for canonical form when SCO config is present. |
| `pos/tillshield_poll.py` | 1 | Default `require_negative_amount: False`. |
| `perception/sku_translator.py` | 3 | Deterministic local SKU → visual query. |
| `perception/episode_selector.py` | 4 | `sco_audit_zone_occupancy()` + episode classification. |
| `reasoning/prompts/sco_basket_match.py` | 5 | Prompt builder (system + user). |
| `reasoning/schemas/sco_basket_match.py` | 5 | Pydantic output schema + `parse_or_fallback`. |
| `reasoning/sco_policy.py` | 6 | `decide_sco()` with 5 strict gates. |
| `app/case_runner.py` | 3, 4, 5, 6 | Wires SKU translator → Falcon categories, runs episode selector, builds SCO prompt, routes to SCO policy. When ROI extras are enabled AND SCO mode active, the user prompt is **caption + SCO prompt** (not Qwen's `DEFAULT_USER_PROMPT`). |
| `app/api/cases.py` | 7a | Gates refund export on `integrations.refund_agent.enabled`. |
| `config.yaml` | all | `sco_checkout` block, `cam_01.zones.sco_audit_zone` + model views, `integrations.tillshield.return_event_types` aligned, `refund_agent.enabled: false`, `reasoning.prompt_version: sco_basket_match_v1`. |

## Test layout

| File | Covers |
|---|---|
| `tests/test_falcon_categories_merge.py` | Phase 0.5 — defaults survive, reserved keys protected, pipeline plumbing. |
| `tests/test_pos_event_normalize.py` | Phase 1 — normalisation rules, SCO mode, legacy fallback. |
| `tests/test_sku_translator.py` | Phase 3 — cleanup, brand preservation, override priority, cache hit, builder. |
| `tests/test_episode_selector.py` | Phase 4 — clean / ambiguous / no-activity / overlapping / anchor-outside / label variants. |
| `tests/test_sco_prompt_and_schema.py` | Phase 5 — prompt content, schema normalisation, round-trip. |
| `tests/test_sco_policy.py` | Phase 6 — all 5 gates + 20+ row outcome matrix + "never HIGH_RISK_REVIEW" sanity. |
| `tests/test_refund_export_disabled.py` | Phase 7a — refund export gating. |
| `tests/test_sco_end_to_end.py` | Phase 8 — POST alias → case → analyse → outcome. |
| `tests/test_sco_active_config.py` | Council fix #6 — real `config.yaml` resolves `sco_audit_zone`, refund_agent disabled, ROI extras compose SCO prompt (not Qwen DEFAULT). |

## How to run

```bash
cd /home/fazil/workspace/projects/sco_vision
PYTHONPATH=. \
  /home/fazil/workspace/projects/fraud_detection_v3/venv/bin/python \
  -m pytest tests/ -q \
  --ignore=tests/smoke_falcon_local.py \
  --ignore=tests/smoke_qwen3_vl_local.py \
  --ignore=tests/smoke_sam2_local.py
```

Why the old venv: the local `sco_vision/venv` was created via `--no-deps`
because of a `requirements.lock` resolution conflict (datasets vs fsspec).
The old venv has every package installed; activation flips PATH back. Either:
finish populating the new venv (`pip install pytest pytest-asyncio pygments`
+ resolve the lock conflict), or keep using the old venv for development.

## Deferred to v1.1

- SCO exporter (`pos/sco_agent_export.py`) — Phase 7b.
- Matched-only-boxes overlay rendering in the audit UI.
- UI labels / titles / `app/main.py` FastAPI title.
- README rewrite.
- Real-camera `sco_audit_zone` ROI geometry tuning.
- Multi-camera SCO scaling.
- Local LLM SKU translator fallback (if deterministic cleanup proves
  insufficient on real POS data).
- Quantity verification (presence-only in v1; VLMs are unreliable at
  counting > 3 of the same class).
- Multi-scenario routing if SCO and refund need to live in one repo again.

## Pre-existing baseline failures

15 tests were already failing before the SCO conversion started (storage
retention, UI handlers, NVR password encoding, schema/concurrency cleanup,
review-safe label drift). They are unrelated to SCO and were intentionally
not fixed as part of v1. Investigation should be a separate ticket.
