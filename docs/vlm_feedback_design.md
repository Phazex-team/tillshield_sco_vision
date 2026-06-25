# VLM Feedback Loop — Design (Phase 1 + Phase 2)

Design for capturing human feedback on VLM (Vision Primary "Q" / Vision Fallback "G")
summaries and using it to improve outputs over time, **without real-time model
training**. Grounded in 2026 industry practice; extends the existing schema
(`ReviewAction`, `VlmRun`, `AuditLog`, the prompt safety scan) rather than
forking parallel structures. Targets the current stack: SQLite, single GB10
box, frozen local models.

> **Status:** design only — not yet implemented. No code changes have been made.

---

## Background: the hard truth

A frozen model does **not** learn from a correction at inference time. Telling
the model "you misunderstood" changes nothing for the next case — weights are
frozen and there is no memory between calls. Improvement must come from what we
do with the feedback downstream: every credible approach is a pipeline of
**capture → store → feed back** via one of four standard mechanisms.

### The standard industry spectrum (cheapest/safest → heaviest)

| # | Approach | What it is | Learns? | Risk |
|---|---|---|---|---|
| 1 | Structured feedback capture | Reviewer marks each VLM summary right/wrong, corrects the specific fields, adds a note. | — (foundation) | none |
| 2 | Dynamic few-shot / RAG over corrections | At inference, retrieve similar corrected cases and inject as worked examples. | "improves" instantly, no training | low |
| 3 | Self-critique / reflection (Reflexion, Self-Refine, LLM-as-judge) | A second pass checks the output against the rules + exemplars and revises. | per-run only | low |
| 4 | Prompt iteration from failure clusters | Cluster failures, update the system prompt (Rules editor). | — (we learn) | low |
| 5 | Periodic offline fine-tune — SFT then DPO/KTO | Batch-train on accumulated corrections, redeploy. KTO learns from thumbs-up/down. | yes (real) | high |

Research consensus: for "the model misread X," **exemplar memory + retrieval
beats fine-tuning** (fine-tuning is forgetting-prone for knowledge injection),
and blanket VLM fine-tuning causes **catastrophic forgetting** of zero-shot
ability. DPO/KTO are the standard *when* you do train.

**Recommendation:** build Phase 1 (capture) now, then Phase 2 (dynamic
few-shot). Treat fine-tuning (Phase 3) as a later, gated step once there is
volume + an eval harness.

---

## Phase 1 — Feedback capture schema

### Design principle

Feedback must pin to the **exact `VlmRun`** that was judged (cases reprocess →
many runs; feedback on a stale output would poison the dataset), capture a
**per-field delta**, and tag **which layer** actually erred — so the VLM is
never "trained" to fix a perception/ROI or policy fault (e.g. the
`customer_present` bug, which was an ROI gap, not a VLM-knowledge gap).

### New table: `vlm_feedback`

| Column | Type | Purpose |
|---|---|---|
| `id` | uuid PK | |
| `vlm_run_id` | FK → `vlm_runs.id` | pins to the versioned output judged |
| `case_id` | FK → `cases.id` | denormalized for querying |
| `reviewer_id`, `created_at` | | provenance |
| `verdict_agreement` | enum `agree` / `disagree` / `partial` | KTO/DPO-ready binary signal (cheap, abundant) |
| `responsible_layer` | enum `vlm` / `perception` / `policy` / `video` / `unknown` | failure attribution — routes the correction to the right fix |
| `field_corrections` | JSON list of `{field, model_value, correct_value, comment}` | structured delta over the VLM output fields (`handover_occurred`, `customer_present`, `item_presented`, `item_count`, `narrative`, `confidence`) |
| `tags` | JSON array | failure-mode labels (`missed_customer`, `hallucinated_handover`, `wrong_item_count`) for clustering |
| `correction_note` | text | free text, safety-scanned (same scanner as prompts) |
| `prompt_version`, `model_name`, `provider` | snapshot from the run | which prompt/model produced the judged output |
| `usable_as_exemplar` | bool, default false | curation gate — nothing feeds Phase 2 until approved |
| `safety_scanned` | bool | |

### Why these choices

- **`verdict_agreement` (ternary)** = the thumbs-up/down that **KTO** consumes
  directly in Phase 3. Collect it now, for free.
- **`field_corrections` (model_value vs correct_value)** is one artifact serving
  three masters: exemplar content (Phase 2), SFT/DPO target (Phase 3), and eval
  label (regression set). Reuse, don't fork.
- **`responsible_layer`** is the failure-attribution guardrail made concrete:
  disagreements attributed to `perception`/`policy` route to ROI/gate tuning;
  **only `vlm`-attributed ones feed the VLM exemplars.**
- **`usable_as_exemplar` gate**: raw reviewer feedback never auto-feeds prompts
  (quality + review-safety). Standard data-curation hygiene.

### Fit with existing tables

- **Complements `ReviewAction`** (case verdict) — do not merge. `ReviewAction` =
  "what the human decided for the case"; `vlm_feedback` = "what was wrong with
  this specific model output." Optionally store `review_action_id`.
- Every write also goes through `audit.record()` → `AuditLog` (existing pattern).
- `correction_note` + exemplar text reuse the existing prompt safety scan.

### API + UI surface

- `POST /api/v1/cases/{case_id}/vlm-feedback` — body: `vlm_run_id` (defaults to
  latest run), `verdict_agreement`, `responsible_layer`, `field_corrections[]`,
  `tags[]`, `note`. Mirrors the existing `review-actions` endpoint.
- **UI:** on the *VLM verdict card*, a "Was this reading correct?" control →
  **agree / correct**. "Correct" expands the structured fields pre-filled with
  the model's values; reviewer edits the wrong ones, picks the responsible
  layer, adds a note. This is the "feedback on every VLM summary" surface.

---

## Phase 2 — Dynamic few-shot retrieval design

### Goal

At inference, retrieve the top-k most similar **curated** corrected cases and
inject them as worked examples into the Vision prompt — frozen models "improve
every time" with **no training**.

### 2.1 The retrieval key (what to embed)

**Hybrid** (recommended):

- **Structured features** (already in the DB, cheap): pos `event_type` + amount
  sign, zone-hit pattern (counter/staff/customer track presence), Falcon label
  histogram (person/item/receipt counts), `camera_id`.
- **+ Visual embedding**: pooled CLIP-style (or the VLM's own vision-encoder)
  embedding of the manifest keyframes — captures scene layout/occlusion.

Do **not** embed the `narrative` — it's the thing being corrected (circular).

### 2.2 Where vectors live (no new infra)

On SQLite + single box: **`sqlite-vec`** (or in-process **FAISS**). Do not stand
up a separate vector-DB service for this scale. Index only
`usable_as_exemplar=true` rows:
`{feedback_id, vlm_run_id, case_id, embedding, exemplar_payload, prompt_version, model, camera_id}`.

### 2.3 Retrieval at inference (inside the VLM stage)

1. Embed the current case (same encoder).
2. Top-N nearest neighbors, filtered: same `camera_id`,
   `usable_as_exemplar=true`, `responsible_layer='vlm'`, exclude the current case.
3. Re-rank for diversity: drop near-dupes; prefer a mix of corrected-negatives
   (known failure modes) + a couple of positives for balance.
4. **Hard cap k = 2–3** — each exemplar costs prompt tokens (and a thumbnail
   costs vision tokens) → competes with frames for the KV cache. Budget it.

### 2.4 Exemplar shape in the prompt (review-safe, anti-anchoring)

A compact block appended to the user prompt, framed as **calibration, not the
answer**:

> *"For calibration — in a similar scene: customer_present=true,
> handover_occurred=true. (The customer was at the right edge, partially
> occluded.) Judge the current frames on their own merits."*

Start text-only; add one thumbnail per exemplar later if lift justifies the
vision-token cost.

### 2.5 Guardrails

- **Anchoring bias** is the #1 risk: few-shot can drag the verdict toward the
  exemplar. Mitigate with mixed positive/negative exemplars, small k,
  calibration framing, and an over-anchoring metric.
- **Versioning/audit:** extend `VlmRun.input_manifest` with
  `retrieved_exemplar_ids` + `exemplar_set_version` → auditability + lift
  measurement + rollback.
- **Cold start is graceful:** no curated feedback → retrieval returns nothing →
  behaves exactly like today.
- **Review-safe:** exemplar text passes the same safety scan; no
  "fraud"/"theft" wording.

### 2.6 Measuring lift (mandatory before trusting it)

- Freeze a holdout of corrected cases as a regression/eval set (Phase 1 data
  doubles as this).
- A/B: same case with vs without exemplars → measure agreement-rate gain **and**
  over-anchoring (does it flip correct AGREE cases to wrong?).
- This same harness is the gate for Phase 3 fine-tuning.

---

## The closed loop

```
Reviewer corrects a VLM summary
  → vlm_feedback row (+ AuditLog)
  → curation gate (usable_as_exemplar=true)
  → embed + index (sqlite-vec/FAISS)
  → next similar case retrieves top-k
  → injected into Vision prompt (logged in input_manifest)
  → new VlmRun → reviewer feedback on THAT → loop
```

`responsible_layer` is the switch: `vlm` → exemplars; `perception`/`policy` →
ROI/gate tuning instead.

---

## Build increments (no new services, no training)

- **Phase 1:** a `vlm_feedback` table + migration + `POST .../vlm-feedback`
  endpoint + a feedback control on the VLM verdict card.
- **Phase 2:** an embedder + a `sqlite-vec` index + retrieve-and-inject in the
  VLM stage, behind a feature flag, with the eval harness as the safety gate.
- **Phase 3 (later, gated):** offline KTO/DPO on accumulated verdicts, evaluated
  against the regression set before redeploy.

## References

- Post-Training in 2026: GRPO, DAPO, RLVR & Beyond — https://llm-stats.com/blog/research/post-training-techniques-2026
- Continual Learning of Vision-Language Models — survey & taxonomy — https://github.com/YuyangSunshine/Awesome-Continual-learning-of-Vision-Language-Models
- ICAL: VLM Agents Generate Their Own Memories — https://arxiv.org/abs/2406.14596
- Synthetic Data is an Elegant GIFT for Continual VLMs — https://arxiv.org/pdf/2503.04229
- Enhanced Continual Learning of VLMs with Model Fusion — https://arxiv.org/pdf/2503.10705
