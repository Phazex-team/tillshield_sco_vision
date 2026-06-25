# Prompt Changelog

Tracks every change to the VLM/Falcon prompts in `classifiers.py` (and any
provider prompt overrides): **what the prompt was, when it changed, why, and
what the current active prompt is.** Newest entries on top.

Prompts live in `classifiers.py` → `CLASSIFIERS[<key>]` (`falcon_prompt`,
`gemma_system`, `gemma_user`) and the shared `_REVIEW_SAFE_JSON_SCHEMA`.
Review-safe constraint (unchanged across all versions): prompts must never use
the words *fraud / fraudulent / theft / suspect* — the admin prompt safety scan
enforces this. The VLM only describes; `reasoning.decision_policy` decides.

---

## Current active prompt — `return_review` (as of 2026-06-25)

**`gemma_user`:**
```
Describe what the camera shows of the return-counter interaction for THE
customer who triggered this session (ignore bystanders):
1) Is the customer's own physical product visible at the counter at ANY point
   in the clip — placed on it, held, or handed to staff? Judge this by the
   product being PRESENT and associated with the customer, NOT by catching the
   exact hand-over motion (the moment of release may fall between frames). It
   need not be a fixture from start_objects.
2) Is that product a tangible item (bag, clothing, box, package, etc.), not
   just a receipt or paper document?
3) Was a receipt or document visible?
4) Was your view obstructed at any point?
5) One sentence: what the camera shows.

Respond with this exact JSON schema only:
{ handover_occurred, physical_item_presented, receipt_visible, items_observed,
  customer_description, narrative, confidence, obstructed, camera_view_clear,
  limitations }
```

**`gemma_system`** (key rules — full text in `classifiers.py`):
- FIXTURE RULE: object labels in `start_objects` are pre-existing fixtures →
  excluded from the handover judgement. *(Note: in the modular `case_runner`
  path `start_objects` is currently always empty — perception does not tag
  tracks with `role="start"`, only keyframes carry roles — so this rule is
  effectively inert in production today.)*
- `handover_occurred=true` when the customer's own product is visible at the
  counter (handed / placed / held), based on **presence, not motion**.
- If only staff and no customer present → `handover_occurred=false`,
  `physical_item_presented=false`, say "no customer present".
- `physical_item_presented=true` if a tangible product (not just paper)
  belonging to the customer is visible at the counter at any point — placed,
  held, or handed; **based on presence, not the motion**.
- Never use fraud/theft/suspect; describe only what is visible; lower
  `confidence` if unsure.

**`falcon_prompt`:**
```
bag, shopping bag, clothing, shirt, box, package, paper, document, receipt,
phone, card, wallet, item, product
```

---

## Change history

### 2026-06-25 — `return_review`: presence-based item/handover questions

**Why:** Investigated a real case (`fab31b98`, txn `…193467`) where a customer's
blue bag was clearly on the counter (Falcon detected it; a direct probe showed
Gemma *sees* it — *"YES, a blue bag, a black shopping bag, and a bottle"*), yet
the pipeline reported `item_count=0 / handover_occurred=false → REVIEW`.

Root cause (proven, not assumed): the **user-turn questions were act-oriented**
— *"Did the customer **hand** any items to staff?"* / *"Was a product visible
**being presented**?"* — and **contradicted** the system prompt's own
presence-based rule (*"…placed on the counter… base on presence, not on
catching the hand motion"*). For returns where the item is already placed
(here the customer arrived ~8 min before the POS event), there is no
hand-over *act* in-frame, so Gemma answered "no" to the act-questions even
though the item was plainly present. Confirmed by probing Gemma on the actual
handover frame with the old questions → "No, no product being presented", vs a
presence-phrased question → "Yes".

Also ruled out by evidence (so they are NOT the cause): window size (a 500s
re-timed window reaching before the customer's arrival did not change it),
`start_objects`/fixture exclusion (empty), Gemma's vision (sees the bag),
Qwen-being-down / Gemma capability (direct question answers correctly).

**Change — `gemma_user` questions 1 & 2:**

Before:
```
1) Did the customer hand any items to staff?
2) Was a physical product visible being presented?
```
After:
```
1) Is the customer's own physical product visible at the counter at ANY point
   in the clip — placed on it, held, or handed to staff? Judge this by the
   product being PRESENT and associated with the customer, NOT by catching the
   exact hand-over motion (the moment of release may fall between frames). It
   need not be a fixture from start_objects.
2) Is that product a tangible item (bag, clothing, box, package, etc.), not
   just a receipt or paper document?
```

**Change — `gemma_system` `physical_item_presented` rule:**

Before:
```
Set physical_item_presented if a tangible product (not just paper/documents)
is visible being presented.
```
After:
```
Set physical_item_presented true if a tangible product (not just
paper/documents) belonging to the customer is visible at the counter at any
point — placed, held, or handed. Base this on the product being PRESENT, not
on catching the hand-over motion.
```

**Safety:** No weakening of false-positive protection. VERIFIED still requires
the Falcon **track-gate** (`physical_item_track` + `item_reaches_counter`) **+
`customer_present`** in `reasoning.decision_policy`. Reporting item-present
cannot, by itself, clear a staff-only refund with no customer.

### (prior, undated) — conservative tuning for false-positive prevention

Earlier in the project the `return_review` prompt was tightened to prevent
false-VERIFIED on staff-only refunds: added the "only describe a customer if a
person is clearly on the customer side; if only staff, set
handover_occurred=false and say 'no customer present'" rule, the act-oriented
"never infer intent" framing, and the FIXTURE RULE. The 2026-06-25 change
above corrects an over-correction from this tuning (act-questions causing false
negatives on already-placed items).
