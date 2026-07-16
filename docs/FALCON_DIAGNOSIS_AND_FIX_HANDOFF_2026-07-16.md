# Temporary Falcon Diagnosis and Fix Handoff

> **Temporary working document.** Keep this file in the repository while the
> Falcon correctness/performance work is incomplete. Delete it only after every
> required acceptance criterion in the final section has been implemented and
> validated. If a future session completes only part of the work, update the
> checklist and leave this document in place.

Date captured: 2026-07-16 (Asia/Dubai)

Repository: `/home/phazex/workspace/tillshield/sco_vision`

## Non-negotiable guardrail

The Qwen3-VL and Gemma VLMs are intentionally disabled by the operator because
their memory usage previously froze the machine. Do not re-enable either VLM,
undo the operator's `start.sh` changes, or make a VLM a dependency of the fix.
The work described here is specifically for the Falcon-only path.

At the time this document was written, the worktree already contained changes
that do not belong to this diagnosis:

```text
 M app/auto_analyzer.py
 M start.sh
```

Preserve and inspect those changes before editing overlapping files. In
particular, `app/auto_analyzer.py` contains a live, uncommitted change that
signals container PID 1 when a wedged reload child must trigger a real container
restart. It was active through WatchFiles but had not yet been validated by a
second watchdog event.

## Executive summary

The recent screenshot is valid evidence of two separate problems:

1. Falcon dominates processing time. The VLM contributed zero milliseconds to
   the measured case.
2. Product names shown in the UI are not verified Falcon classifications. The
   application asks Falcon a product-name query, accepts whatever boxes are
   returned, stamps those boxes with the requested POS label, assigns a
   synthetic score of `0.5`, and then turns the stamped label back into a
   POS-matched item group. This is circular and explains why checkout hardware
   is displayed as a bag or beverage can.

There is also an independent reprocess persistence bug. Reprocessing appends a
second complete perception result, and evidence packaging reads every result for
the case. This exactly doubled the screenshot case from 2,692 to 5,384
detections and from 390 to 780 keyframes.

The correct direction is not simply another threshold tweak. Falcon should be
treated as an unverified object/localization signal unless a separate non-VLM
verifier establishes identity. Workload must also be reduced before attempting
lower-level engine optimization.

## User-visible evidence

Screenshot:

```text
/home/phazex/Documents/Screenshot from 2026-07-16 14-48-49.png
```

The screenshot maps to case:

```text
9ae758a0-2159-45a9-bdd8-c55adb2fe477
```

Visible problems:

- `HD Supermarket Bag` boxes most of the self-checkout kiosk.
- Two `Alkzy Breeze CaffnFr Can 250ml` cards box kiosk/payment/display
  hardware rather than cans.
- `Fresh Hummous` boxes a round food tub and is the only visually plausible
  product association, although it is still not verified.
- The three `not on receipt` cards include a duplicate round tub and two nearly
  identical dark crops.
- The UI reports 5,384 detections and 780 keyframes.
- The final outcome is `REVIEW` with count/identity uncertainty and suspected
  fragmentation.

Saved crops for the case are under:

```text
storage/cases/case_id=9ae758a0-2159-45a9-bdd8-c55adb2fe477/snapshots/
```

Important examples:

```text
item_00_sco_group_001.png  # kiosk labelled HD Supermarket Bag
item_01_sco_group_002.png  # kiosk labelled Alkzy can
item_02_sco_group_003.png  # second Alkzy association / bad crop
item_03_sco_group_004.png  # plausible round hummus tub
```

## Exact case measurements

The case video is 138 seconds, 1920x1080, 5 FPS, and 690 frames. The sampling
policy selected 345 frames per analysis attempt.

Two analysis attempts were persisted:

| Metric | Attempt 1 | Attempt 2 |
|---|---:|---:|
| Video window ID | `1eca889e-3e46-4bf2-a5da-6e8cb07d61a0` | `4d1e8e1f-685f-49f3-9753-4dfc530c8da5` |
| Sampled frames | 345 | 345 |
| Falcon time | 399.091 s | 405.740 s |
| Frame sampling time | 1.742 s | 1.777 s |
| Perception total | 400.873 s | 407.552 s |
| VLM latency | 0 ms (disabled) | 0 ms (disabled) |
| Detections | 2,692 | 2,692 |
| Tracks | 183 | 183 |
| Keyframes | 390 | 390 |

Combined DB/UI totals after reprocessing:

```text
5,384 detections
366 tracks
780 keyframes
170 unique keyframe frame indices
```

The combined label distribution was:

| Label | Detection rows | Frames with at least one box | Score range |
|---|---:|---:|---:|
| `sco_generic_products` | 1,266 | 280 | 0.5 only |
| `item` | 1,242 | 251 | 0.5 only |
| `person` | 762 | 331 | 0.5 only |
| `receipt` | 744 | 336 | 0.5 only |
| `sco_item_006` (bag) | 686 | 341 | 0.5 only |
| `sco_item_000` (Alkzy) | 332 | 166 | 0.5 only |
| `sco_item_001` (Alkzy) | 332 | 166 | 0.5 only |
| `sco_item_003` (hummus) | 20 | 9 | 0.5 only |

The bag query produced a box on 341 of 345 sampled frame indices. That is a
static-background/fixture signal, not evidence of a transaction item.

At the final health check on 2026-07-16, the live app was healthy on port 4001,
no cases were `REPROCESSING`, memory was about 12.2/121.7 GB, and the recorder
was healthy. This is only a timestamped observation, not a guarantee for the
next session.

Across the 17 completed runs carrying stage timings at that point:

```text
average Falcon time: 144.4 s
minimum Falcon time: 46.1 s
maximum completed Falcon time: 405.7 s
```

A separate case, `e41b9c86-e094-43db-a3a3-5eb7f2dc8422`, exceeded the
900-second watchdog and was quarantined with a reprocess timeout.

## Root cause 1: full-window frame x query fan-out

Relevant code:

- `perception/sampling.py:21-29` hardcodes `base_fps=3.0`.
- `perception/sampling.py:56-60` rounds a 5 FPS source to a step of two,
  effectively sampling 2.5 FPS for this clip.
- `perception/pipeline.py:181` constructs `SamplingPolicy()` directly instead
  of loading a Falcon-specific policy from configuration.
- `perception/pipeline.py:503-506` calls `plan_indices` without motion or
  handover timestamps, so the designed burst mechanism is unused.
- `perception/falcon_client.py:182-199` defines three broad default queries.
- `perception/sku_translator.py:204-235` adds a generic product query and one
  category per POS basket line.
- `perception/falcon_client.py:245-288` loops over every sampled frame and every
  distinct query, calling Falcon serially.

The screenshot basket contained seven POS lines but six unique product query
strings because the two Alkzy descriptions were identical. Together with three
defaults and the generic catch-all, current code has ten distinct queries. A
re-run therefore still performs roughly:

```text
345 frames x 10 queries = 3,450 autoregressive Falcon generations
```

Before the recent exact-query deduplication, the two identical Alkzy lines were
also run separately, producing approximately 3,795 calls.

Recent commit `1aa9eaa` deduplicates identical query strings per frame. This is
useful but only removes exact duplicates; it does not remove the broad semantic
overlap among `item`, `sco_generic_products`, and the POS-derived queries, and
it does not address the number of frames.

## Root cause 2: requested query is presented as detected identity

Relevant code:

- `falcon_detector.py:48-52` defines a box result without a confidence score.
- `perception/falcon_client.py:182-192` explains that the adapter stamps a
  requested category onto returned boxes.
- `perception/falcon_client.py:285-313` performs that stamping.
- `perception/falcon_client.py:307` falls back to the literal score `0.5` for
  every box.
- `perception/item_grouping.py:95-117` seeds a POS group from the stamped
  `sco_item_NNN` label.
- `perception/item_grouping.py:312-322` looks up the POS description using that
  label index.
- `perception/item_grouping.py:447-457` calls a single-label POS group
  `medium`; this is not model probability.

The resulting logic is circular:

```text
POS says "HD Supermarket Bag"
  -> ask Falcon for "hd supermarket bag"
  -> Falcon returns a kiosk-sized box
  -> stamp box as sco_item_006
  -> map sco_item_006 back to POS line 6
  -> UI displays "HD Supermarket Bag · medium"
```

There is no independent visual confirmation in that chain. A referring
detector may localize something for a prompt even when the exact product is not
present, especially with blurry overhead imagery and fine-grained SKU/brand
queries.

## Root cause 3: repeated POS lines clone the same physical evidence

`perception/falcon_client.py:245-256` deduplicates identical query computation,
but `perception/falcon_client.py:301-313` fans every resulting box back out to
every label that requested the query.

For two identical Alkzy POS lines, one Falcon box becomes both
`sco_item_000` and `sco_item_001`. The tracker treats those labels separately,
and `perception/item_grouping.py:95-117` seeds one matched group per POS line.
Consequently, a single physical box can satisfy quantity two.

A newer eight-line case showed the same issue more starkly: seven repeated
`Rawabi Trngl Cheese` lines each received exactly the same 113 detections on the
same 113 frames. The performance deduplication was correct, but cloning its
output into seven semantic matches was not.

Repeated POS products must instead be represented as one visual target plus an
expected quantity. Matching must require the corresponding number of spatially
or temporally distinct physical observations.

## Root cause 4: ROI and static kiosk hardware

The configured audit ROI is:

```text
x=662, y=504, w=523, h=457 on a 1920x1080 source
```

See `config.yaml:14-23` and the Falcon view at `config.yaml:39-46`.

This crop includes the self-checkout screen, payment terminal, kiosk body, and
bagging/scale surface. Falcon therefore sees a large amount of fixed hardware
that visually dominates small products.

Other contributing behavior:

- `perception/falcon_client.py:133-148` performs per-label NMS and sorts by box
  area, retaining larger boxes first. This can favor a kiosk-sized false box.
- NMS is only within one label/query. There is no class-agnostic cross-query
  suppression before tracking.
- Falcon's rectangular `union_crop` uses the bounding box of assigned zones.
  A polygon saved in the ROI does not by itself mask pixels outside the polygon
  during Falcon inference.
- There is no pre-roll/background model to identify fixtures that persist
  before, during, and after the transaction.

ROI calibration alone will help but cannot solve the circular identity problem.

## Root cause 5: tracker fragmentation and keyframe multiplication

Relevant code:

- `perception/tracker.py:50-53` requires exact label equality.
- `perception/tracker.py:66-76` uses IoU 0.3 and closes after eight sampled
  misses rather than elapsed time.
- `perception/tracker.py:158-170` exports confirmed closed tracks as well as
  non-closed tentative/lost tracks.
- `perception/keyframes.py:15-61` adds at least first and final frames for every
  exported track, plus receipt/handover/counter roles.

With many noisy per-query boxes, these rules produced 183 tracks and 390
keyframes in one attempt. Keyframes were not globally deduplicated or capped.
This is not the main GPU bottleneck, but it bloats SQLite, packages, the API,
and the review UI.

## Root cause 6: reprocess appends and packages all historical perception

Relevant code:

- `evidence/persistence.py:25-146` only inserts detections, tracks,
  observations, keyframes, and OCR rows. It never replaces or versions the
  currently active perception result.
- `evidence/package.py:89-102` selects all perception rows by `case_id`, without
  filtering to the latest successful `video_window_id` or analysis attempt.
- `evidence/graph.py:104-128` has the same case-wide selection behavior.

For the screenshot case, SQL confirmed exactly 2,692 detections, 183 tracks,
and 390 keyframes under each of two window IDs. The latest package then included
both attempts.

Preferred fix: preserve audit history but introduce an explicit analysis-run or
active-window boundary and make the current package/UI select one successful
attempt. A transactional delete-and-replace is simpler but must delete
`TrackObservation` children first and may conflict with audit-retention goals.
Do not silently mix attempts.

Snapshot artifacts also need run-aware naming or cleanup. Current filenames can
be overwritten while older artifact metadata remains.

## Root cause 7: warmup can overlap real inference

Relevant code:

- `app/main.py:59-68` starts Falcon warmup in a daemon thread.
- The auto-analyzer starts immediately afterward in the same lifespan.
- `perception/falcon_client.py:36` protects resident model construction, but no
  process-wide lock or readiness event protects `detector.detect()`.

Warmup and the first case can therefore call the same inference engine
concurrently. The engine is not established as thread-safe, and overlapping
transient activations can increase memory pressure or contribute to a wedge.

Required fix: one process-wide Falcon inference lock plus a warmup-ready event.
The analyzer should wait for warmup success/failure before entering Falcon, or
both paths should acquire the same lock. Warmup must also pass through the same
memory-admission policy as a real case.

## Root cause 8: current engine path leaves throughput available

Relevant code:

- `falcon_detector.py:25,100` uses `BatchInferenceEngine`.
- `falcon_detector.py:75` defaults `compile=False`.
- `falcon_detector.py:173-215` already implements `detect_batch`.
- `perception/falcon_client.py:112-115` declares `_FALCON_BATCH=8`, but the
  current path does not use it.
- Vendored upstream code uses `PagedInferenceEngine` with compile/CUDA-graph
  options in `Falcon-Perception/demo/perception_benchmark.py:105-143`.

Do not blindly enable batch size eight or the paged server. The GB10 uses unified
memory, and avoiding another freeze is more important than a benchmark win.
First remove unnecessary frames/queries. Then benchmark:

1. Current sequential path.
2. Homogeneous same-query/same-ROI frame batches at B=2.
3. The same at B=4 if peak memory is safe.
4. Paged/compiled inference only after output-parity and memory measurements.

The existing heterogeneous-batch concern does not apply in the same way when a
batch contains consecutive frames with identical dimensions and one identical
prompt. Still, benchmark rather than assume.

Also benchmark reducing `max_new_tokens` from the wrapper's 200
(`falcon_detector.py:71`) toward the upstream batch-engine default of 100. This
requires a recall check; NMS after generation cannot recover truncated objects.

## Operational and test gaps

- Docker Compose is the functioning runtime. The app is on port 4001 and uses
  `storage/sco_vision.sqlite` through `/app/storage/sco_vision.sqlite`.
- `status.sh:9` still defaults to port 3902 and PID-file assumptions, so it can
  report that a healthy Compose deployment is stopped.
- `docker-compose.override.yml:30` enables `uvicorn --reload`. This is helpful
  for development but can reload/reconstruct model state during edits. Use the
  non-override production command for controlled performance benchmarks, after
  coordinating any restart with the operator.
- After paths were rebaked to `/app`, host-side Falcon imports may fail. Run
  Falcon-related tests and diagnostics inside the app container.
- Unit tests stub the detector. `tests/smoke_falcon_local.py` verifies loading
  but does not assert inference accuracy or latency.
- There is no committed real-frame regression for the kiosk false positive,
  repeated-line cloning, or a Falcon latency/frame-budget limit.

Recent related commits that should be understood before changing the same code:

```text
1aa9eaa  perf(perception): run each distinct Falcon query once per frame
c7fac07  feat(perception): de-fragment Falcon over-detections into distinct-item groups
5e70215  fix: unwedge the reprocess queue — warm compile, hang watchdog, orphan reaper
44c1760  feat: per-camera POS→video time offset + resident Falcon weights
8ecf338  feat: saved Falcon detection snapshots in case detail
```

## Required implementation sequence

### Phase A — make evidence semantically honest

- [ ] Separate `query`/`requested_category` from verified identity in schemas,
      persistence, packages, and UI.
- [ ] Do not assign a model-confidence value when Falcon did not provide one.
      Store `null`/unknown or an explicit `score_source=synthetic/absent`; do not
      render it as `medium` model confidence.
- [ ] Stop using `sco_item_NNN` query labels as sufficient proof of a POS match.
- [ ] With VLM disabled, show exact identity as `unknown` unless a non-VLM
      verifier (barcode/OCR, reference-image retrieval, or trained SKU model)
      crosses a calibrated threshold.
- [ ] Keep POS descriptions visible as expectations, clearly separated from
      observations.

Suggested minimal observation vocabulary:

```text
observed_class: unknown_item | container | bag_like | paper_like | person_like
identity_status: unverified | verified_barcode | verified_ocr | verified_model
requested_query: original Falcon prompt
model_score: null when unavailable
```

### Phase B — fix quantity and cross-query duplication

- [ ] Group identical normalized POS queries/SKUs before Falcon inference.
- [ ] Carry `expected_quantity` or the list of POS line IDs with that target.
- [ ] Return one set of physical boxes/tracks per unique target, not one cloned
      set per POS line.
- [ ] Run class-agnostic/cross-query spatial-temporal deduplication before
      deciding physical count.
- [ ] Require two distinct physical observations before satisfying quantity two.

### Phase C — fix reprocess attempt boundaries

- [ ] Introduce/select a single active successful analysis attempt for current
      case evidence.
- [ ] Ensure packages, evidence graph, API, and review UI use that same attempt.
- [ ] Preserve older attempts only as explicitly versioned audit history.
- [ ] Make snapshot names and artifact metadata attempt-aware.
- [ ] Add a reprocess regression test proving totals do not double.

### Phase D — cut workload before engine tuning

- [ ] Add Falcon-specific sampling configuration; do not reuse a Gemma-named FPS
      setting.
- [ ] Start with base sampling in the 0.5-1.0 FPS range.
- [ ] Add a configurable hard cap on Falcon frames (initial test range 32-64).
- [ ] Wire motion/handover/scan timestamps into the existing burst sampler.
- [ ] If item-level POS timestamps exist, run product queries only near their
      scan events rather than across the full transaction.
- [ ] Run broad person/receipt queries at a much sparser cadence or replace them
      with a lightweight dedicated signal if they are required.
- [ ] Remove one of the semantically redundant generic `item` /
      `sco_generic_products` queries after regression testing.

### Phase E — reject fixtures and improve the ROI

- [ ] Recalibrate the Falcon view to focus on scan and bagging surfaces while
      excluding the vertical screen/payment terminal as far as possible.
- [ ] Apply post-detection polygon/exclusion-zone filtering; rectangular crop
      alone is insufficient.
- [ ] Build a static-background/fixture signal from pre-roll and ideally across
      cases for the same camera.
- [ ] Reject product-query boxes that match stable fixture boxes before and
      throughout the transaction.
- [ ] Add configurable maximum-area/aspect constraints, with explicit tests for
      legitimate large bags so they are not accidentally removed.

### Phase F — serialize inference and benchmark throughput

- [ ] Add one process-wide Falcon inference lock/readiness event shared by
      warmup and case inference.
- [ ] Put warmup through memory admission and prevent auto-analysis from racing
      it.
- [ ] Benchmark sequential vs homogeneous B=2/B=4 batches on saved real frames.
- [ ] Record latency, peak unified memory, detection parity, and count/box
      changes.
- [ ] Evaluate paged/compiled inference only after the smaller workload is
      correct and stable.

### Phase G — control tracker/keyframe/UI volume

- [ ] Filter one-hit/tentative/noisy tracks before item grouping.
- [ ] Use elapsed time rather than sampled-frame count for lifecycle timeouts.
- [ ] Dedupe keyframes by frame/role and impose a global case cap.
- [ ] Do not load/render thousands of raw boxes by default in the browser.
- [ ] Keep raw evidence accessible for audit without making it the default UI
      payload.

### Phase H — add durable regressions and correct operations tooling

- [ ] Add a real-image fixture representing the kiosk false positive, subject
      to the project's privacy/data-retention rules.
- [ ] Add repeated-POS-line tests requiring distinct physical evidence.
- [ ] Add static-background, score-unknown, latest-attempt, and keyframe-cap
      tests.
- [ ] Add a benchmark/performance-budget test that is opt-in for GPU CI.
- [ ] Update `status.sh` and startup documentation for Compose port 4001.
- [ ] Keep VLM-disabled operation covered by tests.

## Proposed acceptance criteria

Do not delete this document until all required criteria below pass. Performance
targets are initial engineering targets and may be tightened, but they should
not be weakened without recording the measured reason.

### Correctness

- [ ] Replaying the screenshot case does not label the kiosk as a supermarket
      bag or Alkzy can.
- [ ] A Falcon query hit alone is not displayed or persisted as verified SKU
      identity.
- [ ] No Falcon result is shown with fabricated `0.5`/`medium` model
      confidence.
- [ ] Two identical POS lines require two distinct physical observations; one
      bbox cannot satisfy both.
- [ ] Generic and POS queries cannot create duplicate physical items from the
      same overlapping box/track.
- [ ] VLM models remain disabled and the Falcon-only path remains functional.

### Reprocess/evidence integrity

- [ ] Reprocessing a case does not double the current detection, track,
      keyframe, OCR, artifact, graph, or package counts.
- [ ] Historical attempts, if retained, are explicitly identified and never
      mixed into the active evidence view.
- [ ] Snapshot files and artifact rows refer to the same analysis attempt.

### Performance and safety

- [ ] The 138-second screenshot case shows at least a 75% Falcon-time reduction
      from the 405.7-second baseline (initial ceiling: about 102 seconds).
- [ ] Desired follow-up target is Falcon time under 60 seconds for that case on
      the same GB10, without reducing correctness.
- [ ] No case reaches the 900-second watchdog in the controlled replay set.
- [ ] Peak memory remains below the configured hard admission limit, with no
      swap storm, OOM, or system freeze.
- [ ] Warmup and real inference cannot overlap on the same detector.

### Tests and operations

- [ ] Relevant unit/integration tests pass inside the Compose app container.
- [ ] The real-frame accuracy regression passes.
- [ ] The controlled GPU benchmark records before/after latency and peak memory.
- [ ] `/api/v1/health`, `/api/v1/memory`, and `/api/v1/ops/status` remain healthy.
- [ ] `status.sh` reports the actual Compose service correctly.

## Useful inspection commands

Run from the repository root. These commands are read-only unless explicitly
noted.

```bash
git status --short
docker compose ps
docker compose logs --tail=500 app

curl -sS http://127.0.0.1:4001/api/v1/health
curl -sS http://127.0.0.1:4001/api/v1/memory
curl -sS http://127.0.0.1:4001/api/v1/ops/status
curl -sS http://127.0.0.1:4001/api/v1/cases/9ae758a0-2159-45a9-bdd8-c55adb2fe477/processing-timings
```

Timing aggregate:

```bash
sqlite3 -readonly -header -column storage/sco_vision.sqlite \
  "SELECT COUNT(*) AS timed_runs,
          ROUND(AVG(json_extract(input_manifest,
            '$.processing_timings_ms.perception.falcon_ms'))/1000.0, 1)
            AS avg_falcon_s,
          ROUND(MIN(json_extract(input_manifest,
            '$.processing_timings_ms.perception.falcon_ms'))/1000.0, 1)
            AS min_falcon_s,
          ROUND(MAX(json_extract(input_manifest,
            '$.processing_timings_ms.perception.falcon_ms'))/1000.0, 1)
            AS max_falcon_s
     FROM vlm_runs
    WHERE json_extract(input_manifest,
      '$.processing_timings_ms.perception.falcon_ms') IS NOT NULL;"
```

Confirm attempt-specific counts for the screenshot case:

```bash
sqlite3 -readonly -header -column storage/sco_vision.sqlite \
  "SELECT video_window_id, COUNT(*) AS detections,
          COUNT(DISTINCT frame_idx) AS frames
     FROM detections
    WHERE case_id='9ae758a0-2159-45a9-bdd8-c55adb2fe477'
    GROUP BY video_window_id;"
```

Relevant tests should run inside the container because the current environment
is rebased to `/app`:

```bash
docker compose exec -T app pytest -q \
  tests/test_falcon_categories_merge.py \
  tests/test_processing_timings.py \
  tests/test_item_grouping.py \
  tests/test_reprocess_guard.py
```

Do not trigger production reprocessing or restart Compose merely to benchmark
without coordinating with the operator. Prefer a copied test DB/case fixture or
a controlled maintenance window.

## Start-of-session instructions for the next Claude/Codex agent

1. Read this file completely before editing.
2. Run `git status --short` and preserve unrelated work, especially
   `start.sh` and the current `app/auto_analyzer.py` change.
3. Verify live health and whether any case is actively processing before
   restarting containers or running GPU benchmarks.
4. Keep both VLMs disabled.
5. Work through the implementation phases in order. Accuracy/evidence honesty
   and attempt isolation come before low-level throughput tuning.
6. Add tests alongside each fix and update this document's checkboxes/results.
7. If work remains, leave this document in the repo for the following session.
8. When every acceptance criterion is satisfied and the final benchmark is
   recorded elsewhere in durable project history, delete this file as the last
   cleanup step:

```bash
git rm docs/FALCON_DIAGNOSIS_AND_FIX_HANDOFF_2026-07-16.md
```

Do not delete the handoff merely because implementation has started or one
symptom appears improved.
