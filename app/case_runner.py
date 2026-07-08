"""End-to-end case analyzer.

Given a case id, this:

1. Resolves the CCTV window via ``pos.correlation.plan_window``.
2. Reconstructs the actual window MP4 from the matched immutable
   segments via ``video.window_builder.build_window``. If the build
   fails, the case is marked ``INVALID_VIDEO`` and analysis stops.
3. Runs the perception pipeline on the rebuilt window.
4. Extracts up to ``manifest_max_frames`` frames from the window MP4
   and encodes them as ``data:image/jpeg;base64`` entries in the
   ``EvidenceManifest``. The reasoning chain NEVER sees frames=[] for
   a valid window.
5. Calls the active provider chain (Qwen3-VL primary, Gemma fallback).
6. Persists the VLM run row + the perception evidence + writes a
   versioned evidence package.
7. Wraps the result with the deterministic decision policy.
"""
from __future__ import annotations

import base64
import io
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from sqlalchemy.orm import Session

from db.models import (
    Case,
    PosEvent,
    VideoSegment,
    VideoWindow,
    VlmRun,
)
from pos.correlation import plan_window


log = logging.getLogger(__name__)


def analyze_case(session: Session,
                 case_id: str,
                 *,
                 perception_runner: Optional[Callable] = None,
                 vlm_runner: Optional[Callable] = None,
                 prompt_version: Optional[str] = None,
                 manifest_max_frames: Optional[int] = None,
                 pre_roll_sec: Optional[float] = None,
                 post_roll_sec: Optional[float] = None,
                 cfg=None) -> dict:
    """Run the full analysis for ``case_id`` and persist results.

    ``perception_runner`` and ``vlm_runner`` are dependency-injection
    hooks. Tests usually only override ``vlm_runner`` so they exercise
    real window-building + real frame extraction.

    ``cfg`` is the AppConfig snapshot for this single case. When
    omitted it is loaded ONCE here, at the start of the run, and
    threaded through every helper that would otherwise call
    ``load_config()`` mid-case. This means a ``config.yaml`` edit
    that lands between the time the case starts and the time the
    package is written CANNOT change provider selection / ROI views
    / storage root for the in-flight case. Such edits take effect on
    the NEXT case or reprocess.
    """
    from app import audit
    from evidence.package import write_package
    from evidence.persistence import persist_perception
    from reasoning.decision_policy import (
        OUTCOME_INVALID_VIDEO,
        OUTCOME_VERIFIED,
        decide,
        summary_from_vlm,
    )
    from video.window_builder import build_window

    case = session.get(Case, case_id)
    if case is None:
        raise KeyError(f"case {case_id} not found")
    pos = session.get(PosEvent, case.pos_event_id) if case.pos_event_id else None
    if pos is None:
        raise ValueError("case has no POS event linked")

    # Processing-time observability: lightweight perf_counter timers
    # threaded through every analysis stage. The dict is advisory only
    # (never read by the decision policy) and is persisted into
    # VlmRun.input_manifest at the end so the reviewer UI + evidence
    # package can render a timing legend. ``try/finally`` is used per
    # stage so an exception still records the elapsed time it took to
    # fail. Stages that did not run are OMITTED rather than reported
    # as 0, so the operator can tell apart "didn't run" from "ran <1ms".
    _t_total = time.perf_counter()
    timings: dict[str, Any] = {}

    def _ms_since(t0: float) -> int:
        return int((time.perf_counter() - t0) * 1000)

    # Config snapshot for this case. Every helper below reads from
    # ``cfg`` (or accepts it as a kwarg), so a mid-case config.yaml
    # edit cannot retarget provider/ROI/storage in flight.
    if cfg is None:
        from app.config import load_config
        cfg = load_config()

    # SCO prompt routing is config-driven. Defaults to the v2
    # basket-match prompt. Only an explicit return_review_v1 opt-in
    # can reach the legacy refund policy; unknown/empty prompt versions
    # stay on SCO v2 instead of silently falling through to refund.
    if prompt_version is None:
        prompt_version = (
            (cfg.raw.get("reasoning") or {}).get("prompt_version")
            or "sco_basket_match_v2"
        )
    if prompt_version not in (
        "sco_basket_match_v2", "sco_basket_match_v1", "return_review_v1",
    ):
        log.warning("unknown prompt_version=%r; defaulting to SCO v2",
                    prompt_version)
        prompt_version = "sco_basket_match_v2"

    # ----- 1. Resolve window from segment index ----------------------
    storage_root = _storage_root(cfg)
    _t = time.perf_counter()
    try:
        plan = plan_window(session, case.camera_id, pos.pos_event_at,
                           pos_event_end_at=pos.pos_event_end_at,
                           pre_roll_sec=pre_roll_sec,
                           post_roll_sec=post_roll_sec)
    finally:
        timings["window_resolution_ms"] = _ms_since(_t)
    window = VideoWindow(
        case_id=case.id,
        camera_id=case.camera_id,
        requested_start_at=plan.requested_start,
        requested_end_at=plan.requested_end,
        actual_start_at=plan.actual_start,
        actual_end_at=plan.actual_end,
        segment_ids=list(plan.matched_segment_ids),
        status="PENDING",
    )
    session.add(window)
    session.flush()
    # Release the SQLite write lock before the slow NVR/ffmpeg/model
    # phases below. Holding one transaction open for the whole run
    # serialises all other writers (the POS poller, a second reprocess)
    # and surfaces as "database is locked". expire_on_commit=False keeps
    # ``window``/``case`` usable after each commit.
    session.commit()

    out_path = (storage_root / "cases" / f"case_id={case.id}" / "window"
                / f"window_{window.id}.mp4")

    # ----- 2. Acquire the window MP4: LOCAL segments first, NVR fallback.
    # Local recorded segments are extracted with ffmpeg from files already
    # on disk (near-instant). The NVR path streams playback-by-time at
    # ~real-time and is slow, so it is used ONLY when local coverage is
    # missing — e.g. the event is older than the local retention window,
    # the recorder was not capturing this camera at that time, or the
    # local files were purged. No DB write is held across these slow
    # phases (we committed the PENDING window above), so other writers
    # (the POS poller, a second reprocess) are never blocked.
    build = None
    local_fail_reason = None

    # 2a. Local recorded segments (fast).
    _t = time.perf_counter()
    if plan.is_valid:
        segments = (session.query(VideoSegment)
                    .filter(VideoSegment.id.in_(plan.matched_segment_ids))
                    .order_by(VideoSegment.start_at.asc()).all())
        local_build = build_window(
            segments=segments,
            requested_start=plan.requested_start,
            requested_end=plan.requested_end,
            out_path=out_path,
        )
        if local_build.ok:
            build = local_build
            window.acquisition_source = "local_segments_used"
        else:
            local_fail_reason = local_build.failure_reason
            log.warning("local window build failed (%s); falling back to NVR",
                        local_build.failure_reason)
    timings["window_build_ms"] = _ms_since(_t)

    # 2b. NVR on-demand retrieval (slow) — only when local did not produce
    # a clip. Never raises; annotates window.nvr_metadata either way.
    if build is None:
        _t = time.perf_counter()
        try:
            nvr_clip = _try_nvr_window(case, pos.pos_event_at, window,
                                        storage_root, cfg=cfg)
        finally:
            timings["nvr_acquisition_ms"] = _ms_since(_t)
        if nvr_clip:
            build = _adopt_clip(nvr_clip, plan)
            window.acquisition_source = "nvr_clip_retrieved"

    # Audit the resolution AFTER the attempt so the source is accurate.
    audit.record(session, action="case.window_resolved",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"window_id": window.id,
                        "coverage_ratio": plan.coverage_ratio,
                        "valid": plan.is_valid,
                        "invalid_reason": plan.invalid_reason,
                        "acquisition_source": window.acquisition_source,
                        "nvr": window.nvr_metadata})

    if build is None:
        # Neither local segments nor an NVR clip yielded a usable window.
        window.status = "FAILED"
        reason = (plan.invalid_reason or local_fail_reason
                  or "no local segments and no NVR clip")
        window.failure_reason = reason
        if not window.acquisition_source:
            window.acquisition_source = "local_no_segments"
        return _close_invalid(session, case, reason)

    window.path = build.out_path
    window.sha256 = build.sha256
    window.actual_start_at = build.actual_start_at
    window.actual_end_at = build.actual_end_at
    window.status = "SUCCEEDED"
    session.flush()
    session.commit()  # release lock before perception (GPU) + VLM

    # ----- 3. Perception ---------------------------------------------
    perception_result = None
    if perception_runner is None:
        try:
            from perception.pipeline import run_perception
            from perception.sku_translator import (
                build_falcon_categories_from_pos,
                init_translator,
            )
            # Bind the config snapshot so the perception runner sees
            # the same enable/disable flags as the rest of the case.
            _cfg_snapshot = cfg
            # SCO Phase 3: build Falcon categories from the POS basket
            # so Falcon searches for the line items AND a generic
            # catch-all. The translator is initialised lazily here from
            # config-defined paths so v1 has zero network and a single
            # source of truth.
            sco_cfg = (cfg.raw.get("sco_checkout") if cfg else None) or {}
            init_translator(
                cache_path=sco_cfg.get("sku_cache_path"),
                overrides_path=sco_cfg.get("sku_overrides_path"),
            )
            try:
                _sco_falcon_categories = build_falcon_categories_from_pos(pos)
            except Exception:
                log.exception("sku_translator failed; using generic Falcon "
                              "defaults only")
                _sco_falcon_categories = None

            # SCO SAM-3 experiment: build text concept prompts from the
            # same POS basket. Independent of Falcon: when sam3 is the
            # only backend enabled in config, perception runs SAM-3
            # only and never invokes Falcon. Resolving the ROI crop
            # for SAM-3 follows the same model_roi_views shape Falcon
            # uses.
            _sco_sam3_concepts = None
            _sco_sam3_roi_crop = None
            try:
                from perception.sam3_client import build_concepts_from_pos
                _sco_sam3_concepts = build_concepts_from_pos(pos)
            except Exception:
                log.exception("sam3 concept build failed")
            try:
                from app.camera_rois import model_view
                _sam3_view = model_view(cfg, case.camera_id, "sam3")
                # Use the first resolved zone as the crop window. SAM-3
                # is concept-prompted so it benefits from a tight ROI.
                if _sam3_view and _sam3_view.get("resolved_zones"):
                    z = _sam3_view["resolved_zones"][0]
                    _sco_sam3_roi_crop = (
                        int(z["x"]), int(z["y"]),
                        int(z["x"]) + int(z["w"]),
                        int(z["y"]) + int(z["h"]),
                    )
            except Exception:
                log.exception("sam3 ROI resolution failed (continuing "
                              "without crop)")

            def _bound_perception(s, c, w):
                return run_perception(
                    s, c, w, cfg=_cfg_snapshot,
                    falcon_categories=_sco_falcon_categories,
                    sam3_concepts=_sco_sam3_concepts,
                    sam3_roi_crop=_sco_sam3_roi_crop,
                )
            perception_runner = _bound_perception
        except Exception as exc:
            log.warning("perception pipeline unavailable: %s", exc)
    if perception_runner is not None:
        _t = time.perf_counter()
        try:
            try:
                perception_result = perception_runner(session, case, window)
            except Exception:
                log.exception("perception failed; treating as "
                              "obstructed evidence")
                perception_result = None
        finally:
            timings["perception_total_ms"] = _ms_since(_t)
        if isinstance(perception_result, dict):
            inner = perception_result.get("timings_ms")
            if isinstance(inner, dict):
                timings["perception"] = inner

    # Persist perception evidence so the evidence package + graph see
    # real rows, not in-memory dicts.
    if perception_result:
        try:
            persist_perception(session, case_id=case.id,
                               window_id=window.id,
                               perception=perception_result)
        except Exception:
            log.exception("perception persist failed (continuing)")
    # Release the lock before keyframe extraction (ffmpeg) and the VLM
    # inference, which are the longest phases of the run.
    session.commit()

    # ----- 3.5 Episode selection (SCO Phase 4) -----------------------
    # Derive the SCO customer episode from the perception tracks BEFORE
    # the VLM stage so the prompt + policy receive episode metadata.
    # Falcon already ran on the wide POS window in one pass; the
    # episode-selector only re-uses those tracks (no second detection).
    _sco_episode = None
    try:
        from perception.episode_selector import select_sco_episode
        sco_cfg = (cfg.raw.get("sco_checkout") if cfg else None) or {}
        roi_name = sco_cfg.get("roi_name", "sco_audit_zone")
        _t = time.perf_counter()
        try:
            ep = select_sco_episode(
                (perception_result or {}).get("tracks") or [],
                pos_time=pos.pos_event_at,
                roi_name=roi_name,
                window_start=(build.actual_start_at
                              or plan.requested_start),
                window_end=(build.actual_end_at
                            or plan.requested_end),
            )
        finally:
            timings["sco_episode_select_ms"] = _ms_since(_t)
        _sco_episode = ep.to_dict()
        log.info(
            "sco_episode case=%s reason=%s ambiguous=%s coverage=%.2f",
            case.id, ep.reason, ep.ambiguous, ep.coverage_ratio,
        )
    except Exception:
        log.exception("sco episode selection failed (continuing without)")

    # ----- 4. Build the evidence manifest with REAL frames -----------
    window_start_naive = build.actual_start_at or plan.requested_start
    # VLM frame budget: 1 fps across the built window — every recorded
    # second gets a frame (this is a fast-paced self-checkout counter, so
    # we do NOT sub-sample). 99% of transactions are < 3 min, so this
    # stays modest; the cap only bites the rare long tail, which is then
    # chunked (chunking keeps ALL these frames across in-budget calls, it
    # doesn't drop them). An explicit ``manifest_max_frames`` (e.g. from
    # tests) overrides this auto behaviour.
    _chunked_analysis = bool(
        (cfg.raw.get("reasoning") or {}).get("chunked_analysis"))
    _VLM_FPS = 1.0
    _VLM_FRAME_CAP = 180  # ~3 min at 1 fps; bounds cv2 random-seek cost
    if manifest_max_frames is None:
        if build.actual_start_at and build.actual_end_at:
            _win_secs = (build.actual_end_at
                         - build.actual_start_at).total_seconds()
        else:
            _win_secs = (plan.requested_end
                         - plan.requested_start).total_seconds()
        manifest_max_frames = max(
            24, min(_VLM_FRAME_CAP, int(round((_win_secs or 0) * _VLM_FPS))))
    _t = time.perf_counter()
    try:
        sampled_frames = _extract_keyframe_data_urls(
            window_path=build.out_path,
            window_start_ts=window_start_naive,
            keyframes=(perception_result or {}).get("keyframes") or [],
            max_frames=manifest_max_frames,
        )
    finally:
        timings["manifest_frame_extract_ms"] = _ms_since(_t)

    # ROI-driven labeled crops + caption preamble for the active VLM
    # primary (Qwen3-VL). When no active labeled_crops view exists for
    # this camera, ``vlm_roi_extras`` is None and the manifest keeps
    # its full-frame-only shape — pure back-compat with the K-series.
    _t = time.perf_counter()
    try:
        vlm_roi_extras = _build_vlm_roi_extras(case.camera_id,
                                                sampled_frames,
                                                cfg=cfg)
    finally:
        timings["vlm_roi_prepare_ms"] = _ms_since(_t)

    # ----- 5. Reasoning ----------------------------------------------
    # Build the manifest once. Both the real chain and any injected
    # ``vlm_runner`` receive it as a fourth positional arg, so tests can
    # assert that ``manifest.frames`` is non-empty for a valid window.
    from reasoning.providers.base import EvidenceManifest
    manifest_frames = list(sampled_frames)
    manifest_user_prompt = None
    manifest_system_prompt = None
    manifest_meta: dict = {"prompt_version": prompt_version}
    # SCO Phase 4: hand the episode meta to the VLM (Phase 5 prompt
    # uses ambiguity/coverage to qualify its narrative) and to the
    # policy (Phase 6 demands episode coverage for VERIFIED).
    if _sco_episode is not None:
        manifest_meta["sco_episode"] = _sco_episode

    # SCO Phase 5: when the active prompt is sco_basket_match_v1 / _v2,
    # build the basket-match system + user prompts from POS + Falcon
    # summary + canonical SAM3/Falcon groups + episode metadata. The
    # providers (Qwen3-VL, Gemma) honour manifest.system_prompt /
    # manifest.user_prompt as overrides, so no provider-specific code
    # change is needed here.
    sco_user_prompt_only: Optional[str] = None
    if prompt_version in ("sco_basket_match_v1", "sco_basket_match_v2"):
        try:
            basket = ((pos.raw_payload or {}).get("items")
                      if pos.raw_payload else None) or []
            falcon_summary = _summarise_falcon_for_sco(perception_result)
            manifest_meta["sco_falcon_summary"] = falcon_summary
            # Collapse re-detections of the same physical item into
            # canonical groups BEFORE the VLM sees them. Works for
            # both Falcon detections and SAM3 video propagation
            # output — the grouper sees a uniform (detections,tracks)
            # schema.
            try:
                from perception.item_grouping import group_sco_items
                canonical_groups = group_sco_items(
                    (perception_result or {}).get("detections") or [],
                    (perception_result or {}).get("tracks") or [],
                    pos_basket=basket,
                )
            except Exception:
                log.exception("SCO item grouping failed; VLM will see "
                              "raw detections")
                canonical_groups = []
            manifest_meta["sco_canonical_groups"] = canonical_groups

            if prompt_version == "sco_basket_match_v2":
                from reasoning.prompts.sco_basket_match_v2 import (
                    build_sco_prompts_v2,
                )
                # SAM3 container de-fragmentation step (between
                # canonical grouping and the VLM). Raw SAM3 object
                # IDs commonly fragment one physical container; the
                # merger collapses them into one merged group when
                # they share container family, sizes/aspects, time
                # continuity, and never co-exist in the same frame.
                merged_meta: dict = {}
                merged_groups = list(canonical_groups)
                try:
                    from perception.container_merge import (
                        merge_sam3_containers,
                    )
                    _coverage = float(
                        (_sco_episode or {}).get("coverage_ratio") or 0.0
                    )
                    _merge_res = merge_sam3_containers(
                        canonical_groups,
                        (perception_result or {}).get("detections") or [],
                        pos_basket_size=len(basket) if basket else None,
                        episode_coverage_ratio=_coverage,
                    )
                    merged_meta = _merge_res.to_dict()
                    merged_groups = merged_meta["merged_groups"]
                except Exception:
                    log.exception("SAM3 container merge failed; "
                                  "VLM will see raw canonical groups only")
                manifest_meta["sco_container_merge"] = merged_meta
                manifest_system_prompt, sco_user_prompt_only = (
                    build_sco_prompts_v2(
                        basket=basket,
                        canonical_groups=canonical_groups,
                        merged_container_groups=merged_groups,
                        container_merge_meta=merged_meta,
                        episode_meta=_sco_episode,
                    )
                )
            else:
                from reasoning.prompts.sco_basket_match import (
                    build_sco_prompts,
                )
                manifest_system_prompt, sco_user_prompt_only = (
                    build_sco_prompts(
                        basket=basket,
                        falcon_summary=falcon_summary,
                        episode_meta=_sco_episode,
                        canonical_groups=canonical_groups,
                    )
                )
            manifest_user_prompt = sco_user_prompt_only
        except Exception:
            log.exception("SCO prompt build failed; falling back to "
                          "provider default prompt")
    if vlm_roi_extras is not None:
        # The helper already composes the FINAL ordered frame list so
        # the user-prompt's "Attached images" section matches the
        # actual provider attachment order one-to-one.
        manifest_frames = list(vlm_roi_extras["frames"])
        manifest_meta["rois"] = vlm_roi_extras["roi_descriptors"]
        manifest_meta["roi_caption_text"] = vlm_roi_extras["caption_text"]
        # SCO Phase 5 council fix: when SCO mode is active AND ROI
        # extras are enabled, the helper's default user_prompt suffix
        # is only a provider fallback. The transaction-specific SCO
        # basket prompt must remain the main prompt. Compose:
        # ROI image-legend caption + SCO user prompt.
        if sco_user_prompt_only:
            manifest_user_prompt = (vlm_roi_extras["caption_text"]
                                    + "\n\n" + sco_user_prompt_only)
        else:
            manifest_user_prompt = vlm_roi_extras["user_prompt"]

    manifest = EvidenceManifest(
        case_id=case.id,
        camera_id=case.camera_id,
        window_start_ts=(build.actual_start_at
                         or plan.requested_start).isoformat(),
        window_end_ts=(build.actual_end_at
                       or plan.requested_end).isoformat(),
        frames=manifest_frames,
        tracks=(perception_result or {}).get("tracks") or [],
        ocr=(perception_result or {}).get("ocr") or [],
        system_prompt=manifest_system_prompt,
        user_prompt=manifest_user_prompt,
        metadata=manifest_meta,
    )

    vlm_result_dict: dict = {}
    if vlm_runner is None:
        try:
            from reasoning.providers import build_active_provider
            # Use the snapshot — NOT a fresh load_config() — so a
            # mid-case toggle of qwen3_vl/gemma cannot retarget the
            # provider chain for this case.
            chain = build_active_provider(cfg)
            vlm_runner = lambda s, c, w, m: _adapt_vlm_result(  # noqa: E731
                chain.analyze_evidence(m), prompt_version)
        except Exception as exc:
            log.warning("VLM chain unavailable: %s", exc)

    if vlm_runner is not None:
        _t = time.perf_counter()
        try:
            # Production / new-style runners take (session, case, window,
            # manifest). Old test-style runners take (session, case,
            # window); fall back transparently.
            def _call_vlm(m):
                try:
                    return vlm_runner(session, case, window, m) or {}
                except TypeError:
                    return vlm_runner(session, case, window) or {}

            try:
                # Single-pass is the DEFAULT: a whole short window (~99% of
                # transactions are < 3 min) fits one in-budget VLM call, so
                # we send it as one call — no 3x chunk tax. We only fall back
                # to chunking when the window is long enough that one call
                # would blow the model's context (~65k tokens ≈ >56 frames at
                # our resolution). Chunking then keeps every frame across
                # several in-budget calls.
                _SINGLE_PASS_MAX = 56   # frames that fit one call w/ margin
                _CHUNK_FRAMES = 48      # per-chunk size when we must chunk
                _frames = list(manifest.frames or [])
                _is_sco = prompt_version in ("sco_basket_match_v1",
                                             "sco_basket_match_v2")
                if (_chunked_analysis and _is_sco
                        and len(_frames) > _SINGLE_PASS_MAX):
                    # Window too long for one call — watch it in overlapping
                    # in-budget chunks, then aggregate + guard.
                    import dataclasses

                    from reasoning.chunk_aggregate import (
                        aggregate_chunk_verdicts, partition_frames)
                    frame_chunks = partition_frames(
                        _frames, chunk_frames=_CHUNK_FRAMES, overlap=6)
                    chunk_results = []
                    for _cf in frame_chunks:
                        _cm = dataclasses.replace(manifest, frames=_cf)
                        try:
                            chunk_results.append(_call_vlm(_cm))
                        except Exception as exc:
                            chunk_results.append({"error": str(exc)})
                    _basket = ((pos.raw_payload or {}).get("items")
                               if pos.raw_payload else None) or []
                    _basket_desc = [
                        (it.get("description") or it.get("name"))
                        for it in _basket
                        if isinstance(it, dict)
                        and (it.get("description") or it.get("name"))]
                    vlm_result_dict = aggregate_chunk_verdicts(
                        chunk_results, basket_descriptions=_basket_desc or None)
                    # Carry provider/model metadata from the first good chunk.
                    for _r in chunk_results:
                        if isinstance(_r, dict) and not _r.get("error"):
                            for _k in ("provider", "model_name",
                                       "model_snapshot", "latency_ms"):
                                vlm_result_dict.setdefault(_k, _r.get(_k))
                            break
                    log.info("chunked VLM: %d chunks over %d frames",
                             len(frame_chunks), len(_frames))
                else:
                    vlm_result_dict = _call_vlm(manifest)
            except Exception as exc:
                log.exception("VLM run raised")
                vlm_result_dict = {"error": str(exc),
                                   "provider": "chain",
                                   "model_name": "chain"}
        finally:
            timings["vlm_total_ms"] = _ms_since(_t)

    # Per-provider vlm fingerprint. ``provider`` is the actual provider
    # that returned the result (the ChainProvider wraps each member's
    # ``VLMResult`` and returns it verbatim, so under a Qwen failure +
    # Gemma success this is ``"gemma"`` — not the configured primary).
    timings["vlm"] = {
        "provider": vlm_result_dict.get("provider"),
        "model_name": vlm_result_dict.get("model_name"),
        "model_snapshot": vlm_result_dict.get("model_snapshot"),
        "latency_ms": vlm_result_dict.get("latency_ms"),
        "status": "FAILED" if vlm_result_dict.get("error") else "SUCCEEDED",
        "error": vlm_result_dict.get("error"),
    }

    # ----- 6. Persist VLM run row ------------------------------------
    input_manifest = {
        "window_id": window.id,
        "frame_count": len(sampled_frames),
        "perception": _summarise_perception(perception_result),
    }
    if isinstance(vlm_result_dict.get("provider_metadata"), dict):
        input_manifest["provider_metadata"] = vlm_result_dict["provider_metadata"]
    if isinstance(vlm_result_dict.get("usage"), dict):
        input_manifest["usage"] = vlm_result_dict["usage"]
    # NOTE: ``processing_timings_ms`` is attached AFTER the decision +
    # package_write timings are recorded below; we stamp it on the row
    # just before commit so total_ms reflects real end-to-end work.
    run = VlmRun(
        case_id=case.id,
        provider=vlm_result_dict.get("provider", "chain"),
        model_name=vlm_result_dict.get("model_name", "chain"),
        model_snapshot=vlm_result_dict.get("model_snapshot"),
        prompt_version=prompt_version,
        input_manifest=input_manifest,
        output_json=vlm_result_dict.get("parsed", {}),
        status="FAILED" if vlm_result_dict.get("error") else "SUCCEEDED",
        latency_ms=vlm_result_dict.get("latency_ms"),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        error=vlm_result_dict.get("error"),
    )
    session.add(run)
    session.flush()

    # ----- 7. Decision policy ----------------------------------------
    # SCO Phase 6: when the active prompt is sco_basket_match_v1, route
    # the decision through reasoning.sco_policy (basket-match semantics,
    # strict VERIFIED gates, never emits HIGH_RISK_REVIEW). Otherwise
    # fall back to the legacy refund decision policy. The refund module
    # stays on disk so future multi-scenario routing can re-activate it.
    _t = time.perf_counter()
    try:
        if prompt_version == "sco_basket_match_v2":
            from reasoning.schemas.sco_basket_match_v2 import (
                parse_or_fallback_v2,
            )
            from reasoning.sco_policy_v2 import decide_sco_v2
            parsed_vlm = vlm_result_dict.get("parsed") or {}
            if vlm_result_dict.get("error"):
                sco_vlm = parse_or_fallback_v2({})
            else:
                sco_vlm = parse_or_fallback_v2(parsed_vlm)
            # Hand the merger meta + POS basket size to the policy
            # so physical-count gating honours merged-count range,
            # not raw SAM3 id count.
            _basket_for_policy = ((pos.raw_payload or {}).get("items")
                                   if pos.raw_payload else None) or []
            _container_merge_meta = manifest_meta.get(
                "sco_container_merge") or None
            decision = decide_sco_v2(
                sco_vlm, _sco_episode,
                container_merge_meta=_container_merge_meta,
                pos_basket_size=(len(_basket_for_policy)
                                 if _basket_for_policy else None),
            )
            class _SummaryShim:
                customer_present = False
                contradictions: list = []
            summary = _SummaryShim()
        elif prompt_version == "sco_basket_match_v1":
            from reasoning.schemas.sco_basket_match import parse_or_fallback
            from reasoning.sco_policy import decide_sco
            parsed_vlm = vlm_result_dict.get("parsed") or {}
            if vlm_result_dict.get("error"):
                # VLM error → schema fallback (uncertain, low confidence)
                sco_vlm = parse_or_fallback({})
            else:
                sco_vlm = parse_or_fallback(parsed_vlm)
            decision = decide_sco(sco_vlm, _sco_episode)
            # Keep a minimal `summary` for downstream rows that still
            # read .customer_present (Phase 7a / refund-shaped audit
            # surfaces). For SCO v1 we don't compute customer_present
            # from perception tracks — the episode selector covers that
            # responsibility implicitly.
            class _SummaryShim:
                customer_present = False
                contradictions: list = []
            summary = _SummaryShim()
        elif prompt_version == "return_review_v1":
            summary = summary_from_vlm(
                vlm_result_dict.get("parsed", {}),
                footage_valid=True,
                obstructed=_perception_obstructed(perception_result),
                camera_gap=False,
                perception_result=perception_result,
            )
            if vlm_result_dict.get("error"):
                summary.contradictions.append(
                    f"vlm error: {vlm_result_dict['error']}")
            decision = decide(summary)
        else:
            raise RuntimeError(f"unhandled prompt_version {prompt_version!r}")
    finally:
        timings["decision_ms"] = _ms_since(_t)
    case.outcome = decision.outcome
    case.risk_score = decision.risk_score
    case.risk_reasons = list(decision.reasons)
    case.decision_policy_version = decision.policy_version
    case.status = "CLOSED"
    case.closed_at = datetime.now(timezone.utc)
    # Persist the track-derived customer_present (a real person on the
    # customer side) onto the VLM run so the case grid can render a
    # "Customer" column without recomputing from tracks. Re-assign a new
    # dict — SQLAlchemy's JSON column does not detect in-place mutation.
    run.output_json = {**(run.output_json or {}),
                       "customer_present": bool(summary.customer_present)}

    # Finalise total + persist timings into the VlmRun row BEFORE the
    # package is written so the evidence package can read the same dict
    # the reviewer UI does. SQLAlchemy's default JSON column does NOT
    # detect in-place mutation, so we explicitly assign a *new* dict
    # each time to force the row to be marked dirty.
    timings["total_ms"] = _ms_since(_t_total)
    run.input_manifest = {**input_manifest,
                          "processing_timings_ms": dict(timings)}
    session.flush()

    # ----- 8. Evidence package ---------------------------------------
    _t = time.perf_counter()
    try:
        package = write_package(session, case.id)
    finally:
        timings["package_write_ms"] = _ms_since(_t)
    # Refresh the total now that the package has been written and
    # re-assign (NEW dict) so the row carries the post-package timing.
    timings["total_ms"] = _ms_since(_t_total)
    run.input_manifest = {**input_manifest,
                          "processing_timings_ms": dict(timings)}
    session.flush()

    audit.record(session, action="case.decided",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"outcome": case.outcome,
                        "risk_score": decision.risk_score,
                        "reasons": list(decision.reasons),
                        "package_sha256": package["sha256"]})
    session.commit()
    return {
        "case_id": case.id,
        "outcome": case.outcome,
        "risk_score": case.risk_score,
        "reasons": list(decision.reasons),
        "package": package["uri"],
        "vlm_run_id": run.id,
        "window_id": window.id,
        "window_path": window.path,
    }


def _close_invalid(session: Session, case: Case,
                   reason: Optional[str]) -> dict:
    from app import audit
    from evidence.package import write_package
    from reasoning.decision_policy import OUTCOME_INVALID_VIDEO

    case.outcome = OUTCOME_INVALID_VIDEO
    case.status = "CLOSED"
    case.invalid_reason = reason
    case.decision_policy_version = "v1"
    case.closed_at = datetime.now(timezone.utc)
    package = write_package(session, case.id)
    audit.record(session, action="case.decided",
                 entity_type="case", entity_id=case.id,
                 actor_type="analyzer",
                 after={"outcome": case.outcome,
                        "invalid_reason": reason,
                        "package_sha256": package["sha256"]})
    session.commit()
    return {"case_id": case.id, "outcome": case.outcome,
            "invalid_reason": reason, "package": package["uri"]}


def _storage_root(cfg=None) -> Path:
    if cfg is None:
        from app.config import load_config
        cfg = load_config()
    return cfg.storage_root


def _camera_cfg(camera_id: str, cfg=None) -> dict:
    if cfg is None:
        from app.config import load_config
        cfg = load_config()
    for c in cfg.cameras:
        if c.get("id") == camera_id:
            return c
    return {}


def _try_nvr_window(case, pos_event_at, window, storage_root,
                    *, cfg=None) -> Optional[str]:
    """Run NVR on-demand acquisition for this case window.

    Annotates ``window.acquisition_source`` + ``window.nvr_metadata`` and
    returns a local clip path when the exact historical clip was actually
    exported (Stage B); otherwise returns ``None`` so the caller uses the
    local recorded segments. NEVER raises — NVR is non-blocking."""
    try:
        from video.nvr_dahua import (
            acquire_window, load_nvr_config, STATE_CLIP_RETRIEVED)
        nvr_cfg = load_nvr_config(_camera_cfg(case.camera_id, cfg=cfg))
        out = (storage_root / "cases" / f"case_id={case.id}" / "window"
               / f"nvr_{window.id}.mp4")
        acq = acquire_window(nvr_cfg, pos_event_at,
                             camera_id=case.camera_id, out_path=str(out))
        window.nvr_metadata = acq.metadata
        if acq.attempted:
            window.acquisition_source = acq.state
        if acq.state == STATE_CLIP_RETRIEVED and acq.clip_path:
            return acq.clip_path
    except Exception:
        log.exception("nvr acquisition step failed (non-fatal)")
    return None


@dataclass
class _ClipBuild:
    """build_window-compatible result for an NVR-exported clip."""
    out_path: str
    sha256: Optional[str]
    actual_start_at: Optional[datetime]
    actual_end_at: Optional[datetime]
    ok: bool = True
    failure_reason: Optional[str] = None


def _adopt_clip(clip_path: str, plan) -> _ClipBuild:
    import hashlib
    h = hashlib.sha256()
    with open(clip_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return _ClipBuild(out_path=clip_path, sha256=h.hexdigest(),
                      actual_start_at=plan.requested_start,
                      actual_end_at=plan.requested_end)


def _extract_keyframe_data_urls(*, window_path: str,
                                window_start_ts: datetime,
                                keyframes: list,
                                max_frames: int) -> list[dict]:
    """Decode a small number of frames from the built window and encode
    them as data URLs the provider chain can ship.

    Prefers timestamps recommended by perception keyframes; otherwise
    samples evenly across the window. If decoding fails entirely the
    function returns ``[]`` and the caller logs a limitation.
    """
    try:
        import cv2  # type: ignore
    except Exception:
        log.warning("cv2 unavailable; sending zero frames to VLM")
        return []
    if not window_path or not os.path.exists(window_path):
        log.warning("window file %r missing on disk", window_path)
        return []

    cap = cv2.VideoCapture(window_path)
    if not cap.isOpened():
        log.warning("cv2 could not open window %r", window_path)
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        log.warning("window %r reports zero frames", window_path)
        return []

    # Choose frame indices: pull from keyframe roles first, dedupe,
    # cap at max_frames, then top up with evenly spaced indices.
    indices: list[int] = []
    for kf in keyframes:
        idx = kf.get("frame_idx") if isinstance(kf, dict) else None
        if isinstance(idx, int) and 0 <= idx < frame_count:
            indices.append(idx)
    indices = sorted(set(indices))[:max_frames]
    if len(indices) < max_frames:
        step = max(1, frame_count // max_frames)
        for i in range(0, frame_count, step):
            if i in indices:
                continue
            indices.append(i)
            if len(indices) >= max_frames:
                break
    indices = sorted(indices)[:max_frames]

    # Anchor frame timestamps to the actual CCTV window start so each
    # frame carries a wall-clock value aligned with the POS event,
    # not a synthetic placeholder.
    base_ts = _ensure_naive_utc(window_start_ts)

    out: list[dict] = []
    from PIL import Image
    # Sequential decode — NOT random-seek. ``cap.set(CAP_PROP_POS_FRAMES)``
    # per frame forces a keyframe seek + GOP re-decode on every pick, which
    # is pathologically slow on long windows and on the recorder's gappy
    # segments (broken timestamps make it crawl or hang). Decoding forward
    # once and grabbing the target indices as they pass is far faster and
    # robust. ``indices`` is sorted, so we walk to the last one only.
    target = set(indices)
    max_idx = max(indices) if indices else -1
    cur = 0
    while cur <= max_idx:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if cur in target:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            out.append({
                "frame_id": f"frame_{cur:06d}",
                "frame_idx": int(cur),
                "ts": _ts_for_index(cur, fps, base_ts).isoformat(),
                "image_url": f"data:image/jpeg;base64,{b64}",
            })
        cur += 1
    cap.release()
    return out


def _ensure_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _ts_for_index(idx: int, fps: float, base_ts: datetime) -> datetime:
    """Frame timestamp = ``base_ts + idx/fps``. ``base_ts`` is the
    real CCTV window start."""
    from datetime import timedelta
    return base_ts + timedelta(seconds=idx / max(fps, 1.0))


def _adapt_vlm_result(vlm_result, prompt_version: str) -> dict:
    parsed = dict(vlm_result.parsed or {})
    # ``_model_run`` is stamped by the active provider when it returns a
    # successful result. Lift it out of ``parsed`` (where the VLM body
    # would otherwise sit) into top-level fields so the VlmRun row can
    # persist the provider/runtime fingerprint without polluting
    # ``output_json`` with framework metadata.
    model_run = parsed.pop("_model_run", None) \
        if isinstance(parsed.get("_model_run"), dict) else None
    out = {
        "provider": vlm_result.provider,
        "model_name": vlm_result.model_name,
        "parsed": parsed,
        "latency_ms": vlm_result.latency_ms,
        "error": vlm_result.error,
        "prompt_version": prompt_version,
    }
    if isinstance(model_run, dict):
        meta = model_run.get("provider_metadata")
        if isinstance(meta, dict):
            out["provider_metadata"] = meta
        snap = model_run.get("model_snapshot")
        if isinstance(snap, str) and snap:
            out["model_snapshot"] = snap
        usage = model_run.get("usage")
        if isinstance(usage, dict):
            out["usage"] = usage
    return out


def _summarise_perception(perception_result: Optional[dict]) -> dict:
    if not perception_result:
        return {"tracks": 0, "keyframes": 0, "ocr": 0, "detections": 0}
    return {
        "tracks": len(perception_result.get("tracks") or []),
        "keyframes": len(perception_result.get("keyframes") or []),
        "ocr": len(perception_result.get("ocr") or []),
        "detections": len(perception_result.get("detections") or []),
        "limitations": perception_result.get("limitations") or [],
    }


def _summarise_falcon_for_sco(perception_result: Optional[dict]) -> dict:
    """Count Falcon detections by category prefix so the SCO basket-match
    prompt can hand the VLM a structured evidence summary.

    Buckets (label prefix → meaning):
      ``sco_item_*``       → POS-derived item query fired (matched candidate)
      ``sco_generic_*``    → generic catch-all query fired (possible extra)
      anything else        → default categories (item/person/receipt)
    """
    summary = {
        "matched_count": 0,
        "generic_candidate_count": 0,
        "default_detection_count": 0,
        "queries_run": [],
    }
    if not perception_result:
        return summary
    queries_seen: set[str] = set()
    for d in (perception_result.get("detections") or []):
        label = ""
        if isinstance(d, dict):
            label = str(d.get("label") or "")
        else:
            label = str(getattr(d, "label", "") or "")
        if label.startswith("sco_item_"):
            summary["matched_count"] += 1
            queries_seen.add(label)
        elif label.startswith("sco_generic"):
            summary["generic_candidate_count"] += 1
            queries_seen.add(label)
        else:
            summary["default_detection_count"] += 1
    summary["queries_run"] = sorted(queries_seen)
    return summary


def _build_vlm_roi_extras(camera_id: str,
                          sampled_frames: list[dict],
                          *,
                          cfg=None,
                          ) -> Optional[dict]:
    """Return the full ordered VLM manifest extras when the active
    primary VLM has a ``labeled_crops`` view configured for
    ``camera_id``. Returns ``None`` otherwise so the manifest keeps its
    existing full-frame shape.

    The returned dict carries:

      * ``frames`` — the FINAL ordered list of manifest frames that the
        provider will attach in this exact order: full-frame overview
        entries (if ``include_full_frame_overview`` is on) followed by
        the labeled ROI crops, grouped by source frame and then by zone
        order. The provider's image-list order is identical to this
        list — see ``Qwen3VLProvider._analyze_vllm`` /
        ``GemmaProvider.analyze_evidence``.
      * ``user_prompt`` — a composed text prompt that, BEFORE the
        canonical JSON request, contains an ordered ``Attached images``
        section pairing position 1..N with frame_id, kind (overview |
        roi_crop), roi_id/label/crop_xyxy/source_frame_id where
        applicable. The model is therefore told exactly which image is
        which ROI in the same order the image parts arrive.
      * ``caption_text`` — the same ROI section text exposed for audit
        / persistence purposes.
      * ``roi_descriptors`` — list of full ROI definitions for
        ``manifest.metadata["rois"]``.
      * ``include_full_frame_overview`` — echo of the resolved flag.
      * ``primary_model`` — which provider's view drove the assembly.

    Captions/labels round-trip from stable operator-chosen identifiers.
    """
    if not sampled_frames:
        return None
    try:
        from app.camera_rois import (
            apply_margin, model_view, scale_zones_to_frame,
        )
        from reasoning.providers.qwen3_vl import DEFAULT_USER_PROMPT
    except Exception:
        log.exception("roi extras: imports failed")
        return None
    if cfg is None:
        try:
            from app.config import load_config
            cfg = load_config()
        except Exception:
            log.exception("roi extras: cfg load failed")
            return None
    reasoning_cfg = (cfg.raw.get("reasoning") if cfg else None) or {}
    primary = reasoning_cfg.get("primary_provider") or "qwen3_vl"
    view = model_view(cfg, camera_id, primary)
    if view is None:
        view = model_view(cfg, camera_id, "qwen3_vl")
    if view is None or view.get("mode") != "labeled_crops":
        return None
    zones = view.get("resolved_zones") or []
    if not zones:
        return None
    include_overview = bool(view.get("include_full_frame_overview", True))

    try:
        from PIL import Image
        import io
        import base64
    except Exception:
        log.exception("roi extras: PIL unavailable")
        return None

    def _decode(url: str):
        if not url.startswith("data:image"):
            return None
        try:
            _, b64 = url.split(",", 1)
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw))
            return img.convert("RGB") if img.mode != "RGB" else img
        except Exception:
            log.exception("roi extras: decode failed")
            return None

    sample = _decode(sampled_frames[0].get("image_url") or "")
    if sample is None:
        return None
    src_w, src_h = sample.size
    # Scale zones from their saved source dimensions onto the actual
    # decoded frame size. Operators who calibrated on a high-res
    # snapshot can still produce correct crops when the manifest
    # frames are smaller (e.g. mp4_evidence_width=640).
    scaled_zones = scale_zones_to_frame(zones, int(src_w), int(src_h))
    if not scaled_zones:
        return None

    # Pick source frames to clip from: first + middle keep the manifest
    # compact while still covering more than a single instant.
    n = len(sampled_frames)
    source_indices = sorted({0, n // 2}) if n > 1 else [0]
    margin_pct = float(view.get("margin_pct") or 0.0)
    extra_frames: list[dict] = []
    for src_idx in source_indices:
        src = sampled_frames[src_idx]
        src_img = _decode(src.get("image_url") or "")
        if src_img is None:
            continue
        src_frame_id = src.get("frame_id") or f"frame_{src_idx:06d}"
        for z in scaled_zones:
            x1, y1, x2, y2 = apply_margin(
                (int(z["x"]), int(z["y"]),
                 int(z["x"]) + int(z["w"]), int(z["y"]) + int(z["h"])),
                margin_pct, src_w, src_h)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = src_img.crop((x1, y1, x2, y2))
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            extra_frames.append({
                "frame_id": f"{src_frame_id}__roi_{z['id']}",
                "frame_idx": src.get("frame_idx"),
                "ts": src.get("ts"),
                "image_url": f"data:image/jpeg;base64,{b64}",
                "source_frame_id": src_frame_id,
                "roi_id": z["id"],
                "roi_label": z.get("label") or z["id"],
                "crop_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            })

    if not extra_frames:
        return None

    # Build the FINAL ordered frames list — this is the exact order the
    # provider will attach images in, so the user-prompt manifest below
    # can number each position confidently.
    overview_frames: list[dict] = list(sampled_frames) if include_overview \
        else []
    final_frames = overview_frames + extra_frames

    # Compose the provider fallback prompt. Runtime SCO cases replace
    # this suffix with the transaction-specific basket prompt before
    # sending the manifest.
    image_lines: list[str] = []
    image_lines.append(
        "Attached images (the model receives them in this exact order):")
    for pos, f in enumerate(final_frames, start=1):
        if "roi_id" in f:
            crop = f.get("crop_xyxy") or []
            image_lines.append(
                f"  [{pos}] roi_crop  frame_id={f.get('frame_id')}  "
                f"roi_id={f.get('roi_id')}  "
                f"label={f.get('roi_label')!r}  "
                f"crop_xyxy={list(crop)}  "
                f"source_frame_id={f.get('source_frame_id')}")
        else:
            image_lines.append(
                f"  [{pos}] overview  frame_id={f.get('frame_id')}  "
                f"ts={f.get('ts')}")

    caption_lines: list[str] = []
    if include_overview:
        caption_lines.append(
            "Camera ROI views — the manifest above includes a full-frame "
            "overview AND additional labeled crops.")
    else:
        caption_lines.append(
            "Camera ROI views — the manifest above contains labeled "
            "crops only (no full-frame overview).")
    if view.get("caption"):
        caption_lines.append(view["caption"])
    caption_lines.append("ROI legend:")
    for z in zones:
        bits = [f"- {z['id']} [{z.get('label') or z['id']}]"]
        if z.get("purpose"):
            bits.append(f"({z['purpose']})")
        caption_lines.append(" ".join(bits))

    caption_text = "\n".join(image_lines + [""] + caption_lines)
    user_prompt = caption_text + "\n\n" + DEFAULT_USER_PROMPT

    roi_descriptors = [
        {"id": z["id"],
         "label": z.get("label") or z["id"],
         "purpose": z.get("purpose") or "",
         "x": int(z["x"]), "y": int(z["y"]),
         "w": int(z["w"]), "h": int(z["h"])}
        for z in zones
    ]

    return {
        "frames": final_frames,
        "extra_frames": extra_frames,
        "user_prompt": user_prompt,
        "caption_text": caption_text,
        "roi_descriptors": roi_descriptors,
        "include_full_frame_overview": include_overview,
        "primary_model": primary,
    }


def _perception_obstructed(perception_result: Optional[dict]) -> bool:
    if not perception_result:
        return False
    if perception_result.get("obstructed") is not None:
        return bool(perception_result["obstructed"])
    limitations = perception_result.get("limitations") or []
    return any("obstruct" in str(l).lower() for l in limitations)
