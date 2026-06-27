"""Phoenix / OpenTelemetry tracer for sco_vision.

If ``observability.phoenix_enabled`` is False (or the OTel deps are not
installed) every ``trace_*`` call is a no-op. None of the public
functions ever raise into the caller — failures are logged and
swallowed. The OTel imports are lazy so the app boots even when the
optional dependencies are missing.

Public surface used by the pipeline:
    init(cfg)
    trace_falcon(camera_id, session_id, frame_type, detections)
    trace_gemma(camera_id, session_id, classifier, thinking,
                prompt_tokens, completion_tokens, thinking_tokens,
                raw_response, parsed_result, latency_ms,
                num_frames=None, token_budget=None)
    trace_session(camera_id, session_id, classifier, duration_sec,
                  num_frames, result, confidence)
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager

log = logging.getLogger(__name__)

_INIT_LOCK = threading.Lock()
_INITIALIZED = False
_ENABLED = False
_TRACER = None


def init(cfg: dict) -> None:
    """Initialize the OTel tracer once. Safe to call multiple times."""
    global _INITIALIZED, _ENABLED, _TRACER
    with _INIT_LOCK:
        if _INITIALIZED:
            return
        _INITIALIZED = True
        obs = (cfg or {}).get("observability") or {}
        if not obs.get("phoenix_enabled", False):
            log.info("phoenix tracing disabled "
                     "(observability.phoenix_enabled=false)")
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except Exception as e:
            log.warning("phoenix tracing disabled: import failed (%s); "
                        "pip install arize-phoenix opentelemetry-sdk "
                        "opentelemetry-exporter-otlp", e)
            return
        endpoint = (obs.get("phoenix_url", "http://localhost:6006")
                    .rstrip("/")) + "/v1/traces"
        project = obs.get("phoenix_project", "sco_vision")
        try:
            resource = Resource.create({"service.name": project})
            provider = TracerProvider(resource=resource)
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            trace.set_tracer_provider(provider)
            _TRACER = trace.get_tracer(project)
            _ENABLED = True
            log.info("phoenix tracing enabled -> %s (project=%s)",
                     endpoint, project)
        except Exception:
            log.exception("phoenix tracing init failed; tracing disabled")


@contextmanager
def _safe_span(name: str, **attrs):
    if not _ENABLED or _TRACER is None:
        yield None
        return
    try:
        with _TRACER.start_as_current_span(name) as span:
            for k, v in attrs.items():
                if v is None:
                    continue
                try:
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(k, v)
                    else:
                        span.set_attribute(k, str(v))
                except Exception:
                    pass
            yield span
    except Exception:
        log.exception("trace span failed: %s", name)
        yield None


def trace_falcon(camera_id, session_id, frame_type, detections) -> None:
    if not _ENABLED:
        return
    try:
        with _safe_span(
            "falcon.detect",
            camera_id=camera_id, session_id=session_id,
            frame_type=frame_type,
            num_detections=len(detections) if detections is not None else 0,
        ) as span:
            if span is None:
                return
            for i, d in enumerate(detections or []):
                try:
                    span.set_attribute(f"det.{i}.label", str(getattr(d, "label", "")))
                    bbox = getattr(d, "bbox_px", None)
                    if bbox is not None:
                        span.set_attribute(
                            f"det.{i}.bbox",
                            ",".join(str(int(x)) for x in bbox),
                        )
                except Exception:
                    pass
    except Exception:
        log.exception("trace_falcon failed")


def trace_gemma(camera_id, session_id, classifier, thinking,
                prompt_tokens, completion_tokens, thinking_tokens,
                raw_response, parsed_result, latency_ms,
                num_frames=None, token_budget=None) -> None:
    if not _ENABLED:
        return
    try:
        with _safe_span(
            "gemma.reason",
            camera_id=camera_id, session_id=session_id,
            classifier=classifier, thinking=bool(thinking),
            latency_ms=latency_ms, num_frames=num_frames,
            token_budget=token_budget,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens=thinking_tokens,
        ) as span:
            if span is None:
                return
            try:
                # Cap raw response at 16 KiB to keep span size bounded.
                span.set_attribute("raw_response", (raw_response or "")[:16384])
                if isinstance(parsed_result, dict):
                    for k in ("handover_occurred", "item_count", "confidence",
                              "flag_for_review", "narrative",
                              "items_handed_over", "customer_description"):
                        v = parsed_result.get(k)
                        if v is None:
                            continue
                        if isinstance(v, (list, tuple)):
                            v = ", ".join(map(str, v))[:1024]
                        elif isinstance(v, str):
                            v = v[:2048]
                        try:
                            span.set_attribute(f"result.{k}", v)
                        except Exception:
                            pass
            except Exception:
                pass
    except Exception:
        log.exception("trace_gemma failed")


def trace_session(camera_id, session_id, classifier, duration_sec,
                  num_frames, result, confidence) -> None:
    if not _ENABLED:
        return
    try:
        with _safe_span(
            "session.complete",
            camera_id=camera_id, session_id=session_id,
            classifier=classifier, duration_sec=float(duration_sec or 0),
            num_frames=int(num_frames or 0),
            result=str(result), confidence=str(confidence),
        ):
            pass
    except Exception:
        log.exception("trace_session failed")
