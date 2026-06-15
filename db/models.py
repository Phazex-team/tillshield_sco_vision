"""SQLAlchemy 2.x mapped models.

Designed to be portable between SQLite (dev/test) and PostgreSQL
(production). Enums are stored as plain text so SQLite can hold the
same values; the production migration adds real ENUM types.

Identifiers use UUIDv4 strings to remain portable. JSON columns use
SQLAlchemy's ``JSON`` (maps to ``jsonb`` on Postgres).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


CASE_OUTCOMES = ("VERIFIED", "REVIEW", "HIGH_RISK_REVIEW", "INVALID_VIDEO")
CASE_STATUSES = ("OPEN", "IN_REVIEW", "CLOSED", "REPROCESSING")
RUN_STATUSES = ("PENDING", "RUNNING", "SUCCEEDED", "FAILED", "SKIPPED")
ARTIFACT_TYPES = (
    "SEGMENT", "WINDOW_CLIP", "KEYFRAME", "SNAPSHOT",
    "MASK", "OCR_CROP", "PACKAGE",
)


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class PosBatch(Base):
    __tablename__ = "pos_batches"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    source_system: Mapped[Optional[str]] = mapped_column(String(64))
    store_id: Mapped[Optional[str]] = mapped_column(String(64))
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    batch_start_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    batch_end_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    payload_hash: Mapped[Optional[str]] = mapped_column(String(128), unique=True)
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PosEvent(Base):
    __tablename__ = "pos_events"
    __table_args__ = (
        UniqueConstraint(
            "store_id", "terminal_id", "transaction_id", "line_id",
            name="uq_pos_event_natural_key",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    batch_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("pos_batches.id"))
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    terminal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    transaction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    line_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    pos_event_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    staff_id: Mapped[Optional[str]] = mapped_column(String(64))
    sku: Mapped[Optional[str]] = mapped_column(String(64))
    item_description: Mapped[Optional[str]] = mapped_column(Text)
    quantity: Mapped[Optional[float]] = mapped_column(Float)
    amount: Mapped[Optional[float]] = mapped_column(Float)
    currency: Mapped[Optional[str]] = mapped_column(String(8))
    raw_payload: Mapped[Optional[dict]] = mapped_column(JSON)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    pos_event_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("pos_events.id"), unique=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32),
                                        default="OPEN", nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String(32))
    risk_score: Mapped[Optional[float]] = mapped_column(Float)
    risk_reasons: Mapped[Optional[list]] = mapped_column(JSON)
    decision_policy_version: Mapped[Optional[str]] = mapped_column(String(32))
    opened_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    invalid_reason: Mapped[Optional[str]] = mapped_column(Text)


class VideoSegment(Base):
    __tablename__ = "video_segments"
    __table_args__ = (
        UniqueConstraint("camera_id", "start_at",
                         name="uq_segment_camera_start"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    start_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    duration_sec: Mapped[Optional[float]] = mapped_column(Float)
    fps: Mapped[Optional[float]] = mapped_column(Float)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    frame_count: Mapped[Optional[int]] = mapped_column(Integer)
    has_gap: Mapped[bool] = mapped_column(Boolean, default=False)
    corrupt: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class VideoWindow(Base):
    __tablename__ = "video_windows"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    requested_start_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    requested_end_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    actual_start_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    actual_end_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    segment_ids: Mapped[Optional[list]] = mapped_column(JSON)
    path: Mapped[Optional[str]] = mapped_column(Text)
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    failure_reason: Mapped[Optional[str]] = mapped_column(Text)


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(32), nullable=False)
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[Optional[str]] = mapped_column(String(64))
    mime_type: Mapped[Optional[str]] = mapped_column(String(64))
    frame_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    frame_idx: Mapped[Optional[int]] = mapped_column(Integer)
    artifact_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class VlmRun(Base):
    __tablename__ = "vlm_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_snapshot: Mapped[Optional[str]] = mapped_column(String(64))
    prompt_version: Mapped[Optional[str]] = mapped_column(String(32))
    input_manifest: Mapped[Optional[dict]] = mapped_column(JSON)
    output_json: Mapped[Optional[dict]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="PENDING")
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    error: Mapped[Optional[str]] = mapped_column(Text)


class ReviewAction(Base):
    __tablename__ = "review_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(36))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(String(32))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    labels: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    video_window_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("video_windows.id"))
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_xyxy: Mapped[Optional[list]] = mapped_column(JSON)
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, default=0)
    frame_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    query: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    video_window_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("video_windows.id"))
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    tracker_id: Mapped[Optional[str]] = mapped_column(String(64))
    first_seen_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_seen_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    zones: Mapped[Optional[list]] = mapped_column(JSON)
    events: Mapped[Optional[list]] = mapped_column(JSON)
    physical_item_candidate: Mapped[bool] = mapped_column(
        Boolean, default=False)
    receipt_candidate: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class TrackObservation(Base):
    __tablename__ = "track_observations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    track_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("tracks.id"), nullable=False)
    detection_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("detections.id"))
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, default=0)
    frame_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    bbox_xyxy: Mapped[Optional[list]] = mapped_column(JSON)


class Keyframe(Base):
    __tablename__ = "keyframes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    video_window_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("video_windows.id"))
    role: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    frame_idx: Mapped[int] = mapped_column(Integer, default=0)
    frame_ts: Mapped[Optional[datetime]] = mapped_column(DateTime)
    track_id_ref: Mapped[Optional[str]] = mapped_column(String(64))
    uri: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class OcrResult(Base):
    __tablename__ = "ocr_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    case_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("cases.id"), nullable=False)
    video_window_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("video_windows.id"))
    frame_id: Mapped[str] = mapped_column(String(64), nullable=False)
    bbox_xyxy: Mapped[Optional[list]] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    engine: Mapped[str] = mapped_column(String(64), default="falcon")
    crop_uri: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    actor_id: Mapped[Optional[str]] = mapped_column(String(36))
    actor_type: Mapped[Optional[str]] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[Optional[str]] = mapped_column(String(64))
    entity_id: Mapped[Optional[str]] = mapped_column(String(36))
    before_json: Mapped[Optional[dict]] = mapped_column(JSON)
    after_json: Mapped[Optional[dict]] = mapped_column(JSON)
    ip: Mapped[Optional[str]] = mapped_column(String(64))
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
