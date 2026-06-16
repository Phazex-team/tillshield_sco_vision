-- 0001_core.sql — production Postgres schema for fraud_detection_v3.
--
-- This is the production target. The SQLAlchemy models in db/models.py
-- are schema-compatible with this DDL, but use plain TEXT in place of
-- the ENUM types below so SQLite can hold the same values during dev.
--
-- Apply with:
--   psql "$DATABASE_URL" -f db/migrations/0001_core.sql

CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$ BEGIN
    CREATE TYPE case_outcome AS ENUM
        ('VERIFIED','REVIEW','HIGH_RISK_REVIEW','INVALID_VIDEO');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE case_status AS ENUM
        ('OPEN','IN_REVIEW','CLOSED','REPROCESSING');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE artifact_type AS ENUM
        ('SEGMENT','WINDOW_CLIP','KEYFRAME','SNAPSHOT',
         'MASK','OCR_CROP','PACKAGE');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE run_status AS ENUM
        ('PENDING','RUNNING','SUCCEEDED','FAILED','SKIPPED');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
    CREATE TYPE edge_type AS ENUM (
        'LINKED_TO_TRANSACTION','HAS_WINDOW','HAS_ARTIFACT','OBSERVED_IN',
        'APPEARS_AT','DISAPPEARS_AT','TRACKS_OBJECT','HAS_OCR',
        'SUPPORTS_CLAIM','CONTRADICTS_CLAIM','REVIEWED_AS'
    );
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS pos_batches (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system text,
    store_id text,
    received_at timestamptz,
    batch_start_at timestamptz,
    batch_end_at timestamptz,
    payload_hash text UNIQUE,
    raw_payload jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS pos_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id uuid REFERENCES pos_batches(id),
    store_id text NOT NULL,
    terminal_id text NOT NULL,
    transaction_id text NOT NULL,
    line_id text NOT NULL,
    event_type text NOT NULL,
    pos_event_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    staff_id text,
    sku text,
    item_description text,
    quantity numeric,
    amount numeric,
    currency text,
    raw_payload jsonb,
    UNIQUE (store_id, terminal_id, transaction_id, line_id)
);

CREATE TABLE IF NOT EXISTS cases (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    pos_event_id uuid UNIQUE REFERENCES pos_events(id),
    camera_id text NOT NULL,
    status case_status NOT NULL DEFAULT 'OPEN',
    outcome case_outcome,
    risk_score numeric,
    risk_reasons jsonb,
    decision_policy_version text,
    opened_at timestamptz NOT NULL DEFAULT now(),
    closed_at timestamptz,
    invalid_reason text
);

CREATE TABLE IF NOT EXISTS video_segments (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    camera_id text NOT NULL,
    start_at timestamptz NOT NULL,
    end_at timestamptz NOT NULL,
    path text NOT NULL,
    sha256 text,
    duration_sec numeric,
    fps numeric,
    width int,
    height int,
    frame_count int,
    has_gap boolean NOT NULL DEFAULT false,
    corrupt boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (camera_id, start_at)
);

CREATE TABLE IF NOT EXISTS video_windows (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    camera_id text NOT NULL,
    requested_start_at timestamptz NOT NULL,
    requested_end_at timestamptz NOT NULL,
    actual_start_at timestamptz,
    actual_end_at timestamptz,
    segment_ids uuid[],
    path text,
    sha256 text,
    status run_status NOT NULL DEFAULT 'PENDING',
    failure_reason text,
    acquisition_source text,
    nvr_metadata jsonb
);

CREATE TABLE IF NOT EXISTS artifacts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    artifact_type artifact_type NOT NULL,
    uri text NOT NULL,
    sha256 text,
    mime_type text,
    frame_ts timestamptz,
    frame_idx int,
    metadata jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS vlm_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    provider text NOT NULL,
    model_name text NOT NULL,
    model_snapshot text,
    prompt_version text,
    input_manifest jsonb,
    output_json jsonb,
    status run_status NOT NULL DEFAULT 'PENDING',
    latency_ms int,
    started_at timestamptz,
    finished_at timestamptz,
    error text
);

CREATE TABLE IF NOT EXISTS review_actions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    reviewer_id uuid,
    action text NOT NULL,
    outcome case_outcome,
    notes text,
    labels jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id uuid,
    actor_type text,
    action text NOT NULL,
    entity_type text,
    entity_id uuid,
    before_json jsonb,
    after_json jsonb,
    ip inet,
    user_agent text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS detections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    video_window_id uuid REFERENCES video_windows(id),
    label text NOT NULL,
    score numeric NOT NULL DEFAULT 0,
    bbox_xyxy jsonb,
    frame_id text NOT NULL,
    frame_idx int NOT NULL DEFAULT 0,
    frame_ts timestamptz,
    query text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tracks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    video_window_id uuid REFERENCES video_windows(id),
    label text NOT NULL,
    tracker_id text,
    first_seen_ts timestamptz,
    last_seen_ts timestamptz,
    confidence numeric NOT NULL DEFAULT 0,
    zones jsonb,
    events jsonb,
    physical_item_candidate boolean NOT NULL DEFAULT false,
    receipt_candidate boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS track_observations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    track_id uuid NOT NULL REFERENCES tracks(id),
    detection_id uuid REFERENCES detections(id),
    frame_id text NOT NULL,
    frame_idx int NOT NULL DEFAULT 0,
    frame_ts timestamptz,
    bbox_xyxy jsonb
);

CREATE TABLE IF NOT EXISTS keyframes (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    video_window_id uuid REFERENCES video_windows(id),
    role text NOT NULL,
    frame_id text NOT NULL,
    frame_idx int NOT NULL DEFAULT 0,
    frame_ts timestamptz,
    track_id_ref text,
    uri text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    case_id uuid NOT NULL REFERENCES cases(id),
    video_window_id uuid REFERENCES video_windows(id),
    frame_id text NOT NULL,
    bbox_xyxy jsonb,
    text text NOT NULL DEFAULT '',
    confidence numeric NOT NULL DEFAULT 0,
    engine text NOT NULL DEFAULT 'falcon',
    crop_uri text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- TillShield POS-agent poller cursor + operational counters.
CREATE TABLE IF NOT EXISTS integration_poll_state (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_system text NOT NULL DEFAULT 'tillshield_agent',
    workstation_id text NOT NULL,
    last_txn_at timestamptz,
    last_txn_id text,
    last_poll_at timestamptz,
    last_success_at timestamptz,
    last_error text,
    rows_seen integer NOT NULL DEFAULT 0,
    events_inserted integer NOT NULL DEFAULT 0,
    cases_created integer NOT NULL DEFAULT 0,
    ignored_counts jsonb,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_system, workstation_id)
);

CREATE INDEX IF NOT EXISTS ix_pos_events_at ON pos_events (pos_event_at);
CREATE INDEX IF NOT EXISTS ix_cases_status ON cases (status);
CREATE INDEX IF NOT EXISTS ix_segments_camera_time
    ON video_segments (camera_id, start_at, end_at);
CREATE INDEX IF NOT EXISTS ix_artifacts_case ON artifacts (case_id, artifact_type);
CREATE INDEX IF NOT EXISTS ix_vlm_runs_case ON vlm_runs (case_id);
CREATE INDEX IF NOT EXISTS ix_audit_entity ON audit_log (entity_type, entity_id);
CREATE INDEX IF NOT EXISTS ix_detections_case ON detections (case_id);
CREATE INDEX IF NOT EXISTS ix_tracks_case ON tracks (case_id);
CREATE INDEX IF NOT EXISTS ix_observations_track ON track_observations (track_id);
CREATE INDEX IF NOT EXISTS ix_keyframes_case ON keyframes (case_id);
CREATE INDEX IF NOT EXISTS ix_ocr_case ON ocr_results (case_id);
