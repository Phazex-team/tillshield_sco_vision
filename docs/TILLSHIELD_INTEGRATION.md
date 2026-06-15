# TillShield POS integration

The reviewer/investigation platform exposes a thin compatibility surface
that lets the TillShield agent stream return/refund transactions
directly into this app. The TillShield agent sends rows from its
`tillshield_agent.pos_api_transactions` table; the app converts each
row into a `PosEvent`, opens (or reuses) a case, and runs the existing
transaction-led investigation flow (POS → window → perception →
reasoning → decision policy → evidence package → reviewer workflow).

## Endpoints

| Verb | Path | Purpose |
|------|------|---------|
| POST | `/api/v1/integrations/tillshield/transactions/event` | One TillShield transaction |
| POST | `/api/v1/integrations/tillshield/transactions/batch` | Many TillShield transactions |

Both endpoints return **HTTP 200 on duplicates** so the agent can retry
freely. They are idempotent on the natural key
`(store_id, workstation_id, transaction_id, line_id)`.

## Authentication (optional)

Set a shared secret in either the environment or `config.yaml`:

```bash
export TILLSHIELD_INGEST_TOKEN="phzx_..."
```

or

```yaml
integrations:
  tillshield:
    ingest_token: "phzx_..."
```

The agent then sends the secret on every call:

```
X-PhazeX-Ingest-Token: phzx_...
```

When no token is configured the endpoint is open (dev mode). The token
value is **never** echoed in responses, audit rows, or logs.

## Field mapping

| TillShield (`pos_api_transactions`) | Canonical `PosEvent` | Notes |
|-------------------------------------|----------------------|-------|
| `transaction_id`                    | `transaction_id`     | natural-key component |
| `workstation_id`                    | `terminal_id`        | natural-key component |
| `store_id`                          | `store_id`           | natural-key component |
| —                                   | `line_id = "transaction"` | unless line-level cases enabled (see below) |
| `transaction_type`                  | `event_type` (uppercased, hyphens/spaces → `_`) | RETURN / REFUND / REFUND_RETURN / RETURN_REFUND / VOID_RETURN open cases |
| `transaction_date`                  | `pos_event_at`       | **never** `received_at` — delayed batches are normal |
| `operator_id`                       | `staff_id`           |  |
| `cashier_name`                      | `raw_payload.cashier_name` |  |
| `reference_id`                      | `raw_payload.reference_id` |  |
| `transaction_end_date`              | `raw_payload.transaction_end_at` |  |
| `currency`                          | `currency`           |  |
| `total_items`                       | `quantity`           |  |
| `total_amount`                      | `amount`             |  |
| `items` jsonb                       | `raw_payload.items`  | preserved verbatim |
| `payload` jsonb                     | `raw_payload.raw_payload` | preserved verbatim |
| `source_ip`                         | `raw_payload.source_ip` | falls back to request IP |
| `received_at`                       | `raw_payload.received_at` |  |

`source_system` on the resulting `pos_batches` row is always
`"tillshield_agent"`.

## Transaction-type normalization

| Incoming `transaction_type` | Normalised | Opens case? |
|-----------------------------|------------|-------------|
| `Return`, `RETURN`          | `RETURN`   | yes |
| `Refund`, `REFUND`          | `REFUND`   | yes |
| `Refund-Return`             | `REFUND_RETURN` | yes |
| `Return Refund`             | `RETURN_REFUND` | yes |
| `Void Return`               | `VOID_RETURN`   | yes |
| `SALE`, anything else       | _(unchanged)_   | no (counted in `ignored_non_return_events`) |

Extend or restrict the accepted set in
`config.yaml.integrations.tillshield.return_event_types`.

## Line-level vs transaction-level cases

Default: **one case per transaction** with `line_id = "transaction"`.
Set `integrations.tillshield.line_level_cases: true` to open one case
per `items[]` entry that has a stable `line_id` / `lineId` / `id` /
`sequence`. If item line IDs are missing or duplicated, the normaliser
falls back to the transaction-level case so we never invent unstable
IDs.

## Idempotency + retry

- Replaying the same `(store_id, workstation_id, transaction_id,
  line_id)` returns 200 with `duplicate_batch: true` semantics carried
  through `events_already_present`. No second case is opened.
- Replaying the same exact payload yields the same `pos_batches.id`.

## Timezone expectations

`transaction_date` should be timezone-aware ISO-8601
(`2026-06-15T14:00:00+04:00`). The app normalises every internal
datetime to naive UTC so SQLite + Postgres behave identically. Delayed
events (up to 30 minutes per PRODUCTION_SPEC §8) are normal and the
correlation step uses `transaction_date`, never `received_at`.

## Example single event

```bash
curl -X POST http://127.0.0.1:3902/api/v1/integrations/tillshield/transactions/event \
  -H "content-type: application/json" \
  -H "X-PhazeX-Ingest-Token: phzx_..." \
  -d '{
    "transaction_id": "RTX-2026-000123",
    "reference_id": "REF-99",
    "store_id": "store_001",
    "workstation_id": "WS-12",
    "transaction_type": "RETURN",
    "transaction_date": "2026-06-15T14:00:00+04:00",
    "transaction_end_date": "2026-06-15T14:01:42+04:00",
    "operator_id": "op_77",
    "cashier_name": "Anita",
    "currency": "AED",
    "total_items": 1,
    "total_amount": "49.900",
    "items": [
      {"line_id": "L1", "sku": "SKU-A",
       "description": "shirt", "quantity": 1, "amount": 49.9}
    ],
    "payload": {"channel": "in_store"},
    "source_ip": "10.0.0.42"
  }'
```

Response:

```json
{
  "events_inserted": 1,
  "events_already_present": 0,
  "cases_created": 1,
  "case_ids": ["8a32d3c5-..."],
  "ignored_non_return_events": 0,
  "errors": []
}
```

## Example batch

```bash
curl -X POST http://127.0.0.1:3902/api/v1/integrations/tillshield/transactions/batch \
  -H "content-type: application/json" \
  -H "X-PhazeX-Ingest-Token: phzx_..." \
  -d '{
    "source_system": "tillshield_agent",
    "events": [
      {"transaction_id": "RTX-1", "store_id": "store_001",
       "workstation_id": "WS-12", "transaction_type": "RETURN",
       "transaction_date": "2026-06-15T14:00:00+04:00",
       "total_amount": "49.900"},
      {"transaction_id": "RTX-2", "store_id": "store_001",
       "workstation_id": "WS-12", "transaction_type": "SALE",
       "transaction_date": "2026-06-15T14:01:00+04:00",
       "total_amount": "120.000"}
    ]
  }'
```

The `SALE` row is reported as `ignored_non_return_events: 1` and does
not create a case.

## Verifying a case was created

After the call:

```bash
curl http://127.0.0.1:3902/api/v1/cases | jq '.items[] | {id, status, outcome, pos_event}'
```

Or open the reviewer UI at `/review.html` and look for the new case in
the queue. The POS column shows `transaction_id`; the staff column
shows `operator_id`.

## How the TillShield agent should call this app

Per transaction:

1. Receive the POS payload.
2. Insert it into your own `pos_api_transactions` table.
3. POST the same row to
   `/api/v1/integrations/tillshield/transactions/event`.
4. Retry on 5xx; ignore 200 with `duplicate_batch`-equivalent counts.
5. On 400, fix the payload and resubmit — the response `errors` list
   tells you which field rejected.

Per batch (recommended for replay / catch-up):

1. Collect up to a few hundred rows from the local table.
2. POST as `events: [...]` to
   `/api/v1/integrations/tillshield/transactions/batch`.
3. Track `events_inserted` / `events_already_present` for confidence.

## Assumptions

- `transaction_type` is a string. The normaliser uppercases it and
  replaces hyphens / spaces with underscores before matching against
  `return_event_types`. If the upstream uses numeric codes, map them to
  one of the canonical strings before sending.
- `total_amount` may be a string (PostgreSQL `numeric(12,3)` often
  serializes that way); Pydantic accepts both.
- `transaction_date` must be present. The endpoint returns HTTP 400
  with a clear error when the field is missing or unparseable.
- `items[]` is preserved as-is — no validation beyond JSON. Line-level
  case opening requires each item to expose a stable `line_id` or
  equivalent; otherwise the normaliser opens a single transaction-level
  case.
