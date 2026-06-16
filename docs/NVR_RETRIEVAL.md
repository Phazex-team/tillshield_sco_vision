# NVR on-demand video retrieval (Dahua)

For POS-triggered return cases the app can pull the historical video
**for the transaction window** directly from the Dahua NVR, instead of
relying only on the local continuous recorder. This is **additive and
non-blocking**: the local recorder/segment path remains the fallback and
is never disabled by this feature.

## What works now

| Stage | Capability | Status |
|-------|-----------|--------|
| **A** | Historical **search** via Dahua `mediaFileFind` CGI — find the `.dav` recording(s) covering a tx window | ✅ proven against the live NVR (`DHI-NVR608-32-4KS2`) |
| **B** | **Clip export** of the exact bounded window via RTSP playback-by-time → MP4 | ✅ proven (pulled a 60 s / 1080p hevc clip) |

Export streams at ~real-time (a 5-minute window takes ~5 minutes) and is
hard-bounded to the tx window — **never long-hour pulls** (see
`max_window_sec`, ceiling 15 min). On any failure it falls back to local
segments.

## Channel numbering — live vs playback

The Dahua NVR uses **two numbering schemes for the same camera**:

- **Live RTSP** uses a **1-based** `channel=` (the return counter is
  `channel=15`): `rtsp://…/cam/realmonitor?channel=15&subtype=0`.
- **`mediaFileFind` search** also takes a 1-based `condition.Channel`,
  but the **result `Channel` field and on-disk path are 0-based**.
  Searching `condition.Channel=15` returns results labelled `Channel=14`
  under `/mnt/dvr/.../14/dav/` — **the same physical camera**.

So `live_channel` and `playback_channel` are configured **separately** and
nothing is hardcoded. `playback_channel` is sent verbatim as
`condition.Channel`; the resolved result channel + file path are logged
and stored in the case window's `nvr_metadata` so an operator can confirm
the mapping. For the return counter both happen to be `15` on this NVR.

> The live RTSP URL is confirmed correct as
> `rtsp://<user>:<urlencoded-pass>@<nvr-ip>:554/cam/realmonitor?channel=15&subtype=0`.
> The `14` seen in recording searches is the NVR's internal 0-based
> channel for that same camera — not a different camera.

## Per-camera config

```yaml
cameras:
  - id: cam_return_01
    rtsp_url: 'rtsp://<user>:<urlencoded-pass>@<nvr-ip>:554/cam/realmonitor?channel=15&subtype=0'
    nvr:
      enabled: true
      base_url: http://192.168.1.13
      username: ${NVR_USERNAME}            # prefer env NVR_USERNAME
      password: ${NVR_PASSWORD}        # prefer env NVR_PASSWORD (literal for digest)
      live_channel: 15          # 1-based RTSP channel (live + playback URL)
      playback_channel: 15      # mediaFileFind condition.Channel (1-based)
      subtype: 0
      timezone: Asia/Dubai      # NVR local wall-clock for search/playback
      pre_roll_sec: 120
      post_roll_sec: 180
      max_window_sec: 900       # hard cap: only the tx window (<= 15 min)
      prefer_on_demand_for_pos: true
      export_enabled: true      # Stage B; set false for search-only (Stage A)
```

- **Backward compatible:** a camera with no `nvr:` block behaves exactly
  as before (local recorder only).
- **Credentials** prefer `NVR_USERNAME` / `NVR_PASSWORD` env vars when set
  (matching the repo's secrets pattern); otherwise the config values are
  used. The password is used literally for HTTP digest auth and
  URL-encoded for the RTSP playback URL.
- **Timezone:** `pos_event_at` is stored in UTC; the client converts it to
  the NVR's local wall-clock for both search and playback.

## Retrieval flow & fallback contract

For a qualifying POS case, `app.case_runner.analyze_case`:

1. resolves `camera_id` and the bounded tx window from `pos_event_at`;
2. if the camera has `nvr.enabled` + `prefer_on_demand_for_pos`, queries
   the NVR for recordings overlapping that window;
3. if `export_enabled` and a covering recording exists, exports the exact
   clip via RTSP playback and **uses it** as the analysis window;
4. otherwise records NVR availability metadata and **falls back to the
   local recorded segments** for the actual evidence clip.

The chosen path is recorded explicitly on the case's
`video_windows.acquisition_source` (and full detail in `nvr_metadata`),
surfaced via `GET /api/v1/cases/{id}` under `latest_window`:

| `acquisition_source` | Meaning |
|----------------------|---------|
| `nvr_clip_retrieved` | exact historical clip exported from the NVR (Stage B) |
| `nvr_recording_found_no_export` | NVR has covering footage; export off/failed → local used |
| `nvr_no_recording_found` | NVR reachable, no covering recording → local used |
| `nvr_query_failed` | NVR unreachable/error → local used (`error` captured) |
| `nvr_disabled` | camera has no NVR config |
| `local_segments_used` | local recorder segments used for the clip |
| `local_no_segments` | no NVR + no local segments → `INVALID_VIDEO` |

`nvr_metadata` also carries: `attempted`, `playback_channel`,
`window_start`/`window_end`, `recordings_found`, `first_match`
(file path + start/end + result channel), `export_attempted`,
`export_ok`, `fallback_used`, and `last error`.

## Adding a new NVR-backed camera

1. Add the camera with its live `rtsp_url` and a `nvr:` block.
2. Set `playback_channel` so the search returns *that* camera's
   recordings — verify via the logged result channel / file path.
3. Set `timezone` to the NVR's local zone.
4. Keep the local recorder running for that camera until NVR export is
   confirmed reliable in your environment.

## Operational notes

- The local continuous recorder is **not** disabled by this feature and
  remains the fallback.
- Export is ~real-time and bounded; it never pulls more than the tx
  window (≤ 15 min).
- Unit tests mock all NVR responses (`tests/test_nvr_dahua.py`); they do
  not touch the real NVR.
