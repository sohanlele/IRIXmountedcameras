# API spec: the CameraEvent family

This is the contract between this repo and `irix-mvp-app` (or any other
consumer): a set of structured, typed events, never raw video, never a
statutorily-defined biometric identifier. Source of truth is
`irix/pipeline/schema.py` -- this document is a reference summary; if the
two disagree, the code is correct and this file needs updating (see
`docs/TODO.md`).

## Versioning status

**Current state: unversioned.** There is one flat `CameraEvent` union
type with no `schema_version`/`event_version` field, and `HTTPCloudSync`
(`irix/pipeline/cloud_sync.py`) has no version negotiation. This is a
real gap against the founding brief's "versioned event API" requirement
-- acceptable while `irix-mvp-app` doesn't yet expose a live-ingestion
endpoint to version against (see `docs/PRODUCT_SPEC.md`), but should be
closed before a real integration: recommended approach is a top-level
`schema_version: int` field on every event (bump on any breaking field
change) plus an `event_type` string discriminator (already present on
every event's `to_dict()` output today) so a consumer can dispatch on
type without needing Python's `Union` typing. See `docs/TODO.md`.

## Transport today

`irix.pipeline.edge_buffer.LocalBuffer` (per zone, rolling window,
never holds raw frames) -> `irix.pipeline.aggregator.Aggregator` (drains
every registered zone, forwards to one `CloudSync`) -> `CloudSync`
(`InMemoryCloudSync` for tests/demos; `HTTPCloudSync` is an unwired
sketch -- no real endpoint exists yet to point it at).

## Events

Every event has a `timestamp: float` and a `to_dict()` method (the JSON
shape delivered to a consumer). `member_id` is always a wristband-
assigned id (`CheckoutRegistry.resolve_member`), never a biometric
identifier.

### `RepCompletedEvent`
One rep finished. `member_id`, `station_id`, `exercise`, `rep_count`,
`form_score` (0-1, `None` if unscored), `form_faults` (list of fault
code strings, e.g. `["knee_valgus"]`), `weight_kg`, `duration_s` (time
since previous rep). Velocity is two-tier: `peak/mean_velocity_deg_s`
(camera joint-angle proxy, always available) vs.
`peak/mean_velocity_m_s`/`velocity_loss_pct`/`estimated_rpe` (calibrated,
only when a barbell/dumbbell is tracked for this rep -- `None`
otherwise, consumer should fall back to the deg/s fields).

### `SetCompleteEvent`
A set ended. `member_id`, `station_id`, `exercise`, `total_reps` (the
camera's own count, kept for backward compatibility), plus
`imu_rep_count`/`fused_rep_count`/`rep_count_agreement`/
`rep_count_source` when wristband IMU data was available for the set --
`fused_rep_count` is what a consumer should treat as authoritative when
present (see `docs/SENSOR_FUSION.md`).

### `BandPlacementRequiredEvent`
The member's IMU band needs to move (e.g. wrist -> ankle for a machine
leg exercise) before wristband-derived signals for the next exercise are
trustworthy. `member_id`, `exercise`, `from_placement`/`to_placement`
(`"wrist"` | `"ankle"`). Emitted only on an actual change, not every
exercise (`irix.pipeline.events.BandPlacementTracker`).

### `BandPlacementConfirmedEvent`
The member's IMU band's *actual physical* placement changed and was
confirmed (settled + recalibrated) by `irix.identity.placement.
WristbandPlacementTracker` (Phase 3). Distinct from
`BandPlacementRequiredEvent` above, which is the earlier, top-down "the
next exercise needs a different placement" signal -- this is the
bottom-up confirmation that the physical move actually happened and IMU
fusion has resumed. `wristband_id`, `from_side`/`to_side`
(`"left_wrist"` | `"right_wrist"` | `"left_ankle"` | `"right_ankle"` |
`"unknown"`). See `docs/WRISTBAND_SYSTEM.md`.

### `WeightConfirmedEvent`
A station's weight-recognition check reached agreement on the loaded
weight -- `method` (`"vlm"` | `"color_plate"`) says which one produced
this read: `"vlm"` when a VLM backend is configured for this station
(N-of-M read-confirmation agreement); `"color_plate"` otherwise, from
`irix.weight_recognition.plate_color_check`'s zero-training,
no-API-key color-coded-bumper-plate detection (Phase 3 default).
`member_id`, `station_id`, `exercise`, `weight_kg`, `confidence`, plus
`geometry_consistent`/`geometry_check_reason` (independent geometric
plate-count cross-check against the barbell detector's own read) and
`color_check_consistent`/`color_check_reason` (cross-check between the
VLM and color-plate reads when both ran) -- either pair is `None` when
that particular check wasn't run for this reading.

### `StationHandoffEvent`
A member's authoritative station changed. `member_id`, `from_station`,
`to_station`, `plausible_adjacency` (`False` if `to_station` isn't a
registered neighbor of `from_station` -- worth surfacing to an ops
dashboard as a likely mis-resolved BLE reading rather than trusting an
implausible jump).

### `SetFatigueSummaryEvent`
Pushed alongside `SetCompleteEvent`. `member_id`, `station_id`,
`exercise`, `rep_count`, `velocity_tier` (`"m_s"` | `"deg_s"` |
`"none"`), `velocity_loss_pct`/`velocity_loss_zone` (`"VL10"` |
`"VL20"` | `"VL30"` | `"VL45"`), `tempo_drift_pct`, `mean_form_score`,
`most_common_fault`, plus cross-set context
(`set_to_set_velocity_trend_pct`, `session_fatigue_index`,
`completed_sets_this_session`) once a second set of the same exercise has
happened this session. Descriptive/classifying only -- never a
prescriptive instruction ("reduce the weight"); that judgment belongs to
`irix-mvp-app`'s AI layer, not this repo (see
`docs/PRODUCT_SPEC.md`).

## Confidence fields

Per the founding brief's "every event should include confidence"
requirement: confidence is present per-field where it's actually
meaningful (`form_score`, `WeightConfirmedEvent.confidence`,
`rep_count_agreement`), rather than one blanket per-event score, since a
single number for e.g. `SetFatigueSummaryEvent` (which bundles several
independently-uncertain sub-measurements) would obscure more than it
reveals. `RepCompletedEvent`/`SetCompleteEvent` don't carry an explicit
"tracking confidence" event-level field today, though `RepEvent.
tracking_confidence` exists upstream in `irix.rep_counting.state_machine`
and feeds `RepCountFusion`'s fallback decision -- surfacing it directly
on the outbound event (not just consuming it internally) is a reasonable
follow-up; see `docs/TODO.md`.

## Events not yet implemented (named in the founding brief, not in `schema.py` today)

`TrackingLost`/`TrackingRecovered` -- the closest existing signal is a
`StationHandoffEvent` or a session simply closing on `presence_timeout_s`
lapse (`StationSessionRunner`/`GymSessionRunner`), but neither is emitted
as an explicit lost/recovered pair today. `ExerciseChanged`/
`ExerciseDetected`/`LoadDetected` -- exercise is currently configured per
station (`StationInfo.default_exercise`), not detected/classified, so
there's no natural trigger for an `ExerciseChanged`/`ExerciseDetected`
event yet; `WeightConfirmedEvent` covers the load case under a different
name. `RestStarted`/`RestEnded` -- rest is inferred internally by
`RestGapSetBoundaryDetector` to decide set boundaries but not emitted as
its own event pair. See `docs/TODO.md` for scoping this gap.
