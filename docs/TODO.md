# TODO

Prioritized, actionable. Refreshed during Phase 3 (2026-07-14) -- items
completed this phase removed, new gaps this phase's work surfaced added.

## High priority

- [ ] **Wire `irix.exercise_recognition.recognize_exercise` into session
      start** so a station can auto-detect/confirm the exercise instead
      of only trusting `StationInfo.default_exercise` -- needs a real
      state-machine decision (a detection preamble before `RepSession`
      commits to one exercise's joint triplet). Scoped into the Phase 3
      workout-state-machine work (`ExerciseCandidate`/`ExerciseConfirmed`
      states) rather than bolted onto `RepSession` directly, since that
      state machine is exactly where a "not yet confirmed" exercise
      needs to live. See `irix/exercise_recognition/__init__.py` for the
      extension point.
- [x] ~~Wire `irix.pose.tracker.TrackedPoseEstimator` in as
      `StationSessionRunner`'s default~~ -- done: `_ensure_estimator()`
      now wraps a real (caller-omitted) `PoseEstimator` in
      `TrackedPoseEstimator` by default; any caller that injects its own
      `pose_estimator` (every existing test/demo) is untouched. **Not
      yet validated against a real multi-person camera feed** -- no
      `ultralytics`/camera hardware in this sandbox, so this is
      integration-correct but accuracy-unvalidated; see
      `docs/VALIDATION.md`.
- [x] ~~Wire `irix.fusion.clock_sync.ClockSyncEstimator` into a live
      entry point~~ -- done, but not the way this bullet originally
      envisioned. `StationSessionRunner` now builds one
      `ClockSyncEstimator` per open session and every `add_imu_samples`
      call applies its current correction (`RepSession`/
      `station_runner.py`). What did **not** get built: automatic
      per-set observation derivation from camera-rep-vs-IMU-peak
      timestamp pairing -- tried, and reverted, because a camera's
      rep-*completion* timestamp and an IMU counter's acceleration-
      *peak* timestamp mark different phases of one physical rep, so
      pairing them conflates a fixed phase offset with genuine clock
      drift (see `irix/fusion/clock_sync.py`'s
      `estimate_offset_from_paired_events` docstring and
      `tests/test_rep_session_clock_sync.py` for the full account).
      `StationSessionRunner.calibrate_wristband_clock()` is the real
      entry point now -- a caller supplies a trustworthy
      `(offset_s, confidence)` pair. **Next step**: build that caller --
      a calibration routine using
      `estimate_offset_via_cross_correlation` against camera-tracked
      wrist-keypoint vertical velocity (available every frame from
      `PersonPose`, no barbell required, unlike bar-path velocity) and
      raw wristband vertical accel over the same window. Not yet built.
- [x] ~~Wire camera-calibration undistortion into a live entry point~~ --
      done: `CalibrationProfile.undistort_frame()` (thin wrapper around
      `irix.barbell.calibration.undistort_frame`) is now applied to
      every frame in `StationSessionRunner.tick()` before pose
      estimation, when a `calibration_profile` is configured for the
      station (optional -- `None` skips undistortion unchanged).
- [x] ~~Wire `irix.weight_recognition.plate_color_check` into the
      production weight-recognition path~~ -- done: `RepSession`'s
      weight-check block now runs color-plate detection on every
      periodic check regardless of whether a VLM backend is configured
      (`method="color_plate"`, zero-training, no API key), and
      cross-checks it against a VLM read (`method="vlm"`) when one is
      configured (`WeightConfirmedEvent.color_check_consistent`/
      `color_check_reason`, alongside the existing geometry check).
- [x] ~~Wire `irix.wristband_sim.calibration.calibrate_stationary` into
      a real entry point~~ -- partially done: `irix.identity.placement.
      WristbandPlacementTracker` now calls it on every placement-change
      settle. **Still open**: an ordinary session that never changes
      placement still runs on raw, uncalibrated samples from the start --
      see `docs/WRISTBAND_SYSTEM.md`'s "Calibration" section.
- [ ] **Wristband placement side laterality doesn't reach exercise
      joint-triplet selection.** `irix.identity.placement.BandSide`
      tracks left/right explicitly, but `ExerciseConfig.joint_triplet`
      still hardcodes the left side throughout `irix/rep_counting/
      exercises.py` (a pre-existing simplification, not something Phase
      3's placement work changed) -- so a band confirmed on the *right*
      wrist/ankle doesn't yet change which keypoints `RepSession` reads
      pose angles from. Low priority until multi-person/crowded-station
      accuracy work needs it.
- [ ] **Automatic clock-sync calibration signal.** See the
      `ClockSyncEstimator` entry above -- `estimate_offset_via_cross_
      correlation` against camera-tracked wrist-keypoint vertical
      velocity and raw wristband vertical accel is the identified next
      step, not yet built.
- [ ] **`IdentityResolution` not yet wired into `StationSessionRunner`'s
      trivial (sole-candidate) path.** `irix.identity.resolution` (Phase
      3) exists and is used by nothing yet -- `CrowdedGroupDisambiguator`
      still returns bare `{wristband_id: PersonPose}` and
      `StationSessionRunner.tick`'s single-candidate branch still uses
      the sole present wristband's member_id directly, neither
      constructing an `IdentityResolution`. Natural next step: wire both
      paths through it, and surface it as a real event once the Priority
      6 workout state machine's `identity_candidate`/`identity_confirmed`/
      `identity_degraded` states exist to consume it -- built in that
      order deliberately (a resolution with nowhere to go is just an
      unused data class).
- [ ] **`SessionRecorder` not wired into `GymSessionRunner`.** Wired
      into `StationSessionRunner` only (Phase 3, Priority 8) -- a
      multi-station gym run has no single place recording across
      stations yet; a caller wanting a whole-gym recording currently has
      to construct one `SessionRecorder` per station manually. Also:
      `frames/*.npy` (the opt-in raw-frame path) has no companion video
      encoder -- fine for algorithm-comparison/replay (frame arrays load
      straight back into numpy), not yet a shareable video file.
- [ ] **Wire per-station rep/set events into the gym-wide
      `WorkoutStateMachine`.** `irix.live.gym_runner.GymSessionRunner`
      now drives session/identity/station-transition/health phases
      correctly (Phase 3), but `SET_STARTED`/`record_rep_completed`/
      `SET_ENDED`/`REST_STARTED`/`REST_ENDED` are not yet triggered from
      real `RepSession` events -- each `StationSessionRunner`'s
      `on_events` callback would need to also forward
      `RepCompletedEvent`/`SetCompleteEvent` up to `GymSessionRunner`
      (which already owns the right `WorkoutStateMachine` instance per
      wristband_id) rather than only to that station's own event sink.
- [ ] **"Motion onset" as its own identity-fusion signal.** Named
      explicitly in the founding brief's identity-fusion signal list
      (alongside camera trajectory/IMU motion/timing/clock sync/station
      occupancy/camera zones/previous confirmed identity/BLE context,
      all of which already have a real source -- see `irix.identity.
      resolution`'s module docstring) but not yet its own distinct
      input anywhere: a just-arrived member's first detected movement as
      a corroborating signal, distinct from `irix.identity.
      motion_correlation`'s steady-state periodic correlation.
- [ ] **Add `schema_version` to the `CameraEvent` family** (still open --
      see `docs/API_SPEC.md`).
- [ ] **Barbell/plate object detector** (`FreeWeightDetector`, still an
      untrained stub) -- fine-tune against the Roboflow "Barbells
      Detector" dataset or evaluate a hosted inference API. Once real,
      `irix.weight_recognition.plate_color_check` and
      `plate_geometry_check` both become strictly more useful (a real
      bounding box to crop color regions from, instead of this phase's
      standalone color-blob detection).
- [ ] **Machine weight-stack reading** (Priority 7's "machine stacks") --
      genuinely not started. Plate-loaded free-weight equipment
      (color-coded bumper plates, VLM read) is the only load-bearing
      equipment type this repo can read at all; a selectorized machine's
      pin-in-stack weight (a printed number on a metal plate, read via
      OCR or a small trained detector, not a bumper-plate color) has no
      implementation, no research entry, and no test coverage yet. Real,
      not-yet-attempted work, not a stub -- would need either an OCR
      model (e.g. a lightweight digit-recognition model, since general
      OCR on stamped metal in variable gym lighting is a harder problem
      than printed-page OCR) or example machine-stack images to even
      start on, neither of which exist in this environment.
- [ ] **Printed-plate OCR** (Priority 7's "OCR") as a fallback for
      non-color-coded plates (the common case for commercial-gym cast-
      iron/rubber plates -- see `plate_color_check.py`'s own "what this
      doesn't cover" note) -- same real gap as machine-stack reading
      above, same blocker (no OCR model, no labeled data here).
- [x] ~~Extend benchmarks: identity latency, event latency~~ -- done
      (Priority 9): `benchmark_identity_resolution_latency`,
      `benchmark_event_latency`, and `benchmark_packet_loss_impact`
      (BLE packet loss vs. fusion degradation, not a timing number but a
      behavioral one) added to `irix/benchmark/run_benchmarks.py`.
- [ ] **Real pose-inference + GPU benchmarks** once `ultralytics`/CUDA
      are available (not in this sandboxed environment) --
      `irix.benchmark.benchmark_pose_inference`/`_gpu_available` already
      auto-detect and run for real the moment they're installed; nothing
      else to build, just needs to actually run on target hardware.
- [x] ~~Scope and add missing event types~~ -- done: `TrackingLostEvent`/
      `TrackingRecoveredEvent`/`RestStartedEvent`/`RestEndedEvent`/
      `ExerciseDetectedEvent` all exist now (`irix/pipeline/schema.py`,
      `docs/API_SPEC.md`). Emission wired for `TrackingLost`/
      `TrackingRecovered` in `StationSessionRunner`'s single-candidate
      path only (a consecutive-missed-frame streak) -- **not yet** for
      the crowded-station path (`irix.live.disambiguation.
      CrowdedGroupDisambiguator` returning nothing for a slot while
      buffering looks the same as "actually lost track" from
      `StationSessionRunner`'s side today; needs its own signal to tell
      the two apart before wiring tracking-loss there too).
      `RestStartedEvent`/`RestEndedEvent` are **not yet emitted by
      anything** -- need a live timer comparing "now" against the last
      completed rep's timestamp *between* ticks (the existing
      `RestGapSetBoundaryDetector` only infers a gap retroactively, at
      the next rep -- correct for its own batch/replay job, wrong shape
      for a real-time "resting right now" event). `ExerciseDetectedEvent`
      is **not yet emitted by anything** -- still needs the session-start
      wiring described in this file's first bullet.

## Medium priority

- [ ] **Extend the IWF plate-color table cautiously.** Only the four
      web-search-confirmed weights (10/15/20/25 kg) are mapped in
      `irix.weight_recognition.plate_color_check` -- the standard is
      commonly described as covering lighter plates too (5 kg white,
      2.5 kg black) but that wasn't independently confirmed. Verify
      against an authoritative IWF technical spec (not a retailer blog)
      before adding them.
- [ ] **RSSI smoothing (EMA) ahead of `StationPairing.resolve()`.**
      `irix.topology.handoff`'s `min_consecutive` hysteresis already
      absorbs most flicker at the decision layer; a smoothed RSSI input
      would reduce noise one layer earlier. Modest expected value,
      deferred this phase in favor of higher-impact work (see
      `docs/RESEARCH_LOG.md`'s BLE localization findings).
- [ ] **Camera health beyond binary reconnect** -- frame-rate
      degradation, stuck/frozen-frame detection (still open -- see
      `docs/CAMERA_SYSTEM.md`).
- [ ] **Ground-truth accuracy validation set** -- real gym-floor video +
      labeled reps/sets (still open -- see `docs/VALIDATION.md`); would
      also let `irix.exercise_recognition`'s scoring thresholds be tuned
      against real motion instead of only synthetic data.
- [ ] **BoT-SORT-style ReID for the intra-camera tracker** if a real
      deployment's crowded-station rate turns out to need appearance
      matching beyond IOU+motion -- deliberately not built this phase
      (see `irix/pose/tracker.py`'s module docstring for why IOU+Kalman
      is sufficient at gym-station scale today).
- [x] ~~External config system for per-gym station/camera layout~~ --
      done (Priority 10): `irix.config.gym_config`. Not yet wired: no
      entry point actually loads a `GymConfig` and constructs a live
      `GymSessionRunner`/`StationSessionRunner` set from it end-to-end --
      every current demo/live entry point still builds its `StationInfo`
      list and per-station kwargs inline. That wiring (a `build_gym_
      session_runner(config, ...)`-style factory, hardware bindings
      still supplied by the caller) is the natural next step, not yet
      built.

## Lower priority / longer-term

- [ ] **Real BLE hardware selection + `LiveBLEIMUStream` implementation**
      (still open -- see `docs/WRISTBAND_SYSTEM.md`).
- [ ] **Depth/ToF camera option** for heavy barbell stations (still open
      -- see `docs/RESEARCH_LOG.md`).
- [ ] **Gym-floor "minimap" dashboard** (still open).
- [ ] **3D wrist-orientation estimation (VQF)** for motion-correlation
      disambiguation (still open -- see `docs/SENSOR_FUSION.md`).
- [ ] **BLE Angle-of-Arrival / UWB** upgrade path (still open).
- [ ] **Containerization + Jetson deployment config** (still open -- see
      `docs/DEPLOYMENT.md`; explicitly deprioritized this phase per
      "objective is no longer to improve infrastructure").
- [ ] **Metrics/observability** (structured logging, Prometheus-style
      export) -- still open.
- [ ] **Trained sequence-model exercise recognition** (ST-GCN/temporal-
      transformer) against MM-Fit or real labeled gym video, replacing
      Phase 2's zero-training baseline once a real GPU training pipeline
      and labeled data exist -- see `irix/exercise_recognition/__init__.py`
      for why that wasn't attempted this phase (sandboxed, GPU-less
      environment; would produce an undertrained model that's worse than
      the honest baseline).

## Documentation upkeep

- [ ] Keep `docs/IMPLEMENTATION_STATUS.md` and this file in sync with
      reality after every subsystem change.
- [ ] `docs/API_SPEC.md` is a hand-maintained summary of
      `irix/pipeline/schema.py`; if they drift, fix the doc.
