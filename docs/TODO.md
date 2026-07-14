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
- [ ] **Wire `irix.wristband_sim.calibration.calibrate_stationary` into
      a real entry point** (still open from Phase 1 -- see
      `docs/WRISTBAND_SYSTEM.md`).
- [ ] **Add `schema_version` to the `CameraEvent` family** (still open --
      see `docs/API_SPEC.md`).
- [ ] **Barbell/plate object detector** (`FreeWeightDetector`, still an
      untrained stub) -- fine-tune against the Roboflow "Barbells
      Detector" dataset or evaluate a hosted inference API. Once real,
      `irix.weight_recognition.plate_color_check` and
      `plate_geometry_check` both become strictly more useful (a real
      bounding box to crop color regions from, instead of this phase's
      standalone color-blob detection).
- [ ] **Real pose-inference + GPU benchmarks** once `ultralytics`/CUDA
      are available (not in this sandboxed environment) --
      `irix.benchmark.benchmark_pose_inference`/`_gpu_available` already
      auto-detect and run for real the moment they're installed; nothing
      else to build, just needs to actually run on target hardware.
- [ ] **Scope and add missing event types** (`ExerciseDetected`/
      `RestStarted`/`RestEnded`/`TrackingLost`/`TrackingRecovered`,
      still open -- see `docs/API_SPEC.md`). `ExerciseDetected` now has
      a natural source (`irix.exercise_recognition`) once the session-
      start wiring above exists.

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
- [ ] **External config system** for per-gym station/camera layout
      (still open -- see `docs/DEPLOYMENT.md`).

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
