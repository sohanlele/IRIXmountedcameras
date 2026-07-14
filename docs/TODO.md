# TODO

Prioritized, actionable. Refreshed end of Phase 2 (2026-07-14) -- items
completed this phase removed, new gaps this phase's work surfaced added.

## High priority

- [ ] **Wire `irix.exercise_recognition.recognize_exercise` into session
      start** so a station can auto-detect/confirm the exercise instead
      of only trusting `StationInfo.default_exercise` -- needs a real
      state-machine decision (a detection preamble before `RepSession`
      commits to one exercise's joint triplet), deliberately not bolted
      on with the rest of Phase 2's work to avoid destabilizing
      `RepSession`'s existing, well-tested behavior. See
      `irix/exercise_recognition/__init__.py` for the extension point.
- [ ] **Wire `irix.pose.tracker.TrackedPoseEstimator` in as
      `StationSessionRunner`'s default** (currently opt-in) once its
      behavior has been validated against a real multi-person camera
      feed -- persistent track_id also opens the door to a future
      pixel-based crowded-station disambiguation signal alongside the
      existing motion-correlation one.
- [ ] **Wire `irix.fusion.clock_sync.ClockSyncEstimator` into a live
      entry point** (e.g. `StationSessionRunner`, applying
      `apply_clock_sync` to polled IMU samples before they reach
      `RepSession`) -- the estimator and the simulated ground-truth
      validation (`irix.wristband_sim`'s `clock_drift_ppm`) both exist
      now, but nothing in the live path actually calls it yet.
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
