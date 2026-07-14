# TODO

Prioritized, actionable. Update this file whenever a real task
completes or a new gap is found -- per the founding brief, this should
stay a living document, not a snapshot.

## High priority

- [ ] **Wire `irix.wristband_sim.calibration.calibrate_stationary` into
      a real entry point** (`run_upload.py --imu`, `StationSessionRunner`
      on session start) so calibration is actually applied to samples
      before they reach `irix.fusion`, not just validated in isolation.
      A real deployment would run this once per checkout (band handed
      out, brief stationary period before first use) or once per
      firmware boot.
- [ ] **Add `schema_version` to the `CameraEvent` family**
      (`irix/pipeline/schema.py`) -- currently unversioned; see
      `docs/API_SPEC.md`. Needed before any real integration with
      `irix-mvp-app`.
- [ ] **Barbell/plate detector**: either fine-tune against the Roboflow
      "Barbells Detector" dataset or evaluate a hosted inference API for
      `FreeWeightDetector` (`irix/barbell/detector.py`) -- the last major
      stubbed model-weights gap. See `docs/RESEARCH_LOG.md` for the
      dataset pointer.
- [ ] **Scope and add missing event types**: `ExerciseDetected`/
      `ExerciseChanged` (needs an actual exercise-classification signal,
      not just per-station config), `RestStarted`/`RestEnded` (currently
      inferred internally by `RestGapSetBoundaryDetector` but not
      emitted), `TrackingLost`/`TrackingRecovered` (closest existing
      signal today is a session close on `presence_timeout_s` lapse --
      needs an explicit event pair). See `docs/API_SPEC.md`.
- [ ] **External config system** for per-gym station/camera layout
      (`StationRegistry` is currently built in Python code) -- a real
      per-site deployment shouldn't require a code change to add a
      station. See `docs/DEPLOYMENT.md`.

## Medium priority

- [ ] **Camera-angle correction for bar-path calibration**
      (`irix.barbell.calibration`) -- GymAware-style correction for
      stations whose camera isn't perfectly perpendicular to the bar
      path (partially addressed already, per `docs/ARCHITECTURE.md`'s
      "camera-tilt correction" mention -- confirm coverage is complete
      for every station geometry, not just the frontal case).
- [ ] **Real BLE hardware selection + `LiveBLEIMUStream` implementation**
      once wristband hardware exists -- see `docs/WRISTBAND_SYSTEM.md`'s
      hardware recommendation (Nordic nRF5x + ICM-42xxx/BMI2xx class IMU
      + `bleak` on the edge-box side). This is the one piece
      `irix.wristband_sim` cannot substitute for indefinitely.
- [ ] **Camera health beyond binary reconnect** -- frame-rate
      degradation, stuck/frozen-frame detection, exposure/focus drift.
      See `docs/CAMERA_SYSTEM.md`.
- [ ] **Latency/dropped-frame benchmark mode** for `run_upload.py`/
      `run_live_gym_demo.py`, satisfying the founding brief's "measure
      latency... measure dropped frames" with a real measured number.
      See `docs/VALIDATION.md`.
- [ ] **Ground-truth accuracy validation set** -- real gym-floor video +
      a small labeled set (rep counts, set boundaries at minimum),
      following Kemtai's published-methodology precedent. See
      `docs/VALIDATION.md`.
- [ ] **Cross-clock synchronization** between a real wristband's onboard
      clock and the edge box's clock -- currently assumed already
      shared (true for simulated/recorded data only). See
      `docs/SENSOR_FUSION.md`.

## Lower priority / longer-term

- [ ] **Depth/ToF camera option** for stations doing heavy barbell work
      (Tempo's approach) as a higher-accuracy alternative to single-RGB-
      frame plate-diameter calibration -- a real hardware lift, not a
      near-term software change. See `docs/RESEARCH_LOG.md`.
- [ ] **Gym-floor "minimap" dashboard** (who's authoritative where, live)
      on top of `StationRegistry`/`GymCoordinator` state -- cheap,
      high-visual-impact, no new tracking logic needed. Ops-facing, not
      a member-facing feature (would live outside this repo's scope,
      same boundary as `irix-mvp-app`).
- [ ] **3D wrist-orientation estimation (VQF)** to strip gravity's
      changing projection from the raw accelerometer signal used by
      motion-correlation disambiguation -- only worth doing if real
      crowded-station accuracy turns out to need it. See
      `docs/SENSOR_FUSION.md`/`docs/RESEARCH_LOG.md`.
- [ ] **BLE Angle-of-Arrival / UWB** upgrade path for station-pairing, if
      RSSI-based false pairings become a practical problem in a real
      deployment -- no field data yet to justify this. See
      `docs/WRISTBAND_SYSTEM.md`.
- [ ] **Containerization + Jetson deployment config.** See
      `docs/DEPLOYMENT.md`.
- [ ] **Metrics/observability** (structured logging, Prometheus-style
      export) across every subsystem, per the founding brief's
      "engineering standards" (logging, metrics, configuration).

## Documentation upkeep

- [ ] Keep `docs/IMPLEMENTATION_STATUS.md` and this file in sync with
      reality after every subsystem change -- a stale status table is
      worse than none.
- [ ] `docs/API_SPEC.md` is a hand-maintained summary of
      `irix/pipeline/schema.py`; if they drift, fix the doc, and
      consider whether the summary should instead be generated from the
      dataclasses directly to prevent drift going forward.
