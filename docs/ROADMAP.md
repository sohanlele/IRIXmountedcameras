# Roadmap

Mapped against the founding brief's numbered "by the end of development"
checklist. Each item: current status, and what's left.

1. **Start the edge system.** Partial -- `run_demo.py`/`run_gym_demo.py`/
   `run_live_gym_demo.py` all start and run standalone; there is no
   single top-level "start the edge system" entry point wiring together
   config-driven camera/station layout + all runners for a real
   deployment yet (needs the config system from `docs/DEPLOYMENT.md`).
2. **Connect multiple cameras.** Real for RTSP/USB/file/webcam via
   `ReconnectingFrameSource`; not tested against real camera hardware.
3. **Simulate or connect multiple wristbands.** Real (simulate) as of
   2026-07-14 -- `irix.wristband_sim.SimulatedBLEGateway`,
   demonstrated with 2 concurrent wristbands in
   `run_live_gym_demo.py`. Connect (real hardware): not started, see
   `docs/WRISTBAND_SYSTEM.md`.
4. **Assign wristbands to members through the IRIX Studio API.**
   Partial -- `CheckoutRegistry` is the software-side record of an
   assignment; there is no actual IRIX Studio integration (separate
   project, out of this repo's scope per the founding brief) calling
   into it yet.
5. **Track members as they move throughout the gym.** Real --
   `GymCoordinator`/`GymSessionRunner`, demonstrated end to end in
   `run_live_gym_demo.py` (a real `StationHandoffEvent`).
6. **Detect exercises.** Partial -- `irix.exercise_recognition` (added
   Phase 2) can now classify which configured exercise a pose window
   matches (or honestly report "unknown"/ambiguous), but it isn't wired
   into session start yet -- `StationInfo.default_exercise` is still
   what `RepSession` actually uses. No `ExerciseDetected`/
   `ExerciseChanged` event exists yet (see `docs/API_SPEC.md`,
   `docs/TODO.md`).
7. **Count reps.** Real, camera+IMU fused.
8. **Detect sets.** Real (`RestGapSetBoundaryDetector`, gap-inferred, not
   hand-scripted).
9. **Track rest.** Partial -- rest gaps drive set-boundary detection
   internally; no standalone `RestStarted`/`RestEnded` event pair (see
   `docs/API_SPEC.md`).
10. **Estimate fatigue.** Real (`irix.fatigue`).
11. **Produce structured workout events.** Real (`irix.pipeline.schema`).
12. **Deliver those events to a mock backend.** Real --
    `InMemoryCloudSync` via `Aggregator`, exercised in
    `run_live_gym_demo.py`. Real backend: not started (`HTTPCloudSync`
    unwired, no live endpoint exists at `irix-mvp-app` yet).
13. **Recover from dropped cameras.** Real
    (`ReconnectingFrameSource`, tested against a scripted-failure fake).
14. **Recover from BLE disconnects.** Real (simulated) as of
    2026-07-14 -- `SimulatedBLEGateway.disconnect()`, demonstrated
    surviving a scripted dropout in `run_live_gym_demo.py`. Against real
    hardware: not started.
15. **Generate benchmark reports.** Real as of Phase 2 --
    `python -m irix.benchmark.run_benchmarks` measures pose-tracker/
    exercise-recognition/fusion/EKF/clock-sync timing, full live-pipeline
    throughput, camera-reconnect schedule, BLE-disconnect-recovery
    margin, CPU/memory -- honestly reports GPU/real-pose-inference FPS as
    unavailable in this sandboxed (no CUDA/ultralytics) environment
    rather than fabricating a number.
16. **Generate validation reports.** Partial -- `docs/VALIDATION.md`
    documents current test coverage and known gaps; benchmark reports
    (above) now exist. Still no ground-truth accuracy report (needs real
    labeled gym video -- see `docs/VALIDATION.md`).
17. **Produce documentation explaining every subsystem.** Real as of
    2026-07-14 -- the full `docs/` suite this file is part of, plus the
    pre-existing `docs/ARCHITECTURE.md`.

## Near-term priority (see `docs/TODO.md` for the itemized list)

1. Wire `irix.wristband_sim.calibration` into `run_upload.py`/
   `StationSessionRunner` so a real deployment's samples actually get
   calibrated before reaching fusion, not just validated in isolation.
2. Add `schema_version` to the event API (`docs/API_SPEC.md`).
3. Scope and add the missing event types (`ExerciseDetected`/
   `RestStarted`/`RestEnded`/`TrackingLost`/`TrackingRecovered`).
4. Build the external config system for per-gym station/camera layout.
5. Barbell/plate detector: either fine-tune against the Roboflow
   "Barbells Detector" dataset or evaluate a hosted inference API,
   closing the last major stubbed model-weights gap (see
   `docs/RESEARCH_LOG.md`).
