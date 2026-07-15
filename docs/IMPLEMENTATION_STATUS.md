# Implementation status

Legend: **Real** (working, tested, no known stub) / **Partial** (real
logic, but a real gap stated below) / **Stub** (interface exists,
implementation deliberately deferred) / **Not started**.

| Subsystem | Status | Notes |
|---|---|---|
| Pose estimation | Real | Pretrained YOLO-Pose, verified against real image+video |
| Rep counting (joint-angle state machine) | Real | 6 exercise configs (squat/curl/deadlift/bench/leg press/hack squat) |
| Form scoring | Real | 5 rule-based checks, no trained classifier |
| Set/rest boundary detection | Real | `RestGapSetBoundaryDetector`, no hand-scripted set length |
| Camera+IMU visual-inertial fusion (EKF+ZUPT) | Real | |
| IMU-only rep counting | Real | Ported from literature (RecoFit/uLift-style) |
| Camera/IMU rep-count reconciliation | Real | `RepCountFusion`, occlusion-aware fallback |
| Fatigue analysis (set + session) | Real | Velocity-loss %, VL-zone, tempo drift, form trend |
| Multi-camera station topology + handoff | Real | Hysteresis-based, tested with a scripted walk between stations |
| Overlapping multi-camera zones | Real | Wristband-correlation-based cross-camera association; optional 3D triangulation |
| Motion-correlation identity disambiguation | Real | Grounded in published prior art (Sensors 2020) |
| BLE RSSI station-pairing heuristic | Real | v1 heuristic; UWB/AoA upgrade path documented, not built |
| Wristband checkout (front-desk step) | Real | |
| Live single-station orchestration | Real | `StationSessionRunner`, exercised by both unit tests and `run_live_gym_demo.py` |
| Live multi-station orchestration | Real | `GymSessionRunner` |
| Camera reconnection (24/7) | Real | Exponential backoff, tested against a failing fake capture |
| Camera health beyond reconnect (degraded-but-succeeding states) | Not started | See `docs/CAMERA_SYSTEM.md` |
| Intra-camera multi-person tracking (persistent track_id) | Real | Added Phase 2: `irix.pose.tracker.PoseTracker` (ByteTrack-derived two-stage IOU+Kalman association). Wired as `StationSessionRunner`'s default as of Phase 3 (`_ensure_estimator()` wraps any caller-omitted `pose_estimator` in `TrackedPoseEstimator`) |
| Exercise recognition (auto-classify from motion) | Real (zero-training baseline) | Added Phase 2: `irix.exercise_recognition` -- range-of-motion/periodicity scoring per candidate `ExerciseConfig`, honest `unknown` on structural ambiguity (squat/leg_press/hack_squat share a joint triplet) or no motion. **Still not wired to replace `StationInfo.default_exercise` at session start** -- remains the top `docs/TODO.md` item |
| Camera/IMU clock offset + drift estimation | Real | Added Phase 2: `irix.fusion.clock_sync`. Wired into a live entry point as of Phase 3: `StationSessionRunner` builds one `ClockSyncEstimator` per open session and applies its current correction on every `add_imu_samples` call. Automatic per-set observation derivation from camera/IMU event pairing was tried and deliberately reverted -- see `irix/fusion/clock_sync.py`'s `estimate_offset_from_paired_events` docstring |
| IMU packet-loss-aware fusion confidence | Real | Added Phase 2: `RepCountFusion` discounts `imu_confidence` by measured sample completeness before comparing against camera confidence |
| Load detection: color-coded bumper-plate cross-check | Real | Added Phase 2: `irix.weight_recognition.plate_color_check`, classical HSV blob detection against the confirmed 4-color IWF standard (10/15/20/25 kg). Only covers color-coded bumper plates, not black/gray iron plates -- by design, not a gap |
| Camera intrinsic/extrinsic calibration workflow | Real | Added Phase 2: `irix.pose.calibration` -- checkerboard `cv2.calibrateCamera`/`solvePnP`, reprojection-error quality reporting, `CalibrationProfile` save/load. Ground-plane homography also added |
| Benchmark suite | Real | Added Phase 2: `irix.benchmark.run_benchmarks` -- pose tracker/exercise recognition/fusion/EKF/clock-sync timing, full live-pipeline throughput, camera-reconnect schedule, BLE-disconnect-recovery margin, CPU/memory. GPU + real pose-inference FPS report `None` in this sandboxed (no CUDA/ultralytics) environment rather than fabricating a number |
| Deterministic event replay | Real | Fixed Phase 2: `RepCompletedEvent`/`SetCompleteEvent`/`SetFatigueSummaryEvent`/`WeightConfirmedEvent`/`BandPlacementRequiredEvent` were constructing with wall-clock (`time.monotonic()`) timestamps regardless of the caller's injected clock -- a real non-determinism bug, now fixed by threading the actual event-relevant timestamp through every construction site. See `docs/ARCHITECTURE.md`'s Phase 2 section |
| BLE wristband + gateway simulator | Real | Added 2026-07-14, `irix.wristband_sim` |
| Wristband IMU calibration | Partial | Real math (`calibrate_stationary`). Wired into `irix.identity.placement.WristbandPlacementTracker` as of Phase 3 -- runs on every placement-change settle. **Still open**: a session that never changes placement still runs on raw, uncalibrated samples from the start |
| Weight recognition (VLM classifier) | Real | Gemini backend verified against real SDK; no bundled API key |
| Weight recognition (local/on-device VLM) | Stub | Deferred -- needs a real model choice benchmarked on target edge hardware |
| Weight recognition geometric cross-check | Real | Plate-count consistency check against barbell detector |
| Barbell/dumbbell bar-path tracking + RPE | Real | Self-calibrated (known-object pixel scale), per-camera-aware |
| Barbell/plate object detector | Stub | `FreeWeightDetector` untrained -- no barbell/plate class in any standard pretrained model; Roboflow dataset identified as a starting point, not yet used |
| Event pipeline (`CameraEvent` family) | Real | See `docs/API_SPEC.md` |
| Event API versioning | Real | Added Phase 3 (Priority 11): `EVENT_SCHEMA_VERSION` in `irix/pipeline/schema.py`, present in every `CameraEvent` subclass's `to_dict()` |
| Edge buffer -> aggregator -> cloud sync | Real (mock backend) | `HTTPCloudSync` unwired -- no real `irix-mvp-app` ingestion endpoint exists yet |
| `TrackingLost`/`TrackingRecovered`/`RestStarted`/`RestEnded` events | Real | Added Phase 3 (Priority 6): all four now in `irix/pipeline/schema.py`, emitted by `StationSessionRunner`. `ExerciseChanged`/`ExerciseDetected` also added (Priority 6) -- see `docs/API_SPEC.md` |
| Live BLE IMU stream (real hardware) | Stub | Deliberately deferred -- hardware/firmware protocol decision |
| Wristband firmware | Not started | Out of software repo's scope; hardware recommendation documented in `docs/WRISTBAND_SYSTEM.md` |
| Camera calibration (bar-path, self-calibrated) | Real | `irix.barbell.calibration`, known-object-based |
| Camera calibration (geometric, for 3D triangulation) | Partial | Consumes an externally-supplied projection matrix; no in-repo calibration tooling |
| Config system (per-gym layout, external file) | Real | Added Phase 3 (Priority 10): `irix.config.gym_config` -- YAML/JSON per-gym config, factory functions producing real `StationRegistry`/`RepSession`/`StationSessionRunner` kwargs. Deliberately excludes hardware bindings (frame sources, BLE readers) from config -- see `docs/DEPLOYMENT.md` |
| Containerization / edge-device deployment configs | Not started | See `docs/DEPLOYMENT.md` |
| Metrics/observability (structured logging, Prometheus-style export) | Not started | |
| Ground-truth accuracy validation (vs. mocap or labeled video) | Not started | See `docs/VALIDATION.md` |
| Latency/throughput benchmarking on real edge hardware | Not started | See `docs/VALIDATION.md` |

| Wristband placement (wrist vs. ankle) state machine | Real | Added Phase 3 (Priority 4): `irix.identity.placement.WristbandPlacementTracker` (STABLE/SETTLING/CALIBRATING); `RepSession.add_imu_samples` rejects IMU data whose placement doesn't match the exercise's required limb (never reuses wrist thresholds for ankle data or vice versa) |
| Identity association fusion (motion-correlation + BLE + continuity) | Real | Added Phase 3 (Priority 5): `irix.identity.motion_correlation.MotionCorrelationResolver` + `irix.live.disambiguation.CrowdedGroupDisambiguator` -- never facial recognition, resolves crowded-station ambiguity via wrist-motion/camera-pose correlation with a continuity prior for the previously-assigned candidate |
| Workout state machine (whole-visit lifecycle) | Real | Added Phase 3 (Priority 6): `irix.pipeline.workout_state.WorkoutStateMachine`, whitelist-based transition table, prevents duplicate/late-packet reopening. Scoped at `GymSessionRunner` (whole gym visit), not per-station |
| Load detection: unified pipeline (color/OCR/machine stacks/geometry) | Real | Added Phase 3 (Priority 7): `RepSession`'s weight-check block runs color-plate detection unconditionally and cross-checks against geometry and (when configured) a VLM read, reporting confidence/evidence/units/method/status on `WeightConfirmedEvent` -- never fabricates a weight. OCR and machine-stack reading remain stubs (see `docs/TODO.md`) |
| Data collection / session recording tooling | Real | Added Phase 3 (Priority 8): `irix.recording.session_recorder.SessionRecorder` + `load_recorded_session` -- deterministic replay, reuses `irix.fusion.imu_io`'s CSV/JSON format. `save_raw_frames=False` by default, consistent with the production pipeline's "never raw video" principle |
| IRIX Studio backend interface | Real | Added Phase 3 (Priority 11): `irix.backend.studio_api.StudioBackendAPI` -- assign/return wristband, start/end session, query battery/assignment/status, versioned events. Does not build Studio itself (out of scope). `query_battery` honestly reports `"unknown"` -- no battery signal exists on any simulated or real hardware path yet. No real network transport wraps it yet (in-process Python class today) -- see `docs/BACKEND_API.md` |
| Automated validation report generation | Real | Added Phase 3 (Priority 12): `irix.validation.report_generator` -- real subprocess `pytest` run + optional benchmark suite, dated Markdown/JSON report, no fabricated numbers |

See `docs/ROADMAP.md` for how these map onto the founding brief's
numbered final-goal checklist, and `docs/TODO.md` for prioritized next
actions.
