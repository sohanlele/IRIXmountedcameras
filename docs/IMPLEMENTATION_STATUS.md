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
| Intra-camera multi-person tracking (persistent track_id) | Real | Added Phase 2: `irix.pose.tracker.PoseTracker` (ByteTrack-derived two-stage IOU+Kalman association). Not yet wired as StationSessionRunner's default -- opt-in via `TrackedPoseEstimator` |
| Exercise recognition (auto-classify from motion) | Real (zero-training baseline) | Added Phase 2: `irix.exercise_recognition` -- range-of-motion/periodicity scoring per candidate `ExerciseConfig`, honest `unknown` on structural ambiguity (squat/leg_press/hack_squat share a joint triplet) or no motion. Not yet wired to replace `StationInfo.default_exercise` at session start |
| Camera/IMU clock offset + drift estimation | Real | Added Phase 2: `irix.fusion.clock_sync` (cross-correlation offset + weighted-linear-fit drift), validated against `irix.wristband_sim`'s new `clock_drift_ppm` simulation. Not yet wired into any live entry point's IMU consumption |
| IMU packet-loss-aware fusion confidence | Real | Added Phase 2: `RepCountFusion` discounts `imu_confidence` by measured sample completeness before comparing against camera confidence |
| Load detection: color-coded bumper-plate cross-check | Real | Added Phase 2: `irix.weight_recognition.plate_color_check`, classical HSV blob detection against the confirmed 4-color IWF standard (10/15/20/25 kg). Only covers color-coded bumper plates, not black/gray iron plates -- by design, not a gap |
| Camera intrinsic/extrinsic calibration workflow | Real | Added Phase 2: `irix.pose.calibration` -- checkerboard `cv2.calibrateCamera`/`solvePnP`, reprojection-error quality reporting, `CalibrationProfile` save/load. Ground-plane homography also added |
| Benchmark suite | Real | Added Phase 2: `irix.benchmark.run_benchmarks` -- pose tracker/exercise recognition/fusion/EKF/clock-sync timing, full live-pipeline throughput, camera-reconnect schedule, BLE-disconnect-recovery margin, CPU/memory. GPU + real pose-inference FPS report `None` in this sandboxed (no CUDA/ultralytics) environment rather than fabricating a number |
| Deterministic event replay | Real | Fixed Phase 2: `RepCompletedEvent`/`SetCompleteEvent`/`SetFatigueSummaryEvent`/`WeightConfirmedEvent`/`BandPlacementRequiredEvent` were constructing with wall-clock (`time.monotonic()`) timestamps regardless of the caller's injected clock -- a real non-determinism bug, now fixed by threading the actual event-relevant timestamp through every construction site. See `docs/ARCHITECTURE.md`'s Phase 2 section |
| BLE wristband + gateway simulator | Real | Added 2026-07-14, `irix.wristband_sim` |
| Wristband IMU calibration | Partial | Real math (`calibrate_stationary`), not yet wired into any live/upload entry point |
| Weight recognition (VLM classifier) | Real | Gemini backend verified against real SDK; no bundled API key |
| Weight recognition (local/on-device VLM) | Stub | Deferred -- needs a real model choice benchmarked on target edge hardware |
| Weight recognition geometric cross-check | Real | Plate-count consistency check against barbell detector |
| Barbell/dumbbell bar-path tracking + RPE | Real | Self-calibrated (known-object pixel scale), per-camera-aware |
| Barbell/plate object detector | Stub | `FreeWeightDetector` untrained -- no barbell/plate class in any standard pretrained model; Roboflow dataset identified as a starting point, not yet used |
| Event pipeline (`CameraEvent` family) | Real | See `docs/API_SPEC.md` |
| Event API versioning | Not started | No `schema_version` field yet -- see `docs/API_SPEC.md` |
| Edge buffer -> aggregator -> cloud sync | Real (mock backend) | `HTTPCloudSync` unwired -- no real `irix-mvp-app` ingestion endpoint exists yet |
| `TrackingLost`/`TrackingRecovered`/`ExerciseChanged`/`RestStarted`/`RestEnded` events | Not started | Named in founding brief, not in `schema.py` -- see `docs/API_SPEC.md` |
| Live BLE IMU stream (real hardware) | Stub | Deliberately deferred -- hardware/firmware protocol decision |
| Wristband firmware | Not started | Out of software repo's scope; hardware recommendation documented in `docs/WRISTBAND_SYSTEM.md` |
| Camera calibration (bar-path, self-calibrated) | Real | `irix.barbell.calibration`, known-object-based |
| Camera calibration (geometric, for 3D triangulation) | Partial | Consumes an externally-supplied projection matrix; no in-repo calibration tooling |
| Config system (per-gym layout, external file) | Not started | Currently constructed in Python; see `docs/DEPLOYMENT.md` |
| Containerization / edge-device deployment configs | Not started | See `docs/DEPLOYMENT.md` |
| Metrics/observability (structured logging, Prometheus-style export) | Not started | |
| Ground-truth accuracy validation (vs. mocap or labeled video) | Not started | See `docs/VALIDATION.md` |
| Latency/throughput benchmarking on real edge hardware | Not started | See `docs/VALIDATION.md` |

See `docs/ROADMAP.md` for how these map onto the founding brief's
numbered final-goal checklist, and `docs/TODO.md` for prioritized next
actions.
