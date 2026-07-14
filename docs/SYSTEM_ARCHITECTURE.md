# System architecture (overview)

This is the short index. `docs/ARCHITECTURE.md` is the deep,
section-by-section design document (~1400 lines) mapping every module to
the original technical design doc, every divergence and why, and every
model-weights/hardware boundary -- read that for the actual reasoning.
This file is the map of the map: what talks to what, and where to go for
detail on each piece.

## Data flow

```
 mounted camera(s)  ---frames-->  PoseEstimator ---+
                                                    |
 wristband IMU (BLE)  ---samples-->  IMUStream -----+--> RepSession (per member, per station)
                                                    |         |
 BLE presence  ---readings-->  ble_pairing/topology-+         v
                                                     CameraEvent stream (schema.py)
                                                              |
                                                              v
                                         LocalBuffer -> Aggregator -> CloudSync
                                          (per zone)   (building-level)   (irix-mvp-app)
```

One edge box per zone runs inference locally (no cloud dependency for
inference, except the optional VLM weight-recognition call). Only
derived, structured events cross the edge/cloud boundary -- never raw
video, never a biometric identifier (see `docs/PRODUCT_SPEC.md`'s
non-goals and `irix/pipeline/schema.py`'s module docstring).

## Subsystem index

| Subsystem | Doc | Code |
|---|---|---|
| Camera ingestion, health, reconnection, multi-camera zones | `docs/CAMERA_SYSTEM.md` | `irix/pose`, `irix/live/camera_source.py`, `irix/live/zone_runner.py` |
| Wristband IMU, BLE gateway, simulator, calibration | `docs/WRISTBAND_SYSTEM.md` | `irix/fusion/imu*.py`, `irix/wristband_sim/`, `irix/identity/` |
| Pose estimation, multi-person tracking, identity, handoff | `docs/TRACKING.md` | `irix/pose`, `irix/topology`, `irix/identity` |
| Camera + IMU sensor fusion | `docs/SENSOR_FUSION.md` | `irix/fusion` |
| Exercise detection, rep/set/rest, form scoring | `docs/ARCHITECTURE.md` ("Rep velocity..." / "Form scoring" sections) | `irix/rep_counting`, `irix/form` |
| Load/weight detection (VLM + barbell tracking) | `docs/ARCHITECTURE.md` ("Model weights" / "Barbell and dumbbell tracking" sections) | `irix/weight_recognition`, `irix/barbell` |
| Event pipeline / API contract | `docs/API_SPEC.md` | `irix/pipeline` |
| Live 24/7 orchestration (single + multi-station) | `docs/ARCHITECTURE.md` ("Live station readiness" section) | `irix/live` |
| Deployment | `docs/DEPLOYMENT.md` | -- |
| Validation / benchmarks | `docs/VALIDATION.md` | `tests/` |

## Two orchestration paths, one shared per-member core

`irix.pipeline.rep_session.RepSession` is the per-member logic (rep
count, form, weight, barbell velocity, fatigue) shared by every entry
point:

- `irix/demo/run_upload.py` -- one `RepSession`, driven against an
  already-recorded video file (+ optional wristband export) from start
  to finish. The right shape for offline analysis of a finished workout.
- `irix.live.station_runner.StationSessionRunner` -- one *or more*
  `RepSession`s per station (more than one only when a station is
  crowded), created/closed as checked-out members' wristbands come and
  go, fed live frames and IMU. The right shape for an always-on station.
- `irix.live.gym_runner.GymSessionRunner` -- runs several
  `StationSessionRunner`s together with gym-wide, hysteresis-based
  presence resolution (`irix.topology.handoff.GymCoordinator`), so a
  member walking between stations is handed off rather than
  double-counted by both cameras.
- `irix.live.zone_runner.MultiCameraZoneRunner` -- the generalization for
  several cameras with genuinely overlapping fields of view over one
  shared area, rather than one camera per fixed station.

## Deliberately unimplemented seams (hardware-dependent, not guessed at)

Two seams stay documented stubs on purpose, because guessing the wrong
protocol/model choice would be worse than an honest "not yet":
`irix.fusion.imu_stream.LiveBLEIMUStream` (needs a real wristband's BLE
GATT protocol) and `irix.weight_recognition.vlm_backend.LocalVLMBackend`
(needs a real on-device model choice benchmarked against real edge
hardware). `irix.wristband_sim` exists precisely to let everything
*around* the first seam be exercised without it -- see
`docs/WRISTBAND_SYSTEM.md`.
