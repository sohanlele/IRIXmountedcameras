# Architecture map

This repo implements the pure-software layers of the IRIX Technical
Architecture & Design Document (mounted-camera + wristband form factor).
It is a **software scaffold**: runnable module structure and unit-tested
logic for the algorithms the doc specifies, not a production build. It does
not include camera/network hardware, wristband firmware, Jetson deployment
configs, or trained model weights.

| Design doc section | Repo module | Status |
|---|---|---|
| 4.1 Pose estimation model | `irix/pose/` | YOLO-Pose wrapper (`ultralytics`, optional dep); joint-angle geometry helper |
| 4.2 Rep-counting logic | `irix/rep_counting/` | Joint-angle state machine + per-exercise configs (squat/curl/deadlift) |
| 4.3 Multi-camera fusion & occlusion | -- | Not implemented; `PoseEstimator` returns single-view poses per camera, multi-view reprojection is future work |
| 4.4 Weight & plate recognition | `irix/weight_recognition/` | v1 QR/barcode reader implemented; v2 vision classifier stubbed (`NotImplementedError` by design) |
| 4.5 Bar path & velocity tracking | -- | Not implemented; would reuse `irix/pose` object-detection output for the barbell class |
| 4.6 Visual-inertial sensor fusion | `irix/fusion/` | EKF (position/velocity state) + ZUPT dead-stop correction |
| 5.1 BLE identity linking | `irix/identity/` | RSSI-based station-resolution heuristic (not a BLE radio stack) |
| 5.4 Personalization data flow | -- | Not implemented; would live alongside `irix/pipeline` as a profile-pull step |
| 6.3 Data flow (edge -> aggregator -> cloud) | `irix/pipeline/` | `LocalBuffer` -> `Aggregator` -> `CloudSync`, derived-metrics-only schema |
| 7 / 7.1 Real-time audio coaching | `irix/coaching/` | Coaching-line generation + TTS engine interface (`NullTTSEngine` for tests, `PiperTTSEngine` sketch) |
| 8 Privacy & data handling | `irix/pipeline/schema.py` | `DerivedMetricsEvent` intentionally carries no video/biometric fields |

Out of scope (hardware, not software): camera selection/mounting (Section
3), edge compute sizing and PoE network design (Section 6.1-6.2), cost
modeling (Section 10), rollout plan (Section 11).

## End-to-end demo

`irix/demo/run_demo.py` wires pose -> rep counting -> coaching -> pipeline
into one loop, in two modes:

- `--mock-pose`: synthetic joint-angle stream, no camera or model weights
  needed. This is what the test suite and a from-scratch clone can run
  immediately.
- `--source <index|path>`: real webcam or video file through
  `PoseEstimator` (requires `pip install irix[pose]`, which pulls in
  `ultralytics`/torch).
