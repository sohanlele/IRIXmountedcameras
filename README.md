# IRIX

Camera-based rep tracking: fixed gym cameras + a lightweight wristband,
replacing manual rep logging. This repo is the **software scaffold** for
the camera/edge-side pure-software layers of the system described in
`IRIX_Camera_System_Technical_Design.docx` (mounted-camera + wristband
form factor) -- pose estimation, rep counting, sensor fusion, weight
recognition, and identity linking, all producing structured events over
an edge-to-cloud pipeline. It computes *what happened* at a station; it
does not generate instructions, coaching copy, or any UI -- that's
[jeffreyjy/irix-mvp-app](https://github.com/jeffreyjy/irix-mvp-app)'s job
(FastAPI backend + iOS app). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for the section-by-section mapping, the repo boundary, and what's
deliberately left unimplemented.

**Status:** early scaffold, not production code. No camera/network
hardware, wristband firmware, or trained model weights are included --
those are hardware/deployment concerns outside a software repo's scope.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, opencv, ultralytics, pyzbar, pytest
# or, for a lighter install without live pose inference / QR reading:
pip install -e .
```

## Run the demo

No camera needed -- synthetic joint-angle stream through the full pipeline,
printing the structured events that would be sent to irix-mvp-app:

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat
python -m irix.demo.run_demo --mock-pose --exercise leg_press  # also emits a band-placement event
```

With a real webcam or video file (requires `pip install irix[pose]`):

```bash
python -m irix.demo.run_demo --source 0 --exercise bicep_curl
```

With the wristband IMU cross-check (camera-based count vs. two independent
IMU-only counters on a synthetic wristband signal):

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat --with-imu-crosscheck
```

With barbell tracking (calibrated m/s bar velocity, velocity-loss %, and
estimated RPE -- squat/bench_press/deadlift have published velocity anchors):

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat --with-barbell-tracking
```

With rule-based form scoring (squat/bicep_curl -- see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full fault list and
the prior-art it's grounded in), optionally injecting a specific fault so
the demo shows one actually getting caught:

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat --with-form-scoring
python -m irix.demo.run_demo --mock-pose --exercise squat --with-form-scoring --inject-form-fault knee_valgus
python -m irix.demo.run_demo --mock-pose --exercise bicep_curl --with-form-scoring --inject-form-fault leaning_back
```

Live mode (`--source`) scores form automatically, no flag needed -- it
already gets a full pose from `PoseEstimator` every frame.

## Test

```bash
pytest
```

## Layout

```
irix/
  pose/              pose estimation (YOLO-Pose wrapper) + joint-angle geometry
  rep_counting/       joint-angle state machine + per-exercise configs; each rep carries duration + peak/mean velocity for fatigue tracking, and optionally the buffered poses for form scoring
  form/               rule-based per-rep fault detection (knee valgus, insufficient depth, leaning back, elbow drift, hips-rising-before-chest), populates RepCompletedEvent.form_score/form_faults
  fusion/             visual-inertial EKF + ZUPT dead-stop correction; RecoFit/uLift wristband IMU-only rep counters
  barbell/             self-calibrated (no environment edits) barbell/plate/dumbbell tracking, m/s bar velocity, RPE/velocity-loss estimation
  weight_recognition/ VLM-based plate/load classifier (pluggable local/cloud backend), N-of-M read confirmation, QR reader (reference only, not deployable -- see docs/ARCHITECTURE.md)
  identity/           BLE RSSI station-pairing heuristic
  pipeline/           edge buffer -> aggregator -> cloud sync; structured CameraEvent family (the API contract with irix-mvp-app)
  demo/               end-to-end CLI (mock or live)
tests/                 unit + smoke tests for every module above
docs/ARCHITECTURE.md   design-doc-to-repo section map
```

## License

MIT, see [LICENSE](LICENSE).
