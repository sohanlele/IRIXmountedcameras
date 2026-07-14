# IRIX

Camera-based rep tracking for a real multi-camera gym deployment: fixed
gym cameras + a lightweight wristband IMU per member, replacing manual
rep logging. This repo is the **software scaffold** for the camera/edge-
side pure-software layers of the system originally described in
`IRIX_Camera_System_Technical_Design.docx` (mounted-camera + wristband
form factor) -- pose estimation, rep counting, camera+IMU sensor fusion,
form scoring, weight recognition, fatigue analysis, multi-station
identity/handoff, and identity linking, all producing structured events
over an edge-to-cloud pipeline. It computes *what happened* on the gym
floor -- accurate rep counts (camera and wristband IMU reconciled, not
just compared), which station each member is actually at, set/session
fatigue trends -- and does not generate instructions, coaching copy, or
any UI -- that's
[jeffreyjy/irix-mvp-app](https://github.com/jeffreyjy/irix-mvp-app)'s job
(FastAPI backend + iOS app). Several subsystems below (sensor fusion,
multi-camera topology, fatigue analysis) deliberately diverge from the
original design doc where research turned up a better-supported approach
-- see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
section-by-section mapping, every divergence's reasoning, and what's
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

Two entrypoints: `run_demo.py` for one station in depth, `run_gym_demo.py`
for what only shows up with several stations and several members at once
(station handoff, camera+IMU fusion, fatigue trends). Neither needs a
camera -- both run on synthetic data.

### Multi-station demo (10-camera deployment scenario)

```bash
python -m irix.demo.run_gym_demo
```

Two members, three stations, one runnable trace showing: BLE-based station
handoff with hysteresis (and a spurious adjacent-camera detection
correctly *not* double-counted), two BLE-ambiguous members at one shared
station correctly told apart by correlating each one's camera-tracked
wrist motion against their own wristband's IMU signal, camera+wristband-
IMU rep-count fusion (both the normal-agreement path and a heavily-
occluded set where fusion correctly falls back to the IMU), calibrated
barbell-velocity tracking feeding set + session fatigue analysis across
two consecutive squat sets (real VL-zone progression, not just the
joint-angle proxy), a bicep-curl set with an injected form fault actually
getting caught, and a weight-recognition geometry cross-check (one
plausible VLM read, one flagged as implausible). See
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)'s "Multi-station
deployment" section for what each piece is doing and why.

### Single-station demo

Synthetic joint-angle stream through the full pipeline, printing the
structured events that would be sent to irix-mvp-app:

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
  rep_counting/       joint-angle state machine + per-exercise configs; each rep carries duration + peak/mean velocity for fatigue tracking, tracking_confidence for fusion, and optionally the buffered poses for form scoring
  form/               rule-based per-rep fault detection (knee valgus, insufficient depth, leaning back, elbow drift, hips-rising-before-chest), populates RepCompletedEvent.form_score/form_faults
  fusion/             visual-inertial EKF + ZUPT dead-stop correction; RecoFit/uLift wristband IMU-only rep counters; rep_fusion.py reconciles camera + IMU set-level rep counts into one authoritative count
  fatigue/             set + session-level fatigue analysis (velocity loss %, VL-zone classification, tempo drift, form trend) aggregated for irix-mvp-app's AI context
  topology/            multi-camera station registry (10-camera example layout) + BLE-hysteresis member handoff, gating which station's events are authoritative to prevent double-counting
  identity/            BLE RSSI station-pairing heuristic + motion-correlation disambiguation (camera wrist motion vs. wristband IMU) for when two members' bands are both in range of one station
  barbell/             self-calibrated (no environment edits) barbell/plate/dumbbell tracking, m/s bar velocity, RPE/velocity-loss estimation
  weight_recognition/ VLM-based plate/load classifier (pluggable local/cloud backend), N-of-M read confirmation, geometric plate-count cross-check, QR reader (reference only, not deployable -- see docs/ARCHITECTURE.md)
  pipeline/           edge buffer -> aggregator -> cloud sync; structured CameraEvent family (the API contract with irix-mvp-app)
  demo/               single-station (run_demo.py) and multi-station (run_gym_demo.py) end-to-end CLIs
tests/                 unit + smoke tests for every module above
docs/ARCHITECTURE.md   design-doc-to-repo section map, including every place this repo diverges from the original design doc and why
```

## License

MIT, see [LICENSE](LICENSE).
