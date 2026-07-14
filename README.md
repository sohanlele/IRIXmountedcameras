# IRIX

Camera-based rep tracking: fixed gym cameras + a lightweight wristband,
replacing manual rep logging. This repo is the **software scaffold** for
the pure-software layers of the system described in
`IRIX_Camera_System_Technical_Design.docx` (mounted-camera + wristband
form factor) -- pose estimation, rep counting, sensor fusion, weight
recognition, identity linking, the edge-to-cloud data pipeline, and audio
coaching. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the
section-by-section mapping and what's deliberately left unimplemented.

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

No camera needed -- synthetic joint-angle stream through the full pipeline:

```bash
python -m irix.demo.run_demo --mock-pose --exercise squat
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

## Test

```bash
pytest
```

## Layout

```
irix/
  pose/              pose estimation (YOLO-Pose wrapper) + joint-angle geometry
  rep_counting/       joint-angle state machine + per-exercise configs
  fusion/             visual-inertial EKF + ZUPT dead-stop correction; RecoFit/uLift wristband IMU-only rep counters
  weight_recognition/ VLM-based plate/load classifier (pluggable local/cloud backend), N-of-M read confirmation, QR reader (reference only, not deployable -- see docs/ARCHITECTURE.md)
  identity/           BLE RSSI station-pairing heuristic
  pipeline/           edge buffer -> aggregator -> cloud sync, derived-metrics schema
  coaching/           rep/set coaching text + local TTS engine interface
  demo/               end-to-end CLI (mock or live)
tests/                 unit + smoke tests for every module above
docs/ARCHITECTURE.md   design-doc-to-repo section map
```

## License

MIT, see [LICENSE](LICENSE).
