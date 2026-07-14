# Validation

## Current state

```bash
$ pytest -q
237 passed, 3 skipped in ~1.6s
```

~7,990 lines of `irix/` source, ~4,715 lines of `tests/` (one test file
per module, plus integration/smoke tests) as of this writing. The 3
skips are the tests requiring `google-genai`/a live network call that
this environment doesn't exercise by default -- not failures, and not
representative of untested code (`tests/test_gemini_vlm_backend.py`
itself runs and passes; only the true live-network path is skipped).

## What's genuinely validated (not mocked end to end)

- **Pose estimation**: `tests/test_pose_estimator_integration.py` runs
  the real, pretrained `yolov8n-pose.pt` checkpoint against a real image
  (correctly finds 2 people, >0.5 confidence on clearly-visible joints)
  and a real (synthetic-but-real-codec) video through the full `run_live`
  pipeline (pose -> joint angle -> `RepCounter` -> `FormScorer` ->
  structured events), no mocking anywhere in that chain.
- **Gemini VLM integration**: `tests/test_gemini_vlm_backend.py` mocks
  only `_load_client()` and asserts the real `google.genai.types.Part`/
  config shapes are constructed correctly against the actual SDK's
  current API -- not sketched from memory, checked against a real
  installed SDK version.
- **The live orchestration path** (`StationSessionRunner`/
  `GymSessionRunner`): previously only ever exercised by unit tests
  against hand-built fakes (`_ScriptedPoseEstimator`, `_FakeFrameSource`,
  `_ScriptedBLEReader` in `tests/test_station_runner.py`/
  `test_gym_runner.py`). `irix/demo/run_live_gym_demo.py` (added
  2026-07-14) now exercises the same classes end to end via
  `irix.wristband_sim`'s simulator, and `tests/test_run_live_gym_demo.py`
  asserts the run produces every claimed event type (rep completion, set
  completion, fatigue summary, one correctly-attributed station handoff)
  -- not just that it executes without raising.
- **BLE gateway simulator + calibration**:
  `tests/test_wristband_simulator.py`/`test_imu_calibration.py` verify
  packet-loss/disconnect behavior against exact tick counts (not just
  "roughly happens sometimes") and that `calibrate_stationary` recovers
  a known injected bias to a tight tolerance.

## What's not validated (real limitations, stated plainly)

- **No real camera hardware tested.** Everything above runs against a
  synthetic image/video or a mocked capture object. Real RTSP
  reconnection behavior, real network jitter, and real frame-rate
  degradation under load have not been measured against actual camera
  hardware.
- **No real wristband hardware tested** -- `irix.wristband_sim`
  simulates plausible BLE/IMU behavior (packet loss, disconnects, RSSI
  noise), but no real wristband firmware or radio has been built or
  tested against this pipeline. Real-world packet loss rates, RSSI
  behavior in a cluttered gym environment, and real clock-sync behavior
  are unknown until real hardware exists.
- **No latency/throughput benchmarks against real inference hardware.**
  No measurement yet of per-frame pose-inference latency on a target
  edge device (e.g. Jetson), or of how many concurrent stations one edge
  box can sustain.
- **No accuracy validation against ground truth** (e.g. motion-capture-
  verified rep counts/joint angles, the way Kemtai publishes a mocap-
  validated accuracy methodology -- see `docs/RESEARCH_LOG.md`). All
  current pose/rep-count validation is against synthetic,
  geometrically-self-consistent data (by construction correct) or a
  real pretrained model's output on unlabeled test images/video
  (correctness checked by inspection, not against a labeled ground
  truth set).
- **Barbell/plate detection has no trained model to validate** --
  `FreeWeightDetector` is an untrained stub (see
  `docs/WRISTBAND_SYSTEM.md`/`docs/IMPLEMENTATION_STATUS.md`).

## Recommended next validation steps

1. **Benchmark real pose-inference latency** on a representative edge
   device once one is chosen (see `docs/DEPLOYMENT.md`) -- frames/sec
   sustained, GPU vs. CPU fallback timing, memory footprint at N
   concurrent stations.
2. **Record real gym-floor video + build a small labeled ground-truth
   set** (rep counts, set boundaries, at minimum) to move rep-counting
   accuracy validation from "geometrically self-consistent synthetic
   data" to actual measured accuracy, following Kemtai's published-
   methodology precedent.
3. **Measure real BLE packet-loss/RSSI behavior** once wristband
   hardware exists, and compare against `irix.wristband_sim`'s
   configurable `packet_loss_pct`/RSSI-noise defaults -- tune the
   simulator's defaults to match reality rather than leaving them as
   initial placeholder values.
4. **Add a `--benchmark` mode to `run_upload.py`/`run_live_gym_demo.py`**
   reporting measured frame-processing latency and dropped-frame counts,
   directly satisfying the founding brief's "measure latency... measure
   dropped frames" requirement with a real number, not just an
   architecture that could in principle produce one.
