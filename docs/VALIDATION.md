# Validation

## Current state

```bash
$ pytest -q
402 passed, 3 skipped in ~8s
```

(Was 237 as of Phase 2; grew to 389 across Phase 3's Priorities 1-11,
then to 402 with Priority 12's own validation-expansion work below.)
The 3 skips are the tests requiring `google-genai`/a live network call
that this environment doesn't exercise by default -- not failures, and
not representative of untested code (`tests/test_gemini_vlm_backend.py`
itself runs and passes; only the true live-network path is skipped).

This number stops needing manual upkeep as of Phase 3, Priority 12:
`python -m irix.validation.report_generator` regenerates a dated
Markdown report with the real current pass/fail/skip counts (and,
unless `--skip-benchmarks` is passed, the full `irix.benchmark.
run_benchmarks` performance suite) from a real subprocess `pytest`
run every time it's invoked -- see `docs/BACKEND_API.md`-style
treatment isn't needed here since the tool's own docstring
(`irix/validation/report_generator.py`) covers usage. It deliberately
does not try to auto-generate this file's qualitative sections below
("what's genuinely validated," "what's not") -- those require human
judgment about what a passing test actually proves.

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

- **The config system, end to end with the live orchestration layer**:
  previously, `tests/test_gym_config.py` only checked that
  `irix.config.gym_config`'s factory functions produced kwargs that
  *could* construct a working `RepSession` in isolation -- the config
  system and `StationSessionRunner`/`GymSessionRunner` had never
  actually been run together. `tests/
  test_config_driven_live_pipeline.py` (Priority 12, 2026-07-14) loads
  the real bundled `configs/example_gym.yaml`, builds real runners from
  it, and runs a scripted session end to end, asserting real events
  come out carrying the config's own exercise name and equipment
  settings. This caught a real bug in the process: `station_runner_kwargs_for`
  had included `camera_tilt_deg` since Priority 10, but
  `StationSessionRunner.__init__` had no matching parameter to receive
  it -- any configured camera tilt correction was silently dropped on
  the floor and never reached `RepSession`'s bar-velocity calculation.
  Fixed the same day it was found; see `irix/live/station_runner.py`'s
  `camera_tilt_deg` parameter and its docstring for the account.

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
