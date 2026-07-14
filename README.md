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
hardware or wristband firmware is included -- those are hardware/
deployment concerns outside a software repo's scope. Model weights are a
mixed bag, worth being precise about: pose estimation
(`irix.pose.estimator.PoseEstimator`) uses a real, freely available,
auto-downloading pretrained checkpoint (`yolov8n-pose.pt`, COCO
keypoints) and is verified end-to-end against a real image and video in
`tests/test_pose_estimator_integration.py` -- generic human pose
estimation is a solved, commodity problem, not something this project
needs to train. Weight recognition uses `GeminiVLMBackend` (cloud), a
real integration verified against the current `google-genai` SDK -- no
API key is bundled here, the deployer supplies their own. Barbell/plate
detection (`irix.barbell.detector.FreeWeightDetector`) and the local VLM
backend (`irix.weight_recognition.vlm_backend.LocalVLMBackend`) are still
untrained/unimplemented stubs, left deferred by choice -- see their
module docstrings and `docs/ARCHITECTURE.md`'s "Model weights" section
for what a real fix looks like for each.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, opencv, ultralytics, pyzbar, google-genai, pytest
# or, for a lighter install (pick only the extras you need):
pip install -e .                     # core only
pip install -e ".[pose,qr,vlm,dev]"  # + real pose inference, QR reading, GeminiVLMBackend, tests
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

### Upload mode: a recorded video + wristband file in, the full event stream out

`run_demo.py --source` and `run_gym_demo.py` are the only other places a
real (or fusion-real) result comes out of this repo, but neither takes an
already-*recorded* video and wristband file and runs the full pipeline
against them -- `--source` only ever wires pose -> rep -> form (no IMU
fusion, weight recognition, barbell velocity, or fatigue), and
`run_gym_demo.py` is synthetic-data-only. `run_upload.py` is that missing
entrypoint: give it a video file (and, optionally, a wristband IMU
export), and it runs every real module this repo has -- pose, rep
counting, form scoring, IMU fusion, weight recognition, barbell velocity/
RPE, fatigue analysis, and rest-gap set-boundary detection (nothing here
hand-scripts where one set ends and the next begins, unlike the mock
demos) -- and returns/writes the full `CameraEvent` JSON stream, the
payload `irix-mvp-app`'s AI needs.

```bash
# video only (pose -> rep -> form -> rest-gap-detected sets -> fatigue)
python -m irix.demo.run_upload --video squat.mp4 --exercise squat

# + a real wristband IMU recording (see irix/fusion/imu_io.py for the
# exact CSV/JSON format) -- reconciled against the camera count per set
python -m irix.demo.run_upload --video squat.mp4 --exercise squat --imu wristband.csv

# + weight recognition (needs a real Gemini API key you supply -- none is
# bundled here)
python -m irix.demo.run_upload --video squat.mp4 --exercise squat --gemini-api-key "$GEMINI_API_KEY"

# write the JSON event stream to a file instead of stdout
python -m irix.demo.run_upload --video squat.mp4 --exercise squat --out events.json
```

Barbell velocity/RPE (`--barbell-model path/to/checkpoint.pt`) needs a
real trained barbell/plate detector checkpoint, which isn't bundled with
this repo (see `docs/ARCHITECTURE.md`'s "Model weights" section) -- left
off by default, in which case reps fall back to the deg/s joint-angle
velocity proxy instead of calibrated m/s + RPE.

### Live mode: structured for a 24/7 station, a front-desk checkout, and a live wristband

The real deployment target isn't a video file -- it's a camera that's
always on, and a wristband that gets checked out at the front desk and
tied to an account before anyone's IMU data means anything. Neither
`run_upload.py` (one file, one member, start to finish) nor `run_live`
(exits when its source runs out, no concept of *whose* session it's
watching) is the right shape for that. Four pieces close that gap in the
software (the actual BLE radio stack and wristband firmware are hardware/
firmware scope, same boundary `irix.identity.ble_pairing`'s docstring
already draws -- these are the software-side interfaces that plug into
real hardware once it exists):

- `irix.identity.checkout.CheckoutRegistry` -- the front-desk step: check
  a wristband out to an account, check it back in, resolve a wristband id
  to a member id. Real and complete.
- `irix.fusion.imu_stream.IMUStream` -- a `poll()`-based protocol so live
  and recorded IMU data can be consumed identically. `RecordedIMUStream`
  (real) wraps an already-loaded file; `LiveBLEIMUStream` is a documented
  stub -- same reasoning as `LocalVLMBackend` staying unimplemented,
  which real BLE client library/wristband protocol to use is a hardware
  decision this scaffold can't make correctly by guessing.
- `irix.live.camera_source.ReconnectingFrameSource` -- wraps
  `cv2.VideoCapture` (webcam index, file, or a live stream URL -- the
  same `source` `run_live` already accepts) and reconnects with
  exponential backoff on any read failure instead of exiting. Real,
  tested against a fake capture that fails on cue.
- `irix.live.station_runner.StationSessionRunner` -- the orchestrator:
  resolves BLE presence to a checked-out account via `CheckoutRegistry`,
  starts a fresh `irix.pipeline.rep_session.RepSession` (the same
  per-member logic `run_upload` uses, factored out so both share it) when
  a checked-out member shows up, feeds it frames and live IMU samples for
  as long as they're present, and closes it out (flushing whatever set
  was in progress) once presence lapses past `presence_timeout_s`.

```python
from irix.identity.checkout import CheckoutRegistry
from irix.live.camera_source import ReconnectingFrameSource
from irix.live.station_runner import StationSessionRunner

registry = CheckoutRegistry()
registry.check_out("band-042", member_id="acct_123", timestamp=...)  # front desk

runner = StationSessionRunner(
    station_id="squat-1",
    exercise_name="squat",
    checkout_registry=registry,
    frame_source=ReconnectingFrameSource("rtsp://squat-1-camera/stream"),
    ble_reader=my_ble_reader,       # returns this station's current BLEReadings -- real hardware integration
    imu_stream_factory=my_imu_factory,  # wristband_id -> IMUStream -- real hardware integration
    on_events=push_to_aggregator,   # e.g. irix.pipeline.aggregator.Aggregator
)
runner.run_forever()  # runs indefinitely, one station, all day
```

`ble_reader` and `imu_stream_factory` are the two seams a real deployment
plugs actual hardware into -- everything downstream of them (presence
resolution, session lifecycle, rep counting, fusion, fatigue) is real,
tested code today.

**Multiple stations at once**: a lone `StationSessionRunner` only knows
about its own station -- two adjacent ones would each independently start
a session for the same band mid-walk between them, double-counting reps.
`irix.live.gym_runner.GymSessionRunner` wires in the same station-handoff
hysteresis `run_gym_demo.py` already demonstrates with synthetic data
(`irix.topology.handoff.GymCoordinator`), for real: it resolves presence
*gym-wide* from one raw BLE reading source, decides which single station
is authoritative for each checked-out member, and only tells that
station's `RepSession` about them.

```python
from irix.live.gym_runner import GymSessionRunner
from irix.topology.registry import StationInfo, StationRegistry

registry = StationRegistry([
    StationInfo(station_id="squat-1", camera_id="cam-1", zone="free_weights", adjacent_station_ids=["squat-2"]),
    StationInfo(station_id="squat-2", camera_id="cam-2", zone="free_weights", adjacent_station_ids=["squat-1"]),
])
station_runners = {
    "squat-1": StationSessionRunner(station_id="squat-1", ..., ble_reader=lambda: []),  # not used -- gym runner drives tick() directly
    "squat-2": StationSessionRunner(station_id="squat-2", ..., ble_reader=lambda: []),
}
gym = GymSessionRunner(
    registry=registry,
    checkout_registry=registry_of_checkouts,
    station_runners=station_runners,
    ble_reader=my_gym_wide_ble_reader,  # every station's readings, every band, in one call -- real hardware integration
    on_gym_events=push_station_handoff_events,
)
gym.run_forever()
```

**Crowded stations**: a separate problem from station-to-station
handoff above -- two different checked-out members whose bands both
resolve to the *same* station at once, which RSSI proximity alone can't
tell apart. `StationSessionRunner` now handles this directly: when more
than one checked-out band is present at a station in the same tick, it
buffers a short window of poses/IMU and resolves who's who via
`irix.identity.motion_correlation.MotionCorrelationResolver` (wrist
motion vs. wristband IMU correlation), then routes each detected person
to the right member's session until the present-band group changes.
Frames during that short buffering window aren't attributed to anyone
(missed rather than guessed at) -- see `docs/ARCHITECTURE.md`'s "Crowded
stations" section for the full trade-offs.

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
  fusion/             visual-inertial EKF + ZUPT dead-stop correction; RecoFit/uLift wristband IMU-only rep counters; rep_fusion.py reconciles camera + IMU set-level rep counts into one authoritative count; imu_io.py loads a real recorded wristband export (CSV/JSON) into IMUSamples; imu_stream.py is the live-vs-recorded IMUStream protocol (RecordedIMUStream real, LiveBLEIMUStream a documented hardware-scope stub)
  fatigue/             set + session-level fatigue analysis (velocity loss %, VL-zone classification, tempo drift, form trend) aggregated for irix-mvp-app's AI context
  topology/            multi-camera station registry (10-camera example layout) + BLE-hysteresis member handoff, gating which station's events are authoritative to prevent double-counting
  identity/            BLE RSSI station-pairing heuristic + motion-correlation disambiguation (camera wrist motion vs. wristband IMU) for when two members' bands are both in range of one station; checkout.py is the front-desk wristband-to-account link (CheckoutRegistry)
  barbell/             self-calibrated (no environment edits) barbell/plate/dumbbell tracking, m/s bar velocity, RPE/velocity-loss estimation
  weight_recognition/ VLM-based plate/load classifier (pluggable local/cloud backend), N-of-M read confirmation, geometric plate-count cross-check, QR reader (reference only, not deployable -- see docs/ARCHITECTURE.md)
  pipeline/           edge buffer -> aggregator -> cloud sync; structured CameraEvent family (the API contract with irix-mvp-app); rep_session.py is the per-member pipeline (rep/form/weight/barbell/fatigue) shared by run_upload and the live station runner
  live/               24/7-station pieces: camera_source.py (ReconnectingFrameSource, reconnects on drop instead of exiting), station_runner.py (StationSessionRunner -- ties checkout + BLE presence + live camera + live IMU + RepSession into one continuously-running station), gym_runner.py (GymSessionRunner -- runs several stations together with GymCoordinator-backed handoff so a member walking between them is never double-counted)
  demo/               single-station (run_demo.py), multi-station (run_gym_demo.py), and upload-mode (run_upload.py -- real video + real wristband file in, full event stream out) end-to-end CLIs
tests/                 unit + smoke tests for every module above
docs/ARCHITECTURE.md   design-doc-to-repo section map, including every place this repo diverges from the original design doc and why
```

## License

MIT, see [LICENSE](LICENSE).
