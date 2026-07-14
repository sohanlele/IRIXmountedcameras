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
| 4.4 Weight & plate recognition | `irix/weight_recognition/` | VLM-based classifier (`vision_classifier.py`) is the deployable path -- see below for why QR stickers and OCR were both ruled out; `confirmation.py` adds N-of-M read-confirmation windowing |
| 4.5 Bar path & velocity tracking | -- | Not implemented; would reuse `irix/pose` object-detection output for the barbell class |
| 4.6 Visual-inertial sensor fusion | `irix/fusion/` | EKF (position/velocity state) + ZUPT dead-stop correction; `imu_rep_counting.py` adds two literature IMU-only rep counters (see below) |
| 5.1 BLE identity linking | `irix/identity/` | RSSI-based station-resolution heuristic (not a BLE radio stack) |
| 5.4 Personalization data flow | -- | Not implemented; would live alongside `irix/pipeline` as a profile-pull step |
| 6.3 Data flow (edge -> aggregator -> cloud) | `irix/pipeline/` | `LocalBuffer` -> `Aggregator` -> `CloudSync`, derived-metrics-only schema |
| 7 / 7.1 Real-time audio coaching | `irix/coaching/` | Coaching-line generation + TTS engine interface (`NullTTSEngine` for tests, `PiperTTSEngine` sketch) |
| 8 Privacy & data handling | `irix/pipeline/schema.py` | `DerivedMetricsEvent` intentionally carries no video/biometric fields |

## Wristband IMU-only rep counting (ported from a collaborator's prototype)

`irix/fusion/imu_rep_counting.py` ports two published rep-counting
algorithms -- `RecoFitCounter` (Morris et al., "RecoFit: Exercise Set
Detection and Rep Counting", CHI 2014) and `ULiftCounter` (Lim et al.,
"uLift", IEEE Access 2024) -- from a collaborator's (jeffreyjy/IrixDemo)
Swift implementation, built for a first-person smart-glasses IRIX
prototype (Mentra Live glasses + phone operator app, not the
mounted-camera design this repo scaffolds).

Two things are worth being explicit about:

- **That system's "sensor fusion" is IMU-only, not camera+IMU fusion.**
  Its rep counting runs entirely off the wristband/glasses IMU stream
  (25 Hz over BLE) with no camera involved at rep-counting time -- the
  camera in that system is only used earlier, for VLM-based setup
  guidance (see below). In IRIX's mounted-camera design, `RecoFitCounter`
  / `ULiftCounter` slot in as the Section 5.3 fallback signal (what a
  station reports when its camera is occluded or its edge box is down)
  and as an independent cross-check alongside the joint-angle + EKF
  counter (Section 4.6) -- see `irix/demo/run_demo.py --with-imu-crosscheck`
  for the latter wired end-to-end.
- **Plate/weight recognition in that system is a cloud VLM (Gemini),
  not an open-source model.** It reads the printed weight number off a
  dumbbell via a prompt-engineered `extraction_state` (N-of-M confirm
  window + a `validate_weight_lbs` range/grid validator) rather than any
  vision library.

## Ankle placement for machine leg exercises

The design doc calls out leg press / hack squat as the case where wrist-IMU
fusion (Section 4.6) contributes nothing: "the wrist doesn't move with the
load." On those machines the *foot* is the rigid contact point with the
load (the footplate) -- the same relationship the wrist has to a curl --
so `ExerciseConfig.band_placement` (`irix/rep_counting/exercises.py`) lets
`LEG_PRESS`/`HACK_SQUAT` specify `BandPlacement.ANKLE`, restoring a real
fusion signal for exactly those two exercises. It does not extend to
free-weight squats: feet stay planted there, so an ankle band sees almost
no motion -- the barbell is what's moving, tracked by the camera
(Section 4.5), not the ankle.

`irix/coaching/triggers.py`'s `BandPlacementCoach` is the stateful piece
that turns this into an actual spoken instruction: it tracks where the
band currently is across a session and only prompts a reposition when the
next exercise's `band_placement` differs from the current one, so a
session that never touches a leg machine stays silent about it.

Both ports are unit-tested against synthetic IMU/reading streams in
`tests/test_imu_rep_counting.py` and `tests/test_confirmation.py`.

## Weight recognition: why it ended up VLM-based too

The original plan in Section 4.4 was QR/barcode stickers on each plate
(v1) with a pure-vision classifier as a v2 stretch goal. Two constraints
ruled both out for a real deployment:

- **No gym environment edits except the cameras.** A sticker on every
  plate is an environment edit -- it's not viable regardless of how cheap
  or accurate it is.
- **Printed plate numbers aren't legible at the Section 3.1 camera
  geometry** (3-4m back, 30-45 deg off-axis, chest height). That's a
  framing/resolution problem, not a model-quality problem -- no amount of
  OCR fixes a number that isn't resolvable in the frame.

That leaves classifying plates by appearance (color, relative size)
instead of reading a label. The classical-CV version of that -- a
per-gym calibration profile mapping each gym's specific plate colors/
diameters to weights, built once during install -- is a real option, but
it re-introduces a manual setup step per plate type per gym and doesn't
generalize to equipment the calibration never saw.

`irix/weight_recognition/vision_classifier.py` instead follows
jeffreyjy/IrixDemo's approach directly: ask a vision-language model to
read the scene. A VLM generalizes to whatever plates a given gym actually
has without a calibration step, for the same reason it generalized to
reading an arbitrary printed number in their first-person case -- it's
reasoning about the image, not matching against a pre-registered lookup
table. `ExtractionConfirmer` (ported from their `confirm_n`/`confirm_window`
+ `consistent_field` pattern) keeps a single noisy read from being
trusted outright, same role it plays for the IMU counters above.

Where this repo diverges from theirs: `vlm_backend.py` makes the model
backend pluggable rather than hardcoding a cloud call. `GeminiVLMBackend`
mirrors their actual approach (useful for parity/demo purposes), but
`LocalVLMBackend` -- an on-device open-source VLM served on the zone edge
box -- is the recommended default, because a cloud VLM call means camera
frames leave the building on every read, which conflicts with Section 8's
data-minimization stance (raw video never leaves the building) and adds a
per-call cost + network dependency at every station. `LocalVLMBackend` is
an interface sketch (`NotImplementedError`), not a finished integration --
which local model/serving stack to run on the Jetson boxes from Section 6
is a real decision that needs actual hardware to validate, not something
to guess at in a software scaffold.

`tests/test_vision_classifier.py` exercises the confirmation-windowing
logic against a scripted `FakeVLMBackend` -- no real model call needed to
validate that logic.

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
