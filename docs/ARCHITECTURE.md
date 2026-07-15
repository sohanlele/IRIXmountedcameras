# Architecture map

This repo implements the pure-software layers of the IRIX Technical
Architecture & Design Document (mounted-camera + wristband form factor).
It is a **software scaffold**: runnable module structure and unit-tested
logic for the algorithms the doc specifies, not a production build. It
does not include camera/network hardware, wristband firmware, or Jetson
deployment configs. Trained model weights are a mixed bag rather than a
blanket "not included" -- see "Model weights: what's real vs. still a
stub" just below.

## Model weights: what's real vs. still a stub

Three modules in this repo wrap a model that needs weights to actually
run. They are not equally solved:

- **`irix.pose.estimator.PoseEstimator` -- real, working, done.**
  `model_path="yolov8n-pose.pt"` is a genuine Ultralytics-published
  checkpoint pretrained on COCO keypoints (the same 17-point layout
  `COCO_KEYPOINT_NAMES` already assumes), auto-downloaded on first use --
  no training, no API key, no cost, no gym-specific data collection
  needed. Generic human pose estimation from a single RGB camera is a
  solved, commodity problem; there was never a reason to train a custom
  model for it. Verified end-to-end in this pass:
  `tests/test_pose_estimator_integration.py` runs the real model against
  a real image (correctly finds 2 people, >0.5 confidence on clearly-
  visible joints) and a real (synthetic-but-real-codec) video through the
  full `run_live` pipeline -- pose -> joint angle -> `RepCounter` ->
  `FormScorer` -> structured events -- with no mocking anywhere in that
  chain. Also surfaced and fixed a real bug while verifying this:
  `run_live` used to call `cv2.imshow`/`cv2.waitKey` unconditionally,
  which crashes with "the function is not implemented" against
  `opencv-python-headless` (this repo's pinned dependency) on any
  machine without a GTK/Cocoa/Windows GUI toolkit -- which describes
  both a CI/sandbox environment *and* the real production target (a
  station's edge box has no monitor attached). Display is now opt-in
  (`--display`), off by default, and fails with a clear message instead
  of an uncaught OpenCV exception if requested somewhere that can't
  actually show a window.

- **`irix.barbell.detector.FreeWeightDetector` -- still a stub.**
  `model_path="freeweight_yolo.pt"` needs a checkpoint fine-tuned on
  barbell/plate/dumbbell classes, which don't exist in any standard
  pretrained object-detection model (COCO doesn't have a "barbell"
  class). The module docstring points at the Roboflow "Barbells
  Detector" dataset (92 labeled images, pretrained model available via
  their API) as a starting point, but actually fine-tuning or wiring up
  a hosted inference API is a real decision (account/API-key access,
  cost, accuracy expectations) this repo hasn't made yet. `irix/demo/
  run_upload.py` accepts a `--barbell-model` pointed at a real checkpoint
  and will wire it in (self-calibration, bar velocity, RPE, geometry
  cross-check) once one exists; without it, reps fall back to the deg/s
  joint-angle proxy.

- **`irix.weight_recognition.vlm_backend.GeminiVLMBackend` (cloud) --
  real, verified, chosen path.** Structured JSON output via the actual
  `google-genai` SDK, checked against the current API (not sketched from
  memory): inline frame bytes go through `types.Part.from_bytes(...)`,
  and `_LOAD_READ_SCHEMA`'s lowercase JSON Schema is passed under
  `response_json_schema` (not `response_schema`, which expects a
  Pydantic model or Gemini's own uppercase-typed schema dialect and would
  silently mis-parse a plain lowercase dict). `tests/
  test_gemini_vlm_backend.py` mocks `_load_client()` and asserts the real
  `google.genai.types.Part`/config shapes are constructed correctly and
  the response is parsed correctly -- no live network call, no API key
  used. **No API key is bundled or hardcoded anywhere in this repo**;
  `api_key` is a required constructor argument the deployer supplies.
  Chosen over `LocalVLMBackend` because a call only happens per weight
  *change* during the confirm window (a few times per set, not per
  frame/rep), so the volume of frames leaving the building is small
  relative to the cost of standing up on-device model serving -- see
  "Weight recognition" below for the full tradeoff.

- **`irix.weight_recognition.vlm_backend.LocalVLMBackend` -- still a
  stub, left deferred by choice.** Raises `NotImplementedError`.
  Implementing it for real means picking and integrating an actual local
  VLM (e.g. a small open-weights model run via `transformers` or
  `ollama`) -- a real decision about model choice, latency/hardware
  tradeoffs, and whether CPU inference on an edge box is even fast
  enough, not something to guess at without a target device to
  benchmark against. Revisit if the privacy/uptime tradeoff that
  currently favors cloud stops being acceptable for a given deployment.

## Where this repo ends and irix-mvp-app begins

This repo's job is entirely on the camera/edge side: compute what
happened at a station (a rep completed, a set ended, a weight was
confirmed, a band needs to move) and hand it off as a structured
`CameraEvent` (`irix/pipeline/schema.py`). It does not generate spoken
text, decide what a member should be told, or render any UI -- that's
[jeffreyjy/irix-mvp-app](https://github.com/jeffreyjy/irix-mvp-app) (a
FastAPI backend + iOS frontend), specifically its `backend/app/agents`
layer (AI-generated instructions) and the iOS app (UI). An earlier
version of this repo had a `coaching/` module that generated spoken
lines and had a TTS engine interface -- that's been removed in favor of
`irix/pipeline/events.py`'s `BandPlacementTracker` and the `CameraEvent`
family, which are the actual data contract between the two repos. As of
this writing, irix-mvp-app doesn't yet expose a live-camera-data
ingestion endpoint (its `api/v1` currently covers auth, workout plans,
and workout sessions) -- `HTTPCloudSync` (`irix/pipeline/cloud_sync.py`)
is a placeholder pointed at wherever that endpoint ends up.

| Design doc section | Repo module | Status |
|---|---|---|
| 4.1 Pose estimation model | `irix/pose/` | YOLO-Pose wrapper (`ultralytics`, optional dep); joint-angle geometry helper. Real, working pretrained weights -- see "Model weights" above |
| 4.2 Rep-counting logic | `irix/rep_counting/` | Joint-angle state machine + per-exercise configs (squat/curl/deadlift/leg_press/hack_squat); each completed rep also carries inter-rep duration + peak/mean angular velocity for fatigue tracking (see below) |
| 4.3 Multi-camera fusion & occlusion | `irix/pose/multiview.py` | DLT triangulation of a 3D pose from 2+ calibrated overlapping cameras' 2D keypoints, optionally wired into `MultiCameraZoneRunner` -- see "Overlapping multi-camera zones" below |
| 4.4 Weight & plate recognition | `irix/weight_recognition/` | VLM-based classifier (`vision_classifier.py`) is the deployable path -- see below for why QR stickers and OCR were both ruled out; `confirmation.py` adds N-of-M read-confirmation windowing |
| 4.5 Bar path & velocity tracking | `irix/barbell/` | Self-calibrated (no environment edits) barbell/plate/dumbbell detection, real-unit (m/s) bar-path velocity, and RPE/fatigue estimation -- see "Barbell and dumbbell tracking" below |
| 4.6 Visual-inertial sensor fusion | `irix/fusion/` | EKF (position/velocity state) + ZUPT dead-stop correction; `imu_rep_counting.py` adds two literature IMU-only rep counters (see below) |
| 5.1 BLE identity linking | `irix/identity/` | RSSI-based station-resolution heuristic (not a BLE radio stack) |
| 5.4 Personalization data flow | -- | Not implemented; would live alongside `irix/pipeline` as a profile-pull step |
| 6.3 Data flow (edge -> aggregator -> cloud) | `irix/pipeline/` | `LocalBuffer` -> `Aggregator` -> `CloudSync`, structured `CameraEvent` family (`RepCompletedEvent`, `SetCompleteEvent`, `BandPlacementRequiredEvent`, `WeightConfirmedEvent`) |
| 7 / 7.1 Real-time audio coaching | -- (owned by irix-mvp-app) | Out of scope for this repo -- see "Where this repo ends" above. `BandPlacementTracker` emits the one coaching-adjacent *event*, but not the instruction text itself |
| 8 Privacy & data handling | `irix/pipeline/schema.py` | Every `CameraEvent` subtype intentionally carries no video/biometric fields (tested) |

**Privacy positioning, stated explicitly (2026-07-14 competitive/legal
review):** under Illinois's BIPA and similar state biometric-privacy
laws, *recording video is not the regulated act -- building a faceprint
from it is*. A biometric identifier under BIPA is specifically a scan of
face/hand geometry, retina/iris, fingerprint, or voiceprint; this repo
never computes one -- `PoseEstimator` outputs a skeleton (keypoint
positions, not a face-geometry embedding), and every identity decision
anywhere in this codebase (`CheckoutRegistry`, `GymCoordinator`,
`MotionCorrelationResolver`) resolves *which wristband*, never *whose
face*. That's a real, load-bearing design property here, not an
afterthought bolted on for compliance -- it's the direct consequence of
building identity around a front-desk-issued wristband instead of face
recognition in the first place (see "Multi-station deployment" and "Live
station readiness" below for how). Worth stating plainly in any pitch or
compliance conversation: this design sits outside BIPA's scope by
construction. (GroeFit, a commercial-gym camera-analytics competitor,
markets the same "no facial recognition" property explicitly as a
privacy differentiator -- this repo has the same property for a
different reason: wristband-based identity was already the simpler,
more reliable design before privacy law was a consideration.)

**On real-time audio coaching specifically:** a 2026-07-14 survey of
open-source tooling turned up
[`GetStream/Vision-Agents`](https://github.com/GetStream/Vision-Agents)
(pip: `vision-agents`), an actively maintained, open-source framework
purpose-built for exactly the row above -- real-time pose tracking + an
LLM (Gemini Live, etc.) for live voice feedback over a low-latency WebRTC
edge network, with a runnable gym-coach tutorial. It's a real, credible
candidate for whoever builds irix-mvp-app's audio-coaching layer -- but
deliberately **not** added as a dependency of *this* repo, for two
reasons: (1) it would blur the boundary this section already draws
("this repo doesn't generate spoken text or decide what to tell a
member"), and (2) it's a heavy dependency graph shaped for a standalone
real-time video/voice *service* (`aiortc`, `getstream`, `onnxruntime`,
`fastapi`, `uvicorn`, MCP), not a lightweight library this repo's
pipeline should embed. One concrete gotcha worth flagging for whoever
does pick this up: `vision-agents` 0.6.6's core fails to import on Python
3.10 (`vision_agents/core/observability/collector.py` does `from typing
import Self`, which only exists in Python 3.11+) despite the package's
own classifiers claiming 3.10 support -- verified by actually installing
and importing it, not assumed. Needs Python 3.11+ in whatever service
ends up using it.

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

## Rep velocity and fatigue tracking (feeding irix-mvp-app's AI)

The plan is for irix-mvp-app's AI to run fatigue analysis on a member's
performance and shape the next set's target weight/reps accordingly (a
standard velocity-based-training / autoregulation pattern: e.g. stop a
set, or reduce the next set's load, once rep velocity drops a set
percentage below the first rep's velocity). The *decision* (what to do
about it) is entirely the app's job -- this repo's job is to supply
accurate numbers for it to work with. Originally that meant per-rep
numbers only; see "Fatigue analysis: set and session level" further down
for why that boundary later moved to also include set/session-level
*aggregation and classification* (still not a decision, just less
redundant arithmetic for the app to redo).

`RepCounter` (`irix/rep_counting/state_machine.py`) buffers every
angle/timestamp sample seen during a rep's concentric (bottom -> top)
phase, and on each completed rep reports, alongside the existing
`duration_s` (time since the previous rep -- tempo/cadence):

- `peak_angular_velocity_deg_s` -- fastest instantaneous |d(angle)/dt|
  during the rep
- `mean_angular_velocity_deg_s` -- average speed across the whole
  concentric phase

Both are **joint-angular velocity, in degrees/second** -- a rep-speed
proxy computed from whatever joint the exercise config tracks (knee,
elbow, hip), not a calibrated linear bar velocity in m/s. A calibrated
velocity needs Section 4.5's barbell centroid tracking against
per-station camera geometry, which isn't built (see the table above).
The proxy is good enough for *relative* within-session trend tracking --
is this rep, or this set, slower than the first one -- which is exactly
what velocity-loss-based autoregulation needs; it's not meant for
absolute cross-device comparison against a dedicated VBT sensor.

Both fields flow straight through to `RepCompletedEvent`
(`irix/pipeline/schema.py`), which is what actually reaches irix-mvp-app.
Rest time between *sets* doesn't need its own event -- the app can derive
it from the gap between one `SetCompleteEvent.timestamp` and the next
`RepCompletedEvent.timestamp` for the same member, both of which are
already in the stream.

Fixed a real bug while adding this: `RepCounter` used to seed its
session-start clock from `time.monotonic()` at construction, but every
caller (tests, the mock demo, a real edge pipeline) has its own timestamp
convention -- so the very first rep's `duration_s` came out as a huge
garbage value whenever that convention didn't happen to line up with
wall-clock monotonic time. It's now seeded from the first timestamp
`update()` actually sees. `tests/test_rep_counting.py` has a regression
test for this.

## Barbell and dumbbell tracking, and RPE estimation (Section 4.5)

Built directly on precedent from existing open-source barbell trackers,
adapted to this repo's constraints (cameras-only install, no environment
edits, third-person fixed-camera geometry rather than first-person).

**Detection.** `irix/barbell/detector.py`'s `FreeWeightDetector` is the
same wrapper pattern as `PoseEstimator` -- `ultralytics` is a lazily
imported optional dependency, no bundled weights. Recommended starting
point for the model itself: fine-tune on the Roboflow "Barbells Detector"
dataset (92 labeled images + a pretrained model/API,
[universe.roboflow.com/yolo-project-c2bfs/barbells-detector](https://universe.roboflow.com/yolo-project-c2bfs/barbells-detector))
or a comparable barbell/plate dataset -- the same category of starting
point [mattiolato98/deadlift-visual-analyzer](https://github.com/mattiolato98/deadlift-visual-analyzer)
used for its YOLOv5 barbell class (with mean-shift tracking layered on
top for frame-to-frame continuity when detection confidence dips).
Dumbbell tracking isn't a separate problem -- it's the same detector with
a `dumbbell` class label and the same downstream tracker.

**Calibration, without touching any equipment.**
`irix/barbell/calibration.py` converts pixel measurements to real-world
distances by self-calibrating off a detected object's *already-known
standard dimension* -- a competition bumper plate's 450mm diameter, or a
men's Olympic barbell's 2200mm length -- rather than a physical marker.
[kostecky/VBT-Barbell-Tracker](https://github.com/kostecky/VBT-Barbell-Tracker)
(78 stars) uses the identical self-calibration *principle* -- pixel size
of a known-diameter reference object gives a px-per-mm scale -- but
against a painted marker on the barbell, which isn't usable here since
painting/marking equipment is an environment edit (the same constraint
that ruled out QR plate stickers in `irix/weight_recognition`). Using the
equipment's own manufactured geometry as the reference gets the same
self-calibration property with literally nothing added to the gym floor.
The module also sketches (but doesn't require) a one-time checkerboard
camera-intrinsics calibration, the same `cv2.calibrateCamera`/
`cv2.fisheye` approach VBT-Barbell-Tracker uses in its
`undistort_fisheye.py` -- a legitimate one-time install step (photograph
a checkerboard from the already-mounted camera) rather than an equipment
edit, and worth doing for wide-angle lenses. Known simplification, stated
plainly in the module docstring: the px-per-mm scale is treated as
isotropic across a station's field of view -- no full 3D camera pose
solve. That's the same rigor level the hobbyist projects above use; a
per-station homography (four known reference points at install time)
would remove the limitation and is a reasonable upgrade path, not
attempted here.

**Camera-tilt correction (added 2026-07-14, from VBT hardware
precedent).** `CameraCalibration` now carries an optional
`camera_tilt_deg` (default `0.0`, fully backward compatible) and a
`pixels_to_vertical_m` conversion that `BarPathTracker.push` uses instead
of the plain `pixels_to_m`. This borrows directly from GymAware -- the
linear-position-transducer (LPT) considered the velocity-based-training
gold standard -- which explicitly corrects for the angle between its
cable and the bar's true vertical path, since a sensor not aligned with
the true direction of motion foreshortens the observed distance for a
given real displacement. A camera has the identical problem if it isn't
mounted exactly perpendicular to the bar's vertical plane of travel
(angled down to see a whole rack, or off to one side): the same real
vertical displacement produces a smaller pixel delta than a level camera
would see, so the *uncorrected* conversion underestimates true bar
velocity. `camera_tilt_deg` is a first-order cosine correction for
exactly that, set once per station at install time (e.g. off a
level/inclinometer reading when the camera is mounted) -- same rigor
level as the isotropic-scale simplification above, not a substitute for
the full-homography upgrade path.

**Tracking.** `irix/barbell/tracker.py`'s `BarPathTracker` buffers
calibrated real-world vertical position over time and computes
displacement/peak/mean velocity for any timestamp window -- in **m/s**,
a genuine calibrated measurement, unlike `irix/rep_counting`'s
joint-angular-velocity proxy (deg/s). `RepEvent` (rep_counting/state_machine.py)
now exposes `concentric_start_timestamp` specifically so a caller can
window a `BarPathTracker` query against the *exact* same concentric
phase the joint-angle counter used, rather than approximating it from
`duration_s` (which spans the whole previous-rep-to-this-rep gap). This
is a genuine two-tier design: `RepCompletedEvent` carries both the
always-available `*_deg_s` proxy and the `*_m_s` calibrated fields (None
whenever no free weight is currently being tracked, e.g. a machine
station or before the detector locks on) -- callers should prefer the
`_m_s` fields when present and fall back to `_deg_s` otherwise.

**RPE / fatigue estimation.** `irix/barbell/rpe.py`'s `RPETracker`
produces two signals per rep, feeding the same fatigue-analysis boundary
described above (this repo supplies numbers, irix-mvp-app's AI makes the
training decision):

- `velocity_loss_pct` -- percent velocity loss relative to that set's
  first rep. The primary, well-grounded signal: Sanchez-Medina &
  Gonzalez-Badillo (2011), "Velocity Loss as an Indicator of
  Neuromuscular Fatigue During Resistance Training" (*Med Sci Sports
  Exerc*), found within-set velocity loss correlates strongly with
  independent fatigue markers in the full squat (blood lactate r=0.97,
  ammonia R²=0.85, countermovement-jump-height loss r=0.92). Doesn't
  require knowing anyone's true 1RM -- self-normalizes against that
  lifter's own first rep of that set.
- `estimated_rpe` -- an absolute RPE estimate from published
  population-average velocity-at-1RM anchors (squat 0.23 m/s, bench
  press 0.10 m/s, deadlift 0.14 m/s), from Zourdos et al. (2016),
  "Novel Resistance Training-Specific RPE Scale Measuring Repetitions in
  Reserve" (*J Strength Cond Res* 30(1):267-275) and a follow-on
  replication (Helms et al.) building on the same methodology. Zourdos
  et al. found a strong inverse RPE-velocity relationship (r≈-0.88 for
  the back squat). Stated plainly: this is meaningfully less precise
  than velocity loss -- it's a population average, not this lifter's
  measured load-velocity profile, and a 20-study/434-lifter meta-analysis
  found even *individually calibrated* load-velocity 1RM estimates carry
  a standard error of ~9.8% of 1RM. `estimate_rpe` returns `None` for any
  exercise without a published anchor (e.g. `leg_press`, `bicep_curl` --
  the source literature studied competition powerlifts specifically).
  Mapping a specific velocity-loss percentage to an RPE/RIR delta is a
  genuinely open question in this literature (no source found converges
  on one universal table) -- left to irix-mvp-app's fatigue layer to
  decide rather than guessed at here.

`irix/demo/run_demo.py --with-barbell-tracking` (mock mode, squat/bench_press/deadlift)
wires all of this end-to-end against a synthetic barbell-pixel stream
with rep-over-rep amplitude decay, so `velocity_loss_pct` and
`estimated_rpe` both show a visible fatigue trend across a set without
needing a camera.

## Form scoring: rule-based fault detection (populating a field that's existed since the first commit and never been filled in)

`RepCompletedEvent.form_score` (`irix/pipeline/schema.py`) has been in
the schema since this repo's very first commit, with the comment "0-1,
None if not yet scored" -- and until now, nothing anywhere in this repo
ever scored a rep. Prompted to search broadly for prior art beyond
velocity/RPE ("search online for anything ppl have built for a similar
camera system for gym tracking, incorporate as much as makes sense"),
this is the most direct gap several existing open-source projects target
head-on:

- **github.com/chrisprasanna/Exercise_Recognition_AI** lists, verbatim,
  on its roadmap: "detect poor form (e.g., leaning, fast eccentric
  motion, knees caving in, poor squat depth)". That's close to a
  ready-made spec for exactly this field.
- **github.com/NgoQuocBao1010/Exercise-Correction** trains one classifier
  per exercise *per fault* (bicep curl "lean back" error, lunge "knee
  over toe" error, squat "stage"/depth, plank "all errors") rather than
  one generic form model -- confirmation that per-exercise, per-fault
  scoring is the right granularity, not a single opaque "form" number.
- **github.com/SravB/Computer-Vision-Weightlifting-Coach** scores deadlift
  posture continuously in `[0, 1]` from OpenPose joint positions -- the
  same output shape `form_score` already had.
- **github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting**
  and chrisprasanna's project both do their form/exercise judgments via
  trained LSTM/BiLSTM models over joint-angle sequences rather than
  hand-written geometric rules. Deliberately **not** the approach taken
  here (see below).

### Why rules instead of a trained classifier

Every other module in this repo that touches pose data
(`irix.rep_counting`, `irix.barbell.rpe`) is pure joint-angle/keypoint
geometry: no training data, no model weights, fully unit-testable with
synthetic fixtures, runs identically in CI as on an edge box.
`irix/form/rules.py` follows the same pattern rather than introducing
this repo's first trained-model dependency: each fault is a direct
geometric check over the `PersonPose` keypoints buffered during a rep
(see "wiring" below), not a classifier's opaque judgment. This trades
away whatever a trained model could pick up that a human wouldn't think
to encode as a rule, in exchange for every fault being explainable ("your
knee shifted 0.31 shank-lengths inward at the bottom of the rep") and
gradeable/testable without a labeled video dataset -- which this project
doesn't have. If a labeled dataset materializes later, the LSTM approach
in the two cited repos is the natural fallback to reach for.

### The five checks (`irix/form/rules.py`)

| exercise(s) | fault code | geometric definition |
|---|---|---|
| squat, leg_press, hack_squat | `insufficient_depth` | the rep's minimum hip-knee-ankle angle never got within ~8deg of the exercise's own configured `bottom_angle` (`irix.rep_counting.exercises`) -- reuses the exact threshold that already defines "bottom" for rep counting, rather than a second hardcoded number |
| squat, leg_press, hack_squat | `knee_valgus` | knee's horizontal offset from the ankle (normalized by shank length) shifts more than 0.25 shank-lengths from the rep's own standing baseline -- "knee caving in", chrisprasanna's and NgoQuocBao1010's namesake fault |
| bicep_curl | `leaning_back` | torso (hip->shoulder) angle from vertical deviates more than 15deg from the rep's own standing baseline -- classic momentum-cheat curl, NgoQuocBao1010's "lean back error" |
| bicep_curl | `elbow_drift` | elbow's horizontal offset from the hip (normalized by upper-arm length) shifts more than 0.35 upper-arm-lengths from baseline -- elbow swinging away from the torso recruits the shoulder instead of isolating the bicep |
| deadlift | `hips_rising_before_chest` | hip's and shoulder's vertical trajectories, each normalized to a 0 (bottom)-1 (lockout) progress scale over the rep, diverge by more than 0.25 -- the hips shoot up before the chest rises ("stripper deadlift"), a standard deadlift coaching cue, shifting load off the legs and rounding the lower back |

Every check requires a minimum number of confidently-tracked keypoint
samples (`MIN_VALID_SAMPLES = 3`, keypoint confidence >= 0.3) before it
will report anything at all -- "couldn't assess" returns `None`, same as
an unscored rep, rather than quietly reporting a perfect score when the
data just wasn't there. `FormScorer.score_rep()` (`irix/form/scoring.py`)
runs an exercise's registered checks, turns each detected fault into a
score penalty (`FormAssessment.score`), and lists the triggered fault
codes (`FormAssessment.faults`) -- structured codes like
`"knee_valgus"`, not sentences, matching this event family's existing
"no coaching text originates in this repo" boundary with irix-mvp-app;
the app decides how to phrase it to the member.

`bench_press` has no registered checks yet: a meaningful bench fault
(elbow flare relative to the bar path) needs either a second camera angle
or the barbell-tracking data from `irix.barbell` wired in, neither of
which this scaffold does -- `form_score` correctly stays `None` for
bench_press rather than reporting a number that isn't actually measuring
anything.

**Note on the camera geometry these checks assume**: the horizontal
(x-axis) keypoint shifts `knee_valgus`/`leaning_back`/`elbow_drift` key
off are a frontal-plane *proxy* seen from the same 30-45deg-off-axis
camera angle assumed everywhere else in this repo (Section 3.1) -- a real
inward knee collapse or backward lean will show up as a larger x-shift
than clean form, but the magnitude isn't a calibrated angle the way a
straight-on frontal camera would give. Same tier-2-proxy honesty already
applied to `peak_angular_velocity_deg_s` elsewhere in this repo.

### Wiring: `RepCounter` now optionally buffers poses, not just angles

`RepCounter.update()` (`irix/rep_counting/state_machine.py`) gained an
optional third argument, `pose: Optional[PersonPose] = None`. When a
caller passes one, every pose seen from just after the previous rep's
completion through this rep's completion is buffered and attached to the
completed `RepEvent.poses` -- the *full* eccentric+concentric cycle, not
just the concentric-phase window `peak_angular_velocity_deg_s` uses,
since a fault like `leaning_back` can show up on the way down just as
easily as on the way up. If a caller never passes a pose (e.g. an
IMU-only path, or a frame where `PoseEstimator` wasn't confident enough
to return keypoints), `RepEvent.poses` is `None` and `FormScorer` simply
doesn't score that rep -- this class stays agnostic of form scoring
itself, it just optionally carries the data. The buffer resets (not just
after a completed rep, but also when the angle idles in the "top" zone
without ever reaching bottom) so a lifter standing between sets for a
while doesn't grow it unboundedly.

`irix/demo/run_demo.py --with-form-scoring` (mock mode, squat/bicep_curl
only -- see below) and `run_live` (real camera, any exercise, no flag
needed -- it already gets a full pose from `PoseEstimator` every frame)
both wire `FormScorer` in and populate
`RepCompletedEvent.form_score`/`form_faults` from it.

### Demo: a synthetic full-body pose stream, not just a synthetic angle

The existing mock demo (`synthetic_angle_stream`) only ever produced a
single oscillating angle, not full keypoints -- fine for rep counting,
useless for form scoring, which needs multiple joints' positions, not
just the one tracked angle. `irix/demo/mock_pose.py` adds
`synthetic_pose_stream`, built on a small general-purpose 2-segment
inverse-kinematics helper (`_third_point`): given two known keypoints and
a target joint angle, place the third keypoint so
`irix.pose.geometry.joint_angle` recovers exactly that angle. This keeps
the synthetic pose stream and the synthetic angle stream mathematically
consistent (same angle, geometrically real keypoints) rather than being
two independent, potentially-contradictory fakes. It supports the squat
family (hip-knee-ankle, ankle fixed) and bicep_curl (shoulder-elbow-wrist,
hip fixed) -- not deadlift, which needs different geometry (a translating
body, not an articulation around one fixed base joint) this generator
doesn't build; requesting it for deadlift yields no pose rather than
silently feeding `FormScorer` nonsense keypoints.

`--inject-form-fault {knee_valgus,leaning_back,elbow_drift}` perturbs the
relevant keypoint independently of the tracked joint angle (so rep
counting stays unaffected) so the demo can show a fault actually getting
caught, not just clean reps:

```
python -m irix.demo.run_demo --mock-pose --exercise squat --with-form-scoring --inject-form-fault knee_valgus
python -m irix.demo.run_demo --mock-pose --exercise bicep_curl --with-form-scoring --inject-form-fault leaning_back
```

`tests/test_form_scoring.py` unit-tests all five checks in isolation
(clean + faulted fixtures, plus insufficient-data returns `None`) using
hand-built `PersonPose` fixtures independent of the demo's synthetic
stream; `tests/test_rep_counting.py` and `tests/test_demo_smoke.py` cover
the pose-buffering wiring and the end-to-end demo path respectively.

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

`irix/pipeline/events.py`'s `BandPlacementTracker` is the stateful piece
that turns this into an actual event: it tracks where the band currently
is across a session and only emits a `BandPlacementRequiredEvent` when the
next exercise's `band_placement` differs from the current one, so a
session that never touches a leg machine emits nothing about it. Turning
that event into a spoken instruction is irix-mvp-app's job, not this
repo's -- see "Where this repo ends" above.

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
(cloud) is the chosen, real, verified path -- see "Model weights" above
for exactly what "verified" means here. This is a real tradeoff decided
explicitly rather than a default accepted quietly: a cloud VLM call means
camera frames leave the building, which is in tension with Section 8's
data-minimization stance (raw video never leaves the building). It was
chosen anyway because `VisionPlateClassifier` only calls the backend
during the confirm window at station setup -- a few calls per set, not
per frame or per rep -- so the volume of frames actually leaving the
building is small, and it avoids standing up and maintaining on-device
model-serving infrastructure on the zone edge box for that low call
volume. No API key is bundled or hardcoded in this repo; a deployer
supplies their own via `GeminiVLMBackend(api_key=...)`.
`LocalVLMBackend` -- an on-device open-source VLM served on the zone edge
box, which would close the frames-never-leave-the-building gap -- remains
an interface sketch (`NotImplementedError`), left deferred rather than
guessed at: which local model/serving stack to run on the Jetson boxes
from Section 6 is a real decision that needs actual hardware to
benchmark, and is worth revisiting if the tradeoff above stops holding
for a given deployment.

`tests/test_vision_classifier.py` exercises the confirmation-windowing
logic against a scripted `FakeVLMBackend` -- no real model call needed to
validate that logic.

Out of scope (hardware, not software): camera selection/mounting (Section
3), edge compute sizing and PoE network design (Section 6.1-6.2), cost
modeling (Section 10), rollout plan (Section 11).

## Multi-station deployment (Section 6: what changes with 10 cameras instead of 1)

Prompted directly by a concrete deployment target -- "if I deploy 10
cameras in a gym and every member has a wristband with an IMU I need to
be able to use both to do accurate rep tracking, weight recognizing,
fatigue analysis etc, and feed this to the app" -- the sections below
cover what materially changes once there's more than one camera and more
than one member on the floor simultaneously, not just a bigger version of
the single-station demo. The user explicitly said not to feel bound to
the original design doc's architecture where a better approach exists;
the three subsections below (fusion, topology, fatigue) each diverge from
what Section 4.6/5/6 originally sketched, with the reasoning for each
divergence spelled out inline.

### Camera + wristband IMU: real fusion, not a parallel crosscheck (`irix/fusion/rep_fusion.py`)

Before this pass, "using both" meant `run_demo.py --with-imu-crosscheck`:
run the camera-based `RepCounter` and the wristband-only
`RecoFitCounter`/`ULiftCounter` independently, print both numbers side by
side. That's not fusion -- nothing about the two counts talked to each
other, and nothing decided which one to actually trust if they disagreed.
`irix.fusion.rep_fusion.RepCountFusion` is the real version:

- **Reconciliation is decision-level (late fusion), not continuous-state
  fusion.** The design doc's Section 4.6 sketched folding IMU data into
  the same EKF used for visual-inertial position tracking
  (`irix.fusion.ekf`). That's the right tool for a continuous physical
  quantity (position/orientation) but not for this: "a rep happened" is a
  discrete event, not a smoothly-varying state, so there's no meaningful
  single continuous quantity to run a Kalman filter over between "camera
  thinks rep 4 landed at t=8.1s" and "IMU thinks it landed at t=8.3s".
  This mirrors how published systems that combine camera/video and
  wearable IMU for exercise tracking actually do it: e.g. the ACM paper
  "Wearable IMU-based Gym Exercise Recognition Using Data Fusion Methods"
  fuses multiple IMU placements at the decision level (each sensor
  produces its own read, a fusion step reconciles them), and the broader
  multi-sensor activity-recognition survey literature converges on
  confidence-weighted decision fusion as the standard pattern for
  reconciling independent per-modality event/count streams. `irix.fusion.ekf`/
  `irix.fusion.zupt` remain the right module for the continuous-state
  problem they solve; this is a different problem needing a different
  tool.
- **The unit of analysis is a set, not a rep.** `RecoFitCounter`/
  `ULiftCounter` are themselves batch algorithms -- they need several
  cycles of signal to estimate a period via autocorrelation (see
  `irix/fusion/imu_rep_counting.py`'s own docstring), so running them on
  a single ~2s rep window isn't meaningful. `RepCountFusion.fuse()` is
  called once per completed set (`SetCompleteEvent` granularity),
  matching both algorithms' actual operating range and the granularity
  `irix-mvp-app` needs an authoritative count at anyway.
- **The fusion runs bidirectionally.** The camera's own observed rep
  durations for the set (`RepEvent.duration_s`, already computed) are fed
  in as `camera_rep_durations` and used to derive `RecoFitCounter`'s
  period-bounds search (`min_period`/`max_period`) instead of guessing
  generic 1-4s bounds blind -- the camera improves the IMU algorithm's own
  accuracy, not just the other way around.
- **Reconciliation logic**: if camera and IMU counts agree (within
  `agreement_tolerance`, default ±1), the camera count wins (it also
  carries velocity/form data the IMU alone doesn't). If they disagree,
  whichever source has higher confidence wins -- `RepCounter.tracking_confidence`
  (a new property: fraction of frames with a usable, non-NaN angle this
  session, i.e. how much of the set wasn't occluded) for the camera, and
  `RepResult.confidence` for the IMU. This is exactly the case the
  wristband earns its keep: heavy occlusion drops camera confidence, and
  fusion correctly leans on the IMU instead (see
  `tests/test_run_gym_demo.py::test_leg_press_zone_shows_band_placement_and_imu_fallback_on_occlusion`
  for this exact scenario, and `irix.demo.run_gym_demo`'s `--occlusion`
  simulation).
- If no IMU data is available at all for a set (didn't call `fuse()` with
  samples, or the signal was too short/flat for either algorithm),
  `source="camera_only"` and the camera count is used as-is -- fusion
  degrades gracefully to exactly today's single-signal behavior rather
  than failing.

The reconciled result (`FusedSetRepCount`) flows into
`SetCompleteEvent.fused_rep_count`/`imu_rep_count`/`rep_count_agreement`/
`rep_count_source` -- `total_reps` (the camera's own count) is kept
unchanged for backward compatibility; `fused_rep_count` is what
`irix-mvp-app` should treat as authoritative when it's present.

### Multi-camera station topology and handoff (`irix/topology/`)

A single-station demo never has to answer "which camera is allowed to
report events for this person right now" -- there's only one camera.
Ten cameras on one gym floor do, for two reasons: a member walking from
the squat rack to the leg press briefly sits in two adjacent cameras'
fields of view, and `irix.identity.ble_pairing`'s RSSI-based resolution
is inherently noisy enough (~5-10m typical accuracy indoors, per that
module's own docstring) that naively trusting every snapshot resolution
would flicker a member's assignment back and forth near a station
boundary.

- `irix.topology.registry.StationRegistry` holds the fixed camera/station
  layout for a gym (`build_default_ten_station_gym()` is a concrete
  10-camera example covering every currently-configured exercise: two
  squat racks, two bench stations, one deadlift platform, two dumbbell/
  curl stations, two leg-press machines, one hack-squat machine), plus an
  adjacency graph (`is_adjacent`) -- which stations a member could
  plausibly walk to directly next.
- `irix.topology.handoff.MemberStationTracker` wraps
  `StationPairing.resolve()` (which already existed, resolving one
  snapshot of RSSI readings to a station) with **hysteresis**: a member's
  assigned station only actually changes after `min_consecutive` readings
  in a row favor a different station, absorbing RSSI jitter near a
  boundary rather than emitting a spurious handoff on every noisy
  reading. Fires a `StationHandoffEvent` (added to the `CameraEvent`
  family in `irix.pipeline.schema`) only on a real, sustained move --
  mirroring `BandPlacementTracker`'s existing "only emit on actual
  change" idiom.
- `StationHandoffEvent.plausible_adjacency` flags a resolved handoff that
  isn't between registered-adjacent stations -- a jump across the whole
  gym floor in one reading is much more likely a mis-resolved BLE reading
  (multipath reflection, a different member's band transiently closer)
  than an instant teleport, and is worth surfacing to an ops dashboard
  rather than silently trusting.
- `irix.topology.handoff.GymCoordinator` is the gym-wide layer: one
  instance tracks every member's station via per-member
  `MemberStationTracker`s, and answers the question a station's edge box
  needs answered before pushing a camera-derived event for some
  `member_id`: `is_authoritative(member_id, station_id)`. This is the
  actual anti-double-counting mechanism -- an adjacent camera that
  glimpses a member mid-walk should not push rep/weight events for them,
  because `is_authoritative` for that station returns `False` until a
  real (hysteresis-confirmed) handoff happens.

See `irix.demo.run_gym_demo._demo_station_handoff_and_dedup` for a
runnable trace: a member settles at squat-1, a single noisy RSSI reading
toward squat-2 is correctly absorbed (no handoff, `is_authoritative`
correctly stays `False` for squat-2), then a sustained 3-reading signal
correctly triggers a real handoff.

### Motion-correlation identity disambiguation (`irix/identity/motion_correlation.py`)

BLE RSSI resolution (`irix.identity.ble_pairing`) answers "which station
is this member's band closest to" -- it does not, and structurally
cannot, answer "of the two people my camera sees at this station, which
one is wearing which of these two members' bands". Both members'
readings can legitimately resolve to the same station (a shared curl
rack, two people training together, a crowded free-weights area) --
`GymCoordinator.active_members_at(station_id)` returning more than one
member is exactly that signal.

`MotionCorrelationResolver` disambiguates by cross-correlating a
vision-derived motion signal (the second derivative -- a Savitzky-Golay
smoothed one, not naive double-differencing, which would amplify
keypoint-tracking jitter by roughly `1/dt**2` -- of a tracked wrist
keypoint's vertical position) against each candidate member's raw
wristband accelerometer signal over the same window. The pairing with
the highest correlation is assumed to be the same physical person: two
different people's limb motion is essentially uncorrelated even doing
similar exercises side by side, while a wristband's actual motion should
closely track its wearer's own tracked wrist. This is not an invented
technique -- it mirrors published person-identification systems that
pair a camera-tracked skeleton to a specific wearable IMU this same way
(e.g. the IEEE "Person tracking association using multi-modal systems"
work matching a depth-camera skeleton to an inertial wearable via
movement features), and the broader acceleration-correlation-based
identification/synchronization literature more generally.

Refuses to guess rather than force an assignment: if the best- and
second-best-correlated candidates for a detected person are too close to
call (`min_confidence_margin`), or a candidate's signal doesn't have
enough valid overlapping samples to correlate at all, the result is
`None` for that person -- the same "couldn't assess, don't fabricate an
answer" posture `irix.form.rules` and `irix.fatigue` already take.
Assignment is a simple greedy match (highest-confidence pairs claimed
first, so no member gets assigned to two different detected people) --
exact bipartite optimal matching wasn't judged necessary at the 2-3
co-located-member scale this is meant for.

**Known limitation, stated plainly**: a wearable's raw accelerometer
measures gravity's changing projection as the wrist rotates, on top of
translational motion; the vision-derived signal captures only
translational motion (keypoint *position*, not orientation). That
mismatch degrades the correlation somewhat -- the literature this is
grounded in explicitly calls out gravity-direction compensation (a full
3D wrist-orientation estimate) as the main accuracy lever for a
production system, and this module doesn't attempt it. It leans instead
on vertical motion being the dominant, well-correlated component for gym
exercises specifically (squats/curls/presses/rows are all fundamentally
vertical and cyclic), which is enough to disambiguate a *small* number of
co-located candidates, not a claim of general-purpose, camera-angle-
agnostic re-identification.

`GymCoordinator.disambiguate_by_motion()` is the integration point --
delegates straight to `MotionCorrelationResolver.resolve()`, so a caller
that already has a `GymCoordinator` doesn't need to wire up the identity
module separately. See `irix.demo.run_gym_demo._demo_motion_correlation_disambiguation`
for a runnable trace: two members ("carol", faster tempo; "dave", slower)
both BLE-resolve to `curl-1`, and the camera's two detected skeletons get
correctly matched to the right member from wrist-motion timing alone.

### Barbell tracking in the multi-station demo

`irix.demo.run_gym_demo`'s squat sets run with `with_barbell_tracking=True`
(mirroring `run_demo.py --with-barbell-tracking`): a synthetic barbell-
pixel stream through `BarPathTracker`/`RPETracker` gives each rep a
calibrated `mean_velocity_m_s`, not just the joint-angular `deg_s` proxy.
This matters specifically for `irix.fatigue.SetFatigueAnalyzer`'s tier
selection: without it, every set analysis silently fell back to the
`deg_s` tier, and `velocity_loss_pct` computed from joint-angular
velocity is only a *directional* proxy (see `irix.rep_counting`'s own
docstrings on this) -- the VL10/VL20/VL30/VL45 zone thresholds
`irix.barbell.rpe`'s citations validate were derived from calibrated bar
velocity, not a joint-angle proxy, so classifying a set into one of those
zones is only as trustworthy as the tier backing it. With barbell
tracking wired in, the demo's two squat sets show real VL-zone
progression (VL10 -> VL30 across the two sets, with `session_fatigue_index`
climbing from ~0.09 to ~0.20) instead of the near-zero noise the
deg/s-tier proxy showed on a synthetic angle stream whose rep-to-rep
shape barely changes. `leg_press`/`bicep_curl` still run on the `deg_s`
tier here (no published velocity anchor for either in
`EXERCISE_1RM_VELOCITY_MS`) -- exactly the same two-tier fallback
pattern used everywhere else in this repo, working as intended rather
than silently degrading.

### Fatigue analysis: set and session level (`irix/fatigue/`)

`docs/ARCHITECTURE.md` previously described this repo's fatigue-related
job as "supply the numbers, the app makes the judgment" -- per-rep
velocity/duration fields on `RepCompletedEvent`, nothing aggregated.
That boundary still holds in spirit (nothing in `irix.fatigue` prescribes
what a member should do next -- no "reduce the weight" instruction
originates here), but the boundary was drawn further back than it needed
to be: aggregating and classifying a completed set's fatigue signature
is still descriptive, not prescriptive, and doing it here means
`irix-mvp-app` doesn't have to re-derive the same arithmetic from a raw
rep stream every time it wants context for its AI.

- `irix.fatigue.set_analysis.SetFatigueAnalyzer` aggregates one set's rep
  samples into a `SetFatigueAnalysis`: **velocity loss %** (first rep vs.
  last rep, the same well-supported signal `irix.barbell.rpe` already
  computes per-rep -- see that module's Sanchez-Medina & Gonzalez-Badillo
  2011 citation -- now aggregated to the set level), which of the
  standard **VL-zone thresholds** (VL10/VL20/VL30/VL45, the exact
  thresholds `irix.barbell.rpe`'s docstring already named as standard in
  that literature) the set's loss crossed, **tempo drift %** (rep
  duration lengthening across the set -- a fatigue signal independent of
  velocity, available even with zero calibrated velocity data since
  `duration_s` is always present), and the set's **form-score trend** and
  **most common fault** (from `irix.form.scoring`, aggregated).
- Two-tier velocity, same pattern as everywhere else in this repo:
  prefers `mean_velocity_m_s` (tier 1, calibrated barbell velocity) when
  any rep in the set has it, falls back to `mean_velocity_deg_s` (tier 2,
  joint-angular proxy) otherwise, and reports which tier
  (`velocity_tier`) actually backed the analysis so a caller never
  mistakes a proxy-based number for a calibrated one.
- `irix.fatigue.session_analysis.SessionFatigueTracker` adds the
  cross-set dimension a single set can't see: **set-to-set velocity
  trend** (each set's opening-rep velocity vs. the session's first set --
  catches a member's 3rd set opening 15% slower than their 1st set opened,
  before that 3rd set has shown any velocity loss of its own) and a
  **session fatigue index** (0-1, a transparent heuristic blending
  within-set loss and across-set decline -- explicitly documented in that
  module as a heuristic, not a validated composite score, since no
  published formula for combining the two exists the way it does for
  velocity loss alone).
- Both feed a new `SetFatigueSummaryEvent` (`irix.pipeline.schema`),
  pushed alongside `SetCompleteEvent` -- see `irix.demo.run_gym_demo` for
  two consecutive squat sets showing the session tracker's cross-set
  fields populate on the second set.

### Weight recognition: a geometric cross-check, not a second reading method (`irix.weight_recognition.plate_geometry_check`)

The existing VLM-based `VisionPlateClassifier` remains the primary weight
reading method, for the reasons already documented in the "Weight
recognition: why it ended up VLM-based too" section below -- most
importantly, standardized competition bumper plates are *all the same
450mm diameter regardless of weight* (`COMPETITION_BUMPER_PLATE_DIAMETER_MM`),
distinguishable only by color, which a VLM can reason about and raw plate
geometry structurally cannot. That fact rules out resurrecting classical
CV plate classification as an equal, independent second signal.

What plate geometry *can* still do without solving that harder problem:
sanity-check a VLM read against how many plates are actually visible in
frame. `check_plate_geometry()` decomposes the VLM-read weight into an
*expected* plate count per side (a generic commercial-gym plate-weight
set, greedy decomposition -- an estimate of what should be loaded, not a
claim about which specific plates are on the bar, since multiple
combinations can sum to the same total) and compares it against
`FreeWeightDetector`'s actual detected plate count for that frame. A
badly wrong VLM read -- hallucinated, or a decimal-point misread -- will
usually imply a wildly different plate count than what's actually
visible, even though fine-grained plate-by-plate identification stays out
of reach. `WeightConfirmedEvent` gained `geometry_consistent`/
`geometry_check_reason` fields to carry the result; `consistent=True`
(not `False`) when there's nothing to check against (no plates detected
-- occluded view, or a non-barbell exercise), since "couldn't check" and
"check failed" are different things worth keeping distinct.

## Live station readiness: front-desk checkout, a 24/7 camera, a live wristband

Everything above -- including `run_upload.py` -- assumes the inputs are
already sitting on disk: a finished video file, maybe a finished IMU
export. The actual production target is different in three ways that
nothing in this repo modeled until now: cameras run continuously, not one
video at a time; a wristband only means something once the front desk has
checked it out to an account; and IMU access is supposed to start when a
member's set starts, not be loaded from a file after the fact. This
section is the software-side response to that -- deliberately scoped to
what's genuinely buildable as pure software, with hardware-dependent
pieces left as documented stubs rather than guessed at (same boundary
`irix.identity.ble_pairing`'s own docstring already draws around the BLE
radio stack).

**`irix.identity.checkout.CheckoutRegistry`** is the piece that was
missing entirely: every `member_id` anywhere in this repo, until now, was
just a string a caller already had to know. A real station only ever
observes a wristband's BLE identifier -- it has no idea whose account
that is unless something recorded "the front desk handed this band to
this account." `CheckoutRegistry.check_out`/`check_in`/`resolve_member`
is that record: one active checkout per physical band at a time (mirrors
a real front-desk band cabinet -- a band has to come back before it can
go out again), with `resolve_member(wristband_id)` as the lookup a live
station does before it's willing to attribute any event to an account.

**`irix.fusion.imu_stream.IMUStream`** answers "how does a live band's
IMU data actually get consumed, given it's arriving continuously instead
of sitting in a file." A `poll()`-based protocol, same pattern as
`VLMBackend`/`CloudSync` elsewhere in this repo: `RecordedIMUStream`
(real) wraps an already-loaded list so `irix.demo.run_upload` and a live
caller can share identical consumption code in `RepSession`;
`LiveBLEIMUStream` is a documented stub -- which BLE client
library/wristband firmware protocol a real device exposes is exactly the
kind of hardware detail this scaffold has no way to guess at correctly,
same reasoning `LocalVLMBackend` stayed unimplemented for. What's settled
is the *shape* (`poll() -> List[IMUSample]`) a real implementation has to
satisfy, so nothing downstream needs to change once one exists.

**`irix.pipeline.rep_session.RepSession`** is a refactor, not new
behavior: the per-member state (`RepCounter`, `FormScorer`,
`VisionPlateClassifier`, `BarPathTracker`/`RPETracker`,
`RestGapSetBoundaryDetector`, `SetFatigueAnalyzer`/
`SessionFatigueTracker`) that `run_upload.run_upload()` used to construct
and drive inline, against a whole video, for exactly one member, is now a
class with `process_frame(frame, ts, person)` / `add_imu_samples(...)` /
`close(end_ts)`. `run_upload` is now a thin driver around one
`RepSession` for the length of a video file; `StationSessionRunner`
(below) drives a *sequence* of `RepSession`s, one per member, for as long
as each is actually present. Extracting this doesn't change what either
one produces -- `run_upload`'s existing tests pass unchanged against the
refactored version -- it just means the event-construction logic exists
in exactly one place instead of needing to be kept in sync between two.

**`irix.live.camera_source.ReconnectingFrameSource`** is the 24/7-camera
piece. `cv2.VideoCapture` already accepts a live stream URL exactly like
it accepts a file path or webcam index -- `run_live`'s `--source` never
technically blocked live streams. What every existing frame loop actually
gets wrong for a 24/7 station is what happens when a read *fails*: they
all stop. A live RTSP feed genuinely drops sometimes (network blip,
camera reboot, DHCP renewal); a station shouldn't go dark until someone
notices and restarts the process. `ReconnectingFrameSource.frames()`
yields indefinitely, releasing and reopening with exponential backoff on
any open/read failure instead of raising -- real, tested logic (see
`tests/test_camera_source.py`, which drives it against a fake capture
object that fails on cue), independent of whether a real camera is
available to test against.

**`irix.live.station_runner.StationSessionRunner`** composes all of the
above into what a real station actually runs: per frame tick, resolve
which checked-out band (if any) this station's BLE reader currently sees
(`irix.identity.ble_pairing.BLEReading` gained an optional
`wristband_id` field for this -- `StationPairing`'s own station-selection
job never needed to know *which* band a reading was for, since it's
always called with one band's readings already; a station watching for
*any* checked-out band showing up does need to know). A checked-out band
showing up starts a fresh `RepSession` (and requests that band's live
`IMUStream`); the band going quiet for `presence_timeout_s` closes it
(flushing whatever set was in progress) -- the same "infer the boundary,
don't wait to be told" spirit as `RestGapSetBoundaryDetector`, one level
up: that class decides when a *set* ends within a session, this class
decides when the *session* ends. `ble_reader` and `imu_stream_factory`
are the two seams a real deployment plugs actual hardware into
(`tests/test_station_runner.py` exercises everything downstream of those
seams with fakes: presence starting/ending a session correctly attributed
to the right account, an unchecked-out band never starting one, one
member's session getting preempted the instant another checked-out
member shows up rather than double-attributing events).

What's still explicitly not built: the real `LiveBLEIMUStream` (needs
real hardware to get right) and the real BLE reader a station would use
in production (same boundary -- radio-stack/firmware, not software-
scaffold scope). Both have a settled interface to build against once
that hardware work happens.

**Multiple stations, live.** Everything above is scoped to one station.
That's wrong for the real 10-camera target the same way a single
`RepSession` was wrong for a whole gym floor: two adjacent stations, each
independently resolving BLE presence from only their own local view,
would both start a session for the same band mid-walk between them --
the exact double-counting problem `irix.topology.handoff.GymCoordinator`
already solves for the synthetic multi-station demo (`run_gym_demo.py`),
just never connected to anything live before now.

`StationSessionRunner` gained a `tick(frame, now, present_wristband_ids)`
method, splitting "resolve who's present" from "given a resolution, do
the frame's work" -- `run_forever` still resolves presence locally
(unchanged behavior, existing tests pass unmodified), but `tick()` is now
also callable directly with an externally-resolved presence. Note the
plural: this used to be `present_wristband_id` (singular, at most one
member per station); see the crowded-station section below for why that
changed.

`irix.live.gym_runner.GymSessionRunner` is that external resolver: it
owns one `GymCoordinator` and one `CheckoutRegistry` shared across every
station, pulls one frame from each station per gym-wide tick, resolves
presence for every checked-out band from a single raw BLE reading source
(`GymCoordinator.update_member`, same hysteresis logic `run_gym_demo.py`
already exercises with synthetic readings -- `min_consecutive` readings
favoring a different station before an actual handoff, not on every RSSI
flicker), and calls each station's `tick()` with *every* member
`GymCoordinator.active_members_at(station_id)` currently says is
authoritative there (`_present_wristbands_at`, plural -- may be more than
one at a crowded station). A member's session at their old station closes
the same way any absence does (`presence_timeout_s`, per member, tracked
gym-wide) once `GymCoordinator` re-resolves them to a different station --
no separate "handoff" code path needed in `StationSessionRunner` itself,
the existing absence-timeout logic already does the right thing once fed
the right presence signal. `tests/test_gym_runner.py` verifies this end
to end with a scripted walk between two stations: exactly one
`StationHandoffEvent`, exactly one `SetCompleteEvent` closed out at each
station (proof the session actually moved, not merged or dropped), and a
separate test with two members at two different stations simultaneously
proving no cross-contamination.

**Crowded stations: disambiguating two co-located members.** Two
checked-out members' bands can both legitimately resolve to the *same*
station at once (a shared bench, a crowded curl rack) -- RSSI proximity
alone can't tell them apart there (see `irix.identity.
motion_correlation`'s module docstring for why), which used to leave this
case explicitly unhandled. It's now wired into `StationSessionRunner`
directly: internal per-session state (`_sessions`, `_imu_streams`,
`_last_seen`) generalized from a single active session to dicts keyed by
`wristband_id`, so more than one `RepSession` can be open at a station at
once.

Routing which detected person is which member is decided from *that
tick's* actual present-band count, not from how many sessions happen to
still be open -- a session past its last-seen tick keeps running through
its ordinary `presence_timeout_s` grace period (tolerating a brief radio
dropout, same as the single-session case always did) but simply doesn't
receive a routed frame while it's not the sole band reported; an ordinary
one-after-another handoff at a station (someone leaves, someone else
steps up right after) still therefore routes unambiguously to whoever is
actually seen *that* tick, without paying any disambiguation cost. Only a
genuine same-tick multi-presence (two or more bands reported at once)
triggers buffering.

When that happens, `StationSessionRunner` accumulates a short window
(`disambiguation_window_frames`, default 60 ticks) of every detected
person's pose alongside every ambiguous band's raw IMU samples, then
calls `irix.identity.motion_correlation.MotionCorrelationResolver`
directly (the same class `GymCoordinator.disambiguate_by_motion`
delegates to in the synthetic demo) to resolve which detected-person slot
correlates with which band's wristband motion. The resolution is
"sticky": once resolved, the same slot->band mapping keeps routing frames
until the present-band group actually changes (someone new joins, or
someone leaves), at which point a fresh window starts. Two trade-offs are
stated plainly rather than hidden:

- While a window is buffering (or for any detected-person slot that never
  resolves confidently), frames for the still-ambiguous group aren't
  attributed to anyone -- reps genuinely happening during that short
  window are missed rather than guessed at.
- Routing assumes a detected person's position in `PoseEstimator.
  estimate()`'s returned list stays stable for the duration of one
  buffering window. `PoseEstimator` doesn't run persistent cross-frame
  tracking (`PersonPose.track_id` is just that frame's list index, not a
  stable id), so this is a real assumption -- reasonable over a short
  window with a static camera, not a guarantee over a long session, which
  is exactly why re-resolution happens fresh every time the present-band
  group changes rather than trusting one resolution indefinitely.

`tests/test_station_runner.py::test_crowded_station_disambiguates_two_co_located_members_by_motion`
exercises this end to end (not just the resolver in isolation, already
covered by `tests/test_motion_correlation.py`): two members with
distinguishable curl tempos, both present at one station from the start,
correctly resolved and routed to the right `RepSession` once the
buffering window fills, with neither member's reps ever attributed to the
other.

## Overlapping multi-camera zones (a dense camera array, not one camera per station)

Everything above -- `StationSessionRunner`, `GymSessionRunner`,
`StationRegistry`'s 10-camera example layout -- assumes exactly one
camera per station: a distinct rack/bench each with its own dedicated
view. A real free-weights floor is sometimes covered differently: several
cameras (the design doc's Section 3.1 range makes ~10 cameras over one
busy area plausible) with genuinely **overlapping fields of view**, where
the same physical person can be visible to more than one camera at once,
and several members can be anywhere in that shared space simultaneously
-- not confined to fixed labeled stations. That's a different problem
from station-to-station handoff (`GymCoordinator`, moving *between*
discrete stations) and from single-camera crowding
(`CrowdedGroupDisambiguator` at *one* camera) above; it needed a third
orchestrator rather than stretching either existing one.

**`irix.live.disambiguation.CrowdedGroupDisambiguator`** is first a
*refactor*, not new behavior: the pose/IMU-buffering, `MotionCorrelationResolver`-
calling, sticky-until-group-changes logic that used to live inline in
`StationSessionRunner` moved out into its own class, one instance per
detection source. `StationSessionRunner` now owns exactly one instance
(unchanged behavior -- its existing test suite passes unmodified against
the refactor) and is the reason a multi-camera zone could reuse this
logic directly rather than needing a parallel reimplementation.

**`irix.live.zone_runner.MultiCameraZoneRunner`** is the new
orchestrator for the overlapping-camera case: several `ZoneCamera`s (a
frame source plus pose estimator each) covering one shared zone, each
running its **own** `CrowdedGroupDisambiguator` against the **same**
zone-wide candidate wristband group. Deliberately *not* attempting the
general multi-camera person re-identification problem (matching a
detection in camera A's frame to a detection in camera B's frame by
appearance/geometry, the way sports-analytics systems like SkillCorner do
with jersey-number recognition) -- consistent with this repo's existing
stance (`irix.identity.motion_correlation`'s own docstring) that
wristband-based identity is the simpler, more reliable choice over
vision-only re-ID. Instead, **the wristband IMU signal itself is what
ties multiple cameras' views of one person together**: if camera A's
slot 2 and camera B's slot 0 both correlate best against wristband X's
IMU, they're the same physical person, with neither camera ever needing
to know the other exists or detected anyone that tick. This also gives
occlusion tolerance for free -- a person invisible to one camera's angle
this tick just doesn't get a routed entry from that camera, but still
gets routed correctly via any other camera that currently sees them.

**Avoiding double-counting when cameras agree.** When 2+ cameras
independently resolve a pose for the *same* member in the same tick (a
legitimately overlapping view), `MultiCameraZoneRunner.tick()` feeds
exactly one of them into that member's `RepSession` -- a fixed
camera-priority order (first camera, in the order the runner was
constructed with, to have a routed pose for that member) decides which,
never more than one `process_frame` call per member per tick.

**Bar-path calibration is per-camera-aware.** When routing switches a
member's ongoing set from one physical camera to another mid-set, bar
velocity switches with it -- `RepSession` self-calibrates a separate
px-per-mm `CameraCalibration` independently for each `camera_id` that has
fed it a frame with a detected plate (`RepSession._bar_calibrations`,
keyed by `camera_id`; `None` for the single-camera case, where
`process_frame`'s `camera_id` parameter is never passed), and
`BarPathTracker.push()` takes an explicit per-call `calibration`
override. `MultiCameraZoneRunner.tick()` passes `camera_id=camera.
camera_id` into every `process_frame` call it makes (both the
single-member "no ambiguity" path and the multi-member disambiguated
path), so whichever camera actually produced a given frame is what its
pixels get calibrated/converted against -- never a stale calibration
carried over from whichever camera calibrated first. Since `BarPathTracker`
samples are stored in real-world meters at push time (not raw pixels),
one continuous sample buffer/velocity-window query still works correctly
across a camera switch mid-rep or mid-set -- only the pixel-to-meters
conversion *at push time* needs to know which camera produced that
particular measurement; nothing downstream does. `camera_tilt_deg_by_camera`
(on both `RepSession` and, forwarded through, `MultiCameraZoneRunner`)
lets each camera's tilt-correction angle differ too, since distinct
physical mountings can plausibly be angled differently. Rep counting was
never affected by any of this either way (joint angles are relative
measurements between a single frame's own keypoints, not dependent on any
absolute calibration).

`tests/test_zone_runner.py` covers: two cameras with fully overlapping
views of two co-located members produce exactly as many reps as a
single-camera equivalent run (proving no double-counting); a member
occluded from one camera's angle for a stretch of ticks still gets
correctly attributed reps via the other camera that still sees them; and
a lone member in the zone with multiple cameras watching still only ever
gets fed by one camera per tick. `tests/test_rep_session_camera_calibration.py`
covers the per-camera-calibration fix directly: two cameras with
different pixel scales for the same physical plate each get their own
`CameraCalibration`; a set that switches from camera A to camera B
mid-set produces a bar-path displacement matching each camera's own
calibration for the samples it produced (not camera A's scale misapplied
to camera B's pixels); `camera_tilt_deg_by_camera` overrides thread
through correctly; and the single-camera case (`camera_id=None`
throughout) is unaffected.

**Multi-view 3D pose triangulation (`irix.pose.multiview`).** Everything
above solves *identity* across overlapping cameras (which physical
person is which wristband) and, per tick, still only feeds `RepSession`
one camera's single 2D-pixel pose. `irix.pose.multiview.triangulate_pose`
goes one step further: once 2+ cameras are already known (via that same
identity resolution) to be looking at the same physical person, it
DLT-triangulates (`triangulate_point`, the standard direct-linear-
transform multi-view geometry solution, Hartley & Zisserman ch. 12) each
of the 17 COCO keypoints independently from whichever subset of cameras
currently sees it above a confidence threshold, into one fused 3D
`PersonPose`. `CameraProjection` is a standard calibrated pinhole camera
model (intrinsic `K` + extrinsic `R`/`t` within one shared zone-wide
coordinate frame) -- deliberately a different thing from
`irix.barbell.calibration.CameraCalibration`'s scalar px-per-mm
conversion, which knows nothing about where a camera actually sits in 3D
space. A real deployment derives `R`/`t` once per camera at install time
via a standard extrinsic calibration procedure (checkerboard/ArUco
marker visible to 2+ cameras, solved with OpenCV's
`stereoCalibrate`/`solvePnP`) -- out of scope for this module, which
assumes those numbers are already known.

**Why this matters for rep counting, not just "more accurate in
general."** `RepCounter` counts reps off a joint angle computed from a
single camera's 2D pixel keypoints -- a *projection* of the true 3D
angle onto that one camera's image plane, which foreshortens whenever
the limb isn't moving parallel to that camera's image plane and can be
wrong entirely when a keypoint is self-occluded from that particular
angle (e.g. a barbell blocking the hip at a squat's bottom, exactly the
position that matters most for rep detection). `RepSession.process_frame`
now prefers a triangulated 3D joint angle over the 2D one whenever all 3
of an exercise's needed keypoints (`irix.rep_counting.exercises.
EXERCISES[...].joint_triplet`) triangulated that tick (`PersonPose.xyz`,
via a new optional `z` field on `Keypoint` -- `None` for an ordinary 2D
pose, so this is purely additive and doesn't change single-camera
behavior at all), falling back to the 2D angle otherwise.

**Wiring: opt-in via `MultiCameraZoneRunner(camera_projections=...)`.**
Without it (the default), this runner's behavior is byte-for-byte
unchanged from before multi-view fusion existed. With it, both tick
paths gather *every* camera that resolved a pose for a given member that
tick (not just the priority-winner used for weight/barbell detection --
see `MultiCameraZoneRunner.tick`'s `all_routed_this_tick`/`poses_by_camera`
collections) and triangulate across them; the fused pose (when
triangulation succeeds) is what gets passed into `RepSession.
process_frame`, while the single priority-winning camera's raw frame
still feeds weight/barbell detection unchanged.

`tests/test_multiview.py` covers the triangulation math directly against
synthetic calibrated cameras (recovers a known 3D point exactly from 2 or
3 views; returns `None` for a single view; per-keypoint view-count/
confidence gating; unusable/missing-projection cameras are simply
ignored, never crash). `tests/test_rep_session_multiview.py` proves
`RepSession` actually prefers the 3D angle when available and falls back
correctly when only some of a joint triplet's keypoints triangulated.
`tests/test_zone_runner_multiview.py` is the end-to-end proof: a
synthetic bicep-curl motion whose forearm swings through a camera's
*depth* axis is fed through `MultiCameraZoneRunner` two ways -- one
camera alone (deliberately positioned so its 2D pixel reading is
degenerate: shoulder/elbow/wrist all share world x=0, so that camera's
projected keypoints always land on the same pixel column, making its 2D
"angle" reading meaningless) vs. both cameras with `camera_projections`
configured. The fused run's rep count matches a ground-truth `RepCounter`
run fed the true 3D angle directly; the single-camera run's does not.

## End-to-end demo

Two demo entrypoints, covering the two things worth demonstrating
separately: one station in depth, vs. what only shows up with several.

`irix/demo/run_demo.py` wires pose -> rep counting -> structured events ->
pipeline into one loop for a *single* station, in two modes:

- `--mock-pose`: synthetic joint-angle stream, no camera or model weights
  needed. This is what the test suite and a from-scratch clone can run
  immediately.
- `--source <index|path>`: real webcam or video file through
  `PoseEstimator` (requires `pip install irix[pose]`, which pulls in
  `ultralytics`/torch).

`irix/demo/run_gym_demo.py` (`python -m irix.demo.run_gym_demo`) is the
multi-station companion -- see "Multi-station deployment" above for what
it demonstrates and why it exists as a separate entrypoint rather than
more flags on `run_demo.py`: station handoff/anti-double-counting,
camera+IMU rep-count fusion (both the agreement and the occlusion-fallback
path), set + session fatigue analysis, and the weight-recognition
geometry cross-check, all in one runnable, deterministic trace across two
members and three stations.

`irix/demo/run_upload.py` is a third entrypoint, and a real gap it closes:
neither of the two above takes an already-*recorded* video and wristband
file and runs the real (non-mock) versions of every module against them.
`run_demo.py --source` only ever wires pose -> rep -> form (no IMU
fusion, weight recognition, or barbell velocity -- those only ever ran
against `run_gym_demo.py`'s synthetic streams before this), and there was
no code anywhere that parsed a real wristband export into `IMUSample`s at
all. `run_upload.py` (see its own module docstring for the full picture)
fixes both: `irix/fusion/imu_io.py` loads a real recorded wristband
CSV/JSON export, and `run_upload()` wires pose -> rep -> form -> (if an
IMU file was given) `RepCountFusion` -> a `RestGapSetBoundaryDetector`
segmenting the continuous rep stream into sets (nothing hand-scripts set
length here, unlike the mock demos) -> `SetFatigueAnalyzer`/
`SessionFatigueTracker`, plus periodic weight recognition
(`VisionPlateClassifier`, if a `VLMBackend` is supplied) and, if a real
barbell-detector checkpoint is supplied (none is bundled -- see "Model
weights" above), calibrated bar velocity/RPE upgrading the deg/s proxy.
Output is the full `CameraEvent` JSON stream -- exactly the payload
`irix-mvp-app`'s AI needs, per `irix/pipeline/schema.py`'s module
docstring.

One correctness detail worth calling out: `run_upload` derives frame
timestamps from the video's own frame index and fps
(`frame_index / fps`), not wall-clock processing time. `run_live`
(`run_demo.py --source`) uses `time.monotonic()` instead, which is
correct there because a live camera's frame arrival time *is* wall-clock
time -- but for an uploaded file, how fast this machine happens to
process each frame has nothing to do with the footage's actual timeline,
and using wall-clock time here would silently misalign a real uploaded
IMU file's timestamps against the video.

## Software wristband + BLE gateway simulator (2026-07-14)

Every piece of "Live station readiness" above (`StationSessionRunner`,
`GymSessionRunner`) had real, tested logic behind it, but only ever run
against hand-built fakes inside unit tests -- no demo drove the actual
tick-loop live orchestration end to end the way a real deployment would.
That's a real gap against this project's stated final goal ("simulate or
connect multiple wristbands," "recover from BLE disconnects"), and a
different problem from the one `irix.fusion.imu_stream.LiveBLEIMUStream`
deliberately leaves unimplemented: that stub is unimplemented because
*real* BLE hardware/firmware protocol choice can't be guessed at
correctly from a software scaffold. A *simulator* has no such excuse --
it exists precisely to stand in for hardware that doesn't exist yet, so
this was worth building now rather than deferring further.

`irix/wristband_sim/simulator.py` adds two classes:

- `SimulatedWristband` -- one simulated physical band: a `station_id`
  (ground truth of where its wearer physically is, set directly by a
  test/demo script, never exposed to the pipeline being tested) and a
  continuous IMU generator (`advance(dt) -> List[IMUSample]`) with a
  settable motion program (`"idle"`: gravity + a fixed, known bias +
  noise; `"reps"`: the same oscillating-vertical-accel model
  `irix.demo.mock_pose.synthetic_imu_stream` already uses, so downstream
  consumers see consistent statistics regardless of source). The bias is
  deliberately nonzero and fixed per instance so
  `irix.wristband_sim.calibration.calibrate_stationary` has something
  real to recover, rather than trivially calibrating against a
  already-zero-bias signal.
- `SimulatedBLEGateway` -- owns N wristbands, ticked once per gym-wide
  loop iteration (`tick(now)`). Produces `irix.identity.ble_pairing.
  BLEReading`s (RSSI + Gaussian noise, station-appropriate) via
  `ble_reader()` -- the exact callable shape `GymSessionRunner`'s/
  `StationSessionRunner`'s `ble_reader` constructor arg expects -- and
  buffers each present band's IMU samples, drained through
  `SimulatedBLEIMUStream` (implements `irix.fusion.imu_stream.IMUStream`)
  via `imu_stream_factory(wristband_id)`. `packet_loss_pct` independently
  drops some fraction of both BLE readings and IMU packets per tick (a
  real radio genuinely loses some of each, transmitted on separate
  channels/characteristics). `disconnect(wristband_id, ticks)` schedules
  a scheduled total dropout -- no BLE reading, no IMU samples -- for
  exercising the disconnect-survives-past-`presence_timeout_s` grace
  period this project's final goal explicitly calls for, against the
  real live pipeline rather than asserting the reconnect logic exists in
  isolation.

  One implementation detail worth being explicit about: whether a
  disconnected tick is skipped in `ble_reader()` is tracked via a
  `_disconnected_this_tick` set computed once in `tick()`, not by
  re-checking the (already-decremented) countdown dict from inside
  `ble_reader()` -- checking the post-decrement value directly would
  under-count the disconnect by one tick (the countdown reaches zero on
  the *last* disconnected tick, at which point a naive `> 0` check would
  already read false). Caught by
  `tests/test_wristband_simulator.py::test_gateway_disconnect_drops_ble_and_imu_for_scheduled_ticks`,
  which asserts both ticks of a 2-tick disconnect are actually dropped,
  not just the first.

`irix/wristband_sim/calibration.py` adds
`calibrate_stationary(samples) -> IMUCalibration`: standard strapdown-IMU
static calibration (gyro bias = mean gyro during a stationary period,
since true angular velocity is exactly zero; accel bias = mean accel
minus expected gravity along whichever axis is "up") and
`apply_calibration`/`apply_calibration_batch` to subtract it back out.
Deliberately just bias, not a full multi-orientation scale-factor/
cross-axis-misalignment calibration (which needs a turntable or several
known orientations) -- unnecessary precision for a wrist-worn
consumer-grade IMU doing rep counting rather than dead-reckoning
navigation, where the practically visible failure mode (a stationary
band's gyro integrating into phantom motion) is fixed by bias correction
alone.

`irix/demo/synthetic_live.py` adds the last piece needed to actually
drive `StationSessionRunner.tick()`/`GymSessionRunner.run_forever()` with
synthetic data: `SyntheticFrameSource` (a `.frames()`/`.close()`
drop-in for `ReconnectingFrameSource`, yielding placeholder arrays --
real pixel content is only needed if whatever's paired with it as
`pose_estimator` actually looks at it) and `SyntheticPoseEstimator`
(ignores the frame it's given, instead replaying a pre-generated
`irix.demo.mock_pose.synthetic_pose_stream` sequence, looping once
exhausted since a live station keeps ticking past any fixed-length demo
sequence).

`irix/demo/run_live_gym_demo.py` (`python -m irix.demo.run_live_gym_demo`)
wires all of the above into the first demo that runs the real
`StationSessionRunner`/`GymSessionRunner` classes rather than
`run_gym_demo.py`'s pattern of calling rep-counting/fusion code directly
against synthetic streams: two members, two adjacent stations
(`squat-1`/`squat-2`), checked out via a real `CheckoutRegistry`, a
scripted timeline (Alice starts a set at squat-1; a short BLE disconnect
that's shorter than `presence_timeout_s` and should be survived
transparently; Bob walks up to squat-2, does a set, and leaves; Alice
then walks from squat-1 to squat-2, producing a real
`StationHandoffEvent` through `GymCoordinator`), and every event pushed
through `irix.pipeline.edge_buffer.LocalBuffer` ->
`irix.pipeline.aggregator.Aggregator` ->
`irix.pipeline.cloud_sync.InMemoryCloudSync` -- the mock-backend delivery
path a real deployment's `HTTPCloudSync` would replace.
`tests/test_run_live_gym_demo.py` asserts the run actually produces every
claimed event type (rep completion, set completion, fatigue summary, and
the one expected station handoff with the correct member/from/to), not
just that it executes without raising.

**What this does and doesn't close.** This makes the live orchestration
path (presence resolution, session lifecycle, handoff, disconnect
recovery, mock-backend delivery) genuinely exercised end to end for the
first time, in CI, without hardware. It does not implement
`irix.fusion.imu_stream.LiveBLEIMUStream` itself -- that remains correctly
unimplemented, for the same hardware-dependency reason stated throughout
this document. What changes is that once real wristband hardware and its
BLE protocol exist, wiring in a real `LiveBLEIMUStream` subclass and a
real `ble_reader` is the *only* new work required -- everything else in
the live path this simulator now exercises stays as-is.

## Phase 2: tracking accuracy, exercise recognition, load detection, calibration, benchmarking (2026-07-14)

Phase 1 built the software scaffold and validated it end-to-end
(simulator, live orchestration, full docs). Phase 2's mandate was
different: stop building infrastructure, spend real research time per
subsystem, and improve production accuracy/robustness of the actual
tracking system. Summary of what changed and why -- full reasoning lives
in each module's own docstring; this is the index.

- **`irix.pose.tracker`** -- intra-camera multi-person tracking.
  `PersonPose.track_id` was documented (Phase 1) as "just that frame's
  list index," not a stable identity. Researched SORT/ByteTrack/BoT-SORT;
  adopted ByteTrack's two-stage high/low-confidence association (Zhang et
  al., ECCV 2022) on top of a SORT-style constant-velocity Kalman filter
  (Bewley et al., 2016), explicitly *not* adopting BoT-SORT's deep ReID
  embedding -- unnecessary at gym-station scale (1-4 people, static
  camera). `TrackedPoseEstimator` makes this an opt-in drop-in wrapper
  around any pose estimator; not yet the default (see `docs/TODO.md`).

- **`irix.exercise_recognition`** -- previously nonexistent; exercise
  was (and still is, at session-start) only ever configured per station,
  never classified. Researched ST-GCN/temporal-attention-GCN/pose
  transformers and the one public camera+IMU fitness dataset (MM-Fit,
  IMWUT 2020) built for exactly this problem shape -- and concluded
  training a sequence model in this sandboxed, GPU-less environment
  would produce an undertrained model worse than an honest baseline, not
  better. Built a zero-training range-of-motion + periodicity classifier
  instead, scored per candidate `ExerciseConfig`, with an explicit,
  tested handling of the real structural ambiguity (squat/leg_press/
  hack_squat share a joint triplet -- reported as `unknown`, not guessed).

- **`irix.fusion.clock_sync`** -- camera/wristband clock offset + drift
  estimation, previously entirely unmodeled (every fusion module assumed
  a shared clock). Researched VINS-Fusion's online temporal calibration
  and BLE's own clock-accuracy spec (±20 ppm main clock, ±250 ppm sleep
  clock -- real numbers, cited in the module). Implemented cross-
  correlation-based offset estimation plus a weighted linear drift fit,
  validated against `irix.wristband_sim`'s new `clock_drift_ppm`
  simulation (recovers a configured 180 ppm drift to within 5%). Not yet
  wired into a live entry point (see `docs/TODO.md`).

- **`RepCountFusion` packet-loss awareness** -- `imu_confidence` is now
  discounted by measured sample completeness (actual vs. expected sample
  count at the configured rate) before being compared against camera
  confidence, so a fusion decision under heavy BLE packet loss correctly
  leans back toward the camera instead of trusting a confidently-computed
  count from a visibly incomplete signal.

- **`irix.weight_recognition.plate_color_check`** -- classical HSV
  color-blob detection against the IWF bumper-plate color standard
  (confirmed via search: 10/15/20/25 kg = green/yellow/blue/red).
  Independent of the still-untrained `FreeWeightDetector`; works today
  for color-coded bumper plates, correctly finds nothing on
  non-standard-colored gym iron (by design, not a gap). Never fabricates
  a total: an odd plate count or no confident detections both return
  `total_weight_kg=None` with a reason.

- **`irix.pose.calibration`** -- the actual checkerboard intrinsic
  (`cv2.calibrateCamera`) + extrinsic (`cv2.solvePnP`) calibration
  workflow that `irix.pose.multiview.CameraProjection` and
  `irix.barbell.calibration.undistort_frame` both previously assumed was
  "out of scope here." Reprojection-error-based quality reporting,
  `CalibrationProfile` save/load, plus a lighter-weight ground-plane
  homography path for single-camera zone/position mapping.

- **`irix.benchmark`** -- real timing/throughput measurements (pose
  tracker, exercise recognition, fusion, EKF, clock sync, full simulated
  live-pipeline throughput, camera-reconnect schedule, BLE-disconnect-
  recovery margin, CPU/memory via stdlib `resource`) using only this
  repo's existing dependencies. GPU utilization and real pose-inference
  FPS auto-detect `ultralytics`/CUDA availability and report `None` with
  a clear reason in this sandboxed environment rather than fabricate a
  number -- they'll run for real the moment they're available.

- **Deterministic-replay bug found and fixed.** Writing a byte-for-byte
  event-replay test (`tests/test_run_live_gym_demo.py`) surfaced a real,
  pre-existing correctness bug: `RepCompletedEvent`/`SetCompleteEvent`/
  `SetFatigueSummaryEvent`/`WeightConfirmedEvent`/
  `BandPlacementRequiredEvent` were all constructed without an explicit
  `timestamp=`, silently falling back to `schema.py`'s dataclass default
  (`time.monotonic()`, real wall-clock time) regardless of whatever
  deterministic clock a caller had injected -- only `StationHandoffEvent`
  got this right before now. Fixed by threading the actual
  event-relevant timestamp (a rep's own detected time, a set's close
  time, a frame's own timestamp, a session's `start_ts`) through every
  construction site in `irix.pipeline.rep_session`/`irix.pipeline.events`.
  This matters beyond just this phase's new tests: `run_upload.py` and
  every live entry point were producing events with non-reproducible
  timestamps before this fix, undermining any validation/benchmark run
  that assumed replay determinism.

- **Simulation expanded**: `irix.wristband_sim.simulator.SimulatedWristband`
  gained `clock_drift_ppm` (a band's onboard clock now genuinely diverges
  from true elapsed time at a configurable rate, matching real crystal-
  oscillator behavior); a new multi-member stress test
  (`tests/test_stress_multi_member.py`) verifies 8 concurrent members
  across 4 stations, including a late-joining group, never cross-
  attribute events; deterministic replay is now verified at the full
  serialized-event-payload level, not just event type/count.

**What Phase 2 deliberately did not do**, and why: no trained deep model
for exercise/action recognition (see above -- would need real GPU
training infra and labeled data this environment doesn't have); no
deep-ReID tracking upgrade (unnecessary at this scale, see
`irix/pose/tracker.py`); no Docker/Jetson deployment work (explicitly
out of scope for this phase per the founding brief -- "the objective is
no longer to improve infrastructure").

## Phase 3: production pipeline integration (2026-07-14, complete except Priority 13)

Phase 2 built and unit-tested individual accuracy modules in isolation.
Phase 3's objective is different: stop adding isolated modules and
integrate everything into the one production pipeline every live/replay/
demo entry point actually runs, so accuracy improvements are default
behavior, not opt-in extras nobody's wiring up sees.

**Default production wiring** (`irix.live.station_runner.
StationSessionRunner`, `irix.pipeline.rep_session.RepSession`):
`_ensure_estimator()` now wraps a real default `PoseEstimator` in
`irix.pose.tracker.TrackedPoseEstimator` (persistent track_id) unless a
caller injects its own (every test/demo unaffected); one
`irix.fusion.clock_sync.ClockSyncEstimator` per open session applies its
current correction to every `add_imu_samples` call (an explicit
calibration step, `calibrate_wristband_clock`, is the real entry point --
see below for why this repo does *not* auto-derive observations from
per-set rep timestamps); `irix.pose.calibration.CalibrationProfile.
undistort_frame` runs before pose estimation when a station's calibration
is configured; `irix.weight_recognition.plate_color_check` runs on every
weight check regardless of VLM configuration (`method="color_plate"`
when no VLM backend exists, `"vlm"` cross-checked against it otherwise).

**A real mistake, caught by its own tests, and reverted**: the first
attempt at closing `docs/TODO.md`'s "wire `ClockSyncEstimator` into a
live entry point" gap tried auto-deriving clock-offset observations by
pairing each set's camera-detected rep-completion timestamps against
`RepCountFusion`'s IMU-derived peak timestamps
(`estimate_offset_from_paired_events`, new this phase). Development
tests (`tests/test_rep_session_clock_sync.py`) caught that this
systematically produces a *wrong* offset, not just a noisy one: a
camera's rep-completion timestamp and an IMU counter's acceleration-peak
timestamp mark different phases of one physical rep (top-of-lift vs.
peak concentric acceleration), so the pairing conflates a fixed phase
offset with genuine clock drift. This was reverted -- `RepSession` no
longer auto-populates observations at all; `estimate_offset_from_paired_
events` is kept as a general utility (correct for genuinely comparable
event pairs) with an explicit warning against this exact misuse. Worth
recording here rather than just in a commit message: this is the kind of
mistake "unknown over incorrect" is meant to catch, and it worked.

**Wristband placement state machine** (`irix.identity.placement.
WristbandPlacementTracker`, Priority 4): a real-time `STABLE -> SETTLING
-> CALIBRATING -> STABLE` state machine for a band's actual worn side
(`BandSide`: `left_wrist`/`right_wrist`/`left_ankle`/`right_ankle`/
`unknown`) -- distinct from `ExerciseConfig.band_placement`'s static
wrist-vs-ankle *requirement*. Fastening-motion samples are discarded by
a sliding settle window; once genuinely quiet and gravity-consistent
(magnitude check, not just low variance -- rules out e.g. free-fall
"quiet" readings), the settled window's own data estimates which local
axis is "up" (rather than assuming a fixed convention, which would be
wrong across different physical orientations) and
`irix.wristband_sim.calibration.calibrate_stationary` recalibrates for
the new placement. `RepSession` withholds IMU samples from fusion
entirely while paused (mid-change) or while the confirmed side's limb
type doesn't match the exercise's requirement -- "never reuse wrist
thresholds for ankle data or vice versa" made concrete, not just stated.
Also corrected a real bug surfaced while wiring this: `HACK_SQUAT` was
configured `ANKLE`, which contradicts the founding brief's explicit
per-exercise placement guidance (camera-primary, wrist-band, like
`SQUAT`) -- fixed, and the seven exercises that guidance names but
didn't exist yet (`LUNGE`, `BULGARIAN_SPLIT_SQUAT`, `CALF_RAISE`,
`LEG_EXTENSION`, `LEG_CURL`, `HIP_ABDUCTION`, `HIP_ADDUCTION`) were added.

**Identity association fusion** (Priority 5): `StationSessionRunner.tick`
now feeds `irix.live.disambiguation.CrowdedGroupDisambiguator`
clock-synchronized IMU (not raw) and withholds a band's samples while
its placement tracker reports mid-change (fastening motion is not the
wearer's body-motion signal). `irix.identity.motion_correlation.
MotionCorrelationResolver` gained a `prior_slot_assignment` continuity
hint -- a small ranking bonus toward whichever member correlated to a
detected-person slot last window, enough to break a genuine near-tie
without overruling a clearly different result; `CrowdedGroupDisambiguator`
remembers its last resolved assignment across group changes/resets and
threads it through. New `irix.identity.resolution.IdentityResolution`
(member_id + confidence + ambiguous + evidence) consolidates which of
the founding brief's named identity-fusion signals already have a real
source in this repo (documented in that module's docstring) vs. what's
still a real gap (motion onset as its own distinct signal; wiring
`IdentityResolution` into the live path itself, deliberately deferred
until the workout state machine below exists to consume it).

**Workout state machine** (`irix.pipeline.workout_state.
WorkoutStateMachine`, Priority 6): models the founding brief's 19 named
states as one ordered `WorkoutPhase` lifecycle plus independent
`WorkoutHealth` flags (see that module's docstring for why a single flat
state enum would have been the wrong model -- some states repeat many
times within another, some are conditions rather than steps). Enforces
by construction: no duplicate sessions, no late packets reopening a
completed set, no duplicate reps, a mechanism for preventing
camera-overlap double counting. Owned by `irix.live.gym_runner.
GymSessionRunner`, deliberately not any single `StationSessionRunner` --
a station only ever sees one slice of a member's gym-wide visit, and
`GymSessionRunner` is already the layer that knows when a band first
appears gym-wide and when a real cross-station handoff happens (see both
modules' docstrings). Five new event types
(`TrackingLostEvent`/`TrackingRecoveredEvent`/`RestStartedEvent`/
`RestEndedEvent`/`ExerciseDetectedEvent`) close `docs/TODO.md`'s
previously-open "scope and add missing event types" item; only the
tracking-loss pair is wired to a real emission source so far (a
consecutive-missed-frame streak in `StationSessionRunner`'s
single-candidate path) -- see `docs/TODO.md` for what's left.

**Priorities 7-12, completed the same day**: unified load detection
(Priority 7) -- `RepSession`'s weight-check block now runs color-plate
detection unconditionally (`method="color_plate"`), cross-checked
against a VLM read when configured, with confidence/evidence/units/
method/status on every `WeightConfirmedEvent`, never a fabricated
weight. Session-recording/data-collection tooling (Priority 8) --
`irix.recording.session_recorder.SessionRecorder`, deterministic replay
via `load_recorded_session`, `save_raw_frames=False` by default to stay
consistent with the production pipeline's "never raw video" principle.
Identity-resolution and event-emission latency benchmarks (Priority 9)
added to `irix.benchmark.run_benchmarks`. An external per-gym YAML/JSON
configuration system (Priority 10, `irix.config.gym_config`) moving
every gym-specific assumption (station layout, thresholds, equipment)
out of hardcoded Python, deliberately excluding hardware bindings. The
IRIX Studio backend interface (Priority 11, `irix.backend.studio_api.
StudioBackendAPI`) -- assign/return wristband, start/end session, query
battery/assignment/status, all backed by real `CheckoutRegistry`/
`GymSessionRunner` state; `query_battery` honestly reports `"unknown"`
since no battery signal exists anywhere in this repo. Every
`CameraEvent` now carries `EVENT_SCHEMA_VERSION`. Validation expansion
and a report generator (Priority 12, `irix.validation.
report_generator`) -- a real subprocess-`pytest` run plus the benchmark
suite, dated Markdown/JSON output, no fabricated pass counts; building
its integration test (`tests/test_config_driven_live_pipeline.py`, the
config system run together with the live orchestration layer for the
first time) caught and fixed a real bug: `camera_tilt_deg` had been
threaded into `station_runner_kwargs_for` since Priority 10 but
`StationSessionRunner.__init__` had no matching parameter, silently
dropping it -- same class of gap as the earlier `bar_weight_kg` fix.

All twelve of the founding brief's numbered integration priorities are
now real. What remains is genuinely hardware- and credential-gated:
Priority 13 (verified GitHub push) is blocked on this sandbox having no
SSH key or HTTPS credential helper -- reported honestly on every attempt,
never claimed to succeed when it didn't; see `docs/TODO.md`'s
lower-priority section for the remaining real-hardware-dependent gaps
(barbell/plate detector training data, real camera/wristband hardware
validation, real edge-device latency benchmarking).
