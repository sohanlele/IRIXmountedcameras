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
| 4.3 Multi-camera fusion & occlusion | -- | Not implemented; `PoseEstimator` returns single-view poses per camera, multi-view reprojection is future work |
| 4.4 Weight & plate recognition | `irix/weight_recognition/` | VLM-based classifier (`vision_classifier.py`) is the deployable path -- see below for why QR stickers and OCR were both ruled out; `confirmation.py` adds N-of-M read-confirmation windowing |
| 4.5 Bar path & velocity tracking | `irix/barbell/` | Self-calibrated (no environment edits) barbell/plate/dumbbell detection, real-unit (m/s) bar-path velocity, and RPE/fatigue estimation -- see "Barbell and dumbbell tracking" below |
| 4.6 Visual-inertial sensor fusion | `irix/fusion/` | EKF (position/velocity state) + ZUPT dead-stop correction; `imu_rep_counting.py` adds two literature IMU-only rep counters (see below) |
| 5.1 BLE identity linking | `irix/identity/` | RSSI-based station-resolution heuristic (not a BLE radio stack) |
| 5.4 Personalization data flow | -- | Not implemented; would live alongside `irix/pipeline` as a profile-pull step |
| 6.3 Data flow (edge -> aggregator -> cloud) | `irix/pipeline/` | `LocalBuffer` -> `Aggregator` -> `CloudSync`, structured `CameraEvent` family (`RepCompletedEvent`, `SetCompleteEvent`, `BandPlacementRequiredEvent`, `WeightConfirmedEvent`) |
| 7 / 7.1 Real-time audio coaching | -- (owned by irix-mvp-app) | Out of scope for this repo -- see "Where this repo ends" above. `BandPlacementTracker` emits the one coaching-adjacent *event*, but not the instruction text itself |
| 8 Privacy & data handling | `irix/pipeline/schema.py` | Every `CameraEvent` subtype intentionally carries no video/biometric fields (tested) |

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
