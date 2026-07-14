# Architecture map

This repo implements the pure-software layers of the IRIX Technical
Architecture & Design Document (mounted-camera + wristband form factor).
It is a **software scaffold**: runnable module structure and unit-tested
logic for the algorithms the doc specifies, not a production build. It does
not include camera/network hardware, wristband firmware, Jetson deployment
configs, or trained model weights.

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
| 4.1 Pose estimation model | `irix/pose/` | YOLO-Pose wrapper (`ultralytics`, optional dep); joint-angle geometry helper |
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
percentage below the first rep's velocity). That analysis is entirely the
app's job -- this repo's job is just to supply accurate, per-rep numbers
for it to work with.

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
