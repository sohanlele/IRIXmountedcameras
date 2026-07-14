"""Per-exercise joint-angle configuration (Section 4.2).

The joint-angle state machine is hand-configured per exercise: which joint,
and which angle range counts as a rep. This module holds that configuration
so ``RepCounter`` stays exercise-agnostic. Section 4.7 covers a
class-agnostic fallback (RepNet/TransRAC-style) for exercises that don't
have a config here yet -- out of scope for this scaffold, but the interface
is designed so a future ``RepCounter`` could fall back to one.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BandPlacement(Enum):
    """Where the member should be wearing the IMU band for this exercise.

    Section 4.6 fusion (and the ``RecoFitCounter``/``ULiftCounter``
    wristband-IMU counters in ``irix.fusion.imu_rep_counting``) only add
    signal when the band is on the limb that's rigidly coupled to the
    load. The design doc's default assumption is the wrist -- true for
    curls, presses, rows, bench, deadlift -- but explicitly calls out
    machine leg exercises (leg press, hack squat) as a case where "the
    wrist doesn't move with the load, so the IMU contributes little
    there." On those machines the *foot* is the rigid contact point with
    the load (the footplate), the same relationship the wrist has to a
    curl -- so wearing the band on the ankle instead restores a real
    fusion signal for exactly those exercises. It does not help free-
    weight squats: feet stay planted on the ground there, so an ankle
    band sees almost no motion; the barbell is what's moving, tracked by
    the camera (Section 4.5), not the ankle.
    """

    WRIST = "wrist"
    ANKLE = "ankle"


@dataclass(frozen=True)
class ExerciseConfig:
    name: str
    # Three keypoint names defining the tracked joint angle (a, vertex, c).
    joint_triplet: tuple  # (str, str, str)
    # Angle (degrees) considered fully "bottom" of the rep (max flexion).
    bottom_angle: float
    # Angle (degrees) considered fully "top" of the rep (max extension).
    top_angle: float
    # Hysteresis band (degrees) to avoid double-counting on sensor noise.
    hysteresis: float = 8.0
    # Where the IMU band should be worn for this exercise's IMU signal
    # (wristband fallback / fusion counters) to be trustworthy.
    band_placement: BandPlacement = BandPlacement.WRIST


# Design doc examples (Section 4.2): "knee for squats, elbow for curls,
# hip for deadlifts".
SQUAT = ExerciseConfig(
    name="squat",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=90.0,
    top_angle=170.0,
    # Free-weight squat: feet stay planted, wrist doesn't track the bar
    # either. Camera-only for velocity (Section 4.5); band_placement is
    # nominal here since neither wrist nor ankle IMU adds much signal.
    band_placement=BandPlacement.WRIST,
)

BICEP_CURL = ExerciseConfig(
    name="bicep_curl",
    joint_triplet=("left_shoulder", "left_elbow", "left_wrist"),
    bottom_angle=160.0,  # arm extended
    top_angle=40.0,      # fully curled
    band_placement=BandPlacement.WRIST,
)

DEADLIFT = ExerciseConfig(
    name="deadlift",
    joint_triplet=("left_shoulder", "left_hip", "left_knee"),
    bottom_angle=80.0,   # hip hinged forward
    top_angle=170.0,     # lockout
    band_placement=BandPlacement.WRIST,
)

BENCH_PRESS = ExerciseConfig(
    name="bench_press",
    joint_triplet=("left_shoulder", "left_elbow", "left_wrist"),
    bottom_angle=75.0,   # bar at chest
    top_angle=165.0,     # lockout
    band_placement=BandPlacement.WRIST,
)

# Machines zone (Section 3.2): the foot is the rigid contact point with
# the load on these, so the IMU band moves to the ankle.
LEG_PRESS = ExerciseConfig(
    name="leg_press",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=90.0,
    top_angle=170.0,
    band_placement=BandPlacement.ANKLE,
)

HACK_SQUAT = ExerciseConfig(
    name="hack_squat",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=95.0,
    top_angle=175.0,
    # Phase 3 correction: previously ANKLE here, which was wrong. Unlike
    # leg_press, a hack squat machine's foot plate is fixed but the
    # member's own bodyweight/torso still moves through real vertical
    # travel the way a free-weight squat's does -- the camera is the
    # primary lower-body kinematics source for this one (see
    # irix.identity.placement's module docstring / docs/WRISTBAND_SYSTEM.md
    # for the full wrist-vs-ankle placement rationale by exercise).
    band_placement=BandPlacement.WRIST,
)

# Squat-pattern variants (Phase 3) -- same hip-knee-ankle joint triplet as
# SQUAT/HACK_SQUAT and the same camera-primary, wrist-band rationale:
# feet stay in continuous ground contact (lunges/split squats plant both
# feet; calf raises never leave the floor at all), so an ankle IMU adds
# little the camera doesn't already see, unlike the true machine-footplate
# exercises below.
LUNGE = ExerciseConfig(
    name="lunge",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=90.0,   # front knee ~90 deg at the bottom of the step
    top_angle=170.0,     # standing back up
    band_placement=BandPlacement.WRIST,
)

BULGARIAN_SPLIT_SQUAT = ExerciseConfig(
    name="bulgarian_split_squat",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=80.0,   # rear foot elevated -> deeper front-knee flexion than a standard lunge
    top_angle=170.0,
    band_placement=BandPlacement.WRIST,
)

CALF_RAISE = ExerciseConfig(
    name="calf_raise",
    # Honest limitation, not a placeholder: COCO-17 (irix.pose.estimator's
    # COCO_KEYPOINT_NAMES) has no toe/heel/foot-index keypoint, so there's
    # no joint *below* the ankle to form a real ankle-plantarflexion angle
    # from -- the actual moving joint for this exercise. hip-knee-ankle is
    # the best available proxy (the whole leg tilts a few degrees as the
    # heel rises), but it's a low-sensitivity signal, not a validated one.
    # Matches this exercise's design intent either way: the camera is
    # already stated as secondary here for rep-counting *precision* (see
    # band_placement's docstring group comment above) -- the wrist IMU's
    # role is periodicity/tempo/set-boundaries, not primary rep counting,
    # so neither signal needs to be precise on its own for this exercise.
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=165.0,  # standing flat
    top_angle=175.0,     # up on toes -- narrow band, small real ROM
    hysteresis=3.0,       # tighter than the 8-degree default -- the whole range is only ~10 degrees
    band_placement=BandPlacement.WRIST,
)

# True machine-footplate exercises (Phase 3) -- the foot is the rigid
# contact point with the resisted load here, the same relationship the
# wrist has to a curl, so the IMU band moves to the ankle for a real
# fusion signal (see BandPlacement's module docstring).
LEG_EXTENSION = ExerciseConfig(
    name="leg_extension",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=90.0,   # knee bent under the seat pad
    top_angle=175.0,     # leg extended straight out
    band_placement=BandPlacement.ANKLE,
)

LEG_CURL = ExerciseConfig(
    name="leg_curl",
    # Inverted pattern like BICEP_CURL: "bottom" (start) is the extended,
    # high-angle position; "top" (peak contraction) is the flexed,
    # low-angle position.
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=170.0,  # leg extended, start position
    top_angle=45.0,      # knee curled toward the glute
    band_placement=BandPlacement.ANKLE,
)

# Frontal-plane (lateral) movements -- weaker fit for a single sagittal-
# view joint-angle threshold than the exercises above (irix.pose.geometry.
# joint_angle doesn't know or constrain camera viewing plane), and these
# ROM values are a first-pass estimate, not validated against real
# footage (see docs/VALIDATION.md's ground-truth gap). Ankle placement
# still applies -- the foot/footplate is still the rigid contact point --
# and is the more load-bearing signal for these two specifically, exactly
# because the camera-side angle read is the weaker of the two here.
HIP_ABDUCTION = ExerciseConfig(
    name="hip_abduction",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=160.0,  # legs together, start position
    top_angle=130.0,     # leg abducted outward against the pad
    band_placement=BandPlacement.ANKLE,
)

HIP_ADDUCTION = ExerciseConfig(
    name="hip_adduction",
    # Inverse ROM of HIP_ABDUCTION: starts abducted, "top" is legs together.
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=130.0,
    top_angle=160.0,
    band_placement=BandPlacement.ANKLE,
)

EXERCISES = {
    c.name: c for c in (
        SQUAT, BICEP_CURL, DEADLIFT, BENCH_PRESS, LEG_PRESS, HACK_SQUAT,
        LUNGE, BULGARIAN_SPLIT_SQUAT, CALF_RAISE,
        LEG_EXTENSION, LEG_CURL, HIP_ABDUCTION, HIP_ADDUCTION,
    )
}
