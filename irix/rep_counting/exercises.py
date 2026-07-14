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
    band_placement=BandPlacement.ANKLE,
)

EXERCISES = {c.name: c for c in (SQUAT, BICEP_CURL, DEADLIFT, BENCH_PRESS, LEG_PRESS, HACK_SQUAT)}
