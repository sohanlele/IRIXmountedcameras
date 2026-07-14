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


# Design doc examples (Section 4.2): "knee for squats, elbow for curls,
# hip for deadlifts".
SQUAT = ExerciseConfig(
    name="squat",
    joint_triplet=("left_hip", "left_knee", "left_ankle"),
    bottom_angle=90.0,
    top_angle=170.0,
)

BICEP_CURL = ExerciseConfig(
    name="bicep_curl",
    joint_triplet=("left_shoulder", "left_elbow", "left_wrist"),
    bottom_angle=160.0,  # arm extended
    top_angle=40.0,      # fully curled
)

DEADLIFT = ExerciseConfig(
    name="deadlift",
    joint_triplet=("left_shoulder", "left_hip", "left_knee"),
    bottom_angle=80.0,   # hip hinged forward
    top_angle=170.0,     # lockout
)

EXERCISES = {c.name: c for c in (SQUAT, BICEP_CURL, DEADLIFT)}
