"""Synthetic pose source for testing/demo without a camera or model weights.

Oscillates a joint angle through an exercise's bottom/top range on a sine
wave, wrapped in a minimal PersonPose so the same RepCounter code path used
against real PoseEstimator output can be exercised end-to-end in CI/tests
and by anyone without a webcam or a downloaded YOLO-Pose checkpoint.
"""
from __future__ import annotations

import math
from typing import Iterator

from ..rep_counting.exercises import ExerciseConfig


def synthetic_angle_stream(
    exercise: ExerciseConfig,
    n_frames: int = 300,
    fps: float = 30.0,
    reps_per_second: float = 0.5,
) -> Iterator[tuple]:
    """Yield (timestamp, angle) pairs oscillating between the exercise's
    bottom and top angle, at roughly ``reps_per_second``."""
    mid = (exercise.top_angle + exercise.bottom_angle) / 2
    amp = abs(exercise.top_angle - exercise.bottom_angle) / 2
    for i in range(n_frames):
        t = i / fps
        angle = mid + amp * math.sin(2 * math.pi * reps_per_second * t)
        yield t, angle
