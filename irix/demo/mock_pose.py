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


def synthetic_bar_pixel_stream(
    n_frames: int = 300,
    fps: float = 30.0,
    reps_per_second: float = 0.5,
    amplitude_px: float = 300.0,
    y0_px: float = 1000.0,
    velocity_decay_per_rep: float = 0.0,
) -> Iterator[tuple]:
    """Yield (timestamp, y_px) pairs for a barbell oscillating vertically
    in image coordinates (y decreases as the bar rises), synchronized to
    the same tempo as ``synthetic_angle_stream`` so a demo can run both a
    joint-angle rep counter and a irix.barbell.tracker.BarPathTracker off
    time-aligned synthetic data.

    ``velocity_decay_per_rep`` (0-1) linearly shrinks the oscillation
    amplitude rep-over-rep, simulating within-set fatigue (each rep a bit
    slower than the last) so irix.barbell.rpe.RPETracker.velocity_loss_pct
    has something nonzero to report in a demo.
    """
    period_s = 1.0 / reps_per_second
    for i in range(n_frames):
        t = i / fps
        rep_index = int(t // period_s)
        decayed_amplitude = amplitude_px * max(0.0, 1.0 - velocity_decay_per_rep * rep_index)
        y_px = y0_px - decayed_amplitude * math.sin(2 * math.pi * reps_per_second * t)
        yield t, y_px
