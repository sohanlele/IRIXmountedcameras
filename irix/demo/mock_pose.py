"""Synthetic pose source for testing/demo without a camera or model weights.

Oscillates a joint angle through an exercise's bottom/top range on a sine
wave, wrapped in a minimal PersonPose so the same RepCounter code path used
against real PoseEstimator output can be exercised end-to-end in CI/tests
and by anyone without a webcam or a downloaded YOLO-Pose checkpoint.
"""
from __future__ import annotations

import math
from typing import Iterator, List, Optional

import numpy as np

from ..pose.estimator import Keypoint, PersonPose
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


def _third_point(vertex: np.ndarray, ref_point: np.ndarray, angle_deg: float, length: float) -> np.ndarray:
    """Return a point ``p`` (at distance ``length`` from ``vertex``) such
    that ``joint_angle(ref_point, vertex, p) == angle_deg``.

    General-purpose two-segment inverse-kinematics helper used to place
    a third body keypoint given two known ones and the exact angle a
    synthetic stream wants ``irix.pose.geometry.joint_angle`` to recover
    at that frame -- e.g. given ankle+knee, place the hip for a target
    hip-knee-ankle angle; given shoulder+elbow, place the wrist for a
    target curl angle. One of two possible solutions is picked (fixed
    rotation direction); which one doesn't matter for synthetic data as
    long as it's used consistently.
    """
    v1 = ref_point - vertex
    v1_angle_deg = math.degrees(math.atan2(v1[1], v1[0]))
    phi_deg = v1_angle_deg - angle_deg
    phi = math.radians(phi_deg)
    return vertex + length * np.array([math.cos(phi), math.sin(phi)])


def _pose_from_keypoints(points: dict) -> PersonPose:
    """Build a PersonPose with only the given named keypoints set (high
    confidence); all others in COCO_KEYPOINT_NAMES are omitted/zeroed with
    confidence 0 so form-rule checks that need a keypoint we didn't place
    correctly skip that sample rather than silently using (0, 0)."""
    from ..pose.estimator import COCO_KEYPOINT_NAMES

    keypoints = []
    for name in COCO_KEYPOINT_NAMES:
        if name in points:
            x, y = points[name]
            keypoints.append(Keypoint(x=float(x), y=float(y), confidence=0.9))
        else:
            keypoints.append(Keypoint(x=0.0, y=0.0, confidence=0.0))
    return PersonPose(keypoints=keypoints)


def synthetic_pose_stream(
    exercise: ExerciseConfig,
    n_frames: int = 300,
    fps: float = 30.0,
    reps_per_second: float = 0.5,
    inject_fault: Optional[str] = None,
) -> Iterator[tuple]:
    """Yield (timestamp, angle, PersonPose) triples for a squat/leg-press/
    hack-squat or bicep-curl exercise, geometrically self-consistent (the
    angle recovered by ``joint_angle`` from the emitted keypoints matches
    the yielded ``angle``, which follows the same sine tempo as
    ``synthetic_angle_stream``), for exercising ``irix.form.scoring``
    end-to-end without a camera.

    ``inject_fault`` optionally perturbs the pose (independent of the
    tracked joint angle, which stays a clean sine wave either way) to
    demonstrate a specific ``irix.form.rules`` check catching something:

    - ``"knee_valgus"`` (squat family): shifts the knee inward relative to
      the ankle during the bottom half of each rep.
    - ``"leaning_back"`` (bicep_curl): increases torso lean from vertical
      during the concentric (curling) half of each rep.
    - ``"elbow_drift"`` (bicep_curl): shifts the elbow away from the hip
      during the concentric half of each rep.

    ``None`` (default) yields clean-form poses -- every check in
    ``irix.form.rules`` should stay silent against this stream.
    """
    mid = (exercise.top_angle + exercise.bottom_angle) / 2
    amp = abs(exercise.top_angle - exercise.bottom_angle) / 2
    is_lower_body = exercise.joint_triplet == ("left_hip", "left_knee", "left_ankle")
    is_curl_like = exercise.joint_triplet == ("left_shoulder", "left_elbow", "left_wrist")

    for i in range(n_frames):
        t = i / fps
        phase = math.sin(2 * math.pi * reps_per_second * t)  # -1 (bottom) .. +1 (top)
        angle = mid + amp * phase

        if not is_lower_body and not is_curl_like:
            # This generator's leg/torso geometry only models the squat
            # family (hip-knee-ankle) and curl-like exercises
            # (shoulder-elbow-wrist, e.g. bicep_curl); an exercise like
            # deadlift (shoulder-hip-knee, and a translating body rather
            # than a fixed base joint) needs different geometry this
            # function doesn't build -- yield no pose rather than one that
            # would silently feed irix.form.scoring nonsense keypoints.
            yield t, angle, None
            continue

        # 0 at the bottom of the rep, 1 at the top -- used to gate fault
        # injection to "the half of the rep where a real lifter would
        # actually make this mistake" rather than the whole cycle.
        bottom_progress = _clamp01(1.0 - (phase + 1.0) / 2.0)

        if is_lower_body:
            ankle = np.array([500.0, 1000.0])
            knee = ankle - np.array([0.0, 300.0])  # shank length 300px, roughly vertical
            if inject_fault == "knee_valgus":
                knee = knee + np.array([160.0 * bottom_progress, 0.0])
            hip = _third_point(knee, ankle, angle, length=350.0)  # thigh length 350px
            pose = _pose_from_keypoints({
                "left_ankle": tuple(ankle), "left_knee": tuple(knee), "left_hip": tuple(hip),
            })
        else:
            hip = np.array([500.0, 1000.0])
            lean_deg = 0.0
            if inject_fault == "leaning_back":
                lean_deg = 30.0 * bottom_progress
            lean_rad = math.radians(lean_deg)
            shoulder = hip + 400.0 * np.array([math.sin(lean_rad), -math.cos(lean_rad)])  # torso length 400px
            elbow = shoulder + np.array([0.0, 250.0])  # upper-arm length 250px, hanging down
            if inject_fault == "elbow_drift":
                elbow = elbow + np.array([200.0 * bottom_progress, 0.0])
            wrist = _third_point(elbow, shoulder, angle, length=220.0)  # forearm length 220px
            pose = _pose_from_keypoints({
                "left_hip": tuple(hip), "left_shoulder": tuple(shoulder),
                "left_elbow": tuple(elbow), "left_wrist": tuple(wrist),
            })

        yield t, angle, pose


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
