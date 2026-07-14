"""Geometric fault-detection primitives.

Each ``check_*`` function takes the full sequence of ``PersonPose`` samples
buffered across one rep (``irix.rep_counting.state_machine.RepEvent.poses``)
and returns a ``FormFault`` if the fault is detected, or ``None`` if the
rep looks clean *or* there isn't enough confidently-tracked keypoint data
to judge either way -- this module never reports a fault it isn't
reasonably sure about, and never reports a perfect score just because data
was missing (see ``FormScorer`` in ``scoring.py``, which treats "couldn't
assess" as "no score" rather than "score = 1.0").

All checks are single-side (``left_*`` keypoints only, matching this
repo's existing joint-angle configs in ``irix.rep_counting.exercises``,
which are also left-side-only) and rely on the same 3-4m, 30-45deg-off-axis
camera geometry assumed throughout this repo (Section 3.1 of the design
doc). That means the horizontal (x) keypoint offsets used below for
knee-valgus/elbow-drift/torso-lean are a *frontal-plane proxy* seen from an
angled camera, not a true frontal-view measurement -- directionally
correct (a real inward knee collapse or torso lean will show up as a
larger x-shift than clean form) but not the calibrated angle a
straight-on camera would give. This mirrors the same tier-2-proxy honesty
already used for ``peak_angular_velocity_deg_s`` elsewhere in this repo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..pose.estimator import PersonPose
from ..pose.geometry import joint_angle
from ..rep_counting.exercises import ExerciseConfig

CONFIDENCE_THRESHOLD = 0.3
MIN_VALID_SAMPLES = 3


@dataclass(frozen=True)
class FormFault:
    code: str
    # 0-1, how far past the detection threshold this fault is -- not a
    # calibrated clinical severity, just lets a caller (or a future UI)
    # distinguish "barely over the line" from "way off".
    severity: float


def _xy(pose: PersonPose, name: str, min_conf: float = CONFIDENCE_THRESHOLD) -> Optional[np.ndarray]:
    kp = pose.get(name)
    if kp is None or kp.confidence < min_conf:
        return None
    return np.array([kp.x, kp.y], dtype=float)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def check_squat_depth(poses: List[PersonPose], cfg: ExerciseConfig) -> Optional[FormFault]:
    """Flags a rep that never reached the exercise's configured depth.

    Uses the same hip-knee-ankle angle (``cfg.joint_triplet``) the rep
    counter already tracks, so this reuses the exact geometry that
    defines "bottom" for this exercise rather than a separate threshold.
    """
    a_name, v_name, c_name = cfg.joint_triplet
    angles = []
    for p in poses:
        a, v, c = _xy(p, a_name), _xy(p, v_name), _xy(p, c_name)
        if a is None or v is None or c is None:
            continue
        angles.append(joint_angle(a, v, c))
    if len(angles) < MIN_VALID_SAMPLES:
        return None

    descending_is_decreasing = cfg.top_angle > cfg.bottom_angle
    extreme = min(angles) if descending_is_decreasing else max(angles)
    deficit = (extreme - cfg.bottom_angle) if descending_is_decreasing else (cfg.bottom_angle - extreme)
    deficit = max(0.0, deficit)
    if deficit < 8.0:  # within the rep counter's own hysteresis-ish tolerance
        return None
    return FormFault(code="insufficient_depth", severity=_clamp01(deficit / 40.0))


def check_knee_valgus(poses: List[PersonPose], cfg: Optional[ExerciseConfig] = None, side: str = "left") -> Optional[FormFault]:
    """Flags the knee drifting medially (inward, "caving in") relative to
    the ankle beyond what's expected during a normal squat/leg-press rep.

    Baseline is this rep's own standing (first-valid-frame) knee-ankle
    horizontal offset, normalized by shank length so it's roughly scale
    (distance-from-camera) invariant. A rep is flagged if the offset
    shifts from that baseline by more than 0.25 shank-lengths at any
    point during the rep.
    """
    hip_name, knee_name, ankle_name = f"{side}_hip", f"{side}_knee", f"{side}_ankle"
    baseline: Optional[float] = None
    worst_shift = 0.0
    n_valid = 0
    for p in poses:
        knee, ankle = _xy(p, knee_name), _xy(p, ankle_name)
        if knee is None or ankle is None:
            continue
        shank = float(np.linalg.norm(knee - ankle))
        if shank == 0:
            continue
        n_valid += 1
        offset = (knee[0] - ankle[0]) / shank
        if baseline is None:
            baseline = offset
            continue
        worst_shift = max(worst_shift, abs(offset - baseline))
    if baseline is None or n_valid < MIN_VALID_SAMPLES:
        return None
    if worst_shift < 0.25:
        return None
    return FormFault(code="knee_valgus", severity=_clamp01((worst_shift - 0.25) / 0.35))


def check_torso_lean(poses: List[PersonPose], cfg: Optional[ExerciseConfig] = None, side: str = "left") -> Optional[FormFault]:
    """Flags the torso leaning back beyond its own rep-start baseline --
    the classic bicep-curl "cheat" of using body momentum instead of arm
    strength (github.com/NgoQuocBao1010/Exercise-Correction trains a
    dedicated classifier for exactly this "lean back error")."""
    shoulder_name, hip_name = f"{side}_shoulder", f"{side}_hip"
    baseline: Optional[float] = None
    worst_dev = 0.0
    n_valid = 0
    vertical = np.array([0.0, -1.0])
    for p in poses:
        shoulder, hip = _xy(p, shoulder_name), _xy(p, hip_name)
        if shoulder is None or hip is None:
            continue
        vec = shoulder - hip
        norm = float(np.linalg.norm(vec))
        if norm == 0:
            continue
        n_valid += 1
        cosang = float(np.clip(np.dot(vec, vertical) / norm, -1.0, 1.0))
        lean_deg = math.degrees(math.acos(cosang))
        if baseline is None:
            baseline = lean_deg
            continue
        worst_dev = max(worst_dev, abs(lean_deg - baseline))
    if baseline is None or n_valid < MIN_VALID_SAMPLES:
        return None
    if worst_dev < 15.0:
        return None
    return FormFault(code="leaning_back", severity=_clamp01((worst_dev - 15.0) / 25.0))


def check_elbow_drift(poses: List[PersonPose], cfg: Optional[ExerciseConfig] = None, side: str = "left") -> Optional[FormFault]:
    """Flags the elbow drifting away from the torso during a curl -- strict
    form keeps the elbow pinned near the hip/ribcage throughout; letting it
    swing forward recruits the shoulder and shortens the effective range of
    motion."""
    shoulder_name, elbow_name, hip_name = f"{side}_shoulder", f"{side}_elbow", f"{side}_hip"
    baseline: Optional[float] = None
    worst_shift = 0.0
    n_valid = 0
    for p in poses:
        shoulder, elbow, hip = _xy(p, shoulder_name), _xy(p, elbow_name), _xy(p, hip_name)
        if shoulder is None or elbow is None or hip is None:
            continue
        upper_arm = float(np.linalg.norm(shoulder - elbow))
        if upper_arm == 0:
            continue
        n_valid += 1
        offset = (elbow[0] - hip[0]) / upper_arm
        if baseline is None:
            baseline = offset
            continue
        worst_shift = max(worst_shift, abs(offset - baseline))
    if baseline is None or n_valid < MIN_VALID_SAMPLES:
        return None
    if worst_shift < 0.35:
        return None
    return FormFault(code="elbow_drift", severity=_clamp01((worst_shift - 0.35) / 0.4))


def check_hip_shoulder_rise(poses: List[PersonPose], cfg: Optional[ExerciseConfig] = None, side: str = "left") -> Optional[FormFault]:
    """Flags hips rising faster than the shoulders/chest during a deadlift
    pull -- a well-known fault (sometimes called "the hips shoot up" or
    "stripper deadlift"), which shifts load off the legs and rounds the
    lower back. Detected by normalizing each of the hip's and shoulder's
    vertical trajectory over the rep to a common 0 (bottom) - 1 (lockout)
    scale and checking how far ahead the hip's progress gets."""
    hip_name, shoulder_name = f"{side}_hip", f"{side}_shoulder"
    ys_hip, ys_shoulder = [], []
    for p in poses:
        hip, shoulder = _xy(p, hip_name), _xy(p, shoulder_name)
        if hip is None or shoulder is None:
            continue
        ys_hip.append(hip[1])
        ys_shoulder.append(shoulder[1])
    if len(ys_hip) < MIN_VALID_SAMPLES:
        return None

    def _progress(ys: List[float]) -> np.ndarray:
        arr = np.array(ys, dtype=float)
        lo, hi = arr.min(), arr.max()  # image y: larger y = lower/bottom
        if hi == lo:
            return np.zeros_like(arr)
        return (hi - arr) / (hi - lo)  # 0 at bottom, 1 at lockout

    hip_progress = _progress(ys_hip)
    shoulder_progress = _progress(ys_shoulder)
    max_gap = float((hip_progress - shoulder_progress).max())
    if max_gap < 0.25:
        return None
    return FormFault(code="hips_rising_before_chest", severity=_clamp01((max_gap - 0.25) / 0.35))
