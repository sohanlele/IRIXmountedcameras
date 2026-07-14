"""Tests for irix.form -- rule-based per-rep fault detection.

Uses hand-built PersonPose fixtures directly (not the demo's synthetic
pose stream) so each rule's geometric definition is tested in isolation,
independent of any particular exercise's tempo/config.
"""
import numpy as np

from irix.form.rules import (
    check_elbow_drift,
    check_hip_shoulder_rise,
    check_knee_valgus,
    check_squat_depth,
    check_torso_lean,
)
from irix.form.scoring import FormScorer
from irix.pose.estimator import Keypoint, PersonPose
from irix.rep_counting.exercises import BICEP_CURL, DEADLIFT, SQUAT


def _pose(**named_xy):
    from irix.pose.estimator import COCO_KEYPOINT_NAMES

    keypoints = []
    for name in COCO_KEYPOINT_NAMES:
        if name in named_xy:
            x, y = named_xy[name]
            keypoints.append(Keypoint(x=float(x), y=float(y), confidence=0.9))
        else:
            keypoints.append(Keypoint(x=0.0, y=0.0, confidence=0.0))
    return PersonPose(keypoints=keypoints)


def _squat_poses(depth_angle=90.0, knee_x_offset=0.0, n=10):
    """A short synthetic squat descent: hip-knee-ankle sweeps from 170
    (standing) down to depth_angle, ankle fixed, knee optionally shifted
    laterally by ``knee_x_offset`` at the deepest frames only."""
    from irix.demo.mock_pose import _third_point

    poses = []
    ankle = np.array([500.0, 1000.0])
    knee_base = ankle - np.array([0.0, 300.0])
    for i in range(n):
        frac = i / (n - 1)
        angle = 170.0 - frac * (170.0 - depth_angle)
        knee = knee_base + np.array([knee_x_offset * frac, 0.0])
        hip = _third_point(knee, ankle, angle, length=350.0)
        poses.append(_pose(left_ankle=tuple(ankle), left_knee=tuple(knee), left_hip=tuple(hip)))
    return poses


def test_squat_depth_clean_rep_no_fault():
    poses = _squat_poses(depth_angle=88.0)  # reaches SQUAT.bottom_angle=90 comfortably
    assert check_squat_depth(poses, SQUAT) is None


def test_squat_depth_shallow_rep_flagged():
    poses = _squat_poses(depth_angle=135.0)  # never gets close to 90
    fault = check_squat_depth(poses, SQUAT)
    assert fault is not None
    assert fault.code == "insufficient_depth"
    assert 0.0 < fault.severity <= 1.0


def test_squat_depth_insufficient_samples_returns_none():
    assert check_squat_depth([_pose()], SQUAT) is None


def test_knee_valgus_clean_rep_no_fault():
    poses = _squat_poses(depth_angle=90.0, knee_x_offset=0.0)
    assert check_knee_valgus(poses) is None


def test_knee_valgus_collapsing_knee_flagged():
    poses = _squat_poses(depth_angle=90.0, knee_x_offset=160.0)  # knee shank ~300px -> big relative shift
    fault = check_knee_valgus(poses)
    assert fault is not None
    assert fault.code == "knee_valgus"


def _curl_poses(lean_deg=0.0, elbow_x_offset=0.0, n=10):
    from irix.demo.mock_pose import _third_point

    poses = []
    hip = np.array([500.0, 1000.0])
    for i in range(n):
        frac = i / (n - 1)
        lean_rad = np.radians(lean_deg * frac)
        shoulder = hip + 400.0 * np.array([np.sin(lean_rad), -np.cos(lean_rad)])
        elbow = shoulder + np.array([elbow_x_offset * frac, 250.0])
        wrist = _third_point(elbow, shoulder, 100.0, length=220.0)
        poses.append(_pose(
            left_hip=tuple(hip), left_shoulder=tuple(shoulder),
            left_elbow=tuple(elbow), left_wrist=tuple(wrist),
        ))
    return poses


def test_torso_lean_clean_rep_no_fault():
    poses = _curl_poses(lean_deg=0.0)
    assert check_torso_lean(poses) is None


def test_torso_lean_cheat_curl_flagged():
    poses = _curl_poses(lean_deg=30.0)
    fault = check_torso_lean(poses)
    assert fault is not None
    assert fault.code == "leaning_back"


def test_elbow_drift_clean_rep_no_fault():
    poses = _curl_poses(elbow_x_offset=0.0)
    assert check_elbow_drift(poses) is None


def test_elbow_drift_swinging_elbow_flagged():
    poses = _curl_poses(elbow_x_offset=200.0)
    fault = check_elbow_drift(poses)
    assert fault is not None
    assert fault.code == "elbow_drift"


def _deadlift_poses(hip_leads_by=0.0, n=10):
    """Hip and shoulder both rise from y=1000 (bottom) to y=600 (lockout);
    ``hip_leads_by`` makes the hip's normalized progress race ahead of the
    shoulder's by that fraction at the steepest point of the pull."""
    poses = []
    for i in range(n):
        frac = i / (n - 1)
        hip_progress = min(1.0, frac + hip_leads_by * np.sin(np.pi * frac))
        shoulder_progress = frac
        hip_y = 1000.0 - hip_progress * 400.0
        shoulder_y = 1000.0 - shoulder_progress * 400.0
        poses.append(_pose(left_hip=(500.0, hip_y), left_shoulder=(500.0, shoulder_y)))
    return poses


def test_hip_shoulder_rise_together_no_fault():
    poses = _deadlift_poses(hip_leads_by=0.0)
    assert check_hip_shoulder_rise(poses) is None


def test_hip_shoots_up_early_flagged():
    poses = _deadlift_poses(hip_leads_by=0.5)
    fault = check_hip_shoulder_rise(poses)
    assert fault is not None
    assert fault.code == "hips_rising_before_chest"


def test_form_scorer_no_faults_scores_one():
    scorer = FormScorer()
    assessment = scorer.score_rep("squat", _squat_poses(depth_angle=88.0))
    assert assessment.score == 1.0
    assert assessment.faults == []


def test_form_scorer_penalizes_each_fault():
    scorer = FormScorer()
    assessment = scorer.score_rep("squat", _squat_poses(depth_angle=135.0, knee_x_offset=160.0))
    assert assessment.score < 1.0
    assert set(assessment.faults) == {"insufficient_depth", "knee_valgus"}


def test_form_scorer_unregistered_exercise_returns_none():
    scorer = FormScorer()
    assert scorer.score_rep("bench_press", _squat_poses()) is None


def test_form_scorer_no_poses_returns_none():
    scorer = FormScorer()
    assert scorer.score_rep("squat", None) is None
    assert scorer.score_rep("squat", []) is None


def test_form_scorer_deadlift_hip_shoot_flagged():
    scorer = FormScorer()
    assessment = scorer.score_rep("deadlift", _deadlift_poses(hip_leads_by=0.5))
    assert assessment.faults == ["hips_rising_before_chest"]
