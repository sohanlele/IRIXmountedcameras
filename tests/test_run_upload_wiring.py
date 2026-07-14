"""Business-logic tests for irix.demo.run_upload -- does the full
pipeline (rep counting, set-boundary detection, IMU fusion, fatigue
analysis, weight recognition, barbell velocity/RPE) actually get wired
together correctly, end to end?

Deliberately doesn't need the real ultralytics pose model (that's what
tests/test_run_upload_integration.py is for, gated behind
pytest.importorskip): PoseEstimator is patched with a small scripted
stand-in that returns a precise, hand-computed sequence of joint angles
via real PersonPose/Keypoint objects (same data shapes the real model
would produce), so this test exercises exactly the same code path
run_upload uses in production while staying fast and dependency-light.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from irix.demo.run_upload import run_upload
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.weight_recognition.vlm_backend import FakeVLMBackend

# Squat thresholds (irix.rep_counting.exercises.SQUAT): bottom_angle=90,
# top_angle=170, hysteresis=8 -- so 170 is safely "at top" (>=162) and 90
# is safely "at bottom" (<=98). Each rep ramps smoothly from 90 to 170
# over 10 frames (1s at fps=10) rather than jumping in one frame, so
# RepEvent.duration_s comes out realistic (~1s/rep) -- that duration
# feeds RepCountFusion's period-bounds search, so an unrealistically fast
# scripted rep would make the IMU-fusion test below unable to find a
# sensible period to search.
def _slow_rep(n_frames: int = 10):
    return list(np.linspace(90.0, 170.0, n_frames))


_ONE_REP = _slow_rep()
_TWO_REPS = _ONE_REP + _ONE_REP
_IDLE = [170.0] * 20  # 2.0s at fps=10 -- comfortably past a 1.0s rest_gap_s


def _pose_for_angle(angle_deg: float) -> PersonPose:
    """A PersonPose whose left_hip/left_knee/left_ankle keypoints produce
    exactly `angle_deg` when run through irix.pose.geometry.joint_angle
    -- knee is the vertex, fixed at the origin; hip is fixed along one
    ray; ankle is placed at the angle that makes the hip-knee-ankle angle
    come out to angle_deg exactly."""
    knee = np.array([0.0, 0.0])
    hip = knee + np.array([0.0, -100.0])
    theta = math.radians(-90 + angle_deg)
    ankle = knee + 100.0 * np.array([math.cos(theta), math.sin(theta)])

    keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]

    def _set(name: str, xy: np.ndarray) -> None:
        keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=float(xy[0]), y=float(xy[1]), confidence=0.9)

    _set("left_hip", hip)
    _set("left_knee", knee)
    _set("left_ankle", ankle)
    return PersonPose(keypoints=keypoints, bbox=(0.0, 0.0, 200.0, 200.0))


class _ScriptedPoseEstimator:
    """Stand-in for irix.pose.estimator.PoseEstimator: returns one
    scripted PersonPose per call, in order, matching frame-by-frame."""

    def __init__(self, angles):
        self._poses = [_pose_for_angle(a) for a in angles]
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._poses):
            return []
        pose = self._poses[self._i]
        self._i += 1
        return [pose]


def _write_blank_video(path: str, n_frames: int, fps: float = 10.0) -> None:
    """Content is irrelevant -- PoseEstimator is mocked out -- this just
    needs to be a real, readable video file with the right frame count so
    cv2.VideoCapture's frame-by-frame loop in run_upload behaves exactly
    like it would against a real uploaded video."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (4, 4))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _write_imu_csv(path, samples):
    lines = ["timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"]
    for t, ax, ay, az, gx, gy, gz in samples:
        lines.append(f"{t},{ax},{ay},{az},{gx},{gy},{gz}")
    path.write_text("\n".join(lines))


def _run(tmp_path, angles, **kwargs):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, n_frames=len(angles), fps=10.0)
    estimator = _ScriptedPoseEstimator(angles)
    with patch("irix.pose.estimator.PoseEstimator", return_value=estimator):
        return run_upload(video_path, "squat", "member-1", "station-1", **kwargs)


def test_reps_are_counted_and_form_scored(tmp_path):
    events = _run(tmp_path, _TWO_REPS)
    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert len(rep_events) == 2
    assert [e.rep_count for e in rep_events] == [1, 2]


def test_rest_gap_splits_the_stream_into_two_sets_with_fatigue_summaries(tmp_path):
    angles = _TWO_REPS + _IDLE + _TWO_REPS
    # 1.5s: comfortably above the ~1.0s gap between two back-to-back
    # _slow_rep()s within one block, comfortably below the ~3s gap the
    # _IDLE block creates between the two blocks.
    events = _run(tmp_path, angles, rest_gap_s=1.5)

    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    fatigue_events = [e for e in events if e.to_dict()["event_type"] == "set_fatigue_summary"]

    assert len(set_events) == 2, "the >2s idle gap should have closed set 1 before set 2's reps started"
    assert [e.total_reps for e in set_events] == [2, 2]
    assert len(fatigue_events) == 2
    # session fatigue tracking accumulates across sets within one run_upload call
    assert fatigue_events[0].completed_sets_this_session == 1
    assert fatigue_events[1].completed_sets_this_session == 2


def test_without_rest_gap_everything_is_one_set(tmp_path):
    """Sanity check on the other side of the boundary logic: reps close
    enough together (no >=rest_gap_s gap) should NOT be split."""
    events = _run(tmp_path, _TWO_REPS + _TWO_REPS, rest_gap_s=100.0)
    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert len(set_events) == 1
    assert set_events[0].total_reps == 4


def test_imu_file_populates_fused_rep_count(tmp_path):
    """A real (well-formed synthetic) IMU file, sliced to the set's time
    window, should reach RepCountFusion and populate imu_rep_count /
    fused_rep_count on the SetCompleteEvent -- proving the uploaded IMU
    file actually gets used, not just accepted and ignored.

    Needs a handful of rep cycles, not just one or two: RecoFitCounter/
    ULiftCounter are batch algorithms that need "several cycles of
    signal to reliably estimate a period" (irix.fusion.rep_fusion's own
    module docstring) -- a ~2-cycle window reliably comes back
    confidence=0.0/count=0 even with a clean signal, confirmed
    empirically against irix.demo.mock_pose.synthetic_imu_stream (the
    same generator every other IMU-related test/demo in this repo
    already relies on) before picking these numbers. So this uses 4 reps
    (~3.9s of scripted camera time) against a matching ~1Hz IMU stream.
    """
    from irix.demo.mock_pose import synthetic_imu_stream

    imu_path = tmp_path / "imu.csv"
    imu_samples = synthetic_imu_stream(n_seconds=4.5, fs=50.0, reps_per_second=1.0, amplitude=6.0)
    _write_imu_csv(imu_path, [
        (s.timestamp, *s.accel.tolist(), *s.gyro.tolist()) for s in imu_samples
    ])

    four_reps = _ONE_REP + _ONE_REP + _ONE_REP + _ONE_REP
    events = _run(tmp_path, four_reps, imu_path=str(imu_path))
    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert len(set_events) == 1
    assert set_events[0].total_reps == 4
    # Whatever RecoFit/uLift actually conclude, imu_count should have
    # been populated (not left None, which is what happens with no IMU
    # data at all) -- that's the thing this test is actually proving.
    assert set_events[0].imu_rep_count is not None
    assert set_events[0].rep_count_source != "camera_only"


def test_no_imu_file_leaves_fusion_fields_none(tmp_path):
    events = _run(tmp_path, _TWO_REPS)
    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert set_events[0].imu_rep_count is None
    assert set_events[0].rep_count_source == "camera_only"
    assert set_events[0].fused_rep_count == set_events[0].total_reps


def test_weight_recognition_populates_weight_confirmed_and_rep_weight(tmp_path):
    backend = FakeVLMBackend(responses=[
        {"plates_visible": True, "total_weight_kg": 40.0, "confidence": 0.95},
        {"plates_visible": True, "total_weight_kg": 40.0, "confidence": 0.95},
        {"plates_visible": True, "total_weight_kg": 40.0, "confidence": 0.95},
    ])
    # weight_check_every_n_frames=1 -- checks every frame so all 3 scripted
    # confirmations happen well before the reps do.
    events = _run(tmp_path, _TWO_REPS, vlm_backend=backend, weight_check_every_n_frames=1)

    weight_events = [e for e in events if e.to_dict()["event_type"] == "weight_confirmed"]
    assert len(weight_events) == 1
    assert weight_events[0].weight_kg == 40.0

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    # Both reps happen after the weight was confirmed (confirm_n=3
    # consecutive reads finish before frame_index reaches either rep's
    # frame in _TWO_REPS), so both should carry the confirmed weight.
    assert all(e.weight_kg == 40.0 for e in rep_events)


def test_no_vlm_backend_means_no_weight_events(tmp_path):
    events = _run(tmp_path, _TWO_REPS)
    assert not [e for e in events if e.to_dict()["event_type"] == "weight_confirmed"]
    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert all(e.weight_kg is None for e in rep_events)


def test_barbell_detector_upgrades_velocity_to_m_s_and_estimates_rpe(tmp_path):
    """A configured barbell_detector (duck-typed here rather than a real
    FreeWeightDetector -- no trained checkpoint is bundled with this
    repo, see docs/ARCHITECTURE.md) should self-calibrate off the first
    detected plate, track the barbell centroid every frame, and upgrade
    each rep from the deg/s joint-angle proxy to real m/s velocity + an
    RPE estimate (irix.barbell.rpe has a published anchor for squat)."""
    from irix.barbell.detector import FreeWeightClass, FreeWeightDetection

    class _FakeBarbellDetector:
        """Every frame: one plate detection (for self-calibration, fixed
        size/position) and one barbell detection whose y-pixel steadily
        decreases (i.e. moves "up" in image coordinates) so BarPathTracker
        has real displacement to differentiate."""

        def __init__(self):
            self._frame = 0

        def detect(self, frame):
            i = self._frame
            self._frame += 1
            plate = FreeWeightDetection(
                class_label=FreeWeightClass.PLATE,
                centroid_px=(300.0, 500.0),
                bbox_px=(255.0, 455.0, 345.0, 545.0),  # 90px box -> self-calibration reference
                confidence=0.9,
            )
            bar_y = 1000.0 - 5.0 * i
            barbell = FreeWeightDetection(
                class_label=FreeWeightClass.BARBELL,
                centroid_px=(300.0, bar_y),
                bbox_px=(100.0, bar_y - 10.0, 500.0, bar_y + 10.0),
                confidence=0.9,
            )
            return [plate, barbell]

    events = _run(tmp_path, _TWO_REPS, barbell_detector=_FakeBarbellDetector())
    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert len(rep_events) == 2
    assert all(e.peak_velocity_m_s is not None for e in rep_events)
    assert all(e.mean_velocity_m_s is not None for e in rep_events)
    assert all(e.estimated_rpe is not None for e in rep_events)
    # velocity_loss_pct is None on the very first rep of a set (nothing to
    # compare against yet -- see RPETracker.velocity_loss_pct), populated
    # from the second rep onward.
    assert rep_events[0].velocity_loss_pct is None
    assert rep_events[1].velocity_loss_pct is not None


def test_band_placement_event_emitted_for_ankle_exercise(tmp_path):
    """leg_press requires the ankle band placement (BandPlacement.ANKLE),
    different from the WRIST default a BandPlacementTracker starts at --
    run_upload should surface that as a BandPlacementRequiredEvent right
    at the start, same as the live/mock demos do."""
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, n_frames=1, fps=10.0)
    estimator = _ScriptedPoseEstimator([170])
    with patch("irix.pose.estimator.PoseEstimator", return_value=estimator):
        events = run_upload(video_path, "leg_press", "member-1", "station-1")
    band_events = [e for e in events if e.to_dict()["event_type"] == "band_placement_required"]
    assert len(band_events) == 1
    assert band_events[0].to_placement == "ankle"


def test_unknown_exercise_raises_value_error(tmp_path):
    video_path = str(tmp_path / "video.mp4")
    _write_blank_video(video_path, n_frames=1)
    with pytest.raises(ValueError, match="Unknown exercise"):
        run_upload(video_path, "not-a-real-exercise", "member-1", "station-1")


def test_missing_video_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="Could not open video"):
        run_upload(str(tmp_path / "nope.mp4"), "squat", "member-1", "station-1")


def test_all_events_are_json_serializable(tmp_path):
    import json

    angles = _TWO_REPS + _IDLE + _TWO_REPS
    events = _run(tmp_path, angles, rest_gap_s=1.0)
    payload = [e.to_dict() for e in events]
    json.dumps(payload)  # should not raise
    assert len(payload) > 0
    assert all("event_type" in d for d in payload)
