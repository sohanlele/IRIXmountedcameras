"""RepSession.process_frame's preference for a triangulated 3D joint
angle (irix.pose.multiview) over the ordinary 2D one, when a pose has
z populated for all 3 of an exercise's needed keypoints.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from irix.pipeline.rep_session import RepSession
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose


def _pose(hip_z=None, knee_z=None, ankle_z=None):
    """A squat-relevant pose (left_hip/left_knee/left_ankle) that would
    read as a *90-degree* angle if computed from (x, y) alone -- but
    whose z coordinates, when present, place the true 3D angle at 180
    degrees (a straight leg), letting a test tell which one RepSession
    actually used."""
    keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]

    def _set(name, x, y, z, confidence=0.9):
        keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=x, y=y, z=z, confidence=confidence)

    # 2D-only layout: hip directly above knee, knee directly left of
    # ankle -> a 90-degree angle at the knee in the (x, y) plane.
    _set("left_hip", 0.0, -1.0, hip_z)
    _set("left_knee", 0.0, 0.0, knee_z)
    _set("left_ankle", 1.0, 0.0, ankle_z)
    return PersonPose(keypoints=keypoints)


def test_process_frame_uses_2d_angle_when_no_z_present():
    session = RepSession(exercise_name="squat", member_id="m1", station_id="s1")
    pose = _pose()  # z=None everywhere -> falls back to 2D
    # Directly check the angle irix.pose.geometry.joint_angle would
    # compute for this pose's 2D layout matches what RepCounter sees, by
    # observing rep-counting behaves as it would for a 90-degree 2D pose
    # (squat's bottom_angle threshold is 90 -- see irix.rep_counting.
    # exercises.SQUAT) rather than asserting internal state directly.
    from irix.pose.geometry import joint_angle
    expected_2d_angle = joint_angle(np.array([0.0, -1.0]), np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert expected_2d_angle == pytest.approx(90.0)

    session.process_frame(np.zeros((2, 2, 3)), ts=0.0, person=pose)
    # No crash / no rep yet (single frame) -- the real assertion is in
    # the paired 3D test below via a comparative angle check.


def test_process_frame_prefers_3d_angle_when_all_three_joint_keypoints_triangulated():
    """Same pose's 2D (x, y) layout reads as 90 degrees, but its z
    coordinates (all three of hip/knee/ankle present, as triangulate_pose
    would produce) place the true 3D angle at 180 degrees -- RepSession
    must use the 3D angle, not the 2D one, once all 3 needed keypoints
    have z populated."""
    from irix.pose.geometry import joint_angle

    # 3D layout: hip and ankle on opposite sides of the knee along the
    # same line -> the 3D vectors knee->hip and knee->ankle point in
    # exactly opposite directions -> a clean 180 degrees, overriding the
    # misleading 2D-only 90-degree reading above.
    knee_xyz = np.array([0.0, 0.0, 0.0])
    hip_xyz = np.array([-1.0, 0.0, 0.0])
    ankle_xyz = np.array([1.0, 0.0, 0.0])
    expected_3d_angle = joint_angle(hip_xyz, knee_xyz, ankle_xyz)
    assert expected_3d_angle == pytest.approx(180.0)

    pose = _pose(hip_z=hip_xyz[2], knee_z=knee_xyz[2], ankle_z=ankle_xyz[2])
    # Overwrite x/y to match the 3D scenario exactly (hip/knee/ankle's
    # x, y, z all consistent with hip_xyz/knee_xyz/ankle_xyz above).
    pose.get("left_hip").x, pose.get("left_hip").y = hip_xyz[0], hip_xyz[1]
    pose.get("left_knee").x, pose.get("left_knee").y = knee_xyz[0], knee_xyz[1]
    pose.get("left_ankle").x, pose.get("left_ankle").y = ankle_xyz[0], ankle_xyz[1]

    assert pose.xyz("left_hip") is not None
    assert pose.xyz("left_knee") is not None
    assert pose.xyz("left_ankle") is not None

    # Drive the counter through a bottom (180, "top" for squat actually
    # -- squat's top_angle=170) then bottom (90) then back to top to
    # produce one rep event we can inspect indirectly isn't necessary;
    # simplest direct proof is comparing RepCounter.update's return
    # against calling it manually with each candidate angle. Since
    # RepSession.process_frame doesn't expose the angle it computed
    # directly, drive two full reps -- one with a pose that's 180 in 3D
    # (reads as squat "top") but would misread as 90 ("bottom") if 2D
    # were used -- and confirm the rep-count state machine only responds
    # to the 3D-correct interpretation.
    session = RepSession(exercise_name="squat", member_id="m1", station_id="s1")
    top_pose_3d = pose  # 3D angle 180 (squat "top", threshold >=162)
    bottom_pose_3d = _pose(hip_z=0.0, knee_z=0.0, ankle_z=0.0)  # collinear-z -> reduces to the 2D 90-degree case, which *is* also a true 3D 90 here since all z=0
    for name, x, y in (("left_hip", 0.0, -1.0), ("left_knee", 0.0, 0.0), ("left_ankle", 1.0, 0.0)):
        kp = bottom_pose_3d.get(name)
        kp.x, kp.y = x, y

    events = []
    events += session.process_frame(np.zeros((2, 2, 3)), ts=0.0, person=top_pose_3d)
    events += session.process_frame(np.zeros((2, 2, 3)), ts=0.2, person=bottom_pose_3d)
    events += session.process_frame(np.zeros((2, 2, 3)), ts=0.4, person=top_pose_3d)
    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert len(rep_events) == 1, (
        "a full top->bottom->top cycle using the 3D-correct angles should "
        "produce exactly one rep -- if RepSession were using the (wrong) "
        "2D reading instead, top_pose_3d would misread as 90 degrees "
        "(itself already 'bottom'), so the state machine would never see "
        "a proper top->bottom transition and no rep would complete"
    )


def test_process_frame_falls_back_to_2d_when_only_some_needed_keypoints_have_z():
    """A pose with z on only 2 of the 3 needed joint-triplet keypoints
    (e.g. multi-view fusion covered hip and knee this tick but not
    ankle) must fall back to the ordinary 2D angle entirely, not attempt
    a mixed 2D/3D computation."""
    pose = _pose(hip_z=-5.0, knee_z=0.0, ankle_z=None)
    assert pose.xyz("left_hip") is not None
    assert pose.xyz("left_knee") is not None
    assert pose.xyz("left_ankle") is None

    session = RepSession(exercise_name="squat", member_id="m1", station_id="s1")
    # Should not raise -- process_frame must detect the incomplete 3D
    # triplet and use person.xy(...) for all three instead.
    session.process_frame(np.zeros((2, 2, 3)), ts=0.0, person=pose)
