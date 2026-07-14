"""Business-logic tests for irix.pose.multiview -- does DLT triangulation
actually recover a known 3D point from synthetic calibrated cameras, and
does triangulate_pose correctly fuse/gate per-keypoint view coverage?
"""
from __future__ import annotations

import numpy as np
import pytest

from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.pose.multiview import CameraProjection, triangulate_point, triangulate_pose


def _identity_intrinsic(fx=800.0, fy=800.0, cx=320.0, cy=240.0) -> np.ndarray:
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])


def _camera_looking_along_z(camera_id: str, position: np.ndarray) -> CameraProjection:
    """A camera at world position ``position``, oriented so its optical
    axis is +z, with no rotation relative to world axes (rotation =
    identity) -- world -> camera is just X - position. Enough to build
    a couple of distinctly-positioned synthetic cameras without needing
    real extrinsic calibration machinery."""
    rotation = np.eye(3)
    translation = -rotation @ position
    return CameraProjection(camera_id=camera_id, intrinsic=_identity_intrinsic(), rotation=rotation, translation=translation)


def _project(cam: CameraProjection, point_world: np.ndarray) -> tuple:
    p = cam.projection_matrix
    homog = np.append(point_world, 1.0)
    image = p @ homog
    return (image[0] / image[2], image[1] / image[2])


def test_triangulate_point_recovers_known_3d_point_from_two_cameras():
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    cam_b = _camera_looking_along_z("cam-b", position=np.array([1.0, 0.0, 0.0]))

    true_point = np.array([0.3, -0.2, 5.0])
    obs_a = _project(cam_a, true_point)
    obs_b = _project(cam_b, true_point)

    recovered = triangulate_point([(cam_a, obs_a), (cam_b, obs_b)])
    assert recovered is not None
    assert recovered == pytest.approx(true_point, abs=1e-6)


def test_triangulate_point_returns_none_for_a_single_observation():
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    assert triangulate_point([(cam_a, (320.0, 240.0))]) is None


def test_triangulate_point_uses_all_views_when_more_than_two():
    """Three cameras observing the same point should still recover it
    (DLT's least-squares generalizes past the 2-view case, not just
    tolerates extra views without using them)."""
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    cam_b = _camera_looking_along_z("cam-b", position=np.array([1.0, 0.0, 0.0]))
    cam_c = _camera_looking_along_z("cam-c", position=np.array([0.0, 1.0, 0.0]))

    true_point = np.array([0.1, 0.4, 4.0])
    observations = [(cam, _project(cam, true_point)) for cam in (cam_a, cam_b, cam_c)]
    recovered = triangulate_point(observations)
    assert recovered == pytest.approx(true_point, abs=1e-6)


def _pose_with_single_keypoint(name: str, xy: tuple, confidence: float = 0.9) -> PersonPose:
    keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]
    keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=xy[0], y=xy[1], confidence=confidence)
    return PersonPose(keypoints=keypoints)


def test_triangulate_pose_fuses_a_keypoint_seen_by_two_cameras():
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    cam_b = _camera_looking_along_z("cam-b", position=np.array([1.0, 0.0, 0.0]))
    projections = {"cam-a": cam_a, "cam-b": cam_b}

    true_hip = np.array([0.2, 0.1, 4.0])
    pose_a = _pose_with_single_keypoint("left_hip", _project(cam_a, true_hip))
    pose_b = _pose_with_single_keypoint("left_hip", _project(cam_b, true_hip))

    fused = triangulate_pose({"cam-a": pose_a, "cam-b": pose_b}, projections)
    assert fused is not None
    recovered = fused.xyz("left_hip")
    assert recovered is not None
    assert recovered == pytest.approx(true_hip, abs=1e-5)
    # A keypoint no camera reported anything for stays untriangulated.
    assert fused.xyz("left_knee") is None


def test_triangulate_pose_excludes_a_keypoint_seen_by_only_one_camera():
    """Even with 2 cameras contributing to the fused pose overall, a
    *specific* keypoint only one of them actually detected (e.g.
    occluded from the other's angle) must not get triangulated from a
    single view -- it should be left out (z=None), not hallucinated."""
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    cam_b = _camera_looking_along_z("cam-b", position=np.array([1.0, 0.0, 0.0]))
    projections = {"cam-a": cam_a, "cam-b": cam_b}

    true_hip = np.array([0.2, 0.1, 4.0])
    pose_a = _pose_with_single_keypoint("left_hip", _project(cam_a, true_hip))
    # cam-b's pose has no left_hip detection at all this tick (occluded).
    pose_b = PersonPose(keypoints=[Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES])

    fused = triangulate_pose({"cam-a": pose_a, "cam-b": pose_b}, projections)
    # Nothing triangulated at all in this scenario (only one keypoint
    # total was ever populated, and it only had one view) -> None.
    assert fused is None


def test_triangulate_pose_respects_min_keypoint_confidence():
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    cam_b = _camera_looking_along_z("cam-b", position=np.array([1.0, 0.0, 0.0]))
    projections = {"cam-a": cam_a, "cam-b": cam_b}

    true_hip = np.array([0.2, 0.1, 4.0])
    pose_a = _pose_with_single_keypoint("left_hip", _project(cam_a, true_hip), confidence=0.9)
    # Below the default min_keypoint_confidence (0.3) -- shouldn't count
    # as a usable view even though it's numerically present.
    pose_b = _pose_with_single_keypoint("left_hip", _project(cam_b, true_hip), confidence=0.1)

    fused = triangulate_pose({"cam-a": pose_a, "cam-b": pose_b}, projections)
    assert fused is None


def test_triangulate_pose_returns_none_with_fewer_than_min_views_cameras():
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    pose_a = _pose_with_single_keypoint("left_hip", (320.0, 240.0))
    fused = triangulate_pose({"cam-a": pose_a}, {"cam-a": cam_a})
    assert fused is None


def test_triangulate_pose_ignores_cameras_missing_a_projection():
    """A pose from a camera_id with no entry in ``projections`` is
    simply not usable -- doesn't crash, just can't contribute."""
    cam_a = _camera_looking_along_z("cam-a", position=np.array([0.0, 0.0, 0.0]))
    pose_a = _pose_with_single_keypoint("left_hip", (320.0, 240.0))
    pose_unknown = _pose_with_single_keypoint("left_hip", (330.0, 250.0))
    fused = triangulate_pose({"cam-a": pose_a, "cam-x": pose_unknown}, {"cam-a": cam_a})
    assert fused is None  # only one camera has a projection -> below min_views
