from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from irix.pose.calibration import (
    CalibrationProfile,
    calibrate_extrinsics,
    calibrate_intrinsics,
    compute_ground_plane_homography,
    find_checkerboard_corners,
)

PATTERN_SIZE = (6, 4)  # interior corners -- a 7x5-square board
SQUARE_PX = 40


def _flat_checkerboard(pattern_size=PATTERN_SIZE, square_px=SQUARE_PX, margin_px=60):
    cols, rows = pattern_size
    board_w, board_h = (cols + 1) * square_px, (rows + 1) * square_px
    board = np.zeros((board_h, board_w), dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 255
    canvas = np.full((board_h + 2 * margin_px, board_w + 2 * margin_px), 255, dtype=np.uint8)
    canvas[margin_px:margin_px + board_h, margin_px:margin_px + board_w] = board
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _warped_views(n=10, seed=0):
    """Several slightly different synthetic "captures" of the same flat
    checkerboard -- small random perspective warps, enough independent
    views for cv2.calibrateCamera to solve a well-conditioned (if not
    physically meaningful, since these are undistorted synthetic images)
    intrinsic model."""
    base = _flat_checkerboard()
    h, w = base.shape[:2]
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    views = []
    for _ in range(n):
        jitter = rng.uniform(-0.06, 0.06, size=(4, 2)) * [w, h]
        dst = (src + jitter).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        views.append(cv2.warpPerspective(base, M, (w, h), borderValue=(255, 255, 255)))
    return views


def test_find_checkerboard_corners_on_a_clean_board():
    corners = find_checkerboard_corners(_flat_checkerboard(), PATTERN_SIZE)
    cols, rows = PATTERN_SIZE
    assert corners is not None
    assert corners.shape[0] == cols * rows


def test_find_checkerboard_corners_returns_none_on_a_blank_image():
    blank = np.full((300, 300, 3), 200, dtype=np.uint8)
    assert find_checkerboard_corners(blank, PATTERN_SIZE) is None


def test_calibrate_intrinsics_on_synthetic_views_reports_low_error():
    result = calibrate_intrinsics(_warped_views(n=12), pattern_size=PATTERN_SIZE, square_size_mm=25.0)

    assert result.camera_matrix.shape == (3, 3)
    assert result.dist_coeffs.shape == (5,)
    assert result.n_images_used >= 10
    assert result.reprojection_error_px < 2.0  # synthetic, undistorted views -- should calibrate cleanly
    assert result.quality in ("excellent", "good", "marginal")


def test_calibrate_intrinsics_rejects_and_reports_unusable_images():
    good_views = _warped_views(n=6)
    blank = np.full_like(good_views[0], 255)
    result = calibrate_intrinsics(good_views + [blank, blank], pattern_size=PATTERN_SIZE, min_images=5)

    assert result.n_images_used == 6
    assert result.n_images_rejected == 2
    assert len(result.rejected_reasons) == 2


def test_calibrate_intrinsics_raises_with_too_few_usable_images():
    blanks = [np.full((300, 300, 3), 255, dtype=np.uint8) for _ in range(3)]
    with pytest.raises(ValueError):
        calibrate_intrinsics(blanks, pattern_size=PATTERN_SIZE, min_images=5)


def test_calibrate_extrinsics_returns_none_without_a_visible_board():
    camera_matrix = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    dist_coeffs = np.zeros(5)
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)

    result = calibrate_extrinsics(blank, camera_matrix, dist_coeffs, pattern_size=PATTERN_SIZE)

    assert result is None


def test_calibrate_extrinsics_recovers_a_valid_pose_from_intrinsics():
    intrinsic_result = calibrate_intrinsics(_warped_views(n=12), pattern_size=PATTERN_SIZE, square_size_mm=25.0)
    extrinsic_result = calibrate_extrinsics(
        _flat_checkerboard(), intrinsic_result.camera_matrix, intrinsic_result.dist_coeffs, pattern_size=PATTERN_SIZE,
    )

    assert extrinsic_result is not None
    assert extrinsic_result.rotation.shape == (3, 3)
    # a real rotation matrix is orthonormal: R @ R.T == I
    np.testing.assert_allclose(extrinsic_result.rotation @ extrinsic_result.rotation.T, np.eye(3), atol=1e-6)
    assert extrinsic_result.reprojection_error_px < 5.0


def test_calibration_profile_round_trips_to_camera_projection():
    intrinsic_result = calibrate_intrinsics(_warped_views(n=12), pattern_size=PATTERN_SIZE)
    extrinsic_result = calibrate_extrinsics(
        _flat_checkerboard(), intrinsic_result.camera_matrix, intrinsic_result.dist_coeffs, pattern_size=PATTERN_SIZE,
    )
    profile = CalibrationProfile(camera_id="cam-1", intrinsic=intrinsic_result, extrinsic=extrinsic_result)

    projection = profile.to_camera_projection()

    assert projection.camera_id == "cam-1"
    np.testing.assert_allclose(projection.intrinsic, intrinsic_result.camera_matrix)
    assert projection.projection_matrix.shape == (3, 4)


def test_calibration_profile_without_extrinsics_raises_on_conversion():
    intrinsic_result = calibrate_intrinsics(_warped_views(n=12), pattern_size=PATTERN_SIZE)
    profile = CalibrationProfile(camera_id="cam-1", intrinsic=intrinsic_result)

    with pytest.raises(ValueError):
        profile.to_camera_projection()


def test_calibration_profile_save_and_load_round_trip():
    intrinsic_result = calibrate_intrinsics(_warped_views(n=12), pattern_size=PATTERN_SIZE)
    extrinsic_result = calibrate_extrinsics(
        _flat_checkerboard(), intrinsic_result.camera_matrix, intrinsic_result.dist_coeffs, pattern_size=PATTERN_SIZE,
    )
    profile = CalibrationProfile(camera_id="cam-7", intrinsic=intrinsic_result, extrinsic=extrinsic_result)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cam-7.json")
        profile.save(path)
        loaded = CalibrationProfile.load(path)

    assert loaded.camera_id == "cam-7"
    np.testing.assert_allclose(loaded.intrinsic.camera_matrix, profile.intrinsic.camera_matrix)
    np.testing.assert_allclose(loaded.extrinsic.rotation, profile.extrinsic.rotation)
    assert loaded.quality == profile.quality


def test_ground_plane_homography_recovers_a_known_scale():
    # 1 pixel = 10 mm, no rotation -- a trivial but exact case to verify against.
    image_points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float64)
    world_points = image_points * 10.0

    H, error = compute_ground_plane_homography(image_points, world_points)

    assert error < 1e-6
    # a point not in the fitting set should still map correctly
    test_point = np.array([[50.0, 50.0, 1.0]])
    projected = (H @ test_point.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    np.testing.assert_allclose(projected, [[500.0, 500.0]], atol=1e-3)


def test_ground_plane_homography_requires_at_least_four_points():
    with pytest.raises(ValueError):
        compute_ground_plane_homography(np.zeros((3, 2)), np.zeros((3, 2)))
