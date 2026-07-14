import math

import pytest

from irix.barbell.calibration import (
    CameraCalibration,
    calibrate_from_known_object,
    COMPETITION_BUMPER_PLATE_DIAMETER_MM,
)


def test_calibrate_from_known_plate_diameter():
    # A 450mm plate measuring 180px across in frame -> 0.4 px/mm.
    cal = calibrate_from_known_object(
        pixel_size=180.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id="s1"
    )
    assert cal.pixels_per_mm == pytest.approx(0.4)
    assert cal.pixels_to_mm(180.0) == pytest.approx(450.0)
    assert cal.pixels_to_m(180.0) == pytest.approx(0.45)


def test_calibrate_rejects_non_positive_pixel_size():
    with pytest.raises(ValueError):
        calibrate_from_known_object(pixel_size=0, real_world_size_mm=450.0, station_id="s1")
    with pytest.raises(ValueError):
        calibrate_from_known_object(pixel_size=-10, real_world_size_mm=450.0, station_id="s1")


def test_calibration_roundtrips_a_known_distance():
    cal = CameraCalibration(pixels_per_mm=2.0, station_id="s1")
    # 2 px/mm -> 100px should be 50mm -> 0.05m
    assert cal.pixels_to_mm(100.0) == pytest.approx(50.0)
    assert cal.pixels_to_m(100.0) == pytest.approx(0.05)


def test_default_camera_tilt_is_zero_and_leaves_vertical_conversion_unchanged():
    # Backward compatibility: every pre-existing caller (and every
    # pre-existing test) constructs a CameraCalibration without
    # camera_tilt_deg, so pixels_to_vertical_m must match pixels_to_m
    # exactly when tilt is the default 0.0.
    cal = CameraCalibration(pixels_per_mm=2.0, station_id="s1")
    assert cal.camera_tilt_deg == 0.0
    assert cal.pixels_to_vertical_m(100.0) == pytest.approx(cal.pixels_to_m(100.0))


def test_camera_tilt_inflates_vertical_distance_by_cosine_correction():
    # A camera tilted 30 degrees off perpendicular-to-bar-path
    # foreshortens the observed vertical pixel delta for a given real
    # displacement -- pixels_to_vertical_m corrects for that by dividing
    # by cos(tilt), same first-order correction GymAware applies for its
    # cable-angle-vs-true-vertical-bar-path mismatch.
    cal = CameraCalibration(pixels_per_mm=2.0, station_id="s1", camera_tilt_deg=30.0)
    raw = cal.pixels_to_m(100.0)
    corrected = cal.pixels_to_vertical_m(100.0)
    assert corrected == pytest.approx(raw / math.cos(math.radians(30.0)))
    assert corrected > raw  # correction always inflates, never shrinks, the raw estimate


def test_calibrate_from_known_object_threads_camera_tilt_deg_through():
    cal = calibrate_from_known_object(
        pixel_size=180.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM,
        station_id="s1", camera_tilt_deg=15.0,
    )
    assert cal.camera_tilt_deg == 15.0
    assert cal.pixels_to_vertical_m(180.0) > cal.pixels_to_m(180.0)
