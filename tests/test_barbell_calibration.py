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
