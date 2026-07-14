"""Tests for irix.weight_recognition.plate_geometry_check."""
from irix.barbell.detector import FreeWeightClass, FreeWeightDetection
from irix.weight_recognition.plate_geometry_check import check_plate_geometry, expected_plates_per_side


def _plate(cx=0.0):
    return FreeWeightDetection(FreeWeightClass.PLATE, (cx, 500.0), (cx - 100, 450.0, cx + 100, 550.0), 0.9)


def test_expected_plates_per_side_decomposes_greedily():
    plates = expected_plates_per_side(100.0, bar_weight_kg=20.0)  # 80kg total -> 40kg/side
    assert sum(plates) == 40.0
    assert plates == [25.0, 15.0]


def test_expected_plates_per_side_zero_when_bar_alone():
    assert expected_plates_per_side(20.0, bar_weight_kg=20.0) == []


def test_check_consistent_when_counts_match():
    detections = [_plate(100), _plate(200), _plate(1200), _plate(1300)]  # 2/side = 4 total
    result = check_plate_geometry(100.0, detections, bar_weight_kg=20.0)
    assert result.consistent
    assert result.detected_plate_count == 4
    assert result.expected_plate_count == 4


def test_check_inconsistent_on_large_mismatch():
    detections = [_plate(100), _plate(200), _plate(1200), _plate(1300)]  # 4 detected
    result = check_plate_geometry(180.0, detections, bar_weight_kg=20.0)  # implies 8
    assert not result.consistent
    assert result.reason is not None
    assert "180.0kg" in result.reason


def test_check_consistent_when_no_plates_detected():
    result = check_plate_geometry(100.0, [], bar_weight_kg=20.0)
    assert result.consistent  # "couldn't check" != "check failed"
    assert result.detected_plate_count == 0
    assert "no plates detected" in result.reason


def test_check_ignores_non_plate_detections():
    barbell = FreeWeightDetection(FreeWeightClass.BARBELL, (700, 500), (0, 480, 1400, 520), 0.9)
    detections = [barbell, _plate(100), _plate(200), _plate(1200), _plate(1300)]
    result = check_plate_geometry(100.0, detections, bar_weight_kg=20.0)
    assert result.detected_plate_count == 4  # barbell isn't counted as a plate


def test_check_within_tolerance_stays_consistent():
    # 4 detected, expected 4 -- exact match within default tolerance.
    detections = [_plate(100), _plate(200), _plate(1200), _plate(1300)]
    result = check_plate_geometry(100.0, detections, bar_weight_kg=20.0, count_tolerance_per_side=1)
    assert result.consistent
