from __future__ import annotations

import cv2
import numpy as np
import pytest

from irix.weight_recognition.plate_color_check import (
    IWF_BUMPER_PLATE_COLORS_KG,
    detect_color_plates,
    estimate_load_from_color_plates,
)

# BGR (OpenCV convention) swatches picked to land inside this module's HSV
# ranges for each color.
_BGR_FOR_COLOR = {
    "green": (0, 180, 0),
    "yellow": (0, 220, 220),
    "blue": (200, 0, 0),
    "red": (0, 0, 220),
}


def _canvas(w=600, h=400, bg=(40, 40, 40)):
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    return img


def _draw_plate(img, center, radius, color_name):
    cv2.circle(img, center, radius, _BGR_FOR_COLOR[color_name], thickness=-1)


def test_detects_a_single_color_coded_plate():
    img = _canvas()
    _draw_plate(img, (150, 200), 60, "green")

    detections = detect_color_plates(img)

    assert len(detections) == 1
    assert detections[0].color == "green"
    assert detections[0].weight_kg == 10.0
    assert detections[0].confidence > 0.5


def test_detects_multiple_different_colored_plates():
    img = _canvas()
    _draw_plate(img, (100, 200), 60, "green")
    _draw_plate(img, (300, 200), 55, "blue")
    _draw_plate(img, (500, 200), 50, "red")

    detections = detect_color_plates(img)
    colors = {d.color for d in detections}

    assert colors == {"green", "blue", "red"}


def test_ignores_a_non_standard_colored_object():
    img = _canvas()
    cv2.circle(img, (150, 200), 60, (128, 128, 128), thickness=-1)  # gray -- not any IWF color

    detections = detect_color_plates(img)

    assert detections == []


def test_rejects_elongated_regions_that_share_a_plate_color():
    """A colored rectangle (e.g. a bench pad or wall marking) shouldn't
    be mistaken for a round plate -- circularity filter should reject it."""
    img = _canvas()
    cv2.rectangle(img, (50, 50), (550, 90), _BGR_FOR_COLOR["yellow"], thickness=-1)  # thin wide bar

    detections = detect_color_plates(img)

    assert detections == []


def test_roi_restricts_detection_and_bbox_is_in_full_frame_coordinates():
    img = _canvas()
    _draw_plate(img, (400, 200), 50, "red")  # outside the ROI below

    inside_roi = detect_color_plates(img, roi=(0, 0, 200, 400))
    assert inside_roi == []

    full_frame = detect_color_plates(img, roi=(300, 100, 500, 300))
    assert len(full_frame) == 1
    x1, y1, x2, y2 = full_frame[0].bbox
    assert x1 >= 300 and x2 <= 500  # bbox reported in original frame coordinates, not ROI-local


def test_symmetric_pair_estimate_sums_bar_plus_both_sides():
    img = _canvas()
    _draw_plate(img, (100, 200), 55, "green")
    _draw_plate(img, (500, 200), 55, "green")

    detections = detect_color_plates(img)
    estimate = estimate_load_from_color_plates(detections, bar_weight_kg=20.0)

    assert estimate.total_weight_kg == pytest.approx(20.0 + 10.0 + 10.0)
    assert estimate.confidence > 0.0
    assert estimate.reason is None


def test_odd_plate_count_never_fabricates_a_total():
    img = _canvas()
    _draw_plate(img, (100, 200), 55, "green")  # single, unmatched plate

    detections = detect_color_plates(img)
    estimate = estimate_load_from_color_plates(detections)

    assert estimate.total_weight_kg is None
    assert estimate.reason is not None and estimate.reason.startswith("odd_plate_count_for_colors:")


def test_no_detections_never_fabricates_a_total():
    img = _canvas()
    estimate = estimate_load_from_color_plates(detect_color_plates(img))
    assert estimate.total_weight_kg is None
    assert estimate.reason == "no_color_coded_plates_detected"


def test_mixed_symmetric_colors_sum_correctly():
    img = _canvas(w=800)
    _draw_plate(img, (100, 200), 55, "blue")
    _draw_plate(img, (700, 200), 55, "blue")
    _draw_plate(img, (250, 200), 40, "yellow")
    _draw_plate(img, (550, 200), 40, "yellow")

    detections = detect_color_plates(img)
    estimate = estimate_load_from_color_plates(detections, bar_weight_kg=20.0)

    assert estimate.total_weight_kg == pytest.approx(20.0 + 20.0 + 20.0 + 15.0 + 15.0)


def test_iwf_color_table_only_has_the_four_confirmed_weights():
    # Deliberately not extended with unverified lighter-plate colors --
    # see module docstring.
    assert set(IWF_BUMPER_PLATE_COLORS_KG.keys()) == {10.0, 15.0, 20.0, 25.0}
