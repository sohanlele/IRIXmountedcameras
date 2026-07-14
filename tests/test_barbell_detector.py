from irix.barbell.detector import FreeWeightDetector, FreeWeightDetection, FreeWeightClass


def _plate(diameter=180, center=(100, 100)):
    half = diameter / 2
    return FreeWeightDetection(
        class_label=FreeWeightClass.PLATE,
        centroid_px=center,
        bbox_px=(center[0] - half, center[1] - half, center[0] + half, center[1] + half),
        confidence=0.9,
    )


def test_pixel_diameter_uses_shorter_bbox_axis():
    d = FreeWeightDetection(
        class_label=FreeWeightClass.PLATE, centroid_px=(0, 0), bbox_px=(0, 0, 200, 180), confidence=0.9
    )
    assert d.pixel_diameter == 180
    assert d.pixel_length == 200


def test_largest_plate_picks_biggest_diameter():
    small = _plate(diameter=100)
    big = _plate(diameter=180)
    result = FreeWeightDetector.largest_plate([small, big])
    assert result is big


def test_largest_plate_ignores_non_plate_detections():
    barbell = FreeWeightDetection(
        class_label=FreeWeightClass.BARBELL, centroid_px=(0, 0), bbox_px=(0, 0, 1000, 40), confidence=0.9
    )
    result = FreeWeightDetector.largest_plate([barbell])
    assert result is None


def test_largest_plate_empty_list():
    assert FreeWeightDetector.largest_plate([]) is None
