"""Color-based bumper-plate detection -- a classical-CV, zero-training
signal for load estimation, independent of ``irix.barbell.detector.
FreeWeightDetector`` (which needs a trained checkpoint that doesn't
exist yet -- see that module's docstring and ``docs/IMPLEMENTATION_STATUS.md``).

## Why color, and why it can be a primary signal here (not just a cross-check)

``irix.weight_recognition.plate_geometry_check``'s own module docstring
already states the key fact plainly: **standardized IWF/IPF competition
bumper plates are all the same 450mm diameter regardless of weight** --
only distinguishable by color. That's not a limitation of a
geometry-based approach, it's a structural property of the equipment:
for color-coded bumper plates specifically, diameter carries *zero*
weight information, and color carries all of it. Confirmed IWF color
standard (verified via web search, not assumed from memory):

    10 kg -> green, 15 kg -> yellow, 20 kg -> blue, 25 kg -> red

[Fit at Midlife -- IWF standard color coding](https://fitatmidlife.com/tag/iwf-standard-color-coding/),
[LoadMyBar -- Olympic plate colors guide](https://loadmybar.com/guides/plate-colors.html).
The IWF standard is commonly described as also color-coding plates below
10 kg (e.g. white/black for 5 kg and 2.5 kg), but that wasn't
independently confirmed by the search above -- rather than guess, this
module **only** maps the four weights above. A plate that doesn't match
one of these four colors within tolerance is reported as
unclassified, not silently assumed to be some other weight.

## What this does and doesn't cover

This only works for color-coded bumper plates -- the free-weight-
platform equipment most likely to actually follow this standard.
Commercial-gym cast-iron/rubber-coated plates are frequently
black/gray/unpainted and follow no shared color standard across
manufacturers; this module will correctly find nothing on those (no
color blob clears the detection threshold), not misclassify them. That
is the intended, honest behavior, not a gap -- see the "never fabricate
a detected weight" principle throughout ``irix.weight_recognition``.

## Detection approach

No trained object detector is required: color-coded bumper plates are
large, solid-colored, roughly circular regions -- classical HSV color
thresholding + contour filtering (size, roughly-circular aspect ratio)
finds them directly. This is a genuinely different, and available-today,
signal from ``FreeWeightDetector`` (still an untrained stub) and from
``irix.weight_recognition.vision_classifier.VisionPlateClassifier``
(VLM-based, requires a cloud API key) -- useful as its own load estimate
when the equipment matches the color standard, and as an additional
corroborating signal (alongside ``plate_geometry_check``) against a VLM
read otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover -- opencv-python-headless is a core dependency; defensive only
    cv2 = None

from ..barbell.calibration import MENS_OLYMPIC_BARBELL_WEIGHT_KG

# IWF standard bumper-plate colors -- see module docstring for the citation
# and why only these four are included.
IWF_BUMPER_PLATE_COLORS_KG = {
    10.0: "green",
    15.0: "yellow",
    20.0: "blue",
    25.0: "red",
}

# HSV thresholds (OpenCV convention: H in [0,179], S/V in [0,255]) for each
# color above. Standard, widely-used ranges for saturated primary/
# secondary colors under typical indoor lighting -- not fit to any
# specific gym's cameras/lighting, so a real deployment should expect to
# recalibrate these per-venue (see docs/CAMERA_SYSTEM.md's calibration
# section) rather than treat them as universally correct out of the box.
_COLOR_HSV_RANGES: dict = {
    "red": [((0, 100, 60), (8, 255, 255)), ((172, 100, 60), (179, 255, 255))],  # red wraps hue 0
    "green": [((40, 60, 40), (85, 255, 255))],
    "blue": [((95, 80, 40), (130, 255, 255))],
    "yellow": [((22, 80, 80), (35, 255, 255))],
}

MIN_PLATE_AREA_PX = 400  # a plate should be a substantial blob, not a few stray colored pixels
MIN_CIRCULARITY = 0.55  # 1.0 = perfect circle; plates are round but a side-on camera view foreshortens them
MIN_COLOR_COVERAGE = 0.4  # fraction of the candidate blob's own bounding box the color mask must actually fill


@dataclass
class PlateColorDetection:
    weight_kg: float
    color: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame pixel coordinates
    confidence: float  # 0-1, from area + circularity + fill-ratio -- see detect_color_plates
    pixel_area: int


@dataclass
class LoadColorEstimate:
    total_weight_kg: Optional[float]  # None if no confident, symmetric read was possible
    confidence: float
    plate_detections: List[PlateColorDetection] = field(default_factory=list)
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "total_weight_kg": self.total_weight_kg,
            "confidence": self.confidence,
            "plate_weights_kg": [d.weight_kg for d in self.plate_detections],
            "reason": self.reason,
        }


def _require_cv2():
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python(-headless) is required for irix.weight_recognition.plate_color_check")


def detect_color_plates(
    frame_bgr: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_area_px: int = MIN_PLATE_AREA_PX,
    min_circularity: float = MIN_CIRCULARITY,
) -> List[PlateColorDetection]:
    """Find color-coded bumper-plate-shaped blobs in ``frame_bgr``
    (optionally restricted to a region of interest, e.g. a bounding box
    around the barbell if one is already known from
    ``irix.barbell.tracker``) for each of the four IWF colors above.

    Pure classical CV, no model weights: HSV threshold per color, contour
    detection, filtered by minimum pixel area (rejects small color
    specks -- a shirt, a small sticker) and circularity (rejects
    elongated color regions -- a bench pad, a wall marking -- that
    happen to share a plate's hue but not its roughly-round shape).
    """
    _require_cv2()
    x_offset, y_offset = 0, 0
    image = frame_bgr
    if roi is not None:
        x1, y1, x2, y2 = roi
        image = frame_bgr[y1:y2, x1:x2]
        x_offset, y_offset = x1, y1
    if image.size == 0:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detections: List[PlateColorDetection] = []

    for weight_kg, color in IWF_BUMPER_PLATE_COLORS_KG.items():
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in _COLOR_HSV_RANGES[color]:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue
            perimeter = cv2.arcLength(contour, closed=True)
            if perimeter <= 0:
                continue
            circularity = float(4 * np.pi * area / (perimeter ** 2))
            if circularity < min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox_area = w * h
            fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
            if fill_ratio < MIN_COLOR_COVERAGE:
                continue

            # Confidence: geometric mean of how "round" it is and how
            # solidly it fills its own bounding box -- both matter (a
            # thin crescent could still pass a lenient circularity check
            # alone, but not also a high fill ratio).
            confidence = float(np.sqrt(min(circularity, 1.0) * min(fill_ratio, 1.0)))
            detections.append(
                PlateColorDetection(
                    weight_kg=weight_kg, color=color,
                    bbox=(x + x_offset, y + y_offset, x + w + x_offset, y + h + y_offset),
                    confidence=confidence, pixel_area=int(area),
                )
            )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def estimate_load_from_color_plates(
    detections: List[PlateColorDetection],
    bar_weight_kg: float = MENS_OLYMPIC_BARBELL_WEIGHT_KG,
    min_detection_confidence: float = 0.5,
) -> LoadColorEstimate:
    """Sum detected color-coded plates into a total load estimate.

    A barbell is loaded symmetrically in essentially every real lift --
    each plate on one side has a matching plate on the other. This
    function does not attempt to figure out "which side" a detection
    came from (that needs real bar-position geometry this module
    deliberately doesn't have); instead it takes the conservative,
    never-fabricate stance: an **even** number of confident detections of
    the *same* color is read as symmetric pairs and summed; anything
    else (an odd count, or several different colors with no consistent
    pairing) is reported as ``total_weight_kg=None`` with a reason,
    rather than guessing which detections are "real" plates and which
    are noise.
    """
    confident = [d for d in detections if d.confidence >= min_detection_confidence]
    if not confident:
        return LoadColorEstimate(total_weight_kg=None, confidence=0.0, plate_detections=[], reason="no_color_coded_plates_detected")

    counts: dict = {}
    for d in confident:
        counts.setdefault(d.color, []).append(d)

    total = bar_weight_kg
    used: List[PlateColorDetection] = []
    odd_colors = []
    for color, dets in counts.items():
        n = len(dets)
        if n % 2 != 0:
            odd_colors.append(color)
            n -= 1  # use the even portion, flag the rest below
        weight_kg = dets[0].weight_kg
        total += weight_kg * n
        used.extend(dets[:n])

    if odd_colors:
        return LoadColorEstimate(
            total_weight_kg=None, confidence=0.0, plate_detections=confident,
            reason=f"odd_plate_count_for_colors:{','.join(sorted(odd_colors))}",
        )

    mean_confidence = float(np.mean([d.confidence for d in used])) if used else 0.0
    return LoadColorEstimate(total_weight_kg=total, confidence=mean_confidence, plate_detections=used)
