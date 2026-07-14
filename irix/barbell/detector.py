"""Barbell / plate / dumbbell object detection (Section 4.5).

Section 4.5 specifies bar path is tracked "by detecting the barbell as an
object class and following its centroid across frames" -- this is that
detector. Same wrapper pattern as ``irix.pose.estimator.PoseEstimator``:
``ultralytics`` is an optional, lazily-imported dependency, so the rest
of this package (calibration, tracking, RPE math) can be exercised
without it.

Recommended starting point for the model itself: fine-tune on the
Roboflow "Barbells Detector" dataset
(universe.roboflow.com/yolo-project-c2bfs/barbells-detector, 92 labeled
images with a pretrained model + API already available) or a similarly
labeled barbell/plate dataset -- the same category of starting point
github.com/mattiolato98/deadlift-visual-analyzer used for its YOLOv5
barbell class before layering mean-shift tracking on top for
frame-to-frame continuity. Neither this class nor any bundled weights are
included here; ``model_path`` must point at a real trained checkpoint.

Dumbbell tracking reuses this exact same detector/tracker pair -- a
dumbbell head is just another object class with its own known reference
dimension, not a separate tracking problem.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class FreeWeightClass(Enum):
    BARBELL = "barbell"
    PLATE = "plate"
    DUMBBELL = "dumbbell"


@dataclass
class FreeWeightDetection:
    class_label: FreeWeightClass
    centroid_px: Tuple[float, float]
    bbox_px: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def pixel_diameter(self) -> float:
        """Bounding-box extent along its shorter axis -- a reasonable
        proxy for a round plate/dumbbell-head's pixel diameter, used for
        self-calibration (see irix.barbell.calibration)."""
        x1, y1, x2, y2 = self.bbox_px
        return min(x2 - x1, y2 - y1)

    @property
    def pixel_length(self) -> float:
        """Bounding-box extent along its longer axis -- a proxy for a
        barbell's pixel length when the whole bar is in frame."""
        x1, y1, x2, y2 = self.bbox_px
        return max(x2 - x1, y2 - y1)


class FreeWeightDetector:
    """Detects barbells, plates, and dumbbells in a frame.

    Per-station region-of-interest cropping (same principle as
    ``PoseEstimator``, Section 4.1) should be applied by the caller
    before passing a frame in, so the model only reasons about the one
    station's equipment.
    """

    def __init__(self, model_path: str = "freeweight_yolo.pt", confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "ultralytics is required for free-weight detection. "
                    "Install the 'pose' extra: pip install irix[pose] "
                    "(shared with PoseEstimator's dependency) and point "
                    "model_path at a barbell/plate-trained checkpoint."
                ) from exc
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> List[FreeWeightDetection]:
        model = self._load_model()
        results = model.predict(frame, conf=self.confidence, verbose=False)
        detections: List[FreeWeightDetection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        names = result.names  # class-index -> name, from the model itself
        for box in result.boxes:
            cls_idx = int(box.cls[0])
            label_str = names.get(cls_idx, "") if isinstance(names, dict) else str(names[cls_idx])
            try:
                label = FreeWeightClass(label_str)
            except ValueError:
                continue  # model detected some other class we don't track here
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(
                FreeWeightDetection(
                    class_label=label,
                    centroid_px=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    bbox_px=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                )
            )
        return detections

    @staticmethod
    def largest_plate(detections: List[FreeWeightDetection]) -> Optional[FreeWeightDetection]:
        """The largest detected plate is usually the best self-calibration
        reference (Section above): outermost plate on a loaded bar, most
        consistently visible regardless of load."""
        plates = [d for d in detections if d.class_label == FreeWeightClass.PLATE]
        if not plates:
            return None
        return max(plates, key=lambda d: d.pixel_diameter)
