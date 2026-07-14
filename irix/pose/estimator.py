"""Pose estimation module (Section 4.1).

Wraps a bottom-up, single-stage multi-person pose model (YOLO-Pose family)
selected in the design doc because inference cost stays roughly constant
regardless of how many people are in frame -- the right fit for a gym floor
with several lifters visible to one camera, unlike top-down estimators
(MediaPipe Pose) whose cost scales with person count.

``ultralytics`` is an optional dependency (see pyproject.toml `[pose]`
extra) so the rest of the codebase (rep counting, fusion, pipeline, tests)
can be imported and exercised without pulling in torch/ultralytics.

**This one actually works with real, freely available weights, no
training required.** The default ``model_path="yolov8n-pose.pt"`` is a
real Ultralytics-published checkpoint pretrained on COCO keypoints --
exactly the 17-point layout ``COCO_KEYPOINT_NAMES`` below already assumes
-- and is auto-downloaded on first use (no API key, no cost, no gym-
specific data collection). Generic human pose estimation is a solved,
commodity problem; this module is not a stub waiting for a model that
doesn't exist, unlike ``irix.barbell.detector.FreeWeightDetector`` and
``irix.weight_recognition.vlm_backend.LocalVLMBackend`` (see their
module docstrings). Verified end-to-end against a real image and a real
video file in ``tests/test_pose_estimator_integration.py`` (skipped
unless ``pip install irix[pose]`` is installed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# COCO-17 keypoint layout used by YOLO-Pose / YOLOv8-pose.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
KEYPOINT_INDEX = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass
class PersonPose:
    """One detected person's keypoints for a single frame."""

    keypoints: List[Keypoint]
    track_id: Optional[int] = None
    bbox: Optional[tuple] = field(default=None)  # (x1, y1, x2, y2)

    def get(self, name: str) -> Optional[Keypoint]:
        idx = KEYPOINT_INDEX.get(name)
        if idx is None or idx >= len(self.keypoints):
            return None
        return self.keypoints[idx]

    def xy(self, name: str) -> Optional[np.ndarray]:
        kp = self.get(name)
        if kp is None:
            return None
        return np.array([kp.x, kp.y])


class PoseEstimator:
    """Multi-person pose estimator per Section 4.1 (recommendation: YOLO-Pose).

    Usage::

        estimator = PoseEstimator(model_path="yolov8n-pose.pt")
        people = estimator.estimate(frame)  # list[PersonPose]

    Per-station region-of-interest cropping (Section 4.1) should be applied
    by the caller before passing a frame in, so each camera's model only
    reasons about the 1-3 people who could plausibly be at that station.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt", confidence: float = 0.5):
        self.model_path = model_path
        self.confidence = confidence
        self._model = None  # lazy-loaded

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "ultralytics is required for live pose estimation. "
                    "Install the 'pose' extra: pip install irix[pose]"
                ) from exc
            self._model = YOLO(self.model_path)
        return self._model

    def estimate(self, frame: np.ndarray) -> List[PersonPose]:
        """Run pose estimation on a single BGR frame, return one PersonPose per detected person."""
        model = self._load_model()
        results = model.predict(frame, conf=self.confidence, verbose=False)
        people: List[PersonPose] = []
        if not results:
            return people
        result = results[0]
        if result.keypoints is None:
            return people
        kp_data = result.keypoints.data.cpu().numpy()  # (n_people, 17, 3)
        boxes = None
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
        for i, person_kps in enumerate(kp_data):
            keypoints = [Keypoint(x=float(x), y=float(y), confidence=float(c)) for x, y, c in person_kps]
            bbox = tuple(boxes[i]) if boxes is not None and i < len(boxes) else None
            people.append(PersonPose(keypoints=keypoints, track_id=i, bbox=bbox))
        return people
