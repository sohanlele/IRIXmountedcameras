"""Barbell/dumbbell tracking (Section 4.5) and velocity-based fatigue
signals for irix-mvp-app's AI. See docs/ARCHITECTURE.md for the full
writeup and citations.
"""
from .calibration import (
    CameraCalibration,
    calibrate_from_known_object,
    undistort_frame,
    MENS_OLYMPIC_BARBELL_LENGTH_MM,
    WOMENS_OLYMPIC_BARBELL_LENGTH_MM,
    MENS_OLYMPIC_BARBELL_SLEEVE_LENGTH_MM,
    COMPETITION_BUMPER_PLATE_DIAMETER_MM,
)
from .detector import FreeWeightDetector, FreeWeightDetection, FreeWeightClass
from .tracker import BarPathTracker, BarPathVelocity
from .rpe import RPETracker, RPEEstimate, EXERCISE_1RM_VELOCITY_MS

__all__ = [
    "CameraCalibration", "calibrate_from_known_object", "undistort_frame",
    "MENS_OLYMPIC_BARBELL_LENGTH_MM", "WOMENS_OLYMPIC_BARBELL_LENGTH_MM",
    "MENS_OLYMPIC_BARBELL_SLEEVE_LENGTH_MM", "COMPETITION_BUMPER_PLATE_DIAMETER_MM",
    "FreeWeightDetector", "FreeWeightDetection", "FreeWeightClass",
    "BarPathTracker", "BarPathVelocity",
    "RPETracker", "RPEEstimate", "EXERCISE_1RM_VELOCITY_MS",
]
