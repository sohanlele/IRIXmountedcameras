"""Geometry helpers shared by pose estimation and rep counting."""
from __future__ import annotations

import numpy as np


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle at vertex ``b`` formed by points a-b-c, in degrees.

    This is the primitive the joint-angle rep-counting state machine
    (Section 4.2) runs on: e.g. hip-knee-ankle for a squat, or
    shoulder-elbow-wrist for a curl.
    """
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))
