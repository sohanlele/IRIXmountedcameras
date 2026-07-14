"""Wristband IMU sample type (Section 4.6 / 5.2).

The wristband samples at 100-200+ Hz -- much faster than the camera's
30-60 fps -- and captures fine-grained acceleration and angular velocity
that the fusion filter uses to fill in detail between camera frames and
bridge short occlusion gaps.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class IMUSample:
    timestamp: float
    accel: np.ndarray  # (3,) m/s^2, wristband frame
    gyro: np.ndarray   # (3,) rad/s, wristband frame

    def __post_init__(self):
        self.accel = np.asarray(self.accel, dtype=float)
        self.gyro = np.asarray(self.gyro, dtype=float)
