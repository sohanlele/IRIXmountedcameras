"""Zero-velocity update (ZUPT) detection (Section 4.6).

Every barbell rep has a natural dead-stop at the top or bottom (lockout,
rack position, full stretch). IMU-only velocity trackers such as
OpenBarbell and PUSH Band exploit these dead-stops as zero-velocity
updates to re-zero their estimate each rep rather than drifting further
over a set. Combined with camera-based correction, this gives the fusion
filter two independent anchors instead of one.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .imu import IMUSample


def detect_zero_velocity(
    samples: Sequence[IMUSample],
    accel_std_thresh: float = 0.15,
    gyro_std_thresh: float = 0.05,
) -> bool:
    """Return True if a short window of IMU samples looks stationary.

    Uses a simple variance test on accel/gyro magnitude over the window --
    a stationary wrist (dead-stop) has near-constant readings (gravity only
    on the accelerometer, ~zero on the gyro), while an actively moving
    wrist has substantially higher variance.
    """
    if len(samples) < 2:
        return False
    accel_mag = np.array([np.linalg.norm(s.accel) for s in samples])
    gyro_mag = np.array([np.linalg.norm(s.gyro) for s in samples])
    return bool(accel_mag.std() < accel_std_thresh and gyro_mag.std() < gyro_std_thresh)
