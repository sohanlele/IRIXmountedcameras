"""Wristband IMU static bias calibration (supports Section 4.6 / 5.2).

Every rep-counting/fusion algorithm downstream of a wristband
(``irix.fusion.imu_rep_counting``, ``irix.fusion.ekf``,
``irix.fusion.zupt``) assumes raw accel/gyro samples are at least
bias-corrected -- an uncalibrated gyro's nonzero rest-state bias
integrates into meaningless orientation drift within seconds, and an
uncalibrated accelerometer's bias shows up as a phantom constant
acceleration no ZUPT/EKF step will distinguish from real motion. Nothing
in this repo modeled that calibration step before this module -- ``irix.
fusion.imu_io`` loads already-recorded samples as-is, and the simulator
in ``irix.wristband_sim.simulator`` generates samples with a known
synthetic bias precisely so this module has something real to recover.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..fusion.imu import IMUSample

GRAVITY_M_S2 = 9.80665
MIN_CALIBRATION_SAMPLES = 10


@dataclass
class IMUCalibration:
    """Per-wristband correction, estimated once (front-desk checkout
    time, or firmware boot) from a short stationary period.

    Deliberately just accel/gyro *bias* -- not per-axis scale factor or
    cross-axis misalignment, which would need a multi-orientation
    (turntable-grade) calibration procedure. A wrist-worn consumer-grade
    IMU used for rep counting (relative motion within a set) rather than
    dead-reckoning navigation (absolute position over minutes) doesn't
    need that precision; bias alone is what causes the practically
    visible failure (drifting stillness reads as motion).
    """

    accel_bias: np.ndarray  # (3,) m/s^2, subtract from raw accel
    gyro_bias: np.ndarray  # (3,) rad/s, subtract from raw gyro
    n_samples: int
    accel_noise_std: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_noise_std: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.accel_bias = np.asarray(self.accel_bias, dtype=float)
        self.gyro_bias = np.asarray(self.gyro_bias, dtype=float)
        self.accel_noise_std = np.asarray(self.accel_noise_std, dtype=float)
        self.gyro_noise_std = np.asarray(self.gyro_noise_std, dtype=float)


def calibrate_stationary(
    samples: List[IMUSample],
    gravity_axis: int = 2,
    gravity_sign: float = 1.0,
) -> IMUCalibration:
    """Estimate accel/gyro bias from a short stationary period.

    Standard strapdown-IMU static calibration: while the band sits still
    (on a table, in a front-desk charging cradle), true angular velocity
    is exactly zero, so any measured gyro signal *is* bias; true
    acceleration is exactly gravity along whichever axis is "up" for
    however the band happens to be resting, so any deviation (including
    on the "up" axis, from exactly ``GRAVITY_M_S2``) is accel bias.
    Averaging over many samples cancels sensor noise, leaving the
    systematic bias.

    ``gravity_axis``/``gravity_sign`` describe which of the band's local
    axes points up and in which direction while it's resting for
    calibration -- e.g. axis 2 (z), sign +1 is this module's default and
    matches ``irix.wristband_sim.simulator.SimulatedWristband``'s "idle"
    motion program.

    Raises ``ValueError`` on fewer than ``MIN_CALIBRATION_SAMPLES`` --
    not enough to average sensor noise out into a trustworthy estimate.
    """
    if len(samples) < MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"need at least {MIN_CALIBRATION_SAMPLES} stationary samples to "
            f"calibrate, got {len(samples)}"
        )

    accel = np.stack([s.accel for s in samples])
    gyro = np.stack([s.gyro for s in samples])

    gyro_bias = gyro.mean(axis=0)

    expected_gravity = np.zeros(3)
    expected_gravity[gravity_axis] = gravity_sign * GRAVITY_M_S2
    accel_bias = accel.mean(axis=0) - expected_gravity

    return IMUCalibration(
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        n_samples=len(samples),
        accel_noise_std=accel.std(axis=0),
        gyro_noise_std=gyro.std(axis=0),
    )


def apply_calibration(sample: IMUSample, calibration: IMUCalibration) -> IMUSample:
    """Bias-corrected copy of ``sample`` -- does not mutate the input, so
    a caller can keep the raw sample around (e.g. for debugging/logging)
    alongside the corrected one fed to fusion/rep-counting."""
    return IMUSample(
        timestamp=sample.timestamp,
        accel=sample.accel - calibration.accel_bias,
        gyro=sample.gyro - calibration.gyro_bias,
    )


def apply_calibration_batch(samples: List[IMUSample], calibration: IMUCalibration) -> List[IMUSample]:
    return [apply_calibration(s, calibration) for s in samples]
