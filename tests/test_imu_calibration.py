from __future__ import annotations

import numpy as np
import pytest

from irix.fusion.imu import IMUSample
from irix.wristband_sim.calibration import (
    GRAVITY_M_S2,
    apply_calibration,
    apply_calibration_batch,
    calibrate_stationary,
)


def _stationary_samples(accel_bias, gyro_bias, n=200, seed=0):
    rng = np.random.default_rng(seed)
    samples = []
    gravity = np.array([0.0, 0.0, GRAVITY_M_S2])
    for i in range(n):
        accel = gravity + np.asarray(accel_bias) + rng.normal(0, 0.01, 3)
        gyro = np.asarray(gyro_bias) + rng.normal(0, 0.005, 3)
        samples.append(IMUSample(timestamp=i / 100.0, accel=accel, gyro=gyro))
    return samples


def test_calibrate_stationary_recovers_known_bias():
    true_accel_bias = np.array([0.1, -0.05, 0.2])
    true_gyro_bias = np.array([0.02, -0.01, 0.03])
    samples = _stationary_samples(true_accel_bias, true_gyro_bias, n=500, seed=1)

    calibration = calibrate_stationary(samples)

    np.testing.assert_allclose(calibration.accel_bias, true_accel_bias, atol=0.01)
    np.testing.assert_allclose(calibration.gyro_bias, true_gyro_bias, atol=0.005)
    assert calibration.n_samples == 500


def test_calibrate_stationary_requires_minimum_samples():
    samples = _stationary_samples(np.zeros(3), np.zeros(3), n=5)
    with pytest.raises(ValueError):
        calibrate_stationary(samples)


def test_apply_calibration_removes_bias_without_mutating_input():
    sample = IMUSample(timestamp=0.0, accel=np.array([0.2, 0.1, 9.9]), gyro=np.array([0.03, -0.02, 0.01]))
    from irix.wristband_sim.calibration import IMUCalibration

    calibration = IMUCalibration(
        accel_bias=np.array([0.2, 0.1, -0.1]), gyro_bias=np.array([0.03, -0.02, 0.01]), n_samples=100,
    )

    corrected = apply_calibration(sample, calibration)

    np.testing.assert_allclose(corrected.accel, [0.0, 0.0, 10.0])
    np.testing.assert_allclose(corrected.gyro, [0.0, 0.0, 0.0])
    # original untouched
    np.testing.assert_allclose(sample.accel, [0.2, 0.1, 9.9])


def test_apply_calibration_batch():
    from irix.wristband_sim.calibration import IMUCalibration

    samples = _stationary_samples(np.array([1.0, 0.0, 0.0]), np.zeros(3), n=10)
    calibration = IMUCalibration(accel_bias=np.array([1.0, 0.0, 0.0]), gyro_bias=np.zeros(3), n_samples=10)

    corrected = apply_calibration_batch(samples, calibration)

    assert len(corrected) == len(samples)
    for s in corrected:
        assert abs(s.accel[0]) < 0.1  # x-axis bias removed
        assert abs(s.accel[2] - GRAVITY_M_S2) < 0.1  # z-axis untouched, still pure gravity
