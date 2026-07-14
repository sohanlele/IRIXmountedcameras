import numpy as np

from irix.fusion.ekf import VisualInertialEKF
from irix.fusion.imu import IMUSample
from irix.fusion.zupt import detect_zero_velocity


def test_ekf_tracks_camera_position_updates():
    ekf = VisualInertialEKF(initial_position=0.0, initial_velocity=0.0)
    ekf.predict(accel=0.0, timestamp=0.0)
    ekf.predict(accel=0.0, timestamp=0.1)
    ekf.update(measured_position=1.0)
    assert abs(ekf.position - 1.0) < 0.5


def test_zupt_correct_pulls_velocity_toward_zero():
    ekf = VisualInertialEKF(initial_position=0.0, initial_velocity=2.0)
    ekf.predict(accel=1.0, timestamp=0.0)
    ekf.predict(accel=1.0, timestamp=0.1)
    v_before = ekf.velocity
    ekf.zupt_correct()
    assert abs(ekf.velocity) < abs(v_before)


def test_detect_zero_velocity_stationary_window():
    samples = [
        IMUSample(timestamp=i * 0.01, accel=np.array([0.0, 0.0, 9.81]), gyro=np.array([0.0, 0.0, 0.0]))
        for i in range(10)
    ]
    assert detect_zero_velocity(samples) is True


def test_detect_zero_velocity_moving_window():
    samples = [
        IMUSample(
            timestamp=i * 0.01,
            accel=np.array([0.0, 0.0, 9.81 + i * 2.0]),
            gyro=np.array([0.0, 0.0, i * 0.5]),
        )
        for i in range(10)
    ]
    assert detect_zero_velocity(samples) is False
