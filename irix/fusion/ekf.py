"""Visual-inertial Extended Kalman Filter (Section 4.6).

Fuses camera-derived absolute wrist/bar position (accurate, drift-free, but
only available at frame rate and occasionally lost to occlusion) with
wristband IMU acceleration (high rate, but drifts quickly when integrated
into velocity/position on its own). This is the same class of technique
(visual-inertial odometry) used by AR/VR headsets and drones.

State vector: [position (3), velocity (3)] in a single fixed reference axis
(e.g. vertical bar-path displacement). This scaffold implements a
constant-acceleration EKF with:
  - ``predict`` driven by IMU accel samples (high rate),
  - ``update`` driven by camera position estimates (low rate, corrects drift),
  - ``zupt_correct`` driven by ZUPT detection (Section 4.6 / OpenBarbell-style
    dead-stop re-zeroing) to bound drift between camera corrections.

This is a scaffold for the fusion math, not a production-tuned filter --
process/measurement noise (Q/R) should be re-estimated against real
camera + wristband hardware.
"""
from __future__ import annotations

import numpy as np


class VisualInertialEKF:
    def __init__(
        self,
        initial_position: float = 0.0,
        initial_velocity: float = 0.0,
        process_noise: float = 0.05,
        camera_measurement_noise: float = 0.01,
        zupt_measurement_noise: float = 0.001,
    ):
        # State: [position, velocity]
        self.x = np.array([initial_position, initial_velocity], dtype=float)
        self.P = np.eye(2) * 0.1
        self.Q_scale = process_noise
        self.R_camera = np.array([[camera_measurement_noise]])
        self.R_zupt = np.array([[zupt_measurement_noise]])
        self._last_t: float | None = None

    def predict(self, accel: float, timestamp: float) -> None:
        """Propagate state forward using an IMU accel sample (constant-acceleration model)."""
        if self._last_t is None:
            self._last_t = timestamp
            return
        dt = max(timestamp - self._last_t, 1e-6)
        self._last_t = timestamp

        F = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([0.5 * dt ** 2, dt])
        self.x = F @ self.x + B * accel

        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = self.Q_scale * np.array([[dt4 / 4, dt3 / 2], [dt3 / 2, dt2]])
        self.P = F @ self.P @ F.T + Q

    def update(self, measured_position: float) -> None:
        """Correct state using a camera-derived absolute position estimate."""
        H = np.array([[1.0, 0.0]])
        self._kalman_update(H, np.array([measured_position]), self.R_camera)

    def zupt_correct(self) -> None:
        """Correct state assuming velocity is zero (dead-stop / ZUPT event)."""
        H = np.array([[0.0, 1.0]])
        self._kalman_update(H, np.array([0.0]), self.R_zupt)

    def _kalman_update(self, H: np.ndarray, z: np.ndarray, R: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ H) @ self.P

    @property
    def position(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        return float(self.x[1])
