"""Bar path displacement and velocity tracking (Section 4.5).

Bar path is tracked by following the barbell/dumbbell's centroid across
frames, differentiated against time to get velocity, calibrated to real
units via ``irix.barbell.calibration``. This produces genuine linear
velocity in m/s -- unlike ``irix.rep_counting.state_machine``'s
angular-velocity proxy (deg/s), which exists as a fallback for exercises
or moments where no barbell/dumbbell is being tracked (e.g. a machine
station, or a station whose free-weight detector hasn't locked on yet).

Mirrors the concentric-phase sample-buffering pattern in
``RepCounter``/``_phase_velocity`` (same idea, now over real position
instead of joint angle) so a caller can ask "what was the bar velocity
between these two timestamps" for any rep window the joint-angle counter
already found.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .calibration import CameraCalibration


@dataclass
class BarPathVelocity:
    peak_velocity_m_s: Optional[float]
    mean_velocity_m_s: Optional[float]
    displacement_m: Optional[float]


class BarPathTracker:
    """Buffers calibrated real-world vertical bar/dumbbell position over
    time and computes displacement/velocity for a given time window.

    ``push`` takes a raw pixel y-coordinate (vertical axis; "up" is
    assumed to be decreasing pixel y, standard image coordinates) and
    converts it to meters via the station's ``CameraCalibration`` before
    buffering, so all downstream math is in real units.
    """

    def __init__(self, calibration: CameraCalibration, max_buffer_s: float = 30.0):
        self.calibration = calibration
        self.max_buffer_s = max_buffer_s
        self._samples: List[Tuple[float, float]] = []  # (timestamp, position_m), +y = up

    def push(self, timestamp: float, y_px: float) -> None:
        # Image-coordinate y increases downward; flip sign so "up" (bar
        # ascending, concentric phase of a squat/bench/deadlift) is a
        # positive position change, matching how a lifter would describe it.
        position_m = -self.calibration.pixels_to_m(y_px)
        self._samples.append((timestamp, position_m))
        cutoff = timestamp - self.max_buffer_s
        self._samples = [(t, p) for t, p in self._samples if t >= cutoff]

    def reset(self) -> None:
        self._samples = []

    def velocity_for_window(self, t_start: float, t_end: float) -> BarPathVelocity:
        """Peak/mean velocity and net displacement over [t_start, t_end]."""
        window = [(t, p) for t, p in self._samples if t_start <= t <= t_end]
        if len(window) < 2:
            return BarPathVelocity(None, None, None)

        speeds = []
        for (t0, p0), (t1, p1) in zip(window, window[1:]):
            dt = t1 - t0
            if dt > 0:
                speeds.append((p1 - p0) / dt)
        if not speeds:
            return BarPathVelocity(None, None, None)

        displacement = window[-1][1] - window[0][1]
        peak = max(speeds, key=abs)
        mean = sum(speeds) / len(speeds)
        return BarPathVelocity(peak_velocity_m_s=peak, mean_velocity_m_s=mean, displacement_m=displacement)
