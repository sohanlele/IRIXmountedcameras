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
    converts it to meters via a ``CameraCalibration`` before buffering, so
    all downstream math is in real units. Uses ``CameraCalibration.
    pixels_to_vertical_m`` (not ``pixels_to_m``), so a station with a
    nonzero ``camera_tilt_deg`` gets its bar-path distance (and therefore
    velocity) corrected for that tilt -- see ``irix.barbell.calibration``'s
    module docstring.

    ``calibration`` passed to the constructor is the *default* used when
    ``push()`` isn't given a more specific one -- see ``push()``'s
    docstring for why a caller (``irix.pipeline.rep_session.RepSession``,
    for a member tracked across more than one camera in ``irix.live.
    zone_runner.MultiCameraZoneRunner``) might supply a different
    calibration per call rather than relying on this one for every push.
    """

    def __init__(self, calibration: CameraCalibration, max_buffer_s: float = 30.0):
        self.calibration = calibration
        self.max_buffer_s = max_buffer_s
        self._samples: List[Tuple[float, float]] = []  # (timestamp, position_m), +y = up

    def push(self, timestamp: float, y_px: float, calibration: Optional[CameraCalibration] = None) -> None:
        """``calibration``, if given, overrides the tracker's default for
        *this call only* -- the right thing to pass whenever the pixel
        measurement being pushed came from a different camera than
        whichever one this tracker was originally constructed against
        (different camera = different actual px-per-mm scale and
        possibly a different mounting tilt, even for the same physical
        bar). Every sample already accumulated stays in real-world
        meters regardless of which calibration produced it, so a single
        continuous buffer/velocity computation still works correctly
        across a camera switch -- only the pixel-to-meters conversion at
        push time needs to know which camera this particular pixel
        measurement came from; nothing downstream does.
        """
        active_calibration = calibration if calibration is not None else self.calibration
        # Image-coordinate y increases downward; flip sign so "up" (bar
        # ascending, concentric phase of a squat/bench/deadlift) is a
        # positive position change, matching how a lifter would describe it.
        position_m = -active_calibration.pixels_to_vertical_m(y_px)
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
