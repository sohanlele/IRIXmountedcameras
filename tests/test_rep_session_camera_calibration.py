"""RepSession.process_frame's camera_id parameter -- proves the fix for
the multi-camera-zone known limitation flagged in irix.live.zone_runner:
when a member's ongoing set gets routed to a different physical camera
mid-set, bar velocity must switch to *that camera's own* calibration
rather than keep applying whichever camera calibrated first.

Uses RepSession directly (not the full MultiCameraZoneRunner) since the
thing being proven -- calibration correctness across a camera_id switch
-- is a RepSession/BarPathTracker concern, and driving it directly here
keeps the test precise and fast rather than routed through disambiguation
machinery irrelevant to this fix.
"""
from __future__ import annotations

import numpy as np
import pytest

from irix.barbell.calibration import calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from irix.barbell.detector import FreeWeightClass, FreeWeightDetection
from irix.pipeline.rep_session import RepSession


class _TwoCameraBarbellDetector:
    """Stands in for a real FreeWeightDetector fed frames from two
    different physical cameras -- the frame's [0, 0, 0] pixel value is
    the test's own marker for "which camera this came from" (0.0 =
    camera A, 1.0 = camera B); a real detector would naturally observe a
    different pixel_diameter for the *same* physical plate from each
    camera's distinct distance/focal length/angle, this just encodes that
    difference explicitly instead of standing up two real detector
    models."""

    def __init__(self, diameter_by_marker):
        self._diameter_by_marker = diameter_by_marker
        self._frame_i = 0

    def detect(self, frame: np.ndarray):
        i = self._frame_i
        self._frame_i += 1
        marker = float(frame[0, 0, 0])
        plate_diameter = self._diameter_by_marker[marker]
        half = plate_diameter / 2.0
        plate = FreeWeightDetection(
            class_label=FreeWeightClass.PLATE,
            centroid_px=(300.0, 500.0),
            bbox_px=(300.0 - half, 500.0 - half, 300.0 + half, 500.0 + half),
            confidence=0.9,
        )
        bar_y = 1000.0 - 5.0 * i
        barbell = FreeWeightDetection(
            class_label=FreeWeightClass.BARBELL,
            centroid_px=(300.0, bar_y),
            bbox_px=(100.0, bar_y - 10.0, 500.0, bar_y + 10.0),
            confidence=0.9,
        )
        return [plate, barbell]


def _frame_for(marker: float) -> np.ndarray:
    frame = np.zeros((2, 2, 3), dtype=np.float64)
    frame[0, 0, 0] = marker
    return frame


def test_each_camera_self_calibrates_independently():
    """Camera A and camera B see the same physical plate at different
    pixel sizes (different physical cameras) -- each camera_id must get
    its own CameraCalibration, not share camera A's."""
    detector = _TwoCameraBarbellDetector({0.0: 180.0, 1.0: 360.0})
    session = RepSession(
        exercise_name="squat", member_id="m1", station_id="zone-1", barbell_detector=detector,
    )
    session.process_frame(_frame_for(0.0), ts=0.0, person=None, camera_id="cam-a")
    session.process_frame(_frame_for(1.0), ts=1.0 / 30, person=None, camera_id="cam-b")

    assert "cam-a" in session._bar_calibrations
    assert "cam-b" in session._bar_calibrations
    cal_a = session._bar_calibrations["cam-a"]
    cal_b = session._bar_calibrations["cam-b"]
    # cam-b's plate was detected at double the pixel diameter of cam-a's
    # -> cam-b's px-per-mm should be double cam-a's (same real plate).
    assert cal_b.pixels_per_mm == pytest.approx(cal_a.pixels_per_mm * 2.0, rel=1e-6)


def test_velocity_uses_the_camera_that_produced_each_pixel_measurement():
    """The core regression this fix targets: a set that starts on camera
    A and (mid-set) switches to camera B must compute velocity using each
    camera's own calibration for the samples it produced -- not misapply
    camera A's scale to camera B's pixels (or vice versa)."""
    diameter_by_marker = {0.0: 180.0, 1.0: 360.0}  # cam-b appears "closer" -> larger pixel plate
    detector = _TwoCameraBarbellDetector(diameter_by_marker)
    session = RepSession(
        exercise_name="squat", member_id="m1", station_id="zone-1", barbell_detector=detector,
    )

    fps = 30.0
    # First half of the set: routed via cam-a. Second half: routed via cam-b.
    for i in range(16):
        session.process_frame(_frame_for(0.0), ts=i / fps, person=None, camera_id="cam-a")
    for i in range(16, 31):
        session.process_frame(_frame_for(1.0), ts=i / fps, person=None, camera_id="cam-b")

    # Compute what each camera's own calibration says the bar-path
    # displacement over the whole window should be, using the exact
    # pixel schedule _TwoCameraBarbellDetector produces (bar_y = 1000 -
    # 5*i), then compare against what BarPathTracker actually recorded.
    cal_a = calibrate_from_known_object(
        pixel_size=180.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id="zone-1",
    )
    cal_b = calibrate_from_known_object(
        pixel_size=360.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id="zone-1",
    )
    expected_positions = []
    for i in range(31):
        bar_y = 1000.0 - 5.0 * i
        cal = cal_a if i < 16 else cal_b
        expected_positions.append(-cal.pixels_to_vertical_m(bar_y))

    result = session.bar_tracker.velocity_for_window(0.0, 30.0 / fps)
    expected_displacement = expected_positions[-1] - expected_positions[0]
    assert result.displacement_m == pytest.approx(expected_displacement, rel=1e-6)

    # Sanity: if the switch had been ignored (cam-a's calibration
    # misapplied to cam-b's pixels the whole way, the pre-fix bug this
    # test targets), the displacement would come out different, since
    # cal_a and cal_b have different px-per-mm scales.
    wrong_displacement = (-cal_a.pixels_to_vertical_m(1000.0 - 5.0 * 30)) - (-cal_a.pixels_to_vertical_m(1000.0))
    assert result.displacement_m != pytest.approx(wrong_displacement, rel=1e-6)


def test_camera_tilt_deg_by_camera_threads_through_self_calibration():
    """A per-camera tilt override (camera_tilt_deg_by_camera) should be
    used when a given camera_id self-calibrates, falling back to the
    shared camera_tilt_deg for any camera_id not in the map."""
    detector = _TwoCameraBarbellDetector({0.0: 180.0, 1.0: 180.0})
    session = RepSession(
        exercise_name="squat",
        member_id="m1",
        station_id="zone-1",
        barbell_detector=detector,
        camera_tilt_deg=0.0,
        camera_tilt_deg_by_camera={"cam-b": 30.0},
    )
    session.process_frame(_frame_for(0.0), ts=0.0, person=None, camera_id="cam-a")
    session.process_frame(_frame_for(1.0), ts=1.0 / 30, person=None, camera_id="cam-b")

    assert session._bar_calibrations["cam-a"].camera_tilt_deg == pytest.approx(0.0)
    assert session._bar_calibrations["cam-b"].camera_tilt_deg == pytest.approx(30.0)


def test_single_camera_case_unaffected_camera_id_defaults_to_none():
    """The common single-camera path (StationSessionRunner, run_upload)
    never passes camera_id -- confirms that still self-calibrates once,
    under the None key, exactly as before this fix."""
    detector = _TwoCameraBarbellDetector({0.0: 180.0})
    session = RepSession(
        exercise_name="squat", member_id="m1", station_id="zone-1", barbell_detector=detector,
    )
    for i in range(5):
        session.process_frame(_frame_for(0.0), ts=i / 30.0, person=None)
    assert list(session._bar_calibrations.keys()) == [None]
    assert session.bar_tracker is not None
