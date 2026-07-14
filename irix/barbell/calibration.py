"""Camera calibration for converting pixel measurements to real-world
distances (Section 4.5 / 4.4's "calibrated against the known camera
geometry for that station").

Two calibration strategies, following real precedent from existing
open-source barbell trackers:

1. **Self-calibration from a detected object's known standard dimension**
   (the default approach here). Olympic barbells and competition plates
   are manufactured to standardized sizes, so a detected barbell/plate's
   pixel size, combined with its known real-world size, gives a
   pixels-per-mm scale factor with no extra install step and no equipment
   modification -- consistent with the cameras-only install constraint
   that already shaped ``irix/weight_recognition`` (see
   ``vision_classifier.py``).

   github.com/kostecky/VBT-Barbell-Tracker uses the identical principle
   -- it measures the pixel diameter of a marker of known physical size
   to derive a px-per-mm scale -- just against a painted marker instead
   of the equipment's own geometry, which isn't usable here since
   painting/marking equipment is an environment edit. Using a plate's
   already-known diameter, or the bar's already-known length, gets the
   same self-calibration property without touching the equipment.

2. **One-time camera intrinsic calibration** (focal length, lens
   distortion) via a checkerboard -- the standard
   ``cv2.calibrateCamera``/``cv2.fisheye`` workflow, also used by
   VBT-Barbell-Tracker (its ``undistort_fisheye.py``). This is a
   legitimate one-time install-time step (photograph a checkerboard from
   the already-mounted camera, nothing added to the gym floor
   afterward), and materially improves accuracy for wide-angle lenses
   where straight lines bow near the frame edges. Sketched here as
   ``undistort_frame``; not required for strategy 1 to work at all, just
   a precision upgrade path.

**Known limitation, stated plainly**: this treats the local px-per-mm
scale as isotropic across a station's field of view (same scale
horizontally and vertically, no full 3D camera pose/perspective
correction). That's the same level of rigor VBT-Barbell-Tracker itself
uses, and is reasonable given a station's limited working-distance range
(Section 3.1: 3-4m, fixed per station) -- but it is a simplification, not
a full photogrammetric solve. A future upgrade path is a per-station
homography computed at install time (four known floor/rack points),
which would remove this limitation; out of scope for this scaffold.

**Camera-tilt correction for vertical bar-path distance**, borrowed from
real velocity-based-training hardware: GymAware -- the linear-position-
transducer (LPT) considered the VBT gold standard -- explicitly corrects
for the angle between its cable and the bar's true vertical path, since a
sensor not perfectly aligned with the true direction of motion
foreshortens the observed distance for a given real displacement. A
camera has the same problem: if it isn't mounted with its optical axis
exactly perpendicular to the bar's vertical plane of travel (angled down
to see a whole rack, or off to one side), the same real vertical
displacement produces a smaller vertical pixel delta than a level,
perpendicular camera would see, so a raw pixel-to-mm conversion
*underestimates* true vertical bar velocity. ``camera_tilt_deg`` on
``CameraCalibration`` (default ``0.0``, i.e. no correction -- backward
compatible) is a first-order cosine correction for exactly this, applied
via ``pixels_to_vertical_m`` -- set once per station at install time
(e.g. read off a level/inclinometer when the camera is mounted), not
derived automatically. Same rigor level as the isotropic-scale
simplification above: a planar approximation assuming tilt is the
dominant source of foreshortening, not a substitute for the full-
homography upgrade path already noted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Reference dimensions for standardized gym equipment (mm). These are
# common manufacturing standards, not universal across every gym -- plate
# diameter in particular varies somewhat (roughly 400-450mm) across
# non-competition/cast-iron plate manufacturers, per the same variability
# noted in irix/weight_recognition's reasoning for why plate *color/text*
# isn't reliable either. Treat these as reasonable defaults, not ground
# truth for a specific gym's equipment -- override with a measured value
# for that gym's plates/bars where precision matters.
MENS_OLYMPIC_BARBELL_LENGTH_MM = 2200.0
WOMENS_OLYMPIC_BARBELL_LENGTH_MM = 2010.0
MENS_OLYMPIC_BARBELL_SLEEVE_LENGTH_MM = 415.0
COMPETITION_BUMPER_PLATE_DIAMETER_MM = 450.0  # IWF standard, same across all weights
MENS_OLYMPIC_BARBELL_WEIGHT_KG = 20.0
WOMENS_OLYMPIC_BARBELL_WEIGHT_KG = 15.0


@dataclass
class CameraCalibration:
    """A per-station px-per-mm scale factor, derived once (or refreshed
    periodically) from a detected reference object of known size.

    ``camera_tilt_deg``: the angle (degrees) between this station's
    camera's optical axis and true horizontal/perpendicular-to-bar-path --
    see the module docstring's "camera-tilt correction" section. Defaults
    to ``0.0`` (no correction), so every existing caller and every
    pre-existing test is unaffected unless a station explicitly supplies
    a measured tilt.
    """

    pixels_per_mm: float
    station_id: str
    camera_tilt_deg: float = 0.0

    def pixels_to_mm(self, pixels: float) -> float:
        return pixels / self.pixels_per_mm

    def pixels_to_m(self, pixels: float) -> float:
        return self.pixels_to_mm(pixels) / 1000.0

    def pixels_to_vertical_m(self, pixels: float) -> float:
        """Like ``pixels_to_m``, but corrected for ``camera_tilt_deg`` --
        use this (not ``pixels_to_m``) when converting a *vertical bar-path*
        pixel delta to a real-world distance, since that's the measurement
        a camera tilt actually foreshortens (see the module docstring).
        Not appropriate for a lateral measurement like a plate's pixel
        diameter, which ``pixels_to_m`` already handles correctly.
        """
        return self.pixels_to_m(pixels) / math.cos(math.radians(self.camera_tilt_deg))


def calibrate_from_known_object(
    pixel_size: float,
    real_world_size_mm: float,
    station_id: str,
    camera_tilt_deg: float = 0.0,
) -> CameraCalibration:
    """Derive a CameraCalibration from one detected reference measurement.

    ``pixel_size`` is the measured pixel extent of some reference feature
    in a frame (e.g. a detected plate's pixel diameter, or a detected
    barbell's pixel length); ``real_world_size_mm`` is that feature's
    known real-world size (e.g. ``COMPETITION_BUMPER_PLATE_DIAMETER_MM``).
    Call this once per station at install time, or periodically re-derive
    it from fresh detections to catch camera/zoom drift.

    ``camera_tilt_deg`` (default ``0.0``, no correction) is this station's
    measured camera-mounting tilt -- see ``CameraCalibration.
    pixels_to_vertical_m`` and the module docstring.
    """
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")
    return CameraCalibration(
        pixels_per_mm=pixel_size / real_world_size_mm,
        station_id=station_id,
        camera_tilt_deg=camera_tilt_deg,
    )


def undistort_frame(frame: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> np.ndarray:
    """Apply a one-time OpenCV lens-distortion correction (strategy 2
    above). ``camera_matrix``/``dist_coeffs`` come from a standard
    ``cv2.calibrateCamera`` checkerboard calibration run at install time
    -- see the module docstring. Precision upgrade, not required for
    ``calibrate_from_known_object`` to function.
    """
    import cv2

    return cv2.undistort(frame, camera_matrix, dist_coeffs)
