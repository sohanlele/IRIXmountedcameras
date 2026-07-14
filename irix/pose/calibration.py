"""Camera intrinsic/extrinsic calibration workflow.

## Why this exists

``irix.pose.multiview.CameraProjection`` (intrinsic ``K`` + extrinsic
``R``/``t``) and ``irix.barbell.calibration.undistort_frame``
(``camera_matrix``/``dist_coeffs``) both *consume* a calibrated camera
model -- neither one *produces* it. Both modules' own docstrings state
this plainly ("a real deployment derives rotation/translation... via a
standard multi-camera extrinsic calibration procedure... out of scope
here"). This module is that missing procedure: the actual
checkerboard-based intrinsic + extrinsic calibration workflow, using
OpenCV's standard, well-established implementation (Zhang's method,
``cv2.calibrateCamera``; ``cv2.solvePnP`` for a calibrated camera's pose
relative to a shared world frame) -- not a novel algorithm, a real
production procedure every multi-camera CV deployment (including
``irix.pose.multiview``'s own stated assumption) runs at install time.

## Workflow

1. **Intrinsic calibration** (``calibrate_intrinsics``): capture several
   (>=5, ideally 15-20 for a well-conditioned solve) images of a planar
   checkerboard from different distances/angles with *one* camera,
   fixed focal length/zoom throughout. Recovers that camera's ``K``
   (focal length, principal point) and lens distortion coefficients --
   intrinsic to the camera/lens, independent of where it's mounted.
2. **Extrinsic calibration** (``calibrate_extrinsics``): with intrinsics
   already known, one image of the *same* checkerboard placed at a fixed
   position on the gym floor (the shared world-frame origin every
   camera in a zone calibrates against) recovers that specific camera's
   ``R``/``t`` -- where it is and how it's oriented relative to that
   shared origin. Every camera in one ``irix.live.zone_runner.
   MultiCameraZoneRunner`` zone needs to calibrate against the *same*
   physical checkerboard placement for their resulting ``R``/``t`` to
   compose into one shared 3D frame (a wrong/inconsistent placement is
   the single most common real-world multi-camera calibration mistake --
   worth stating plainly for whoever runs this at install time).
3. **Quality is reported, not assumed** (``IntrinsicCalibrationResult.
   quality``/``ExtrinsicCalibrationResult.reprojection_error_px``): mean
   reprojection error in pixels is the standard, interpretable
   calibration-quality metric -- how far off, in pixels, the calibrated
   model's own predicted corner positions land from where the corners
   were actually detected. Sub-pixel (<0.5px) is excellent for a good
   checkerboard/lighting setup; >2px means recalibrate before trusting
   this camera's numbers for triangulation (``irix.pose.multiview``) or
   bar-velocity work (``irix.barbell``).
4. **Store the result** (``CalibrationProfile.save``/``.load``) so
   calibration is a one-time (or periodic) install-time step, not
   something recomputed at runtime.

``compute_ground_plane_homography`` is the simpler, single-camera
alternative for zones that don't need full 3D triangulation: a planar
homography (pixel <-> gym-floor-plane coordinates) from as few as 4
known point correspondences (e.g. taped floor markers at measured
positions), useful for station-position/zone-boundary mapping without a
full checkerboard-based 3D calibration.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .multiview import CameraProjection

DEFAULT_MIN_CALIBRATION_IMAGES = 5


def _checkerboard_object_points(pattern_size: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    """The checkerboard's own interior-corner grid in its local 3D frame
    (Z=0, since it's planar), in the units ``square_size_mm`` is given in
    -- e.g. millimeters, so recovered translations come out in
    millimeters too, consistent with the rest of this repo's convention
    (``irix.barbell.calibration`` also works in mm)."""
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_mm
    return objp


def find_checkerboard_corners(image, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Sub-pixel-refined interior checkerboard corners, or ``None`` if
    the pattern wasn't found in this image -- never raises on a bad
    image, since "this particular capture didn't have a clean view of
    the board" is an expected, common outcome during a real calibration
    capture session, not an error."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


@dataclass
class IntrinsicCalibrationResult:
    camera_matrix: np.ndarray  # 3x3 K
    dist_coeffs: np.ndarray  # (5,) [k1, k2, p1, p2, k3] -- OpenCV's standard plumb-bob model
    image_size: Tuple[int, int]  # (width, height)
    reprojection_error_px: float  # RMS across every used image -- the quality signal, see module docstring
    n_images_used: int
    n_images_rejected: int
    rejected_reasons: List[str] = field(default_factory=list)

    @property
    def quality(self) -> str:
        if self.reprojection_error_px < 0.5:
            return "excellent"
        if self.reprojection_error_px < 1.0:
            return "good"
        if self.reprojection_error_px < 2.0:
            return "marginal"
        return "poor"

    def to_dict(self) -> dict:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "reprojection_error_px": self.reprojection_error_px,
            "quality": self.quality,
            "n_images_used": self.n_images_used,
            "n_images_rejected": self.n_images_rejected,
            "rejected_reasons": self.rejected_reasons,
        }

    @staticmethod
    def from_dict(d: dict) -> "IntrinsicCalibrationResult":
        return IntrinsicCalibrationResult(
            camera_matrix=np.array(d["camera_matrix"]), dist_coeffs=np.array(d["dist_coeffs"]),
            image_size=tuple(d["image_size"]), reprojection_error_px=d["reprojection_error_px"],
            n_images_used=d["n_images_used"], n_images_rejected=d["n_images_rejected"],
            rejected_reasons=d.get("rejected_reasons", []),
        )


def calibrate_intrinsics(
    images: List[np.ndarray],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
    min_images: int = DEFAULT_MIN_CALIBRATION_IMAGES,
) -> IntrinsicCalibrationResult:
    """Standard checkerboard intrinsic calibration (Zhang's method, via
    ``cv2.calibrateCamera``) across several images of one camera.

    Raises ``ValueError`` if fewer than ``min_images`` images actually
    had a detectable checkerboard -- a calibration computed from too few
    views is unreliable (under-constrained, especially for distortion
    coefficients) and reporting a falsely-precise-looking result from it
    would be worse than failing loudly with a clear reason.
    """
    object_points_template = _checkerboard_object_points(pattern_size, square_size_mm)
    obj_points_list, img_points_list = [], []
    rejected_reasons = []
    image_size: Optional[Tuple[int, int]] = None

    for i, image in enumerate(images):
        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            rejected_reasons.append(f"image {i}: size {(w, h)} != first image's {image_size}")
            continue
        corners = find_checkerboard_corners(image, pattern_size)
        if corners is None:
            rejected_reasons.append(f"image {i}: checkerboard pattern {pattern_size} not found")
            continue
        obj_points_list.append(object_points_template)
        img_points_list.append(corners)

    if len(obj_points_list) < min_images:
        raise ValueError(
            f"only {len(obj_points_list)} of {len(images)} images had a usable checkerboard "
            f"(need >= {min_images}): {rejected_reasons}"
        )

    rms_error, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points_list, img_points_list, image_size, None, None,
    )

    return IntrinsicCalibrationResult(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs.flatten(), image_size=image_size,
        reprojection_error_px=float(rms_error), n_images_used=len(obj_points_list),
        n_images_rejected=len(rejected_reasons), rejected_reasons=rejected_reasons,
    )


@dataclass
class ExtrinsicCalibrationResult:
    rotation: np.ndarray  # 3x3, world -> camera
    translation: np.ndarray  # (3,), world -> camera, same units as square_size_mm
    reprojection_error_px: float

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "reprojection_error_px": self.reprojection_error_px,
        }

    @staticmethod
    def from_dict(d: dict) -> "ExtrinsicCalibrationResult":
        return ExtrinsicCalibrationResult(
            rotation=np.array(d["rotation"]), translation=np.array(d["translation"]),
            reprojection_error_px=d["reprojection_error_px"],
        )


def calibrate_extrinsics(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_mm: float = 25.0,
) -> Optional[ExtrinsicCalibrationResult]:
    """One checkerboard image (already-intrinsically-calibrated camera)
    -> that camera's pose relative to the checkerboard's own coordinate
    frame, via ``cv2.solvePnP``. Every camera in a zone must calibrate
    against the *same physical checkerboard placement* for the results
    to share one consistent world frame -- see module docstring.

    Returns ``None`` (not a raised exception) if the checkerboard wasn't
    found in this image -- an expected, retryable outcome during a real
    capture session (recapture the extrinsic image and call again),
    distinct from ``calibrate_intrinsics``' harder failure (too few
    usable images across a whole intrinsic capture set).
    """
    corners = find_checkerboard_corners(image, pattern_size)
    if corners is None:
        return None
    object_points = _checkerboard_object_points(pattern_size, square_size_mm)

    ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not ok:
        return None

    rotation, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = float(np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean())

    return ExtrinsicCalibrationResult(rotation=rotation, translation=tvec.flatten(), reprojection_error_px=error)


@dataclass
class CalibrationProfile:
    """A stored, complete calibration for one camera -- the artifact a
    real install-time calibration workflow produces and every later
    process (triangulation, undistortion, velocity tracking) loads
    rather than recalibrating."""

    camera_id: str
    intrinsic: IntrinsicCalibrationResult
    extrinsic: Optional[ExtrinsicCalibrationResult] = None
    calibrated_at: float = field(default_factory=time.time)

    @property
    def quality(self) -> str:
        """The worse of intrinsic/extrinsic quality -- a camera is only
        as well-calibrated as its weakest link; a great intrinsic
        calibration paired with a poor extrinsic solve still produces
        bad triangulated 3D positions."""
        levels = ["excellent", "good", "marginal", "poor"]
        worst = levels.index(self.intrinsic.quality)
        if self.extrinsic is not None:
            extrinsic_bucket = (
                "excellent" if self.extrinsic.reprojection_error_px < 0.5 else
                "good" if self.extrinsic.reprojection_error_px < 1.0 else
                "marginal" if self.extrinsic.reprojection_error_px < 2.0 else "poor"
            )
            worst = max(worst, levels.index(extrinsic_bucket))
        return levels[worst]

    def to_camera_projection(self) -> CameraProjection:
        """Convert to the shape ``irix.pose.multiview`` triangulation
        needs. Requires extrinsics -- raises ``ValueError`` (not a silent
        identity-transform guess) if only intrinsics have been run so
        far, since an uncalibrated position/orientation is worse than an
        explicit error."""
        if self.extrinsic is None:
            raise ValueError(
                f"camera {self.camera_id!r} has no extrinsic calibration yet -- "
                "run calibrate_extrinsics() before converting to a CameraProjection"
            )
        return CameraProjection(
            camera_id=self.camera_id, intrinsic=self.intrinsic.camera_matrix,
            rotation=self.extrinsic.rotation, translation=self.extrinsic.translation,
        )

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "intrinsic": self.intrinsic.to_dict(),
            "extrinsic": self.extrinsic.to_dict() if self.extrinsic else None,
            "calibrated_at": self.calibrated_at,
            "quality": self.quality,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> "CalibrationProfile":
        with open(path) as f:
            d = json.load(f)
        return CalibrationProfile(
            camera_id=d["camera_id"],
            intrinsic=IntrinsicCalibrationResult.from_dict(d["intrinsic"]),
            extrinsic=ExtrinsicCalibrationResult.from_dict(d["extrinsic"]) if d.get("extrinsic") else None,
            calibrated_at=d["calibrated_at"],
        )


def compute_ground_plane_homography(
    image_points: np.ndarray, world_points: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Planar homography from >=4 known pixel<->gym-floor-plane point
    correspondences (e.g. taped floor markers at measured positions) --
    the lighter-weight alternative to full checkerboard-based 3D
    calibration for a single camera that only needs pixel<->floor-
    position mapping (station/zone-boundary layout), not 3D
    triangulation. Returns ``(H, mean_reprojection_error_px_equivalent)``
    -- the second value is in ``world_points``' own units (e.g. meters),
    not pixels, since ``H`` maps pixels to world coordinates.
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    world_points = np.asarray(world_points, dtype=np.float64)
    if len(image_points) < 4:
        raise ValueError(f"need >= 4 point correspondences for a homography, got {len(image_points)}")

    method = cv2.RANSAC if len(image_points) > 4 else 0
    H, _mask = cv2.findHomography(image_points, world_points, method=method)

    ones = np.ones((len(image_points), 1))
    homogeneous = np.hstack([image_points, ones])
    projected = (H @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    error = float(np.linalg.norm(projected - world_points, axis=1).mean())
    return H, error
