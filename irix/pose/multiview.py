"""Multi-view 3D pose triangulation for overlapping-FOV camera zones.

`docs/ARCHITECTURE.md`'s Section 4.3 capability table long carried "Not
implemented; PoseEstimator returns single-view poses per camera,
multi-view reprojection is future work" -- this module is that future
work. Before this, `irix.live.zone_runner.MultiCameraZoneRunner` already
solved *identity* across overlapping cameras (which physical person is
which wristband, via `irix.identity.motion_correlation`'s IMU
correlation -- see that module's docstring), but geometrically it only
ever fed *one* camera's single 2D-pixel pose per member into `RepSession`
each tick (first-camera-priority-wins, see zone_runner's own docstring on
avoiding double-counting). This module goes one step further: once 2+
cameras are already known (via that same identity resolution) to be
looking at the same physical person, fuse their independent 2D
observations of each keypoint into one triangulated 3D position.

**Why this matters for rep counting specifically, not just "more
accurate in general."** `irix.rep_counting.state_machine.RepCounter`
counts reps off a joint angle (`irix.pose.geometry.joint_angle`, hip-
knee-ankle for a squat, shoulder-elbow-wrist for a curl, etc.) computed
from a single camera's 2D pixel keypoints. A 2D angle is a *projection*
of the true 3D joint angle onto that one camera's image plane -- it's
foreshortened whenever the limb isn't moving roughly parallel to that
camera's image plane, and it can be wrong entirely when one of the three
keypoints is self-occluded from that particular angle (e.g. a barbell or
the lifter's own torso blocking the hip during a squat's bottom
position, common exactly at the position that matters most for rep
detection). A 3D angle computed from triangulated keypoints doesn't have
either problem: it's the actual joint angle regardless of any single
camera's viewing angle, and it only needs the keypoint visible to
*any two* of the zone's cameras, not the one camera currently "assigned"
to that member.

**Camera model.** `CameraProjection` is a standard calibrated pinhole
camera: a 3x3 intrinsic matrix `K` (focal length, principal point) plus
an extrinsic rotation/translation (`R`, `t`) placing that camera within
one shared zone-wide 3D coordinate frame, world point `X` mapping to
camera-local `R @ X + t` and then to a homogeneous image point via `K`.
This is a *geometric* calibration -- camera position/orientation in 3D
space -- and is deliberately a different thing from
`irix.barbell.calibration.CameraCalibration`, which is just a scalar
px-per-mm conversion along one camera's own image plane and knows
nothing about where that camera physically sits relative to any other
camera. A real deployment derives `rotation`/`translation` once per
camera at install time via a standard multi-camera extrinsic calibration
procedure (e.g. a checkerboard or ArUco marker visible to 2+ cameras at
once, solved with OpenCV's `stereoCalibrate`/`solvePnP`) -- that
procedure itself is out of scope here; this module assumes the resulting
numbers are already known and only does the triangulation math against
them.

**Triangulation method: direct linear transform (DLT).** The standard
multi-view geometry solution (Hartley & Zisserman, *Multiple View
Geometry in Computer Vision*, ch. 12) for recovering a 3D point from 2+
calibrated 2D observations of it: each view contributes two linear
constraint rows (from the cross-product form of the projection
equation), every view's rows get stacked into one matrix, and the 3D
point (in homogeneous coordinates) is the singular vector for that
matrix's smallest singular value. Generalizes cleanly from the classic
2-view case to N>=2 views without any special-casing -- more views just
means more (generally over-determined, least-squares-resolved) rows.

**Per-keypoint, not per-pose, view requirements -- same "don't fabricate
an answer" posture used elsewhere in this repo** (`irix.identity.
motion_correlation`, `irix.form.rules`, `irix.fatigue`). Not every
keypoint needs to be visible to every contributing camera in a given
tick: `triangulate_pose` triangulates each of the 17 COCO keypoints
independently, using whichever subset of the zone's cameras currently
sees that specific keypoint above `min_keypoint_confidence`. A keypoint
seen by fewer than `min_views` cameras this tick is left untriangulated
(its `Keypoint.z` stays `None`) rather than guessed at -- a caller
needing a specific 3-keypoint joint triplet (e.g.
`irix.rep_counting.exercises.EXERCISES`) still gets a fully 3D angle
whenever those particular 3 keypoints happen to be well covered, even if
other keypoints on the same person aren't this tick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .estimator import COCO_KEYPOINT_NAMES, Keypoint, PersonPose


@dataclass
class CameraProjection:
    """One calibrated camera's pinhole projection model within a shared
    zone-wide 3D coordinate frame -- see this module's docstring for the
    camera-model details and how this differs from
    `irix.barbell.calibration.CameraCalibration`."""

    camera_id: str
    intrinsic: np.ndarray  # 3x3 (K)
    rotation: np.ndarray  # 3x3 (R), world -> camera
    translation: np.ndarray  # (3,) (t), world -> camera

    @property
    def projection_matrix(self) -> np.ndarray:
        """3x4 `P = K [R | t]` -- maps a homogeneous world point
        `(X, Y, Z, 1)` to a homogeneous image point `(u, v, w)` (pixel
        coordinates are `(u/w, v/w)`)."""
        rt = np.hstack([self.rotation, np.asarray(self.translation, dtype=float).reshape(3, 1)])
        return self.intrinsic @ rt


def triangulate_point(
    observations: List[Tuple[CameraProjection, Tuple[float, float]]],
) -> Optional[np.ndarray]:
    """DLT-triangulate one 3D point from 2+ calibrated cameras' pixel
    observations of it (see module docstring for the method). Returns
    `None` if fewer than 2 observations are given (a single view can't
    recover depth at all) or the resulting linear system is degenerate
    (homogeneous scale coordinate too close to zero to normalize --
    happens for near-parallel viewing rays, e.g. two cameras placed
    almost on top of each other)."""
    if len(observations) < 2:
        return None
    rows = []
    for cam, (px, py) in observations:
        p = cam.projection_matrix
        rows.append(px * p[2] - p[0])
        rows.append(py * p[2] - p[1])
    a = np.stack(rows)
    _, _, vt = np.linalg.svd(a)
    x_h = vt[-1]
    if abs(x_h[3]) < 1e-9:
        return None
    return x_h[:3] / x_h[3]


def triangulate_pose(
    poses_by_camera: Dict[str, PersonPose],
    projections: Dict[str, CameraProjection],
    min_keypoint_confidence: float = 0.3,
    min_views: int = 2,
) -> Optional[PersonPose]:
    """Fuse 2+ cameras' single-view 2D poses of the *same
    already-identity-resolved person* into one 3D-triangulated
    `PersonPose`.

    This does no re-identification of its own -- it assumes the caller
    (`irix.live.zone_runner.MultiCameraZoneRunner`) already knows every
    entry in ``poses_by_camera`` is the same physical person, resolved
    the same way single-view routing already was (wristband-IMU motion
    correlation, see `irix.identity.motion_correlation`).

    ``poses_by_camera``: `camera_id -> that camera's PersonPose for this
    person this tick`. ``projections``: `camera_id -> CameraProjection`
    for every camera in the zone (a superset of ``poses_by_camera``'s
    keys is fine; only cameras present in both are used).

    Returns `None` if fewer than ``min_views`` cameras (after
    intersecting with ``projections``) contributed a pose at all, or if
    every individual keypoint failed its own per-keypoint view-count
    check (nothing usable to build a fused pose from).
    """
    usable = {cid: pose for cid, pose in poses_by_camera.items() if cid in projections}
    if len(usable) < min_views:
        return None

    fused_keypoints: List[Keypoint] = []
    any_triangulated = False
    for name in COCO_KEYPOINT_NAMES:
        obs_for_dlt: List[Tuple[CameraProjection, Tuple[float, float]]] = []
        confidences: List[float] = []
        for cid, pose in usable.items():
            kp = pose.get(name)
            if kp is not None and kp.confidence >= min_keypoint_confidence:
                obs_for_dlt.append((projections[cid], (kp.x, kp.y)))
                confidences.append(kp.confidence)

        point = triangulate_point(obs_for_dlt) if len(obs_for_dlt) >= min_views else None
        if point is not None:
            fused_keypoints.append(
                Keypoint(x=float(point[0]), y=float(point[1]), z=float(point[2]),
                         confidence=float(np.mean(confidences)))
            )
            any_triangulated = True
        else:
            # Not enough agreeing views (or a degenerate triangulation)
            # for this keypoint this tick -- leave it out (z=None,
            # confidence=0.0) rather than guess. x/y are unused
            # placeholders here (irix.pose.estimator.PersonPose.xyz
            # never reads them when z is None).
            fused_keypoints.append(Keypoint(x=0.0, y=0.0, z=None, confidence=0.0))

    if not any_triangulated:
        return None
    return PersonPose(keypoints=fused_keypoints)
