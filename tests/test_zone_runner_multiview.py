"""End-to-end test proving MultiCameraZoneRunner's optional multi-view 3D
pose fusion (irix.pose.multiview, wired in per the module docstring's
"Optional multi-view 3D pose fusion" section) is actually exercised in
the live tick loop, not just correct in isolation (that's what
tests/test_multiview.py already covers).

Ground truth: a bicep-curl elbow angle that genuinely varies in 3D depth
(the forearm swings through the camera's depth axis, not just its image
plane) -- both shoulder and elbow are placed at world x=0, so a single
camera positioned at x=0 and looking straight down +z sees shoulder,
elbow, and wrist all projected onto the *same vertical pixel column*
every frame (their world x is always 0, and neither shoulder nor elbow's
world y/z ever moves). That makes that one camera's raw 2D pixel "angle"
degenerate -- a same-column 2D vector only ever points straight up or
straight down, so the 2D angle it reads out is always 0 or 180 degrees,
never the true in-between values a real elbow angle takes throughout a
curl. A second camera off to the side (non-zero x) breaks the degeneracy
just by seeing a different projection, and triangulating between the two
recovers the true, continuously varying 3D angle exactly (already proven
in isolation by test_multiview.py) -- this test's job is to prove
MultiCameraZoneRunner's tick() loop actually plugs that fused pose into
RepSession and gets correct rep counts out, where a single camera alone
would not.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.zone_runner import MultiCameraZoneRunner, ZoneCamera
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.pose.multiview import CameraProjection
from irix.rep_counting.exercises import BICEP_CURL
from irix.rep_counting.state_machine import RepCounter

FPS = 30.0


def _intrinsic(fx=800.0, fy=800.0, cx=320.0, cy=240.0) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def _camera_at(camera_id: str, position: np.ndarray) -> CameraProjection:
    rotation = np.eye(3)  # looking along +z, no tilt, for simplicity
    translation = -rotation @ position
    return CameraProjection(camera_id=camera_id, intrinsic=_intrinsic(), rotation=rotation, translation=translation)


def _project(cam: CameraProjection, point_world: np.ndarray) -> tuple:
    p = cam.projection_matrix
    homog = np.append(point_world, 1.0)
    image = p @ homog
    return (float(image[0] / image[2]), float(image[1] / image[2]))


def _true_angle(t: float, reps_per_second: float) -> float:
    mid = (BICEP_CURL.top_angle + BICEP_CURL.bottom_angle) / 2  # 100
    amp = abs(BICEP_CURL.top_angle - BICEP_CURL.bottom_angle) / 2  # 60
    phase = math.sin(2 * math.pi * reps_per_second * t)
    return mid + amp * phase


def _world_points(theta_deg: float):
    """shoulder/elbow fixed at world x=0; wrist swings through the
    elbow's depth (z) axis as theta varies -- see module docstring."""
    elbow = np.array([0.0, -0.25, 5.0])
    shoulder = elbow + np.array([0.0, 0.5, 0.0])  # direction elbow->shoulder = (0, 1, 0) exactly
    theta = math.radians(theta_deg)
    forearm_length = 0.22
    wrist = elbow + forearm_length * np.array([0.0, math.cos(theta), math.sin(theta)])
    return shoulder, elbow, wrist


def _pose_from_world(cam: CameraProjection, shoulder_w, elbow_w, wrist_w) -> PersonPose:
    keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]

    def _set(name, point_world):
        px, py = _project(cam, point_world)
        keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=px, y=py, confidence=0.9)

    _set("left_shoulder", shoulder_w)
    _set("left_elbow", elbow_w)
    _set("left_wrist", wrist_w)
    return PersonPose(keypoints=keypoints)


class _ScriptedCameraPoseEstimator:
    def __init__(self, script):
        self._script = script
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._script):
            return []
        pose = self._script[self._i]
        self._i += 1
        return [pose]


class _FakeFrameSource:
    def __init__(self, n_frames):
        self._n_frames = n_frames

    def frames(self, max_frames=None, sleep=None):
        limit = self._n_frames if max_frames is None else min(self._n_frames, max_frames)
        for _ in range(limit):
            yield np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeClock:
    def __init__(self, step):
        self.t = 0.0
        self.step = step

    def __call__(self):
        val = self.t
        self.t += self.step
        return val


def _build_scripts(n_frames, reps_per_second, cam_a, cam_b):
    script_a, script_b = [], []
    for i in range(n_frames):
        t = i / FPS
        theta = _true_angle(t, reps_per_second)
        shoulder_w, elbow_w, wrist_w = _world_points(theta)
        script_a.append(_pose_from_world(cam_a, shoulder_w, elbow_w, wrist_w))
        script_b.append(_pose_from_world(cam_b, shoulder_w, elbow_w, wrist_w))
    return script_a, script_b


def _rep_events(events):
    return [e for e in events if e.to_dict()["event_type"] == "rep_completed"]


def _ground_truth_rep_count(n_frames: int, reps_per_second: float) -> int:
    """The rep count RepCounter itself would produce if fed the true 3D
    angle trace directly (no camera/pixel geometry involved at all) --
    the independent reference this test compares both the fused and the
    single-degenerate-camera run against."""
    counter = RepCounter(BICEP_CURL)
    count = 0
    for i in range(n_frames):
        t = i / FPS
        theta = _true_angle(t, reps_per_second)
        if counter.update(theta, timestamp=t) is not None:
            count += 1
    return count


def test_multiview_fusion_recovers_correct_rep_count_where_a_single_degenerate_camera_cannot():
    n_frames = 220  # ~7.3s at fps=30 -- a few full curl cycles at reps_per_second=0.5
    reps_per_second = 0.5

    cam_a = _camera_at("cam-a", position=np.array([0.0, 0.0, 0.0]))  # sits at world x=0 -> degenerate view
    cam_b = _camera_at("cam-b", position=np.array([1.2, 0.3, -1.0]))  # off to the side -> breaks the degeneracy

    # -- Control run: cam-a alone, no camera_projections (pre-existing,
    # single-camera behavior). Its raw 2D pixel angle is degenerate (see
    # module docstring): shoulder/elbow/wrist all project onto the same
    # pixel column every frame, so the 2D angle only ever reads out as
    # (numerically near) 0 or 180 degrees -- never the true in-between
    # values -- and should fail to produce the correct rep count.
    script_a_only, _ = _build_scripts(n_frames, reps_per_second, cam_a, cam_b)
    registry_solo = CheckoutRegistry()
    registry_solo.check_out("band-erin", "member-erin", timestamp=0.0)
    solo_events = []
    solo_cam = ZoneCamera("cam-a", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script_a_only)))
    present = [BLEReading(station_id="zone-1", rssi=-40.0, timestamp=0.0, wristband_id="band-erin")]
    solo_runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=registry_solo,
        cameras=[solo_cam],
        ble_reader=lambda: present,
        presence_timeout_s=1.0,
        on_events=solo_events.extend,
        clock=_FakeClock(step=1.0 / FPS),
    )
    solo_runner.run_forever(max_frames=n_frames)
    solo_runner.close()
    solo_rep_count = len(_rep_events(solo_events))

    # -- Fusion run: both cameras + camera_projections configured.
    # Should recover the true, continuously varying 3D angle and count
    # reps correctly regardless of cam-a's individual degeneracy.
    script_a, script_b = _build_scripts(n_frames, reps_per_second, cam_a, cam_b)
    registry = CheckoutRegistry()
    registry.check_out("band-erin", "member-erin", timestamp=0.0)
    fused_events = []
    cam_a_zone = ZoneCamera("cam-a", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script_a)))
    cam_b_zone = ZoneCamera("cam-b", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script_b)))
    fusion_runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=registry,
        cameras=[cam_a_zone, cam_b_zone],
        ble_reader=lambda: present,
        presence_timeout_s=1.0,
        on_events=fused_events.extend,
        clock=_FakeClock(step=1.0 / FPS),
        camera_projections={"cam-a": cam_a, "cam-b": cam_b},
    )
    fusion_runner.run_forever(max_frames=n_frames)
    fusion_runner.close()
    fused_rep_count = len(_rep_events(fused_events))

    ground_truth = _ground_truth_rep_count(n_frames, reps_per_second)
    assert ground_truth >= 2, "sanity check on the scenario itself, not the fix"

    # The actual point of this test: triangulating both cameras recovers
    # the same rep count RepCounter would get fed the true 3D angle
    # directly, while cam-a's single degenerate 2D view (same pixel
    # column every frame -- see module docstring) does not.
    assert fused_rep_count == ground_truth
    assert solo_rep_count != ground_truth
