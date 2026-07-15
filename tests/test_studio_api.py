"""Tests for irix.backend.studio_api.StudioBackendAPI -- the concrete
backend surface a future IRIX Studio calls (Priority 11). Two tiers:
plain CheckoutRegistry-only tests for the operations that don't need a
live gym loop, and one live-runner test (same fixture pattern as
tests/test_gym_runner.py) that exercises the operations which do.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from irix.backend.studio_api import StudioAPIError, StudioBackendAPI
from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.gym_runner import GymSessionRunner
from irix.live.station_runner import StationSessionRunner
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.topology.registry import StationInfo, StationRegistry


# ---------------------------------------------------------------------
# Checkout-only tests (no live GymSessionRunner needed)
# ---------------------------------------------------------------------

def test_assign_and_query_assignment():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    result = api.assign_wristband("band-1", "member-alice", at_time=100.0)
    assert result["member_id"] == "member-alice"

    status = api.query_assignment("band-1")
    assert status["is_checked_out"] is True
    assert status["member_id"] == "member-alice"


def test_return_wristband_without_live_runner():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    api.assign_wristband("band-1", "member-alice", at_time=0.0)
    result = api.return_wristband("band-1", at_time=10.0)
    assert result["was_active"] is True
    assert api.query_assignment("band-1")["is_checked_out"] is False


def test_return_wristband_already_returned_is_not_an_error():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    api.assign_wristband("band-1", "member-alice", at_time=0.0)
    api.return_wristband("band-1", at_time=10.0)
    result = api.return_wristband("band-1", at_time=20.0)  # double call
    assert result["was_active"] is False


def test_query_battery_is_always_honestly_unknown():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    api.assign_wristband("band-1", "member-alice", at_time=0.0)
    result = api.query_battery("band-1")
    assert result["status"] == "unknown"
    assert "reason" in result


def test_start_session_raises_for_unassigned_band():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    with pytest.raises(StudioAPIError):
        api.start_session("band-nonexistent")


def test_operations_requiring_gym_session_runner_raise_clearly_without_one():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)  # no gym_session_runner
    api.assign_wristband("band-1", "member-alice", at_time=0.0)
    with pytest.raises(StudioAPIError):
        api.end_session("band-1")
    with pytest.raises(StudioAPIError):
        api.request_placement_change("band-1", "wrist", at_time=0.0)


def test_query_wristband_status_degrades_gracefully_without_live_runner():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    api.assign_wristband("band-1", "member-alice", at_time=0.0)
    status = api.query_wristband_status("band-1")
    assert status["is_checked_out"] is True
    assert status["current_station_id"] is None
    assert status["workout_phase"] is None
    assert status["battery"]["status"] == "unknown"


def test_query_wristband_status_for_unassigned_band():
    registry = CheckoutRegistry()
    api = StudioBackendAPI(checkout_registry=registry)
    status = api.query_wristband_status("band-ghost")
    assert status["is_checked_out"] is False
    assert status["member_id"] is None


# ---------------------------------------------------------------------
# Live-runner tests (same fixture pattern as tests/test_gym_runner.py)
# ---------------------------------------------------------------------

def _pose_for_angle(angle_deg: float) -> PersonPose:
    knee = np.array([0.0, 0.0])
    hip = knee + np.array([0.0, -100.0])
    theta = math.radians(-90 + angle_deg)
    ankle = knee + 100.0 * np.array([math.cos(theta), math.sin(theta)])
    keypoints = [Keypoint(x=0.0, y=0.0, confidence=0.0) for _ in COCO_KEYPOINT_NAMES]

    def _set(name, xy):
        keypoints[KEYPOINT_INDEX[name]] = Keypoint(x=float(xy[0]), y=float(xy[1]), confidence=0.9)

    _set("left_hip", hip)
    _set("left_knee", knee)
    _set("left_ankle", ankle)
    return PersonPose(keypoints=keypoints, bbox=(0.0, 0.0, 200.0, 200.0))


class _ScriptedPoseEstimator:
    def __init__(self, angles):
        self._poses = [_pose_for_angle(a) for a in angles]
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._poses):
            return []
        pose = self._poses[self._i]
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
    def __init__(self, step=0.5):
        self.t = 0.0
        self.step = step

    def __call__(self):
        val = self.t
        self.t += self.step
        return val


class _ScriptedBLEReader:
    def __init__(self, readings_by_tick):
        self._script = readings_by_tick
        self._i = 0

    def __call__(self):
        if not self._script:
            return []
        idx = min(self._i, len(self._script) - 1)
        readings = self._script[idx]
        self._i += 1
        return readings


def _one_station_registry():
    return StationRegistry([
        StationInfo(station_id="squat-1", camera_id="cam-1", zone="free_weights", adjacent_station_ids=[]),
    ])


def _build_live_runner(n_ticks, min_consecutive=1):
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    station_runners = {
        "squat-1": StationSessionRunner(
            station_id="squat-1",
            exercise_name="squat",
            checkout_registry=registry,
            frame_source=_FakeFrameSource(n_ticks),
            ble_reader=lambda: [],
            pose_estimator=_ScriptedPoseEstimator([90.0] * n_ticks),
            presence_timeout_s=5.0,
        )
    }
    script = [[
        BLEReading(station_id="squat-1", rssi=-40.0, timestamp=float(t), wristband_id="band-1"),
    ] for t in range(n_ticks)]
    gym_runner = GymSessionRunner(
        registry=_one_station_registry(),
        checkout_registry=registry,
        station_runners=station_runners,
        ble_reader=_ScriptedBLEReader(script),
        presence_timeout_s=5.0,
        min_consecutive=min_consecutive,
        clock=_FakeClock(step=0.5),
    )
    return registry, gym_runner


def test_query_wristband_status_reflects_live_workout_phase():
    from irix.pipeline.workout_state import WorkoutPhase

    registry, gym_runner = _build_live_runner(n_ticks=5)
    gym_runner.run_forever(max_frames=5)
    api = StudioBackendAPI(checkout_registry=registry, gym_session_runner=gym_runner)

    status = api.query_wristband_status("band-1")
    assert status["current_station_id"] == "squat-1"
    assert status["workout_phase"] == WorkoutPhase.EXERCISE_CONFIRMED.value
    assert status["health"]["camera_connected"] is True


def test_start_session_confirms_an_already_active_session():
    registry, gym_runner = _build_live_runner(n_ticks=5)
    gym_runner.run_forever(max_frames=5)
    api = StudioBackendAPI(checkout_registry=registry, gym_session_runner=gym_runner)

    result = api.start_session("band-1")
    assert result["session_active"] is True
    assert result["member_id"] == "member-alice"


def test_end_session_ends_the_workout_state_machine_without_checkin():
    registry, gym_runner = _build_live_runner(n_ticks=5)
    gym_runner.run_forever(max_frames=5)
    api = StudioBackendAPI(checkout_registry=registry, gym_session_runner=gym_runner)

    result = api.end_session("band-1")
    assert result["session_was_active"] is True

    from irix.pipeline.workout_state import WorkoutPhase
    assert gym_runner._workout_states["band-1"].phase == WorkoutPhase.SESSION_ENDED
    # end_session must NOT check the band back in -- it's still assigned.
    assert registry.is_checked_out("band-1") is True


def test_return_wristband_with_live_runner_also_ends_the_session():
    registry, gym_runner = _build_live_runner(n_ticks=5)
    gym_runner.run_forever(max_frames=5)
    api = StudioBackendAPI(checkout_registry=registry, gym_session_runner=gym_runner)

    api.return_wristband("band-1", at_time=100.0)
    assert registry.is_checked_out("band-1") is False
    assert "band-1" not in gym_runner._workout_states
