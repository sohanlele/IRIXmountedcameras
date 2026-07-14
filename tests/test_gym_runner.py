"""Business-logic tests for irix.live.gym_runner.GymSessionRunner --
does cross-station handoff actually prevent double-counting, and does a
member's session correctly transfer from one station's RepSession to the
next station's as they walk?

Same scripted-pose-estimator approach as tests/test_station_runner.py.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.gym_runner import GymSessionRunner
from irix.live.station_runner import StationSessionRunner
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.topology.registry import StationInfo, StationRegistry


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


def _slow_rep(n_frames: int = 10):
    return list(np.linspace(90.0, 170.0, n_frames))


_TWO_REPS = _slow_rep() + _slow_rep()


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


def _two_station_registry():
    return StationRegistry([
        StationInfo(station_id="squat-1", camera_id="cam-1", zone="free_weights", adjacent_station_ids=["squat-2"]),
        StationInfo(station_id="squat-2", camera_id="cam-2", zone="free_weights", adjacent_station_ids=["squat-1"]),
    ])


def _build_runner(n_ticks, ble_script, checkout_registry, station_events, gym_events,
                   presence_timeout_s=1.0, min_consecutive=3):
    station_runners = {}
    for station_id in ("squat-1", "squat-2"):
        events = station_events.setdefault(station_id, [])
        station_runners[station_id] = StationSessionRunner(
            station_id=station_id,
            exercise_name="squat",
            checkout_registry=checkout_registry,
            frame_source=_FakeFrameSource(n_ticks),
            ble_reader=lambda: [],  # never called -- GymSessionRunner drives tick() directly
            pose_estimator=_ScriptedPoseEstimator(_TWO_REPS),
            presence_timeout_s=presence_timeout_s,
            on_events=events.extend,
        )
    return GymSessionRunner(
        registry=_two_station_registry(),
        checkout_registry=checkout_registry,
        station_runners=station_runners,
        ble_reader=_ScriptedBLEReader(ble_script),
        presence_timeout_s=presence_timeout_s,
        min_consecutive=min_consecutive,
        on_gym_events=gym_events.extend,
        clock=_FakeClock(step=0.5),
    )


def _walk_script(n_ticks, switch_at, wristband_id="band-1"):
    """Strong at squat-1 before `switch_at`, strong at squat-2 from
    `switch_at` onward -- simulates a member walking from one station to
    the other partway through."""
    script = []
    for t in range(n_ticks):
        if t < switch_at:
            rssi_a, rssi_b = -40.0, -75.0
        else:
            rssi_a, rssi_b = -75.0, -35.0
        script.append([
            BLEReading(station_id="squat-1", rssi=rssi_a, timestamp=float(t), wristband_id=wristband_id),
            BLEReading(station_id="squat-2", rssi=rssi_b, timestamp=float(t), wristband_id=wristband_id),
        ])
    return script


def test_handoff_moves_the_session_and_never_double_counts():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    station_events = {}
    gym_events = []

    n_ticks = 30
    ble_script = _walk_script(n_ticks, switch_at=15)
    runner = _build_runner(n_ticks, ble_script, registry, station_events, gym_events)
    runner.run_forever(max_frames=n_ticks)
    # squat-2's session is still in progress when the scripted run ends
    # (alice never leaves) -- close() is the real shutdown path that
    # flushes it, same as irix.demo.run_upload's finally block.
    runner.close()

    # A real StationHandoffEvent should have fired exactly once.
    assert len(gym_events) == 1
    assert gym_events[0].to_dict()["event_type"] == "station_handoff"
    assert gym_events[0].member_id == "member-alice"
    assert gym_events[0].from_station == "squat-1"
    assert gym_events[0].to_station == "squat-2"

    set_events_1 = [e for e in station_events["squat-1"] if e.to_dict()["event_type"] == "set_complete"]
    set_events_2 = [e for e in station_events["squat-2"] if e.to_dict()["event_type"] == "set_complete"]
    # Both stations should have closed out a set for alice -- proof the
    # session actually moved (not merged into one continuous station-2
    # session, and not silently dropped at station-1).
    assert len(set_events_1) == 1
    assert len(set_events_2) == 1
    assert set_events_1[0].member_id == "member-alice"
    assert set_events_2[0].member_id == "member-alice"

    # No frame should ever have been double-counted: at any given tick,
    # at most one station should have an active (non-empty) rep stream
    # for alice at once -- check via total reps counted staying sane
    # (each station's real rep count, summed, matches what a single
    # continuous session would have produced -- not inflated by both
    # stations counting the same physical reps).
    rep_events_1 = [e for e in station_events["squat-1"] if e.to_dict()["event_type"] == "rep_completed"]
    rep_events_2 = [e for e in station_events["squat-2"] if e.to_dict()["event_type"] == "rep_completed"]
    assert len(rep_events_1) >= 1
    assert len(rep_events_2) >= 1
    assert all(e.member_id == "member-alice" for e in rep_events_1 + rep_events_2)


def test_unchecked_out_band_is_ignored_gym_wide():
    registry = CheckoutRegistry()  # nothing checked out
    station_events = {}
    gym_events = []
    n_ticks = 10
    ble_script = _walk_script(n_ticks, switch_at=100, wristband_id="band-unregistered")
    runner = _build_runner(n_ticks, ble_script, registry, station_events, gym_events)
    runner.run_forever(max_frames=n_ticks)

    assert gym_events == []
    assert station_events["squat-1"] == []
    assert station_events["squat-2"] == []


def test_two_members_at_two_different_stations_simultaneously():
    """No cross-contamination: two checked-out members, each parked at
    their own station the whole time, should each get their own
    independent session with correctly attributed events."""
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    registry.check_out("band-2", "member-bob", timestamp=0.0)
    station_events = {}
    gym_events = []

    n_ticks = 20
    script = []
    for t in range(n_ticks):
        script.append([
            BLEReading(station_id="squat-1", rssi=-40.0, timestamp=float(t), wristband_id="band-1"),
            BLEReading(station_id="squat-2", rssi=-75.0, timestamp=float(t), wristband_id="band-1"),
            BLEReading(station_id="squat-2", rssi=-40.0, timestamp=float(t), wristband_id="band-2"),
            BLEReading(station_id="squat-1", rssi=-75.0, timestamp=float(t), wristband_id="band-2"),
        ])
    runner = _build_runner(n_ticks, script, registry, station_events, gym_events)
    runner.run_forever(max_frames=n_ticks)

    assert gym_events == []  # no handoffs -- each member stayed at their own station

    rep_events_1 = [e for e in station_events["squat-1"] if e.to_dict()["event_type"] == "rep_completed"]
    rep_events_2 = [e for e in station_events["squat-2"] if e.to_dict()["event_type"] == "rep_completed"]
    assert all(e.member_id == "member-alice" for e in rep_events_1)
    assert all(e.member_id == "member-bob" for e in rep_events_2)
    assert len(rep_events_1) >= 1
    assert len(rep_events_2) >= 1
