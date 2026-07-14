"""Business-logic tests for irix.live.station_runner.StationSessionRunner
-- does checkout resolution + BLE presence + session start/stop actually
tie together correctly?

Same approach as tests/test_run_upload_wiring.py: PoseEstimator is
patched with a scripted stand-in (real PersonPose/Keypoint shapes,
hand-computed joint angles) so this exercises the real code path without
needing the actual ultralytics model. The clock and BLE reader are also
injected/scripted, since "does a 5-second presence timeout fire" has to
be deterministic in a test that runs in milliseconds, not real time.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.station_runner import StationSessionRunner
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose


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
    """Returns one scripted pose per call; once the script runs out,
    returns no people (same as a real model losing tracking) rather than
    raising -- station_runner may call estimate() a few extra times while
    waiting out a presence timeout after the last real angle."""

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
    """Advances by `step` seconds every call -- lets presence_timeout_s
    be tested deterministically without waiting on real wall-clock time."""

    def __init__(self, step=0.1):
        self.t = 0.0
        self.step = step

    def __call__(self):
        val = self.t
        self.t += self.step
        return val


class _ScriptedBLEReader:
    """readings_by_frame[i] is the list[BLEReading] returned on the i-th
    call; calling past the end repeats the last entry (steady-state)."""

    def __init__(self, readings_by_frame):
        self._script = readings_by_frame
        self._i = 0

    def __call__(self):
        if not self._script:
            return []
        idx = min(self._i, len(self._script) - 1)
        readings = self._script[idx]
        self._i += 1
        return readings


def _run(n_frames, ble_script, angles=_TWO_REPS, presence_timeout_s=0.5, checkout_registry=None, **kwargs):
    events = []
    registry = checkout_registry or CheckoutRegistry()
    estimator = _ScriptedPoseEstimator(angles)
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(n_frames),
        ble_reader=_ScriptedBLEReader(ble_script),
        pose_estimator=estimator,
        presence_timeout_s=presence_timeout_s,
        on_events=events.extend,
        clock=_FakeClock(step=0.1),
        **kwargs,
    )
    runner.run_forever(max_frames=n_frames)
    return events, registry


def test_presence_starts_a_session_attributed_to_the_checked_out_member(tmp_path=None):
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    # present for 20 ticks (both scripted reps), then goes quiet for the
    # rest -- 10 more ticks at step=0.1 = 1.0s of absence, comfortably
    # past presence_timeout_s=0.5.
    ble_script = [present] * 20 + [[]] * 10
    events, _ = _run(n_frames=30, ble_script=ble_script, checkout_registry=registry)

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert len(rep_events) == 2
    assert all(e.member_id == "member-alice" for e in rep_events)
    assert len(set_events) == 1
    assert set_events[0].member_id == "member-alice"
    assert set_events[0].total_reps == 2


def test_unchecked_out_band_never_starts_a_session():
    """A band that isn't checked out to anyone shouldn't start tracking
    reps for it -- there's no account to attribute events to."""
    present_but_unregistered = [
        BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-unregistered")
    ]
    ble_script = [present_but_unregistered] * 20
    events, _ = _run(n_frames=20, ble_script=ble_script)

    assert events == []


def test_presence_timeout_closes_the_session_with_a_fatigue_summary():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]
    ble_script = [present] * 20 + [[]] * 10
    events, _ = _run(n_frames=30, ble_script=ble_script, checkout_registry=registry)

    fatigue_events = [e for e in events if e.to_dict()["event_type"] == "set_fatigue_summary"]
    assert len(fatigue_events) == 1
    assert fatigue_events[0].member_id == "member-alice"


def test_a_second_checked_out_member_preempts_and_starts_their_own_session():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    registry.check_out("band-2", "member-bob", timestamp=0.0)
    alice_present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]
    bob_present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-2")]

    # alice present for 10 ticks (one rep's worth), then bob shows up
    # immediately (no gap) -- should end alice's session right away
    # (preemption), not wait for a timeout, and start bob's.
    ble_script = [alice_present] * 10 + [bob_present] * 10
    events, _ = _run(n_frames=20, ble_script=ble_script, checkout_registry=registry)

    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert len(set_events) == 1
    assert set_events[0].member_id == "member-alice"
    assert set_events[0].total_reps == 1

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert rep_events[0].member_id == "member-alice"
    assert rep_events[-1].member_id == "member-bob"


def test_close_flushes_an_in_progress_session():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    events = []
    estimator = _ScriptedPoseEstimator(_TWO_REPS)
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(20),
        ble_reader=_ScriptedBLEReader([present] * 20),
        pose_estimator=estimator,
        on_events=events.extend,
        clock=_FakeClock(step=0.1),
    )
    runner.run_forever(max_frames=20)  # still present the whole time -- no timeout fired
    assert not [e for e in events if e.to_dict()["event_type"] == "set_complete"]

    runner.close()
    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert len(set_events) == 1
    assert set_events[0].total_reps == 2
