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

from irix.demo.mock_pose import synthetic_imu_stream, synthetic_pose_stream
from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.station_runner import StationSessionRunner
from irix.pose.estimator import COCO_KEYPOINT_NAMES, KEYPOINT_INDEX, Keypoint, PersonPose
from irix.rep_counting.exercises import BICEP_CURL


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


class _ChunkedIMUStream:
    """Doles out a prerecorded sample list a fixed-size chunk at a time
    per ``poll()`` call, mimicking how a real live BLE stream would only
    ever have delivered "however much arrived since the last poll" --
    unlike ``irix.fusion.imu_stream.RecordedIMUStream`` (which, correctly
    for its own offline use case, hands back the *entire* file on the
    very first poll), this keeps a disambiguation buffer's accumulated
    IMU window aligned to the same real-time span as the pose buffer
    it's being correlated against, which matters once >1 session is
    active and buffering starts."""

    def __init__(self, samples, samples_per_poll):
        self._samples = list(samples)
        self._samples_per_poll = samples_per_poll
        self._i = 0

    def poll(self):
        chunk = self._samples[self._i : self._i + self._samples_per_poll]
        self._i += self._samples_per_poll
        return chunk


class _TwoPersonScriptedPoseEstimator:
    """Every tick returns both people's next scripted pose, in a fixed
    slot order (poses_a always slot 0, poses_b always slot 1) -- a
    reasonable stand-in for a real per-frame detector's list ordering
    over a short, static-camera window (see the "person-slot-index
    stability" caveat in irix/live/station_runner.py's module
    docstring)."""

    def __init__(self, poses_a, poses_b):
        self._poses_a = poses_a
        self._poses_b = poses_b
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._poses_a):
            return []
        pair = [self._poses_a[self._i], self._poses_b[self._i]]
        self._i += 1
        return pair


def test_crowded_station_disambiguates_two_co_located_members_by_motion():
    """Two checked-out members ('carol', faster tempo, and 'dave', slower)
    both show up at the same station's BLE readings at once -- RSSI alone
    can't say which detected skeleton is which (see irix.identity.
    motion_correlation's module docstring), so StationSessionRunner has to
    buffer a window of poses/IMU and resolve it via
    MotionCorrelationResolver before it can safely route camera detections
    to the right RepSession. This exercises that whole path for real,
    through StationSessionRunner.run_forever -- not just the resolver in
    isolation (already covered by tests/test_motion_correlation.py)."""
    registry = CheckoutRegistry()
    registry.check_out("band-carol", "member-carol", timestamp=0.0)
    registry.check_out("band-dave", "member-dave", timestamp=0.0)

    fps = 30.0
    window_frames = 180  # 6s -- same window size proven to resolve reliably in test_motion_correlation.py
    n_frames = window_frames + 240  # 6s to fill the buffer, 8s more to observe correctly-routed reps after

    poses_carol = [p for _, _, p in synthetic_pose_stream(BICEP_CURL, n_frames=n_frames, fps=fps, reps_per_second=0.6)]
    poses_dave = [p for _, _, p in synthetic_pose_stream(BICEP_CURL, n_frames=n_frames, fps=fps, reps_per_second=0.3)]
    # fs chosen as an exact multiple of fps (90 = 3 * 30) so each tick's
    # poll() hands back a fixed-size chunk with no fractional drift.
    imu_fs = 90.0
    samples_per_tick = int(imu_fs / fps)
    imu_carol = synthetic_imu_stream(n_seconds=n_frames / fps, fs=imu_fs, reps_per_second=0.6, seed=12)
    imu_dave = synthetic_imu_stream(n_seconds=n_frames / fps, fs=imu_fs, reps_per_second=0.3, seed=13)
    imu_by_band = {
        "band-carol": _ChunkedIMUStream(imu_carol, samples_per_tick),
        "band-dave": _ChunkedIMUStream(imu_dave, samples_per_tick),
    }

    both_present = [
        BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-carol"),
        BLEReading(station_id="station-1", rssi=-41.0, timestamp=0.0, wristband_id="band-dave"),
    ]
    events = []
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="bicep_curl",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(n_frames),
        ble_reader=_ScriptedBLEReader([both_present] * n_frames),
        imu_stream_factory=lambda wid: imu_by_band[wid],
        pose_estimator=_TwoPersonScriptedPoseEstimator(poses_carol, poses_dave),
        presence_timeout_s=1.0,
        on_events=events.extend,
        clock=_FakeClock(step=1.0 / fps),
        disambiguation_window_frames=window_frames,
    )
    runner.run_forever(max_frames=n_frames)

    # The scripted estimator always puts carol's pose in slot 0 and dave's
    # in slot 1 -- correlation against each candidate's (distinctly-paced)
    # wristband IMU should recover exactly that mapping. Checked *before*
    # close(): tearing down a session that's part of the currently-
    # resolved group correctly invalidates that group's assignment (a
    # real mid-run member-leaves case should force fresh disambiguation
    # for whoever's left), which would otherwise make this assertion
    # observe a wiped-out {} rather than the resolution that actually
    # drove routing for the whole rest of the run.
    assert runner._disambiguator.slot_assignment == {0: "band-carol", 1: "band-dave"}
    runner.close()

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    carol_reps = [e for e in rep_events if e.member_id == "member-carol"]
    dave_reps = [e for e in rep_events if e.member_id == "member-dave"]
    # Nothing gets attributed to anyone during the buffering window (see
    # the trade-off documented in irix/live/station_runner.py), but once
    # resolved, both members should get some correctly-attributed reps --
    # carol's faster tempo should produce more of them than dave's in the
    # same post-resolution window.
    assert carol_reps and dave_reps
    assert len(carol_reps) >= len(dave_reps)
    assert not any(e.member_id == "member-dave" for e in carol_reps)
    assert not any(e.member_id == "member-carol" for e in dave_reps)

    set_events = [e for e in events if e.to_dict()["event_type"] == "set_complete"]
    assert {e.member_id for e in set_events} == {"member-carol", "member-dave"}


def test_single_member_station_never_triggers_disambiguation_buffering():
    """The common case (one member at a station) shouldn't pay any
    buffering cost or ever populate disambiguation state -- only a
    genuine multi-band tick does."""
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
    runner.run_forever(max_frames=20)
    assert runner._disambiguator._pending_wristband_ids is None
    assert runner._disambiguator.slot_assignment == {}
    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert len(rep_events) == 2
    assert all(e.member_id == "member-alice" for e in rep_events)
