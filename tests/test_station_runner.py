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
from irix.fusion.imu import IMUSample
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


class _RecordingCalibrationProfile:
    """Stand-in for irix.pose.calibration.CalibrationProfile: records
    every frame it's asked to undistort and returns a distinguishable
    (but still valid) array, so a test can assert the pose estimator
    actually received the undistorted frame, not the raw one."""

    def __init__(self):
        self.frames_seen = []

    def undistort_frame(self, frame):
        self.frames_seen.append(frame)
        return frame + 1  # distinguishable from the raw all-zero input frame


class _FrameRecordingPoseEstimator:
    def __init__(self):
        self.frames_seen = []

    def estimate(self, frame):
        self.frames_seen.append(frame)
        return []


def test_calibration_profile_undistorts_frames_before_pose_estimation():
    """StationSessionRunner's Camera Streams -> Pose Estimation stage
    should run every frame through the station's calibration_profile
    first, when one is configured -- the "Camera Calibration" stage of
    the authoritative pipeline (see irix/pipeline/rep_session.py /
    docs/ARCHITECTURE.md). No calibration_profile (the default) must
    leave frames untouched, unchanged from pre-Phase-3 behavior."""
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    calibration = _RecordingCalibrationProfile()
    pose_estimator = _FrameRecordingPoseEstimator()
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(3),
        ble_reader=_ScriptedBLEReader([present] * 3),
        pose_estimator=pose_estimator,
        calibration_profile=calibration,
        clock=_FakeClock(step=0.1),
    )
    runner.run_forever(max_frames=3)

    assert len(calibration.frames_seen) == 3
    # Every frame the pose estimator actually saw must be the
    # *undistorted* one (raw + 1), not the raw all-zero frame.
    assert len(pose_estimator.frames_seen) == 3
    assert all(f.max() == 1 for f in pose_estimator.frames_seen)


def test_calibrate_wristband_clock_corrects_that_bands_subsequent_imu_samples():
    """calibrate_wristband_clock() is the intended entry point for an
    explicit clock-sync calibration step (see its docstring for why this
    repo doesn't auto-derive the observation from rep timestamps) --
    once called for an open session's band, that band's RepSession
    should apply the correction to every later add_imu_samples() batch."""
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    true_lead_s = 0.35
    imu_samples = synthetic_imu_stream(n_seconds=1.0, fs=30.0, reps_per_second=0.5, seed=1)
    shifted = [
        s.__class__(timestamp=s.timestamp + true_lead_s, accel=s.accel, gyro=s.gyro) for s in imu_samples
    ]
    imu_stream = _ChunkedIMUStream(shifted, samples_per_poll=5)

    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(10),
        ble_reader=_ScriptedBLEReader([present] * 10),
        imu_stream_factory=lambda wid: imu_stream,
        pose_estimator=_ScriptedPoseEstimator([]),
        clock=_FakeClock(step=0.1),
    )

    # First tick opens the session (band becomes present) but hasn't
    # calibrated yet -- its first polled batch should be stored raw.
    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.0, present_wristband_ids=["band-1"])
    session = runner._sessions["band-1"]
    assert session._imu_samples[0].timestamp == shifted[0].timestamp

    calibrated = runner.calibrate_wristband_clock("band-1", offset_s=-true_lead_s, confidence=1.0, at_time=0.1)
    assert calibrated is True

    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.2, present_wristband_ids=["band-1"])
    corrected_batch = session._imu_samples[5:10]
    assert corrected_batch[0].timestamp == pytest.approx(shifted[5].timestamp - true_lead_s, abs=1e-9)

    # A band with no open session is a no-op, not an error.
    assert runner.calibrate_wristband_clock("band-nonexistent", offset_s=0.1, confidence=1.0) is False


def test_leg_press_session_withholds_imu_until_placement_is_moved_to_ankle():
    """A default-fresh band starts at BandSide.LEFT_WRIST (irix.identity.
    placement.WristbandPlacementTracker's own documented default) -- a
    leg_press session (ExerciseConfig.band_placement == ANKLE) should not
    trust any IMU sample for fusion until request_wristband_placement_
    change() moves it to an ankle side and the tracker confirms settling
    + recalibration. This is the "never reuse wrist thresholds for ankle
    data or vice versa" rule made concrete at the StationSessionRunner
    level (irix.pipeline.rep_session.RepSession.add_imu_samples does the
    actual gating -- see tests/test_wristband_placement.py for the
    tracker's own state-machine tests in isolation)."""
    from irix.identity.placement import BandSide
    from irix.wristband_sim.calibration import GRAVITY_M_S2

    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    fs = 50.0
    mismatched_samples = synthetic_imu_stream(n_seconds=1.0, fs=fs, reps_per_second=0.5, seed=1)
    imu_stream = _ChunkedIMUStream(mismatched_samples, samples_per_poll=10)

    events = []
    runner = StationSessionRunner(
        station_id="leg-press-1",
        exercise_name="leg_press",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(10),
        ble_reader=_ScriptedBLEReader([present] * 10),
        imu_stream_factory=lambda wid: imu_stream,
        pose_estimator=_ScriptedPoseEstimator([]),
        clock=_FakeClock(step=0.1),
        on_events=events.extend,
    )

    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.0, present_wristband_ids=["band-1"])
    session = runner._sessions["band-1"]
    tracker = runner._placement_trackers["band-1"]
    assert tracker.current_side == BandSide.LEFT_WRIST
    assert session._imu_samples == []  # wrist samples withheld -- exercise needs ankle

    moved = runner.request_wristband_placement_change("band-1", BandSide.LEFT_ANKLE, at_time=0.1)
    assert moved is True
    assert tracker.paused is True

    # Feed genuinely still, gravity-consistent samples (band now resting
    # in its new ankle position) long enough to settle + calibrate.
    rng = np.random.default_rng(7)
    quiet_samples = []
    t = 0.1
    for _ in range(150):
        accel = np.array([0.0, -GRAVITY_M_S2, 0.0]) + rng.normal(scale=0.05, size=3)
        gyro = rng.normal(scale=0.05, size=3)
        quiet_samples.append(IMUSample(timestamp=t, accel=accel, gyro=gyro))
        t += 1.0 / fs
    still_stream = _ChunkedIMUStream(quiet_samples, samples_per_poll=len(quiet_samples))
    runner._imu_streams["band-1"] = still_stream

    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.2, present_wristband_ids=["band-1"])

    assert tracker.current_side == BandSide.LEFT_ANKLE
    assert tracker.paused is False

    from irix.pipeline.schema import BandPlacementConfirmedEvent

    confirmed = [e for e in events if isinstance(e, BandPlacementConfirmedEvent)]
    assert len(confirmed) == 1
    assert confirmed[0].wristband_id == "band-1"
    assert confirmed[0].from_side == "left_wrist"
    assert confirmed[0].to_side == "left_ankle"

    # Now that placement matches the exercise's requirement, a fresh
    # batch should actually be stored.
    more_samples = synthetic_imu_stream(n_seconds=0.2, fs=fs, reps_per_second=0.5, seed=2)
    runner._imu_streams["band-1"] = _ChunkedIMUStream(more_samples, samples_per_poll=len(more_samples))
    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.3, present_wristband_ids=["band-1"])
    assert len(session._imu_samples) > 0


def test_request_wristband_placement_change_for_unopened_band_is_a_no_op():
    from irix.identity.placement import BandSide

    registry = CheckoutRegistry()
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(1),
        ble_reader=_ScriptedBLEReader([[]]),
        pose_estimator=_ScriptedPoseEstimator([]),
        clock=_FakeClock(step=0.1),
    )
    assert runner.request_wristband_placement_change("band-nonexistent", BandSide.LEFT_ANKLE) is False


class _RecordingMotionResolver:
    """Duck-typed stand-in that just records the IMU streams it was
    handed (via detected_people_poses/candidate_imu_streams) instead of
    doing any real correlation -- used to verify StationSessionRunner.
    tick() feeds the disambiguator clock-synced samples (Priority 5's
    "fuse ... clock synchronization" requirement), not raw ones."""

    def __init__(self):
        self.seen_imu_streams = []

    def resolve(self, candidate_imu_streams, detected_people_poses, pose_fps, prior_slot_assignment=None):
        self.seen_imu_streams.append({k: list(v) for k, v in candidate_imu_streams.items()})
        return [None] * len(detected_people_poses)


def test_disambiguator_receives_clock_synced_imu_not_raw_samples():
    from irix.live.disambiguation import CrowdedGroupDisambiguator

    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    registry.check_out("band-2", "member-bob", timestamp=0.0)
    present = [
        BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1"),
        BLEReading(station_id="station-1", rssi=-41.0, timestamp=0.0, wristband_id="band-2"),
    ]

    fs = 50.0
    true_lead_s = 0.4
    raw_alice = synthetic_imu_stream(n_seconds=0.2, fs=fs, reps_per_second=0.5, seed=1)
    imu_alice_stream = _ChunkedIMUStream(raw_alice, samples_per_poll=len(raw_alice))
    imu_bob_stream = _ChunkedIMUStream([], samples_per_poll=1)

    resolver = _RecordingMotionResolver()
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(3),
        ble_reader=_ScriptedBLEReader([present] * 3),
        imu_stream_factory=lambda wid: imu_alice_stream if wid == "band-1" else imu_bob_stream,
        pose_estimator=_TwoPersonScriptedPoseEstimator([_pose_for_angle(90.0)], [_pose_for_angle(90.0)]),
        clock=_FakeClock(step=0.1),
        motion_resolver=resolver,
        disambiguation_window_frames=1,
    )

    # Pre-calibrate band-1's clock before any tick, so the very first
    # poll already has a correction to apply -- isolates "was the
    # correction applied" from "had a set closed yet to produce one".
    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.0, present_wristband_ids=["band-1", "band-2"])
    runner.calibrate_wristband_clock("band-1", offset_s=-true_lead_s, confidence=1.0, at_time=0.0)

    imu_alice_stream2 = _ChunkedIMUStream(raw_alice, samples_per_poll=len(raw_alice))
    runner._imu_streams["band-1"] = imu_alice_stream2
    runner.tick(frame=np.zeros((2, 2, 3), dtype=np.uint8), now=0.1, present_wristband_ids=["band-1", "band-2"])

    assert len(resolver.seen_imu_streams) >= 1
    seen_alice = resolver.seen_imu_streams[-1]["member-alice"]
    assert len(seen_alice) == len(raw_alice)
    assert seen_alice[0].timestamp == pytest.approx(raw_alice[0].timestamp - true_lead_s, abs=1e-9)


class _IntermittentPoseEstimator:
    """Returns a fixed pose for the first `visible_frames` calls, then
    empty (nobody detected) for `missing_frames` calls, then the fixed
    pose again -- simulates a real detector losing and regaining track."""

    def __init__(self, pose, visible_frames, missing_frames):
        self._pose = pose
        self._visible_frames = visible_frames
        self._missing_frames = missing_frames
        self._i = 0

    def estimate(self, frame):
        self._i += 1
        if self._i <= self._visible_frames:
            return [self._pose]
        if self._i <= self._visible_frames + self._missing_frames:
            return []
        return [self._pose]


def test_tracking_lost_and_recovered_fire_after_a_missed_frame_streak():
    from irix.pipeline.schema import TrackingLostEvent, TrackingRecoveredEvent

    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    n_frames = 30
    estimator = _IntermittentPoseEstimator(_pose_for_angle(90.0), visible_frames=5, missing_frames=15)
    events = []
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(n_frames),
        ble_reader=_ScriptedBLEReader([present] * n_frames),
        pose_estimator=estimator,
        on_events=events.extend,
        clock=_FakeClock(step=0.1),
        tracking_lost_after_frames=10,
    )
    runner.run_forever(max_frames=n_frames)

    lost = [e for e in events if isinstance(e, TrackingLostEvent)]
    recovered = [e for e in events if isinstance(e, TrackingRecoveredEvent)]
    assert len(lost) == 1
    assert lost[0].member_id == "member-alice"
    assert lost[0].consecutive_missed_frames == 10
    assert len(recovered) == 1
    assert recovered[0].member_id == "member-alice"
    assert recovered[0].gap_duration_s > 0


def test_a_brief_single_frame_miss_does_not_fire_tracking_lost():
    from irix.pipeline.schema import TrackingLostEvent

    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    present = [BLEReading(station_id="station-1", rssi=-40.0, timestamp=0.0, wristband_id="band-1")]

    n_frames = 10
    estimator = _IntermittentPoseEstimator(_pose_for_angle(90.0), visible_frames=3, missing_frames=1)
    events = []
    runner = StationSessionRunner(
        station_id="station-1",
        exercise_name="squat",
        checkout_registry=registry,
        frame_source=_FakeFrameSource(n_frames),
        ble_reader=_ScriptedBLEReader([present] * n_frames),
        pose_estimator=estimator,
        on_events=events.extend,
        clock=_FakeClock(step=0.1),
        tracking_lost_after_frames=10,
    )
    runner.run_forever(max_frames=n_frames)

    assert not [e for e in events if isinstance(e, TrackingLostEvent)]
