"""Business-logic tests for irix.live.zone_runner.MultiCameraZoneRunner --
does a dense, overlapping-FOV multi-camera zone correctly segregate
multiple co-located members' reps to the right wristband/account, without
double-counting when several cameras see the same person at once, and
without losing someone entirely when only *some* cameras can currently
see them (partial occlusion)?

Same scripted-pose-estimator / chunked-IMU-stream approach as
tests/test_station_runner.py's crowded-station test.
"""
from __future__ import annotations

import numpy as np
import pytest

from irix.demo.mock_pose import synthetic_imu_stream, synthetic_pose_stream
from irix.identity.ble_pairing import BLEReading
from irix.identity.checkout import CheckoutRegistry
from irix.live.zone_runner import MultiCameraZoneRunner, ZoneCamera
from irix.rep_counting.exercises import BICEP_CURL


class _ChunkedIMUStream:
    """Doles out a prerecorded sample list a fixed-size chunk at a time
    per ``poll()`` call, mimicking a real live BLE stream's "however much
    arrived since the last poll" delivery -- see the identical helper (and
    its rationale) in tests/test_station_runner.py."""

    def __init__(self, samples, samples_per_poll):
        self._samples = list(samples)
        self._samples_per_poll = samples_per_poll
        self._i = 0

    def poll(self):
        chunk = self._samples[self._i : self._i + self._samples_per_poll]
        self._i += self._samples_per_poll
        return chunk


class _ScriptedCameraPoseEstimator:
    """One camera's per-tick detected-people list, driven from a script:
    ``script[i]`` is the list of PersonPose objects this camera detects
    at tick ``i`` (already in whatever order/subset simulates that
    camera's current view, including occlusion -- an empty list or a
    shorter list than another camera's is how occlusion is modeled)."""

    def __init__(self, script):
        self._script = script
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._script):
            return []
        people = self._script[self._i]
        self._i += 1
        return people


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


FPS = 30.0
WINDOW_FRAMES = 180  # 6s -- same window proven reliable in test_motion_correlation.py


def _curl_poses(reps_per_second, n_frames):
    return [p for _, _, p in synthetic_pose_stream(BICEP_CURL, n_frames=n_frames, fps=FPS, reps_per_second=reps_per_second)]


def _imu_for(reps_per_second, n_frames, seed, fs=90.0):
    return synthetic_imu_stream(n_seconds=n_frames / FPS, fs=fs, reps_per_second=reps_per_second, seed=seed)


def test_two_overlapping_cameras_segregate_two_members_without_double_counting():
    """Both cameras see the *same* two co-located members the whole time
    (genuinely overlapping FOVs) -- reps must still be attributed to the
    right member, and seeing the same person twice (once per camera)
    each tick must never double-count a rep."""
    registry = CheckoutRegistry()
    registry.check_out("band-carol", "member-carol", timestamp=0.0)
    registry.check_out("band-dave", "member-dave", timestamp=0.0)

    n_frames = WINDOW_FRAMES + 240
    poses_carol = _curl_poses(0.6, n_frames)
    poses_dave = _curl_poses(0.3, n_frames)
    imu_fs = 90.0
    samples_per_tick = int(imu_fs / FPS)
    imu_carol = _imu_for(0.6, n_frames, seed=12, fs=imu_fs)
    imu_dave = _imu_for(0.3, n_frames, seed=13, fs=imu_fs)
    imu_by_band = {
        "band-carol": _ChunkedIMUStream(imu_carol, samples_per_tick),
        "band-dave": _ChunkedIMUStream(imu_dave, samples_per_tick),
    }

    # Both cameras detect exactly the same two people, same order, every tick.
    script = [[poses_carol[i], poses_dave[i]] for i in range(n_frames)]
    cam_front = ZoneCamera("cam-front", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))
    cam_side = ZoneCamera("cam-side", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))

    both_present = [
        BLEReading(station_id="zone-1", rssi=-40.0, timestamp=0.0, wristband_id="band-carol"),
        BLEReading(station_id="zone-1", rssi=-41.0, timestamp=0.0, wristband_id="band-dave"),
    ]
    events = []
    runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=registry,
        cameras=[cam_front, cam_side],
        ble_reader=_ScriptedBLEReader([both_present] * n_frames),
        imu_stream_factory=lambda wid: imu_by_band[wid],
        presence_timeout_s=1.0,
        on_events=events.extend,
        clock=_FakeClock(step=1.0 / FPS),
        disambiguation_window_frames=WINDOW_FRAMES,
    )
    runner.run_forever(max_frames=n_frames)
    runner.close()

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    carol_reps = [e for e in rep_events if e.member_id == "member-carol"]
    dave_reps = [e for e in rep_events if e.member_id == "member-dave"]
    assert carol_reps and dave_reps
    assert not any(e.member_id == "member-dave" for e in carol_reps)
    assert not any(e.member_id == "member-carol" for e in dave_reps)

    # Compare against a single-camera equivalent: rep count should match
    # what one camera alone would produce for the same underlying motion,
    # not be inflated by both cameras independently feeding frames for
    # the same physical reps.
    single_camera_events = []
    solo_registry = CheckoutRegistry()
    solo_registry.check_out("band-carol", "member-carol", timestamp=0.0)
    solo_registry.check_out("band-dave", "member-dave", timestamp=0.0)
    single_cam = ZoneCamera("cam-solo", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))
    solo_runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=solo_registry,
        cameras=[single_cam],
        ble_reader=_ScriptedBLEReader([both_present] * n_frames),
        imu_stream_factory=lambda wid: _ChunkedIMUStream(
            imu_carol if wid == "band-carol" else imu_dave, samples_per_tick
        ),
        presence_timeout_s=1.0,
        on_events=single_camera_events.extend,
        clock=_FakeClock(step=1.0 / FPS),
        disambiguation_window_frames=WINDOW_FRAMES,
    )
    solo_runner.run_forever(max_frames=n_frames)
    solo_runner.close()
    solo_rep_events = [e for e in single_camera_events if e.to_dict()["event_type"] == "rep_completed"]

    assert len(rep_events) == len(solo_rep_events)


def test_occluded_camera_still_gets_member_routed_via_the_other_camera():
    """cam-front always sees both members; cam-side loses sight of dave
    (occluded from that angle) for a stretch of ticks after resolution.
    Dave's reps must keep being attributed correctly the whole time,
    routed via cam-front during the occlusion window."""
    registry = CheckoutRegistry()
    registry.check_out("band-carol", "member-carol", timestamp=0.0)
    registry.check_out("band-dave", "member-dave", timestamp=0.0)

    n_frames = WINDOW_FRAMES + 240
    poses_carol = _curl_poses(0.6, n_frames)
    poses_dave = _curl_poses(0.3, n_frames)
    imu_fs = 90.0
    samples_per_tick = int(imu_fs / FPS)
    imu_carol = _imu_for(0.6, n_frames, seed=20, fs=imu_fs)
    imu_dave = _imu_for(0.3, n_frames, seed=21, fs=imu_fs)
    imu_by_band = {
        "band-carol": _ChunkedIMUStream(imu_carol, samples_per_tick),
        "band-dave": _ChunkedIMUStream(imu_dave, samples_per_tick),
    }

    front_script = [[poses_carol[i], poses_dave[i]] for i in range(n_frames)]
    # cam-side matches cam-front (so it resolves the same group correctly
    # during the shared buffering window), then "loses" dave (always
    # slot 1, never slot 0, so carol's slot-0 mapping stays valid --see
    # the module docstring's documented person-slot-index-stability
    # caveat for why occluding the *later* slot, not the earlier one, is
    # the safe way to simulate this) for ticks
    # [WINDOW_FRAMES + 50, WINDOW_FRAMES + 100).
    occlusion_start, occlusion_end = WINDOW_FRAMES + 50, WINDOW_FRAMES + 100
    side_script = []
    for i in range(n_frames):
        if occlusion_start <= i < occlusion_end:
            side_script.append([poses_carol[i]])
        else:
            side_script.append([poses_carol[i], poses_dave[i]])

    cam_front = ZoneCamera("cam-front", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(front_script))
    cam_side = ZoneCamera("cam-side", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(side_script))

    both_present = [
        BLEReading(station_id="zone-1", rssi=-40.0, timestamp=0.0, wristband_id="band-carol"),
        BLEReading(station_id="zone-1", rssi=-41.0, timestamp=0.0, wristband_id="band-dave"),
    ]
    events = []
    runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=registry,
        cameras=[cam_front, cam_side],
        ble_reader=_ScriptedBLEReader([both_present] * n_frames),
        imu_stream_factory=lambda wid: imu_by_band[wid],
        presence_timeout_s=1.0,
        on_events=events.extend,
        clock=_FakeClock(step=1.0 / FPS),
        disambiguation_window_frames=WINDOW_FRAMES,
    )
    runner.run_forever(max_frames=n_frames)
    runner.close()

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    dave_reps = [e for e in rep_events if e.member_id == "member-dave"]
    carol_reps = [e for e in rep_events if e.member_id == "member-carol"]
    # Dave (the occluded-from-cam-side member) still gets reps counted
    # overall -- cam-front alone kept him covered.
    assert dave_reps
    assert carol_reps
    assert not any(e.member_id == "member-carol" for e in dave_reps)
    assert not any(e.member_id == "member-dave" for e in carol_reps)


def test_single_member_in_zone_only_one_camera_feeds_per_tick():
    """No ambiguity with one member zone-wide -- but with two cameras
    both seeing them, only one process_frame call per tick should ever
    happen (checked indirectly: rep count matches a single-camera run,
    not inflated)."""
    registry = CheckoutRegistry()
    registry.check_out("band-alice", "member-alice", timestamp=0.0)

    n_frames = 60
    poses_alice = _curl_poses(0.5, n_frames)
    script = [[poses_alice[i]] for i in range(n_frames)]

    cam_a = ZoneCamera("cam-a", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))
    cam_b = ZoneCamera("cam-b", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))

    present = [BLEReading(station_id="zone-1", rssi=-40.0, timestamp=0.0, wristband_id="band-alice")]
    events = []
    runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=registry,
        cameras=[cam_a, cam_b],
        ble_reader=_ScriptedBLEReader([present] * n_frames),
        presence_timeout_s=1.0,
        on_events=events.extend,
        clock=_FakeClock(step=1.0 / FPS),
    )
    runner.run_forever(max_frames=n_frames)
    runner.close()

    rep_events = [e for e in events if e.to_dict()["event_type"] == "rep_completed"]
    assert rep_events
    assert all(e.member_id == "member-alice" for e in rep_events)

    # Single-camera control run over the identical pose script.
    solo_events = []
    solo_registry = CheckoutRegistry()
    solo_registry.check_out("band-alice", "member-alice", timestamp=0.0)
    solo_cam = ZoneCamera("cam-solo", _FakeFrameSource(n_frames), _ScriptedCameraPoseEstimator(list(script)))
    solo_runner = MultiCameraZoneRunner(
        zone_id="free-weights-zone",
        exercise_name="bicep_curl",
        checkout_registry=solo_registry,
        cameras=[solo_cam],
        ble_reader=_ScriptedBLEReader([present] * n_frames),
        presence_timeout_s=1.0,
        on_events=solo_events.extend,
        clock=_FakeClock(step=1.0 / FPS),
    )
    solo_runner.run_forever(max_frames=n_frames)
    solo_runner.close()
    solo_rep_events = [e for e in solo_events if e.to_dict()["event_type"] == "rep_completed"]

    assert len(rep_events) == len(solo_rep_events)


def test_requires_at_least_one_camera():
    with pytest.raises(ValueError):
        MultiCameraZoneRunner(
            zone_id="z",
            exercise_name="bicep_curl",
            checkout_registry=CheckoutRegistry(),
            cameras=[],
            ble_reader=lambda: [],
        )
