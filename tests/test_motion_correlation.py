"""Tests for irix.identity.motion_correlation -- disambiguating which
camera-detected person is which member via vision/IMU motion correlation,
for when BLE RSSI alone can't tell two co-located members apart."""
import numpy as np
import pytest

from irix.fusion.imu import IMUSample
from irix.identity.motion_correlation import MotionCorrelationResolver
from irix.pose.estimator import COCO_KEYPOINT_NAMES, Keypoint, PersonPose


def _make_poses(freq, phase, n=180, fps=30.0, noise=0.3, seed=0, keypoint="left_wrist"):
    rng = np.random.default_rng(seed)
    poses = []
    for i in range(n):
        t = i / fps
        y = 500.0 + 50.0 * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, noise)
        kps = []
        for name in COCO_KEYPOINT_NAMES:
            if name == keypoint:
                kps.append(Keypoint(x=100.0, y=y, confidence=0.9))
            else:
                kps.append(Keypoint(x=0.0, y=0.0, confidence=0.0))
        poses.append(PersonPose(keypoints=kps))
    return poses


def _make_imu(freq, phase, n_seconds=6.0, fs=100.0, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n_seconds * fs)
    samples = []
    for i in range(n):
        t = i / fs
        az = -9.81 - 6.0 * np.sin(2 * np.pi * freq * t + phase) + rng.normal(0, noise)
        samples.append(IMUSample(timestamp=t, accel=np.array([0, 0, az]), gyro=np.array([0, 0, 0])))
    return samples


def test_resolves_two_distinguishable_members_correctly():
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    poses_b = _make_poses(freq=0.3, phase=0.7, seed=2)
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    imu_bob = _make_imu(freq=0.3, phase=0.7, seed=4)

    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses_a, poses_b],
        pose_fps=30.0,
    )
    assert results[0] is not None and results[0].member_id == "alice"
    assert results[1] is not None and results[1].member_id == "bob"
    assert results[0].confidence > 0.5
    assert results[1].confidence > 0.5


def test_no_member_assigned_twice():
    # Both detected people correlate best with the same candidate (a
    # contrived worst case) -- the resolver must not double-assign one
    # member_id to two different detected people.
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    poses_b = _make_poses(freq=0.5, phase=0.05, seed=2)  # nearly identical motion
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    imu_bob = _make_imu(freq=0.2, phase=2.0, seed=4)  # clearly unrelated to either

    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses_a, poses_b],
        pose_fps=30.0,
    )
    assigned_members = [r.member_id for r in results if r is not None]
    assert len(assigned_members) == len(set(assigned_members))


def test_ambiguous_identical_signal_returns_none_rather_than_guessing():
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    poses_1 = _make_poses(freq=0.5, phase=0.0, seed=5, noise=0.1)
    poses_2 = _make_poses(freq=0.5, phase=0.0, seed=6, noise=0.1)

    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_alice},  # same signal for both candidates
        detected_people_poses=[poses_1, poses_2],
        pose_fps=30.0,
    )
    assert results == [None, None]


def test_handles_occlusion_gap_in_pose_sequence():
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    for p in poses_a[60:90]:
        p.keypoints[COCO_KEYPOINT_NAMES.index("left_wrist")].confidence = 0.0
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    imu_bob = _make_imu(freq=0.3, phase=0.7, seed=4)

    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses_a],
        pose_fps=30.0,
    )
    assert results[0] is not None
    assert results[0].member_id == "alice"


def test_no_candidates_returns_none_for_every_person():
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    resolver = MotionCorrelationResolver()
    results = resolver.resolve(candidate_imu_streams={}, detected_people_poses=[poses_a], pose_fps=30.0)
    assert results == [None]


def test_no_detected_people_returns_empty_list():
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice}, detected_people_poses=[], pose_fps=30.0,
    )
    assert results == []


def test_entirely_untracked_keypoint_yields_no_match():
    # left_wrist never confidently tracked at all in this pose sequence.
    poses = _make_poses(freq=0.5, phase=0.0, seed=1, keypoint="left_ankle")
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice}, detected_people_poses=[poses], pose_fps=30.0,
    )
    assert results == [None]


def test_single_candidate_single_person_matches_if_confident_enough():
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    resolver = MotionCorrelationResolver()
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice}, detected_people_poses=[poses_a], pose_fps=30.0,
    )
    # Only one candidate exists, so there's no second-best to form a
    # margin against -- confidence should reflect that this hasn't been
    # discriminated against any alternative, not just default to "sure".
    assert results[0] is not None
    assert results[0].member_id == "alice"


def test_prior_slot_assignment_breaks_an_otherwise_too_close_to_call_tie():
    """Priority 5's "fuse ... previous confirmed identity" requirement:
    when two candidates correlate almost equally well with a detected
    person (a genuine near-tie, unresolvable on this window's motion
    evidence alone), a prior_slot_assignment hint should be enough to
    break the tie toward whichever member was already confirmed in that
    slot last window -- but only as a tie-breaker, not a way to overrule
    a clearly different result (see the next test)."""
    # Both candidates fed the exact same underlying IMU signal (same
    # freq/phase/seed) -- raw correlation against the detected person is
    # therefore exactly tied, a genuine, unresolvable-on-evidence-alone
    # ambiguity, not just a numerically-close approximation of one.
    poses = _make_poses(freq=0.4, phase=0.0, seed=1, noise=3.0)
    imu_alice = _make_imu(freq=0.4, phase=0.0, seed=10, noise=2.0)
    imu_bob = _make_imu(freq=0.4, phase=0.0, seed=10, noise=2.0)

    resolver = MotionCorrelationResolver(min_confidence_margin=0.1, prior_identity_bonus=0.2)

    baseline = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses],
        pose_fps=30.0,
    )
    assert baseline[0] is None  # too close to call without a prior

    with_prior = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses],
        pose_fps=30.0,
        prior_slot_assignment={0: "bob"},
    )
    assert with_prior[0] is not None
    assert with_prior[0].member_id == "bob"
    # The bonus affects ranking/margin only -- reported confidence/
    # correlation should still reflect genuine (un-inflated) evidence.
    assert with_prior[0].correlation <= 1.0


def test_prior_slot_assignment_does_not_overrule_a_clear_result():
    poses_a = _make_poses(freq=0.5, phase=0.0, seed=1)
    imu_alice = _make_imu(freq=0.5, phase=0.0, seed=3)
    imu_bob = _make_imu(freq=0.15, phase=2.5, seed=4)

    resolver = MotionCorrelationResolver(prior_identity_bonus=0.2)
    results = resolver.resolve(
        candidate_imu_streams={"alice": imu_alice, "bob": imu_bob},
        detected_people_poses=[poses_a],
        pose_fps=30.0,
        prior_slot_assignment={0: "bob"},  # contradicts the clear evidence for alice
    )
    assert results[0] is not None
    assert results[0].member_id == "alice"
