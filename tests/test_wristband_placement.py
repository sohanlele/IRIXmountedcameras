"""irix.identity.placement.WristbandPlacementTracker's STABLE ->
SETTLING -> CALIBRATING -> STABLE lifecycle, in isolation from RepSession/
StationSessionRunner (see tests/test_rep_session_placement.py and
tests/test_station_runner.py for those integrations)."""
from __future__ import annotations

import numpy as np
import pytest

from irix.fusion.imu import IMUSample
from irix.identity.placement import BandSide, PlacementState, WristbandPlacementTracker, limb_type_of
from irix.rep_counting.exercises import BandPlacement
from irix.wristband_sim.calibration import GRAVITY_M_S2


def _still_samples(n, fs=50.0, start_t=0.0, axis=2, sign=1.0, noise=0.05, seed=0, bias=None):
    rng = np.random.default_rng(seed)
    samples = []
    gravity = np.zeros(3)
    gravity[axis] = sign * GRAVITY_M_S2
    if bias is not None:
        gravity = gravity + np.asarray(bias)
    for i in range(n):
        accel = gravity + rng.normal(scale=noise, size=3)
        gyro = rng.normal(scale=noise, size=3)
        samples.append(IMUSample(timestamp=start_t + i / fs, accel=accel, gyro=gyro))
    return samples


def _fastening_motion_samples(n, fs=50.0, start_t=0.0, seed=1):
    rng = np.random.default_rng(seed)
    return [
        IMUSample(
            timestamp=start_t + i / fs,
            accel=rng.normal(scale=8.0, size=3),  # large, noisy -- being unstrapped/carried/restrapped
            gyro=rng.normal(scale=6.0, size=3),
        )
        for i in range(n)
    ]


def test_limb_type_of_maps_sides_to_wrist_or_ankle_or_none():
    assert limb_type_of(BandSide.LEFT_WRIST) == BandPlacement.WRIST
    assert limb_type_of(BandSide.RIGHT_WRIST) == BandPlacement.WRIST
    assert limb_type_of(BandSide.LEFT_ANKLE) == BandPlacement.ANKLE
    assert limb_type_of(BandSide.RIGHT_ANKLE) == BandPlacement.ANKLE
    assert limb_type_of(BandSide.UNKNOWN) is None


def test_starts_stable_and_unpaused_with_no_change_requested():
    tracker = WristbandPlacementTracker("band-1")
    assert tracker.state == PlacementState.STABLE
    assert tracker.paused is False
    assert tracker.current_side == BandSide.LEFT_WRIST  # documented default


def test_requesting_the_same_side_already_stable_at_is_a_no_op():
    tracker = WristbandPlacementTracker("band-1", initial_side=BandSide.LEFT_WRIST)
    status = tracker.request_change(BandSide.LEFT_WRIST)
    assert status.state == PlacementState.STABLE
    assert status.paused is False


def test_a_requested_change_pauses_until_settled_and_calibrated():
    tracker = WristbandPlacementTracker(
        "band-1", initial_side=BandSide.LEFT_WRIST, settle_still_duration_s=1.0, min_calibration_samples=10,
    )
    tracker.request_change(BandSide.LEFT_ANKLE, at_time=0.0)
    assert tracker.paused is True
    assert tracker.state == PlacementState.SETTLING

    # Fastening motion: noisy, high-variance -- must not settle the tracker.
    fastening = _fastening_motion_samples(60, start_t=0.0)
    status = tracker.feed_samples(fastening)
    assert status.paused is True
    assert tracker.state == PlacementState.SETTLING

    # Now it goes still (band resting in its new position) for long
    # enough to fill the settle window with genuinely quiet samples.
    quiet = _still_samples(120, start_t=fastening[-1].timestamp + 0.02, axis=1, sign=-1.0)
    status = tracker.feed_samples(quiet)

    assert status.state == PlacementState.STABLE
    assert status.paused is False
    assert tracker.current_side == BandSide.LEFT_ANKLE
    assert status.confidence == pytest.approx(1.0)
    assert status.calibration is not None
    # The estimated calibration should recover ~zero bias against
    # whichever axis it correctly identified as "up" for this window.
    assert np.allclose(status.calibration.accel_bias, 0.0, atol=0.6)


def test_quiet_but_not_gravity_consistent_samples_do_not_confirm():
    """Low variance alone isn't enough -- e.g. free-fall or a sensor
    fault reads as "quiet" (near-zero variance) without being a
    trustworthy stationary read. Must not confirm placement from that."""
    tracker = WristbandPlacementTracker(
        "band-1", settle_still_duration_s=1.0, min_calibration_samples=10, gravity_tolerance_m_s2=1.0,
    )
    tracker.request_change(BandSide.RIGHT_WRIST, at_time=0.0)

    fake_still = [
        IMUSample(timestamp=i / 50.0, accel=np.array([0.01, 0.01, 0.01]), gyro=np.zeros(3))
        for i in range(120)  # comfortably more than 1.0s of settle window at fs=50
    ]
    status = tracker.feed_samples(fake_still)

    assert status.state == PlacementState.CALIBRATING
    assert status.paused is True
    assert tracker.current_side == BandSide.LEFT_WRIST  # unchanged -- never confirmed


def test_a_short_burst_of_stillness_shorter_than_the_settle_window_does_not_confirm_early():
    tracker = WristbandPlacementTracker("band-1", settle_still_duration_s=2.0, min_calibration_samples=10)
    tracker.request_change(BandSide.LEFT_ANKLE, at_time=0.0)

    brief_quiet = _still_samples(20, start_t=0.0, axis=2)  # only ~0.4s of samples at fs=50
    status = tracker.feed_samples(brief_quiet)

    assert status.paused is True
    assert tracker.current_side == BandSide.LEFT_WRIST
