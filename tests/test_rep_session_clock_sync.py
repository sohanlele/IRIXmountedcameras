"""RepSession's Phase 3 clock-sync integration: when a shared
ClockSyncEstimator is supplied, every add_imu_samples() call applies the
estimator's *current* best correction before storing samples.

RepSession deliberately does NOT auto-derive observations from its own
camera-rep-vs-IMU-peak timestamps -- see rep_session.py's __init__
docstring and clock_sync.py's estimate_offset_from_paired_events
docstring for why that was tried and reverted (phase-offset confound).
Observations must come from an explicit calibration step instead; here
that's simulated with a direct add_observation() call, exactly like a
real calibration step would make.
"""
from __future__ import annotations

import pytest

from irix.demo.mock_pose import synthetic_imu_stream
from irix.fusion.clock_sync import ClockSyncEstimator
from irix.fusion.imu import IMUSample
from irix.pipeline.rep_session import RepSession

TRUE_WRISTBAND_LEAD_S = 0.35  # the wristband's clock reads this far ahead of the camera's


def _shifted_imu(n_seconds, reps_per_second, seed, shift_s):
    samples = synthetic_imu_stream(n_seconds=n_seconds, reps_per_second=reps_per_second, seed=seed)
    return [IMUSample(timestamp=s.timestamp + shift_s, accel=s.accel, gyro=s.gyro) for s in samples]


def test_without_a_clock_sync_estimator_behavior_is_unchanged():
    session = RepSession(exercise_name="squat", member_id="alice", station_id="squat-1")
    samples = _shifted_imu(2.0, 0.5, seed=1, shift_s=TRUE_WRISTBAND_LEAD_S)
    session.add_imu_samples(samples)
    assert session._imu_samples[0].timestamp == samples[0].timestamp


def test_with_no_observations_yet_correction_is_a_no_op():
    estimator = ClockSyncEstimator(min_confidence=0.0)
    session = RepSession(exercise_name="squat", member_id="alice", station_id="squat-1", clock_sync_estimator=estimator)
    samples = _shifted_imu(2.0, 0.5, seed=1, shift_s=TRUE_WRISTBAND_LEAD_S)
    session.add_imu_samples(samples)
    assert session._imu_samples[0].timestamp == samples[0].timestamp


def test_a_calibrated_offset_is_applied_to_subsequently_added_samples():
    # Simulates what an explicit calibration step (not RepSession itself)
    # would do: determine the wristband's clock is 0.35s ahead and record
    # that as an observation before any IMU samples are fed in.
    estimator = ClockSyncEstimator(min_confidence=0.0)
    estimator.add_observation(at_time=0.0, offset_s=-TRUE_WRISTBAND_LEAD_S, confidence=1.0)
    session = RepSession(exercise_name="squat", member_id="alice", station_id="squat-1", clock_sync_estimator=estimator)

    raw_samples = _shifted_imu(2.0, 0.5, seed=1, shift_s=TRUE_WRISTBAND_LEAD_S)
    raw_first_ts = raw_samples[0].timestamp
    session.add_imu_samples(raw_samples)

    corrected_first_ts = session._imu_samples[0].timestamp
    # Correcting a fast (ahead) wristband clock should pull its
    # timestamps backward toward the camera's clock.
    assert corrected_first_ts == pytest.approx(raw_first_ts - TRUE_WRISTBAND_LEAD_S, abs=1e-9)


def test_later_batches_pick_up_a_correction_recorded_after_earlier_ones_were_stored():
    estimator = ClockSyncEstimator(min_confidence=0.0)
    session = RepSession(exercise_name="squat", member_id="alice", station_id="squat-1", clock_sync_estimator=estimator)

    first_batch = _shifted_imu(1.0, 0.5, seed=1, shift_s=TRUE_WRISTBAND_LEAD_S)
    session.add_imu_samples(first_batch)
    uncorrected_first_ts = session._imu_samples[0].timestamp
    assert uncorrected_first_ts == pytest.approx(first_batch[0].timestamp)

    # Calibration completes mid-session; a later batch should now be corrected.
    estimator.add_observation(at_time=1.0, offset_s=-TRUE_WRISTBAND_LEAD_S, confidence=1.0)
    second_batch = _shifted_imu(1.0, 0.5, seed=2, shift_s=TRUE_WRISTBAND_LEAD_S)
    session.add_imu_samples(second_batch)
    corrected_second_ts = session._imu_samples[-len(second_batch)].timestamp
    assert corrected_second_ts == pytest.approx(second_batch[0].timestamp - TRUE_WRISTBAND_LEAD_S, abs=1e-9)


def test_paired_event_estimator_recovers_offset_for_genuinely_comparable_events():
    # Sanity check that estimate_offset_from_paired_events itself is
    # correct when given events with matching physical semantics on both
    # sides (unlike camera-rep-completion vs. IMU-peak, which it must
    # NOT be used for -- see its docstring). Here both lists are the
    # "same" detector's peaks, just clock-shifted.
    from irix.fusion.clock_sync import estimate_offset_from_paired_events

    true_events = [1.0, 3.0, 5.0, 7.0]
    shifted_events = [t + TRUE_WRISTBAND_LEAD_S for t in true_events]
    offset, confidence = estimate_offset_from_paired_events(true_events, shifted_events)
    assert offset == pytest.approx(-TRUE_WRISTBAND_LEAD_S, abs=1e-9)
    assert confidence == pytest.approx(1.0, abs=1e-6)
