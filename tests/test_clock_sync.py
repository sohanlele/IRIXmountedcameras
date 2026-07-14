from __future__ import annotations

import numpy as np
import pytest

from irix.fusion.clock_sync import (
    ClockSyncEstimator,
    apply_clock_sync,
    estimate_offset_via_cross_correlation,
)
from irix.fusion.imu import IMUSample


def _sine(t, freq=0.5, phase=0.0):
    return np.sin(2 * np.pi * freq * t + phase)


def test_cross_correlation_recovers_known_positive_offset():
    ref_t = np.linspace(0, 10, 500)
    ref_signal = _sine(ref_t)

    true_offset = 0.3  # target stream is 0.3s "ahead" -- see convention below
    target_t = ref_t - true_offset
    target_signal = _sine(ref_t)  # same underlying signal, just re-timestamped

    offset_s, confidence = estimate_offset_via_cross_correlation(ref_t, ref_signal, target_t, target_signal)

    assert offset_s == pytest.approx(true_offset, abs=0.05)
    assert confidence > 0.8


def test_cross_correlation_recovers_known_negative_offset():
    ref_t = np.linspace(0, 10, 500)
    ref_signal = _sine(ref_t)

    true_offset = -0.4
    target_t = ref_t - true_offset
    target_signal = _sine(ref_t)

    offset_s, confidence = estimate_offset_via_cross_correlation(ref_t, ref_signal, target_t, target_signal)

    assert offset_s == pytest.approx(true_offset, abs=0.05)
    assert confidence > 0.8


def test_cross_correlation_low_confidence_when_signals_dont_correlate():
    ref_t = np.linspace(0, 10, 500)
    ref_signal = _sine(ref_t, freq=0.5)

    target_t = ref_t.copy()
    rng = np.random.default_rng(0)
    target_signal = rng.normal(0, 1, len(ref_t))  # pure noise, no relationship to ref_signal

    _, confidence = estimate_offset_via_cross_correlation(ref_t, ref_signal, target_t, target_signal)

    assert confidence < 0.5


def test_cross_correlation_flat_signal_returns_zero_confidence():
    ref_t = np.linspace(0, 10, 200)
    ref_signal = _sine(ref_t)
    target_t = ref_t.copy()
    target_signal = np.zeros_like(ref_t)  # flat -- nothing to correlate

    offset_s, confidence = estimate_offset_via_cross_correlation(ref_t, ref_signal, target_t, target_signal)

    assert confidence == 0.0
    assert offset_s == 0.0


def test_estimator_single_observation_has_no_drift_yet():
    estimator = ClockSyncEstimator(min_confidence=0.3)
    accepted = estimator.add_observation(at_time=0.0, offset_s=0.1, confidence=0.9)
    assert accepted

    result = estimator.estimate()
    assert result.offset_s == pytest.approx(0.1)
    assert result.drift_ppm is None
    assert result.n_observations == 1


def test_estimator_rejects_low_confidence_observations():
    estimator = ClockSyncEstimator(min_confidence=0.6)
    accepted = estimator.add_observation(at_time=0.0, offset_s=0.1, confidence=0.2)
    assert not accepted
    assert estimator.estimate().n_observations == 0


def test_estimator_fits_drift_rate_from_multiple_observations():
    """Simulate a wristband clock drifting at a known rate (e.g. a cheap
    crystal running fast) -- offset grows linearly with elapsed time."""
    true_drift_ppm = 50.0  # within BLE's spec-allowed range for the sleep clock
    true_slope = true_drift_ppm / 1e6

    estimator = ClockSyncEstimator(min_confidence=0.3)
    for t in [0.0, 60.0, 120.0, 180.0, 240.0]:
        offset = true_slope * t
        estimator.add_observation(at_time=t, offset_s=offset, confidence=0.9)

    result = estimator.estimate(at_time=300.0)

    assert result.drift_ppm == pytest.approx(true_drift_ppm, abs=1.0)
    assert result.offset_s == pytest.approx(true_slope * 300.0, abs=0.001)
    assert result.n_observations == 5


def test_estimator_reset_clears_observations():
    estimator = ClockSyncEstimator()
    estimator.add_observation(at_time=0.0, offset_s=0.1, confidence=0.9)
    estimator.reset()
    assert estimator.estimate().n_observations == 0


def test_apply_clock_sync_shifts_timestamps_without_mutating_input():
    from irix.fusion.clock_sync import ClockSyncEstimate

    samples = [IMUSample(timestamp=1.0, accel=np.zeros(3), gyro=np.zeros(3))]
    estimate = ClockSyncEstimate(offset_s=0.25, drift_ppm=None, confidence=0.9, n_observations=1)

    shifted = apply_clock_sync(samples, estimate)

    assert shifted[0].timestamp == pytest.approx(1.25)
    assert samples[0].timestamp == 1.0  # original untouched
