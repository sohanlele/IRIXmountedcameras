"""Tests for the RecoFit / uLift IMU-only rep counters.

Ported from jeffreyjy/IrixDemo (Swift) -- see irix/fusion/imu_rep_counting.py
module docstring for provenance. These tests exercise the Python port
against a synthetic wristband accelerometer signal: a primary rep-rate
sinusoid plus a small higher-frequency jitter component (grip/muscle
tremor) and sensor noise -- a pure noise-free sinusoid is *not*
representative here, since both algorithms' amplitude-percentile peak
filters are calibrated against noisier real IMU data with several
candidate peaks per true rep (see the flat-signal test below for why).
"""
import numpy as np

from irix.fusion.imu import IMUSample
from irix.fusion.imu_rep_counting import RecoFitCounter, ULiftCounter


def synthetic_imu_stream(n_seconds=16.0, fs=100.0, reps_per_second=0.5, amplitude=6.0, jitter=0.6, seed=0):
    """Vertical accel oscillating like repeated concentric/eccentric reps,
    riding on gravity, with a bit of higher-frequency jitter (grip/muscle
    tremor) and sensor noise -- both algorithms' amplitude-percentile
    filters expect this kind of noise floor, not a clean sinusoid."""
    rng = np.random.default_rng(seed)
    n = int(n_seconds * fs)
    t = np.arange(n) / fs
    az = (
        -9.81
        + amplitude * np.sin(2 * np.pi * reps_per_second * t)
        + jitter * np.sin(2 * np.pi * 4.3 * t)
        + rng.normal(0, 0.15, n)
    )
    ax = rng.normal(0, 0.1, n)
    ay = rng.normal(0, 0.1, n)
    gyro_noise = rng.normal(0, 0.05, (n, 3))
    return [
        IMUSample(timestamp=float(t[i]), accel=np.array([ax[i], ay[i], az[i]]), gyro=gyro_noise[i])
        for i in range(n)
    ]


def test_recofit_counts_reps_within_tolerance():
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5)
    counter = RecoFitCounter(min_period=1.0, max_period=4.0)
    result = counter.count(samples)
    # 16s @ 0.5 reps/sec -> 8 true cycles.
    assert 6 <= result.count <= 10
    assert 0.0 < result.confidence <= 1.0


def test_recofit_handles_faster_tempo_with_matched_period_bounds():
    samples = synthetic_imu_stream(n_seconds=12.0, reps_per_second=1.0, amplitude=8.0)
    counter = RecoFitCounter(min_period=0.5, max_period=2.0)
    result = counter.count(samples)
    # 12s @ 1 rep/sec -> 12 true cycles.
    assert 9 <= result.count <= 14


def test_recofit_low_count_on_stationary_signal():
    # RecoFit's amplitude filter is self-referential (percentile of its
    # own candidate peaks), so it has no explicit "nothing is happening"
    # gate -- pure sensor noise still has a tallest 40%. This mirrors the
    # source system's design: DEMO.md's whole "tap Start Set right before
    # the descent" procedure exists because the analysis window is
    # expected to be tightly bounded to the active set by the operator,
    # not self-gated by the DSP. So the bar here is "low", not "zero".
    rng = np.random.default_rng(1)
    n = 300
    samples = [
        IMUSample(timestamp=i / 100.0, accel=np.array([0.0, 0.0, -9.81]) + rng.normal(0, 0.02, 3), gyro=np.zeros(3))
        for i in range(n)
    ]
    counter = RecoFitCounter()
    result = counter.count(samples)
    assert result.count <= 3


def test_recofit_too_short_buffer_returns_zero():
    counter = RecoFitCounter()
    assert counter.count([]).count == 0
    assert counter.count([IMUSample(timestamp=0.0, accel=np.zeros(3), gyro=np.zeros(3))]).count == 0


def test_ulift_counts_reps_within_tolerance_no_period_config():
    # uLift needs no exercise-specific period bounds -- same synthetic
    # stream, no min/max period passed in.
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5)
    counter = ULiftCounter()
    result = counter.count(samples)
    assert 6 <= result.count <= 10
    assert 0.0 < result.confidence <= 1.0


def test_ulift_handles_faster_tempo():
    samples = synthetic_imu_stream(n_seconds=12.0, reps_per_second=0.75, amplitude=7.0)
    counter = ULiftCounter()
    result = counter.count(samples)
    # 12s @ 0.75 reps/sec -> 9 true cycles. uLift is the exercise-agnostic
    # fallback (Section 4.7) and is less precise than RecoFit when a
    # proper period config is available -- wider tolerance is expected.
    assert 6 <= result.count <= 13
