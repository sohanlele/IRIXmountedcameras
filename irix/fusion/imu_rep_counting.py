"""Wristband IMU-only rep counting: RecoFit and uLift algorithms.

Ported from jeffreyjy/IrixDemo (private repo, Swift), which built these for
a first-person smart-glasses form factor where a phone-side DSP counts
reps purely from the wristband's IMU stream (no camera fusion at all in
that system -- unlike IRIX's mounted-camera design, which fuses camera
pose with the IMU via ``VisualInertialEKF``/``zupt.py``). The algorithms
themselves are camera-agnostic: they only consume an IMU time series, so
they port over unchanged and serve as the wristband IMU-only fallback
signal for Section 5.3 (fallback redundancy) -- what a station reports
when its camera is occluded or its edge box is down -- and as an
independent cross-check against the joint-angle + EKF counter described
in Section 4.6.

Two algorithms, both from the published literature (reimplemented here,
not vendored from any package):

- ``RecoFitCounter`` -- Morris et al., "RecoFit: Exercise Set Detection
  and Rep Counting", CHI 2014. Needs exercise-specific period bounds
  (``min_period``/``max_period``); bandpass-filters vertical acceleration,
  then does multi-pass peak detection: greedy min-period spacing ->
  autocorrelation period refinement -> amplitude filter.
- ``ULiftCounter`` -- Lim et al., "uLift", IEEE Access 2024. Exercise-
  agnostic: no hardcoded period bounds. Estimates the workout rate from
  sliding-window weighted autocorrelation across all three axes, then
  auto-selects the axis with the largest range for peak counting.

Both return a ``RepResult(count, confidence)`` where confidence is a
heuristic in [0, 1] (more accepted peaks -> higher confidence), matching
the original Swift implementation's calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import ellip, find_peaks, sosfiltfilt

from .imu import IMUSample


@dataclass
class RepResult:
    count: int
    confidence: float
    # Absolute timestamps (same clock as the input IMUSample.timestamp) of
    # each accepted peak -- lets a caller line these up event-by-event
    # against camera-derived rep timestamps (irix.fusion.rep_fusion), not
    # just compare final counts.
    peak_timestamps: List[float] = None

    def __post_init__(self):
        if self.peak_timestamps is None:
            self.peak_timestamps = []


def _imu_buffer_arrays(samples: Sequence[IMUSample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a list of IMUSample into (t, ax, ay, az) float arrays."""
    t = np.array([s.timestamp for s in samples], dtype=float)
    ax = np.array([s.accel[0] for s in samples], dtype=float)
    ay = np.array([s.accel[1] for s in samples], dtype=float)
    az = np.array([s.accel[2] for s in samples], dtype=float)
    return t, ax, ay, az


def _resample_uniform(signal: np.ndarray, time: np.ndarray, target_fs: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Linear-interpolate a non-uniformly-sampled 1D signal onto a uniform grid."""
    dt = 1.0 / target_fs
    if len(time) < 2 or time[-1] <= time[0]:
        return np.array([]), np.array([]), dt
    n = int((time[-1] - time[0]) / dt)
    if n <= 0:
        return np.array([]), np.array([]), dt
    t_uniform = time[0] + np.arange(n) * dt
    sig_uniform = np.interp(t_uniform, time, signal)
    return sig_uniform, t_uniform, dt


def _percentile(arr: np.ndarray, p: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, p))


def _autocorrelation(signal: np.ndarray) -> np.ndarray:
    """Normalized autocorrelation (positive lags only), ac[0] == 1."""
    n = len(signal)
    if n == 0:
        return np.array([])
    centered = signal - signal.mean()
    ac = np.correlate(centered, centered, mode="full")[n - 1:]
    if ac[0] != 0:
        ac = ac / ac[0]
    return ac


def _greedy_select_by_amplitude(candidates: np.ndarray, amps: np.ndarray, min_spacing: int) -> np.ndarray:
    """Accept peaks in descending amplitude order, rejecting any within
    ``min_spacing`` samples of an already-accepted peak (matches the
    Swift implementation's greedy spacing filter)."""
    order = np.argsort(-amps)
    accepted: List[int] = []
    for idx in order:
        c = candidates[idx]
        if all(abs(int(c) - a) >= min_spacing for a in accepted):
            accepted.append(int(c))
    return np.array(sorted(accepted))


class RecoFitCounter:
    """RecoFit counting algorithm (Morris et al., CHI 2014).

    Requires exercise-specific period bounds. Good fit for IRIX's
    per-exercise config (``irix.rep_counting.exercises``) -- min/max
    period can be derived from the same joint-angle exercise config used
    by the camera-based counter.
    """

    def __init__(
        self,
        min_period: float = 1.0,
        max_period: float = 4.0,
        target_fs: float = 50.0,
        bandpass_low_hz: float = 0.15,
        bandpass_high_hz: float = 11.0,
        filter_fs: float = 200.0,
    ):
        self.min_period = min_period
        self.max_period = max_period
        self.target_fs = target_fs
        self.filter_fs = filter_fs
        # 4th-order elliptic bandpass, computed directly with scipy rather
        # than the hardcoded SOS coefficients the Swift port carries
        # (which were precomputed offline for lack of a scipy equivalent
        # on-device).
        nyquist = filter_fs / 2.0
        self._sos = ellip(
            4, 1, 40,
            [bandpass_low_hz / nyquist, bandpass_high_hz / nyquist],
            btype="band", output="sos",
        )

    def count(self, samples: Sequence[IMUSample]) -> RepResult:
        if len(samples) < 2:
            return RepResult(0, 0.0)
        t, _, _, az = _imu_buffer_arrays(samples)
        neg_az = -az

        sig_hi, t_hi, _ = _resample_uniform(neg_az, t, self.filter_fs)
        if sig_hi.size <= 400:
            return RepResult(0, 0.0)
        filtered_hi = sosfiltfilt(self._sos, sig_hi)

        filtered, t_filtered, dt = _resample_uniform(filtered_hi, t_hi, self.target_fs)
        if filtered.size <= int(self.target_fs * 2):
            return RepResult(0, 0.0)

        min_samples = max(1, round(self.min_period / dt))
        max_samples = max(1, round(self.max_period / dt))

        candidates, _ = find_peaks(filtered)
        if candidates.size == 0:
            return RepResult(0, 0.0)

        amps = filtered[candidates]
        accepted = _greedy_select_by_amplitude(candidates, amps, min_samples)
        if accepted.size == 0:
            return RepResult(0, 0.0)

        window_half = max_samples * 2
        periods = []
        for peak_idx in accepted:
            lo = max(0, peak_idx - window_half)
            hi = min(len(filtered), peak_idx + window_half)
            chunk = filtered[lo:hi]
            ac = _autocorrelation(chunk)
            lag_lo = min(min_samples, len(ac) - 1)
            lag_hi = min(max_samples, len(ac) - 1)
            if lag_lo >= lag_hi:
                periods.append(self.min_period)
                continue
            search = ac[lag_lo:lag_hi + 1]
            best_offset = int(np.argmax(search))
            periods.append((lag_lo + best_offset) * dt)

        period = float(np.median(periods)) if periods else self.min_period
        refined_min = max(1, round(0.75 * period / dt))

        amps2 = filtered[accepted]
        refined = _greedy_select_by_amplitude(accepted, amps2, refined_min)
        if refined.size == 0:
            return RepResult(0, 0.0)

        ref_amps = filtered[refined]
        thresh = 0.5 * _percentile(ref_amps, 40)
        final = refined[ref_amps >= thresh]

        count = int(final.size)
        confidence = min(1.0, count / 20.0 + 0.5) if count else 0.0
        peak_ts = t_filtered[final].tolist() if count else []
        return RepResult(count, confidence, peak_timestamps=peak_ts)


class ULiftCounter:
    """uLift counting algorithm (Lim et al., IEEE Access 2024).

    Exercise-agnostic -- no hardcoded period bounds. Derives the workout
    rate from sliding-window weighted autocorrelation across all three
    axes, then auto-selects the axis with the largest range for peak
    counting. Useful as a fallback when an exercise doesn't have a
    ``RecoFitCounter``-compatible period config yet (cf. Section 4.7's
    class-agnostic rep counting discussion).
    """

    def __init__(self, target_fs: float = 50.0):
        self.target_fs = target_fs

    def count(self, samples: Sequence[IMUSample]) -> RepResult:
        if len(samples) < 2:
            return RepResult(0, 0.0)
        t, ax_raw, ay_raw, az_raw = _imu_buffer_arrays(samples)
        az_raw = -az_raw

        ax, t_u, _ = _resample_uniform(ax_raw, t, self.target_fs)
        ay, _, _ = _resample_uniform(ay_raw, t, self.target_fs)
        az, _, _ = _resample_uniform(az_raw, t, self.target_fs)
        n = ax.size
        if n <= int(self.target_fs * 3):
            return RepResult(0, 0.0)

        workout_rate = self._estimate_workout_rate(ax, ay, az)
        if workout_rate <= 0:
            return RepResult(0, 0.0)

        final_indices = self._count_with_best_axis(ax, ay, az, workout_rate)
        count = int(final_indices.size)
        confidence = min(1.0, count / 20.0 + 0.5) if count else 0.0
        peak_ts = t_u[final_indices].tolist() if count else []
        return RepResult(count, confidence, peak_timestamps=peak_ts)

    def _estimate_workout_rate(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> float:
        n = ax.size
        w4 = int(4.0 * self.target_fs)
        w8 = int(8.0 * self.target_fs)
        step = max(1, int(0.2 * self.target_fs))

        all_periods = []
        for window_size in (w4, w8):
            if n < window_size:
                continue
            end = window_size
            while end <= n:
                win_ax = ax[end - window_size:end]
                win_ay = ay[end - window_size:end]
                win_az = az[end - window_size:end]
                acf = self._weighted_acf(win_ax, win_ay, win_az)
                period = self._extract_period_from_acf(acf)
                if period is not None:
                    all_periods.append(period)
                end += step

        if not all_periods:
            return 0.0
        p45 = _percentile(np.array(all_periods), 45)
        p95 = _percentile(np.array(all_periods), 95)
        return (p45 + p95) / 2.0

    def _weighted_acf(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
        acfs = [_autocorrelation(axis) for axis in (ax, ay, az)]
        energies = np.array([np.sqrt(np.sum(ac ** 2)) for ac in acfs])
        weights = np.exp(energies - energies.max())
        weights = weights / weights.sum()
        length = min(len(ac) for ac in acfs)
        combined = np.zeros(length)
        for w, ac in zip(weights, acfs):
            combined += w * ac[:length]
        return combined

    def _extract_period_from_acf(self, acf: np.ndarray) -> Optional[float]:
        min_lag = int(0.5 * self.target_fs)
        if min_lag >= len(acf):
            return None
        search = acf[min_lag:]
        peaks, _ = find_peaks(search)
        if peaks.size == 0:
            return None
        best = peaks[np.argmax(search[peaks])]
        return (best + min_lag) / self.target_fs

    def _count_with_best_axis(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray, workout_rate: float) -> np.ndarray:
        """Returns the accepted peak *indices* (into the target_fs-uniform
        grid), not just a count -- callers map these back to timestamps
        via the uniform time array from the first resample in ``count()``."""
        axes = [ax - ax.mean(), ay - ay.mean(), az - az.mean()]
        ranges = [float(a.max() - a.min()) if a.size else 0.0 for a in axes]
        best_axis = int(np.argmax(ranges))
        sig = axes[best_axis]

        candidates, _ = find_peaks(sig)
        if candidates.size == 0:
            return np.array([], dtype=int)

        cand_amps = sig[candidates]
        amp_thresh = _percentile(cand_amps, 80)
        above = candidates[cand_amps >= amp_thresh]
        if above.size == 0:
            return np.array([], dtype=int)

        wr_half_samples = max(1, round(workout_rate / 2.0 * self.target_fs))
        final = _greedy_select_by_amplitude(above, sig[above], wr_half_samples)
        return final
