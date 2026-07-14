"""Camera/wristband clock offset + drift estimation (Section 4.6
support).

## Why this exists

Every fusion module in this repo (``irix.fusion.ekf``,
``irix.fusion.rep_fusion``, ``RepSession.add_imu_samples``) assumes
camera frame timestamps and wristband ``IMUSample.timestamp`` are
already on one shared clock -- true for ``RecordedIMUStream``/
``irix.wristband_sim`` (both timestamp everything against the same
process clock) but **not yet validated against real hardware**, and
almost certainly false for it: a real wristband free-runs its own
onboard crystal oscillator, not synchronized to the edge box's clock by
anything, and every offset estimate starts to drift the moment it's
made. This was a stated, undocumented-as-solved gap (see
``docs/SENSOR_FUSION.md``'s "What's not built" section as of Phase 1) --
this module closes it.

## Approach, and why

Two real numbers ground this: BLE's own core specification allows up to
±20 ppm drift on a device's main clock (and up to ±250 ppm on the
separate low-power sleep clock some BLE stacks use while idle) --
[Bluetooth clock accuracy requirements](https://www.fujicrystal.com/Application_details/30.html);
even a well-behaved 10 ppm crystal accumulates roughly a millisecond of
drift per minute unsynchronized. Over a 20-60 minute gym session that is
easily tens to hundreds of milliseconds -- enough to matter for
frame-level (30-60 fps => 16-33 ms/frame) visual-inertial fusion and for
correctly attributing a ZUPT dead-stop to the right camera frame, even
though it's nowhere near enough to matter for coarse BLE presence/
station-handoff timing (seconds-scale).

Hardware time-sync (e.g. a shared trigger line) isn't available to a BLE
wristband -- the standard answer in visual-inertial literature is
**online temporal calibration**: estimate the offset (and, over multiple
estimates, the drift rate) directly from how well two independently-
timestamped signals correlate, rather than trusting either clock's
labels. VINS-Fusion (Qin, Cao, Pan & Shen -- see the temporal-calibration
extension of Qin & Shen, IROS 2018) does exactly this between a camera
and a hardware-synchronized IMU; this module generalizes the same
cross-correlation idea to work from *any* two roughly-correlated motion
signals -- e.g. camera-tracked bar-path vertical velocity
(``irix.barbell.tracker``) against wristband vertical accel
(``irix.fusion.imu``) during the same rep -- since a gym wristband has no
hardware sync line to calibrate against in the first place.

``ClockSyncEstimator`` accumulates offset estimates over multiple
windows (e.g. once per set) and fits a linear drift model, so a longer
session's growing skew gets corrected rather than re-derived from
scratch (and noisily) every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .imu import IMUSample

# Bluetooth Core Specification clock-accuracy requirements -- cited above,
# used as realistic defaults for docs/simulation, not hardcoded into any
# estimation logic here (real drift is *estimated* from data, never
# assumed to equal these spec limits).
BLE_MAIN_CLOCK_DRIFT_PPM_MAX = 20.0
BLE_SLEEP_CLOCK_DRIFT_PPM_MAX = 250.0

MIN_OVERLAP_SAMPLES = 8


@dataclass
class ClockSyncEstimate:
    offset_s: float  # add this to the target stream's timestamps to align with the reference stream
    drift_ppm: Optional[float]  # None until >=2 observations exist to fit a slope against
    confidence: float  # 0-1, normalized cross-correlation peak height of the most recent observation
    n_observations: int


def estimate_offset_via_cross_correlation(
    ref_t: np.ndarray,
    ref_signal: np.ndarray,
    target_t: np.ndarray,
    target_signal: np.ndarray,
    sample_rate_hz: float = 50.0,
    max_offset_s: float = 1.0,
) -> Tuple[float, float]:
    """Resample both signals onto a common uniform time grid and
    cross-correlate to find the lag that best aligns them.

    Returns ``(offset_s, confidence)``: ``offset_s`` should be *added* to
    ``target_t`` to align the target stream with the reference stream;
    ``confidence`` (0-1) is the normalized cross-correlation peak height
    -- low when the two signals just don't resemble each other (e.g. one
    is flat/idle), not merely when timing is uncertain, so a caller can
    reject a bad estimate instead of trusting a meaningless best-fit lag.
    """
    lo = max(ref_t.min(), target_t.min() - max_offset_s)
    hi = min(ref_t.max(), target_t.max() + max_offset_s)
    if hi - lo < MIN_OVERLAP_SAMPLES / sample_rate_hz:
        return 0.0, 0.0

    grid = np.arange(lo, hi, 1.0 / sample_rate_hz)
    if len(grid) < MIN_OVERLAP_SAMPLES:
        return 0.0, 0.0

    ref_resampled = np.interp(grid, ref_t, ref_signal)
    target_resampled = np.interp(grid, target_t, target_signal)

    ref_norm = ref_resampled - ref_resampled.mean()
    target_norm = target_resampled - target_resampled.mean()
    ref_energy = np.linalg.norm(ref_norm)
    target_energy = np.linalg.norm(target_norm)
    if ref_energy < 1e-9 or target_energy < 1e-9:
        return 0.0, 0.0  # one signal is flat -- nothing to correlate against

    correlation = np.correlate(ref_norm, target_norm, mode="full")
    correlation /= ref_energy * target_energy  # normalized cross-correlation, in [-1, 1]

    max_lag_samples = int(max_offset_s * sample_rate_hz)
    center = len(target_norm) - 1
    search_lo = max(0, center - max_lag_samples)
    search_hi = min(len(correlation), center + max_lag_samples + 1)
    windowed = correlation[search_lo:search_hi]

    best_idx = int(np.argmax(windowed)) + search_lo
    lag_samples = best_idx - center  # target leads reference by this many samples if positive
    offset_s = lag_samples / sample_rate_hz
    confidence = float(np.clip(windowed[best_idx - search_lo], 0.0, 1.0))
    return offset_s, confidence


class ClockSyncEstimator:
    """Accumulates offset observations (each from
    ``estimate_offset_via_cross_correlation`` or supplied directly, e.g.
    from a known-event correlation elsewhere) and fits a linear
    offset-vs-time model, so ``estimate()`` reflects accumulated drift,
    not just the most recent single measurement.
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._observations: List[Tuple[float, float, float]] = []  # (at_time, offset_s, confidence)

    def add_observation(self, at_time: float, offset_s: float, confidence: float) -> bool:
        """Record one offset observation. Returns whether it was accepted
        (confidence below ``min_confidence`` is discarded rather than
        allowed to corrupt the drift fit with a spurious lag from two
        signals that didn't actually correlate)."""
        if confidence < self.min_confidence:
            return False
        self._observations.append((at_time, offset_s, confidence))
        return True

    def estimate(self, at_time: Optional[float] = None) -> ClockSyncEstimate:
        """Current best offset estimate, projected to ``at_time`` (default:
        the most recent observation's time) using the fitted drift rate
        once >=2 observations exist."""
        if not self._observations:
            return ClockSyncEstimate(offset_s=0.0, drift_ppm=None, confidence=0.0, n_observations=0)

        times = np.array([o[0] for o in self._observations])
        offsets = np.array([o[1] for o in self._observations])
        latest_confidence = self._observations[-1][2]
        target_time = at_time if at_time is not None else times[-1]

        if len(self._observations) < 2:
            return ClockSyncEstimate(
                offset_s=float(offsets[-1]), drift_ppm=None,
                confidence=latest_confidence, n_observations=1,
            )

        # Weighted linear fit (offset = intercept + slope * time), weighted
        # by each observation's cross-correlation confidence so a noisy
        # low-confidence estimate influences the drift fit less than a
        # sharp, high-confidence one.
        weights = np.array([o[2] for o in self._observations])
        A = np.vstack([times, np.ones_like(times)]).T
        W = np.diag(weights)
        try:
            coeffs, *_ = np.linalg.lstsq(W @ A, W @ offsets, rcond=None)
            slope, intercept = coeffs
        except np.linalg.LinAlgError:
            return ClockSyncEstimate(
                offset_s=float(offsets[-1]), drift_ppm=None,
                confidence=latest_confidence, n_observations=len(self._observations),
            )

        projected_offset = float(intercept + slope * target_time)
        drift_ppm = float(slope * 1e6)
        return ClockSyncEstimate(
            offset_s=projected_offset, drift_ppm=drift_ppm,
            confidence=latest_confidence, n_observations=len(self._observations),
        )

    def reset(self) -> None:
        self._observations = []


def apply_clock_sync(samples: List[IMUSample], estimate: ClockSyncEstimate) -> List[IMUSample]:
    """Shift every sample's timestamp by ``estimate.offset_s`` -- does not
    mutate the input, same convention as
    ``irix.wristband_sim.calibration.apply_calibration``."""
    return [IMUSample(timestamp=s.timestamp + estimate.offset_s, accel=s.accel, gyro=s.gyro) for s in samples]
