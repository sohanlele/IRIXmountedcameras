"""Camera + wristband IMU rep-count fusion (Section 4.6, done as decision-
level/late fusion rather than the low-level EKF state fusion that section
originally sketched -- see the module docstring below for why).

This is the actual point of wearing a wristband IMU *and* having a
camera watch the same lift: a single set-level rep count that's more
trustworthy than either source alone, not two independent counters
printed side by side (which is all the pre-existing
``irix.demo.run_demo._run_imu_crosscheck`` did).

Design, and why it's decision-level rather than continuous-state fusion:
multiple published systems that combine camera/video and wearable IMU
data for exercise tracking do so at the *decision* level -- each modality
independently produces its own count/label, and a fusion step reconciles
them by confidence, rather than fusing raw signals into one continuous
state estimate (see e.g. the ACM "Wearable IMU-based Gym Exercise
Recognition Using Data Fusion Methods" paper, which fuses multiple IMU
placements this way, and the general multi-sensor activity-recognition
survey literature on confidence-weighted decision fusion). A "rep" is a
discrete event, not a smoothly-varying physical quantity like position or
orientation -- there's no meaningful single continuous state to run a
Kalman filter over between "camera thinks a rep happened at t=4.2s" and
"IMU thinks a rep happened at t=4.4s". Reconciling two independent
per-set counts (with confidence scores) is both the simpler and the
better-supported-by-prior-art approach for this specific problem;
``irix.fusion.ekf``/``irix.fusion.zupt`` remain the right tool for the
continuous-state problem they solve (visual-inertial position tracking),
just not for this one.

The fusion also runs *bidirectionally*, not just "compute both, then
pick one": the camera's own observed rep durations (``RepEvent.duration_s``
across a set) are used as a prior to constrain ``RecoFitCounter``'s
period-bounds search, which the IMU-only crosscheck never had access to
and which measurably narrows RecoFit's search space versus guessing
generic 1-4s bounds blind.

**Packet-loss awareness (Phase 2).** ``RecoFitCounter``/``ULiftCounter``'s
own ``confidence`` reflects how clean/periodic *the samples they were
given* look -- it has no way to know whether those samples are the whole
set or a packet-loss-degraded fraction of it (``irix.wristband_sim``'s
``packet_loss_pct`` can drop IMU packets same as a real radio would).
A sparse-but-locally-clean-looking signal (e.g. every third packet lost,
but the surviving ones still trace a plausible periodic shape) could
otherwise report unjustified confidence. ``fuse()`` now discounts
``imu_confidence`` by ``_sample_completeness`` -- the ratio of samples
actually present to how many should exist at the expected sample rate
over the set's observed time span -- before comparing it against the
camera's confidence, so a fusion decision under heavy packet loss
correctly leans back toward the camera rather than trusting a
confidently-computed count over a visibly incomplete signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence

import numpy as np

from ..fusion.imu import IMUSample
from ..fusion.imu_rep_counting import RecoFitCounter, RepResult, ULiftCounter

FusionSource = Literal[
    "camera_only", "camera_imu_agreement", "camera_preferred_on_disagreement",
    "imu_preferred_on_disagreement", "imu_only",
]


@dataclass
class FusedSetRepCount:
    camera_count: int
    camera_confidence: float  # RepCounter.tracking_confidence over the set
    imu_count: Optional[int]
    imu_confidence: Optional[float]
    imu_algorithm: Optional[str]  # "recofit" | "ulift", whichever fusion actually used
    imu_peak_timestamps: List[float] = field(default_factory=list)
    # Fraction (0-1) of expected samples (at RepCountFusion's configured
    # imu_sample_rate_hz) actually present over the set's observed IMU
    # time span -- 1.0 for a complete stream, lower under packet loss.
    # Surfaced (not just used internally) so a caller/ops dashboard can
    # see *why* a fusion decision leaned toward the camera under heavy
    # packet loss, not just that it did.
    imu_sample_completeness: Optional[float] = None
    fused_count: int = 0
    agreement: bool = True
    source: FusionSource = "camera_only"

    def to_dict(self) -> dict:
        return {
            "camera_count": self.camera_count,
            "camera_confidence": self.camera_confidence,
            "imu_count": self.imu_count,
            "imu_confidence": self.imu_confidence,
            "imu_algorithm": self.imu_algorithm,
            "imu_sample_completeness": self.imu_sample_completeness,
            "fused_count": self.fused_count,
            "agreement": self.agreement,
            "source": self.source,
        }


class RepCountFusion:
    """Reconciles a completed set's camera-derived rep count against a
    wristband IMU-derived one.

    Call ``fuse()`` once per completed set (mirrors ``SetCompleteEvent``),
    not per rep -- ``RecoFitCounter``/``ULiftCounter`` are themselves
    batch algorithms that need several cycles of signal to reliably
    estimate a period (see their module docstring), so they aren't
    meaningful run per-rep on a ~2s window; a whole set (typically
    15-40s / 5-15 reps) is the right unit of analysis for them, and
    happens to be exactly the granularity irix-mvp-app needs the
    authoritative rep count at.
    """

    def __init__(
        self,
        agreement_tolerance: int = 1,
        min_imu_confidence: float = 0.35,
        default_min_period: float = 1.0,
        default_max_period: float = 4.0,
        imu_sample_rate_hz: float = 100.0,
        completeness_floor: float = 0.7,
    ):
        self.agreement_tolerance = agreement_tolerance
        self.min_imu_confidence = min_imu_confidence
        self.default_min_period = default_min_period
        self.default_max_period = default_max_period
        # Expected wristband sample rate -- see irix.fusion.imu's module
        # docstring ("100-200+ Hz"); irix.wristband_sim.simulator's
        # default matches this. Only used to *measure* packet loss
        # (actual vs. expected sample count), never to fabricate missing
        # samples.
        self.imu_sample_rate_hz = imu_sample_rate_hz
        # Completeness at/above this fraction gets zero confidence
        # discount -- ordinary packet loss (a real radio typically drops
        # a small, tolerable fraction of packets even in good conditions)
        # shouldn't discount a fusion decision at all; only meaningfully
        # degraded streams below this floor should.
        self.completeness_floor = completeness_floor

    def _period_bounds(self, camera_rep_durations: Sequence[float]) -> tuple:
        """Derive RecoFitCounter's period search bounds from the camera's
        own observed rep tempo for this set, when available -- narrower
        and better-centered than a generic exercise-agnostic guess. Falls
        back to wide defaults when there's no camera timing to work from
        at all (the exact situation -- heavy occlusion, camera down --
        where leaning on the IMU matters most, so it needs to still work
        reasonably blind)."""
        durations = [d for d in camera_rep_durations if d and d > 0]
        if not durations:
            return self.default_min_period, self.default_max_period
        min_period = max(0.3, min(durations) * 0.5)
        max_period = max(durations) * 2.0
        if max_period <= min_period:
            max_period = min_period * 2.0
        return min_period, max_period

    def _best_imu_result(self, imu_samples: Sequence[IMUSample], period_bounds: tuple) -> tuple:
        """Try RecoFit (period-bounded, generally more precise when the
        bounds are decent) first; fall back to uLift (exercise-agnostic,
        no period assumption) if RecoFit isn't confident. Returns
        (RepResult, algorithm_name)."""
        min_period, max_period = period_bounds
        recofit = RecoFitCounter(min_period=min_period, max_period=max_period)
        result = recofit.count(imu_samples)
        algorithm = "recofit"
        if result.confidence < self.min_imu_confidence:
            ulift_result = ULiftCounter().count(imu_samples)
            if ulift_result.confidence > result.confidence:
                result, algorithm = ulift_result, "ulift"
        return result, algorithm

    def fuse(
        self,
        camera_count: int,
        camera_confidence: float,
        imu_samples: Optional[Sequence[IMUSample]] = None,
        camera_rep_durations: Sequence[float] = (),
    ) -> FusedSetRepCount:
        if not imu_samples:
            return FusedSetRepCount(
                camera_count=camera_count, camera_confidence=camera_confidence,
                imu_count=None, imu_confidence=None, imu_algorithm=None,
                fused_count=camera_count, agreement=True, source="camera_only",
            )

        period_bounds = self._period_bounds(camera_rep_durations)
        imu_result, algorithm = self._best_imu_result(imu_samples, period_bounds)
        completeness = self._sample_completeness(imu_samples)
        effective_imu_confidence = imu_result.confidence * min(1.0, completeness / self.completeness_floor)

        if imu_result.confidence <= 0.0 and imu_result.count == 0:
            # IMU signal unusable (e.g. band not worn, or too short/flat)
            # -- fall back to camera alone rather than reporting a
            # confidently-wrong zero.
            return FusedSetRepCount(
                camera_count=camera_count, camera_confidence=camera_confidence,
                imu_count=None, imu_confidence=None, imu_algorithm=None,
                imu_sample_completeness=completeness,
                fused_count=camera_count, agreement=True, source="camera_only",
            )

        agree = abs(camera_count - imu_result.count) <= self.agreement_tolerance
        if agree:
            fused_count, source = camera_count, "camera_imu_agreement"
        elif camera_confidence >= effective_imu_confidence:
            fused_count, source = camera_count, "camera_preferred_on_disagreement"
        else:
            fused_count, source = imu_result.count, "imu_preferred_on_disagreement"

        return FusedSetRepCount(
            camera_count=camera_count, camera_confidence=camera_confidence,
            imu_count=imu_result.count, imu_confidence=effective_imu_confidence,
            imu_algorithm=algorithm, imu_peak_timestamps=imu_result.peak_timestamps,
            imu_sample_completeness=completeness,
            fused_count=fused_count, agreement=agree, source=source,
        )

    def _sample_completeness(self, imu_samples: Sequence[IMUSample]) -> float:
        """Fraction of expected samples (at ``imu_sample_rate_hz`` over
        the observed span from the first to the last sample) actually
        present -- 1.0 for a complete stream, lower under packet loss.
        A single sample (zero span) can't measure a rate, so it's treated
        as complete (nothing to discount against) rather than div-by-zero.
        """
        if len(imu_samples) < 2:
            return 1.0
        span = imu_samples[-1].timestamp - imu_samples[0].timestamp
        if span <= 0:
            return 1.0
        expected = span * self.imu_sample_rate_hz
        if expected <= 0:
            return 1.0
        return float(np.clip(len(imu_samples) / expected, 0.0, 1.0))
