"""Velocity-based fatigue signals for irix-mvp-app's AI (feeds the fatigue
analysis the app uses to shape the next set's target weight/reps -- see
docs/ARCHITECTURE.md's "Rep velocity and fatigue tracking" section for
the joint-angle-proxy version of this same idea). This repo computes the
numbers; the app's AI makes the training decision.

Two signals, in order of how much they should be trusted:

1. **Velocity loss (%)**, relative to the first rep of the set --
   ``RPETracker.velocity_loss_pct``. The well-established, most robust
   fatigue signal: Sanchez-Medina & Gonzalez-Badillo (2011), "Velocity
   Loss as an Indicator of Neuromuscular Fatigue During Resistance
   Training" (Med Sci Sports Exerc), found velocity loss within a set
   correlates strongly with independent fatigue markers -- blood lactate
   (r=0.97), ammonia (R^2=0.85), and countermovement-jump-height loss
   (r=0.92) in the full squat. It doesn't require knowing the lifter's
   true 1RM velocity -- it self-normalizes against that lifter's own
   first rep of that set, which is exactly why it's the primary signal
   here rather than the absolute estimate below. VL10/VL20/VL30/VL45 are
   the commonly used training-zone thresholds in that literature (10%,
   20%, 30%, 45% velocity loss); this module reports the raw percentage
   and leaves zone/threshold decisions to the app.

2. **An absolute RPE estimate**, from published population-average
   velocity-at-1RM anchors -- ``RPETracker.estimate_rpe``. Zourdos et al.
   (2016), "Novel Resistance Training-Specific RPE Scale Measuring
   Repetitions in Reserve" (J Strength Cond Res 30(1):267-275) validated
   an RPE/RIR scale (RPE 10 = 0 RIR) and found a strong inverse
   relationship between average concentric velocity (ACV) and RPE (r
   around -0.88 for the back squat in that study; follow-on work in
   comparable populations reports correlations from -0.88 to -1.00
   depending on lift and sample). Population-average ACV-at-1RM (RPE 10)
   values used below -- squat 0.23 m/s, bench press 0.10 m/s, deadlift
   0.14 m/s -- come from Helms et al.'s replication building on the same
   Zourdos methodology.

   This estimate is meaningfully less precise than velocity loss: it's
   built from population averages, not this specific lifter's measured
   load-velocity profile. A pooled meta-analysis across 20 studies (434
   lifters) found individualized load-velocity-profile 1RM estimates
   carry a standard error of ~9.8% of 1RM even when personally
   calibrated -- a population-average anchor without that calibration
   step should be treated as a coarse, directional estimate, not a
   measurement. Exposed here because a rough number is more useful input
   to an autoregulation algorithm than nothing, not because it's
   precise. Mapping a velocity-loss percentage to a specific RPE/RIR
   delta is a genuinely open design question in this literature (the
   sources above don't converge on one universal table) -- left to
   irix-mvp-app's fatigue-analysis layer to decide, rather than guessed
   at here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

# Average concentric velocity (m/s) at 1RM / RPE 10 / 0 RIR -- population
# averages, not individually calibrated. See module docstring for sources.
EXERCISE_1RM_VELOCITY_MS: Dict[str, float] = {
    "squat": 0.23,
    "bench_press": 0.10,
    "deadlift": 0.14,
}

# A "comfortable" reference point (~RPE 6 / 4 RIR) is commonly cited in
# VBT practitioner literature as roughly 2x the 1RM velocity -- used only
# to anchor the other end of the linear interpolation below. Less rigorously
# sourced than the 1RM anchors; treat estimate_rpe's output accordingly.
_RPE_AT_ANCHOR_LOW = 6.0
_RPE_AT_ANCHOR_HIGH = 10.0
_ANCHOR_LOW_VELOCITY_MULTIPLIER = 2.0  # x the 1RM velocity


@dataclass
class RPEEstimate:
    estimated_rpe: Optional[float]  # None if exercise has no published anchor
    velocity_loss_pct: Optional[float]  # None on the first rep of a set (nothing to compare against)
    basis: str


class RPETracker:
    """Tracks fatigue signals across one set for one exercise.

    Call ``reset()`` at the start of each new set (or between rest
    periods long enough to be a genuinely new set), then ``estimate()``
    once per completed rep with that rep's mean concentric bar velocity
    (irix.barbell.tracker.BarPathVelocity.mean_velocity_m_s).
    """

    def __init__(self, exercise: str):
        self.exercise = exercise
        self._first_rep_velocity: Optional[float] = None

    def reset(self) -> None:
        self._first_rep_velocity = None

    def velocity_loss_pct(self, mean_velocity_m_s: float) -> Optional[float]:
        if self._first_rep_velocity is None:
            self._first_rep_velocity = mean_velocity_m_s
            return None
        if self._first_rep_velocity <= 0:
            return None
        loss = (self._first_rep_velocity - mean_velocity_m_s) / self._first_rep_velocity * 100.0
        return loss

    def estimate_rpe(self, mean_velocity_m_s: float) -> Optional[float]:
        """Linear interpolation between a ~RPE 6 reference velocity and
        the published RPE 10 (1RM) velocity for this exercise. Returns
        None if this exercise has no published anchor (see
        EXERCISE_1RM_VELOCITY_MS) -- most machine/isolation exercises
        don't, since the literature this is built on studied
        competition powerlifts specifically."""
        v_10 = EXERCISE_1RM_VELOCITY_MS.get(self.exercise)
        if v_10 is None:
            return None
        v_6 = v_10 * _ANCHOR_LOW_VELOCITY_MULTIPLIER
        if v_6 == v_10:
            return None
        # Linear interpolation/extrapolation in velocity, clamped to a
        # sane RPE range -- this is explicitly an approximation, see
        # module docstring.
        frac = (v_6 - mean_velocity_m_s) / (v_6 - v_10)
        rpe = _RPE_AT_ANCHOR_LOW + frac * (_RPE_AT_ANCHOR_HIGH - _RPE_AT_ANCHOR_LOW)
        return max(1.0, min(10.0, rpe))

    def estimate(self, mean_velocity_m_s: float) -> RPEEstimate:
        return RPEEstimate(
            estimated_rpe=self.estimate_rpe(mean_velocity_m_s),
            velocity_loss_pct=self.velocity_loss_pct(mean_velocity_m_s),
            basis=(
                "population-average velocity anchor (Zourdos et al. 2016 / Helms et al.), "
                "not individually calibrated -- see irix.barbell.rpe module docstring"
            ),
        )
