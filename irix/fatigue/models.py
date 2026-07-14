"""Data shapes for fatigue analysis."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RepFatigueSample:
    """The subset of a completed rep's signal fatigue analysis needs --
    decoupled from irix.pipeline.schema.RepCompletedEvent so this package
    doesn't have to import the whole pipeline just to analyze a list of
    reps (and so tests can build fixtures without constructing full
    pipeline events)."""

    rep_number: int
    duration_s: Optional[float] = None
    mean_velocity_m_s: Optional[float] = None  # tier 1: calibrated barbell velocity (irix.barbell)
    mean_velocity_deg_s: Optional[float] = None  # tier 2: joint-angular velocity proxy (irix.rep_counting)
    form_score: Optional[float] = None
    form_faults: List[str] = field(default_factory=list)

    @classmethod
    def from_rep_completed_event(cls, event) -> "RepFatigueSample":
        """Convenience constructor from a irix.pipeline.schema.RepCompletedEvent."""
        return cls(
            rep_number=event.rep_count,
            duration_s=event.duration_s,
            mean_velocity_m_s=event.mean_velocity_m_s,
            mean_velocity_deg_s=event.mean_velocity_deg_s,
            form_score=event.form_score,
            form_faults=list(event.form_faults),
        )


@dataclass
class SetFatigueAnalysis:
    exercise: str
    rep_count: int
    # Which velocity signal this analysis actually ran on: "m_s" (tier 1,
    # calibrated barbell velocity) is preferred when available for any
    # rep in the set, "deg_s" (tier 2, joint-angular proxy) is the
    # fallback, "none" if neither was ever populated (e.g. IMU/camera
    # both failed, or this exercise doesn't track velocity at all).
    velocity_tier: str = "none"
    first_rep_velocity: Optional[float] = None
    last_rep_velocity: Optional[float] = None
    # Classic VBT fatigue signal: % drop from the set's first rep to its
    # last, in whichever velocity tier was used. See irix.barbell.rpe
    # module docstring for the Sanchez-Medina & Gonzalez-Badillo (2011)
    # citation underlying this as a fatigue indicator.
    velocity_loss_pct: Optional[float] = None
    # Per-rep cumulative loss vs. the first rep (same units as
    # velocity_loss_pct), one entry per rep in order -- lets a caller
    # plot/inspect the trend, not just read the final number.
    velocity_loss_trend_pct: List[Optional[float]] = field(default_factory=list)
    # Which commonly-used VL training-zone threshold this set's overall
    # loss crossed (VL10/VL20/VL30/VL45, the same thresholds
    # irix.barbell.rpe's docstring names as the standard ones in that
    # literature) -- a descriptive classification, not a prescription;
    # None if under VL10 or velocity wasn't tracked.
    velocity_loss_zone: Optional[str] = None
    # % change in rep duration from first to last rep -- positive means
    # reps got slower/longer (tempo drift), a fatigue signal independent
    # of velocity that's available even with zero calibrated velocity
    # data (just needs the always-present duration_s).
    tempo_drift_pct: Optional[float] = None
    mean_form_score: Optional[float] = None
    form_score_trend: List[Optional[float]] = field(default_factory=list)
    # The fault code (irix.form.rules) that showed up on the most reps in
    # this set, if any -- a cheap "what's the one thing worth flagging"
    # summary without re-deriving it from every rep's fault list.
    most_common_fault: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "exercise": self.exercise,
            "rep_count": self.rep_count,
            "velocity_tier": self.velocity_tier,
            "first_rep_velocity": self.first_rep_velocity,
            "last_rep_velocity": self.last_rep_velocity,
            "velocity_loss_pct": self.velocity_loss_pct,
            "velocity_loss_trend_pct": self.velocity_loss_trend_pct,
            "velocity_loss_zone": self.velocity_loss_zone,
            "tempo_drift_pct": self.tempo_drift_pct,
            "mean_form_score": self.mean_form_score,
            "form_score_trend": self.form_score_trend,
            "most_common_fault": self.most_common_fault,
        }


@dataclass
class SessionFatigueSummary:
    member_id: str
    exercise: str
    completed_sets: int
    set_analyses: List[SetFatigueAnalysis] = field(default_factory=list)
    # Each completed set's first-rep velocity expressed as a % change
    # from the very first set's first-rep velocity this session -- lets
    # the app see fatigue accumulating *before* a set even starts (e.g.
    # set 3 opens 15% slower than set 1 opened, independent of within-set
    # velocity loss).
    set_to_set_velocity_trend_pct: List[Optional[float]] = field(default_factory=list)
    # 0-1 heuristic blending the most recent set's within-set velocity
    # loss with the across-set opening-velocity decline -- a single
    # number for the app's AI to use as one input among others, not a
    # replacement for looking at the underlying sets. See
    # SessionFatigueTracker docstring for exactly how it's computed and
    # why it's explicitly labeled a heuristic, not a calibrated measure.
    session_fatigue_index: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "member_id": self.member_id,
            "exercise": self.exercise,
            "completed_sets": self.completed_sets,
            "set_analyses": [s.to_dict() for s in self.set_analyses],
            "set_to_set_velocity_trend_pct": self.set_to_set_velocity_trend_pct,
            "session_fatigue_index": self.session_fatigue_index,
        }
