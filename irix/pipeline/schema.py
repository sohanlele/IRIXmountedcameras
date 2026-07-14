"""Structured camera events -- the API contract with irix-mvp-app (Section 6.3 / 8.2).

Camera -> zone edge box (pose + object detection + rep logic, all local)
-> local buffer -> aggregator -> jeffreyjy/irix-mvp-app's backend. This
repo computes what happened on the gym floor; the app (FastAPI backend +
iOS frontend, github.com/jeffreyjy/irix-mvp-app) owns the UI and the
AI-generated instructions/coaching copy. Nothing here generates spoken
text or decides what to tell a member -- that's the app's `agents/`
layer, not this one. Each event type below is a distinct, well-typed
payload rather than one generic blob, so it maps cleanly onto whatever
Pydantic schemas / DB models the app's `backend/app/schemas` end up using
once it exposes a live-camera-data ingestion endpoint (doesn't exist yet
as of this writing -- the app currently only models `workout_plan` and
`workout_session`).

Raw video never persists beyond the local debug buffer, and no field on
any event here carries video or a statutorily-defined biometric
identifier (Section 8.1) -- only rep counts, form scores, and a
wristband-assigned member id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union


def _now() -> float:
    import time

    return time.monotonic()


@dataclass
class RepCompletedEvent:
    """One rep just finished (irix.rep_counting.state_machine.RepEvent,
    turned into something worth sending off-station).

    ``duration_s`` (time since the previous rep) and the two velocity
    fields exist for fatigue-trend analysis on the irix-mvp-app side --
    e.g. velocity loss across a set (a standard velocity-based-training /
    autoregulation signal) to shape the next set's target weight/reps.
    This repo only supplies the per-rep numbers; the fatigue judgment
    itself, and what to do about it, is the app's AI's job.

    The velocity fields are joint-angular velocity (deg/s) -- a rep-speed
    *proxy* derived from the camera-tracked joint angle, not a calibrated
    linear bar velocity in m/s. A calibrated velocity needs Section 4.5's
    barbell centroid tracking against per-station camera geometry, which
    isn't built yet (see docs/ARCHITECTURE.md). Good enough for relative
    within-session trend tracking (is this rep slower than the first rep
    of the set?), not for absolute cross-device VBT comparison.
    """

    member_id: str  # wristband-assigned id, not a biometric identifier
    station_id: str
    exercise: str
    rep_count: int
    form_score: Optional[float] = None  # 0-1, None if not yet scored
    weight_kg: Optional[float] = None
    duration_s: Optional[float] = None  # time since the previous rep (tempo/cadence)

    # Tier 2 (fallback): joint-angular velocity proxy, always available
    # from the camera-tracked joint angle alone (irix.rep_counting).
    peak_velocity_deg_s: Optional[float] = None
    mean_velocity_deg_s: Optional[float] = None

    # Tier 1 (preferred, when a barbell/dumbbell is being tracked):
    # calibrated linear velocity in m/s from irix.barbell.tracker, plus
    # the fatigue signals derived from it (irix.barbell.rpe). None
    # whenever no free weight is being tracked for this rep (e.g. a
    # machine station, or before the detector has locked on) -- callers
    # should fall back to the deg/s fields above in that case.
    peak_velocity_m_s: Optional[float] = None
    mean_velocity_m_s: Optional[float] = None
    velocity_loss_pct: Optional[float] = None
    estimated_rpe: Optional[float] = None

    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "rep_completed",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "rep_count": self.rep_count,
            "form_score": self.form_score,
            "weight_kg": self.weight_kg,
            "duration_s": self.duration_s,
            "peak_velocity_deg_s": self.peak_velocity_deg_s,
            "mean_velocity_deg_s": self.mean_velocity_deg_s,
            "peak_velocity_m_s": self.peak_velocity_m_s,
            "mean_velocity_m_s": self.mean_velocity_m_s,
            "velocity_loss_pct": self.velocity_loss_pct,
            "estimated_rpe": self.estimated_rpe,
            "timestamp": self.timestamp,
        }


@dataclass
class SetCompleteEvent:
    """A set has ended (station-level signal, e.g. rep rate dropped to
    zero for N seconds, or an operator/BLE trigger closed it out)."""

    member_id: str
    station_id: str
    exercise: str
    total_reps: int
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "set_complete",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "total_reps": self.total_reps,
            "timestamp": self.timestamp,
        }


@dataclass
class BandPlacementRequiredEvent:
    """The member's IMU band needs to move before the next exercise's
    IMU signal (wristband fallback / fusion counters) is trustworthy --
    see irix.rep_counting.exercises.BandPlacement and
    irix.pipeline.events.BandPlacementTracker, which decides *when* to
    emit this (only on an actual change, not every exercise)."""

    member_id: str
    exercise: str
    from_placement: str  # "wrist" | "ankle"
    to_placement: str
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "band_placement_required",
            "member_id": self.member_id,
            "exercise": self.exercise,
            "from_placement": self.from_placement,
            "to_placement": self.to_placement,
            "timestamp": self.timestamp,
        }


@dataclass
class WeightConfirmedEvent:
    """A station's VisionPlateClassifier (Section 4.4) reached
    confirm_n-of-confirm_window agreement on the loaded weight."""

    member_id: str
    station_id: str
    exercise: str
    weight_kg: float
    confidence: float
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "weight_confirmed",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "weight_kg": self.weight_kg,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


CameraEvent = Union[RepCompletedEvent, SetCompleteEvent, BandPlacementRequiredEvent, WeightConfirmedEvent]
