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
from typing import List, Optional, Union


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
    # Fault codes from irix.form.scoring.FormScorer, e.g. ["knee_valgus"].
    # Structured tags, not sentences -- irix-mvp-app turns these into
    # whatever copy its AI coach decides to say, matching the rest of
    # this event family (see module docstring: no spoken text originates
    # here). Empty list if scored clean or not scored at all; check
    # form_score is not None to tell "clean" from "unscored".
    form_faults: List[str] = field(default_factory=list)
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
            "form_faults": self.form_faults,
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
    zero for N seconds, or an operator/BLE trigger closed it out).

    ``total_reps`` is the camera's own count, kept as-is for backward
    compatibility with anything already reading this field. When wristband
    IMU data was available for the set, irix.fusion.rep_fusion.RepCountFusion
    reconciles the two into ``fused_rep_count`` -- the count irix-mvp-app
    should actually treat as authoritative, since it accounts for camera
    occlusion (falls back toward the IMU when camera tracking_confidence
    was low) rather than blindly trusting whichever source happened to run
    first.
    """

    member_id: str
    station_id: str
    exercise: str
    total_reps: int
    timestamp: float = field(default_factory=_now)
    imu_rep_count: Optional[int] = None
    fused_rep_count: Optional[int] = None  # None if no IMU data was available -- caller should use total_reps
    rep_count_agreement: Optional[bool] = None
    rep_count_source: Optional[str] = None  # see irix.fusion.rep_fusion.FusionSource

    def to_dict(self) -> dict:
        return {
            "event_type": "set_complete",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "total_reps": self.total_reps,
            "timestamp": self.timestamp,
            "imu_rep_count": self.imu_rep_count,
            "fused_rep_count": self.fused_rep_count,
            "rep_count_agreement": self.rep_count_agreement,
            "rep_count_source": self.rep_count_source,
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
class BandPlacementConfirmedEvent:
    """The member's IMU band's *actual physical* placement changed and
    was confirmed by ``irix.identity.placement.WristbandPlacementTracker``
    (settled + recalibrated -- see that module's state machine). Distinct
    from ``BandPlacementRequiredEvent`` above, which is a top-down signal
    ("the next exercise needs a different placement than before") fired
    the moment an exercise transition is known about, before anyone has
    necessarily moved anything; this one is the bottom-up confirmation
    that a real, physical move actually happened and IMU fusion has
    resumed trusting the band again."""

    wristband_id: str
    from_side: str  # BandSide.value, e.g. "left_wrist"
    to_side: str
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "band_placement_confirmed",
            "wristband_id": self.wristband_id,
            "from_side": self.from_side,
            "to_side": self.to_side,
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
    # Independent geometric sanity check (irix.weight_recognition.
    # plate_geometry_check) against irix.barbell.detector's plate count in
    # the same frame(s) -- catches a badly wrong VLM read even though it
    # can't do fine-grained plate identification on its own (see that
    # module's docstring for why). None if the check was never run (e.g.
    # no barbell detector configured for this station).
    geometry_consistent: Optional[bool] = None
    geometry_check_reason: Optional[str] = None
    # Independent color-coded-bumper-plate check (irix.weight_recognition.
    # plate_color_check) -- the *primary* method when no VLM backend is
    # configured (method == "color_plate" then), or a cross-check against
    # a VLM read when one is (method == "vlm"). None if color-plate
    # detection found nothing usable (unmarked/non-standard equipment) or
    # was never run.
    color_check_consistent: Optional[bool] = None
    color_check_reason: Optional[str] = None
    # How weight_kg/confidence were produced -- "vlm" (irix.
    # weight_recognition.vision_classifier) or "color_plate" (irix.
    # weight_recognition.plate_color_check, zero-training, no API key
    # needed). Never fabricated -- an unconfident/ambiguous read simply
    # never produces this event at all (see RepSession.process_frame's
    # weight-check block), same "unknown over incorrect" principle as
    # everywhere else in this repo.
    method: str = "vlm"

    def to_dict(self) -> dict:
        return {
            "event_type": "weight_confirmed",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "weight_kg": self.weight_kg,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "geometry_consistent": self.geometry_consistent,
            "geometry_check_reason": self.geometry_check_reason,
            "color_check_consistent": self.color_check_consistent,
            "color_check_reason": self.color_check_reason,
            "method": self.method,
        }


@dataclass
class StationHandoffEvent:
    """A member's authoritative station changed -- irix.topology.handoff's
    MemberStationTracker decides *when* this actually fires (with
    hysteresis, to absorb BLE RSSI jitter near a station boundary rather
    than emitting on every noisy reading). Distinct from
    BandPlacementRequiredEvent: this is about which station's camera is
    currently allowed to emit events for this member at all, gating
    against two adjacent cameras double-counting the same person mid-walk
    between stations."""

    member_id: str
    from_station: Optional[str]
    to_station: str
    timestamp: float = field(default_factory=_now)
    # False if to_station isn't a registered neighbor of from_station
    # (irix.topology.registry.StationRegistry.is_adjacent) -- an
    # implausible jump across the gym floor in one reading is more likely
    # a mis-resolved BLE reading than an instant teleport; worth surfacing
    # to the app/ops dashboard rather than silently trusting it.
    plausible_adjacency: bool = True

    def to_dict(self) -> dict:
        return {
            "event_type": "station_handoff",
            "member_id": self.member_id,
            "from_station": self.from_station,
            "to_station": self.to_station,
            "timestamp": self.timestamp,
            "plausible_adjacency": self.plausible_adjacency,
        }


@dataclass
class SetFatigueSummaryEvent:
    """A completed set's fatigue analysis (irix.fatigue.SetFatigueAnalyzer),
    pushed alongside SetCompleteEvent -- the structured context
    irix-mvp-app's AI uses to shape the member's next set (target weight/
    reps), per the fatigue-analysis boundary described in
    irix.fatigue's module docstring: descriptive/classifying (velocity
    loss %, which VL-zone it landed in, tempo drift, form trend), never
    prescriptive (no "reduce the weight" instruction originates here)."""

    member_id: str
    station_id: str
    exercise: str
    rep_count: int
    velocity_tier: str  # "m_s" | "deg_s" | "none"
    velocity_loss_pct: Optional[float] = None
    velocity_loss_zone: Optional[str] = None  # "VL10" | "VL20" | "VL30" | "VL45" | None
    tempo_drift_pct: Optional[float] = None
    mean_form_score: Optional[float] = None
    most_common_fault: Optional[str] = None
    # Cross-set context (irix.fatigue.SessionFatigueTracker), None until a
    # second set of the same exercise has happened this session.
    set_to_set_velocity_trend_pct: Optional[float] = None
    session_fatigue_index: Optional[float] = None
    completed_sets_this_session: int = 1
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "event_type": "set_fatigue_summary",
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "rep_count": self.rep_count,
            "velocity_tier": self.velocity_tier,
            "velocity_loss_pct": self.velocity_loss_pct,
            "velocity_loss_zone": self.velocity_loss_zone,
            "tempo_drift_pct": self.tempo_drift_pct,
            "mean_form_score": self.mean_form_score,
            "most_common_fault": self.most_common_fault,
            "set_to_set_velocity_trend_pct": self.set_to_set_velocity_trend_pct,
            "session_fatigue_index": self.session_fatigue_index,
            "completed_sets_this_session": self.completed_sets_this_session,
            "timestamp": self.timestamp,
        }


CameraEvent = Union[
    RepCompletedEvent, SetCompleteEvent, BandPlacementRequiredEvent, WeightConfirmedEvent,
    StationHandoffEvent, SetFatigueSummaryEvent,
]
