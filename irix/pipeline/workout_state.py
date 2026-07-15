"""One authoritative workout state machine per wristband session
(Priority 6) -- the missing piece that names and orders everything
``RepSession``/``StationSessionRunner`` already do informally, and adds
the explicit duplicate/late-event guards the founding brief calls out by
name: no duplicate reps, no duplicate sets, no duplicate sessions, no
duplicate identities, no camera-overlap double counting, no late packet
reopening a completed set.

## Why a separate module rather than baking this into RepSession

``RepSession`` already gets rep counting, set boundaries, and fatigue
right (well-tested, unchanged by this module). What it does *not* have
is a single place that says, authoritatively, "is a `rep_completed` even
legal right now" -- today a late/duplicate IMU-fused signal reaching
``RepSession`` after a set has already closed simply wouldn't have
anywhere to go (the set's already been flushed and reset -- see
``RepSession._close_set``), which happens to prevent the worst case
accidentally, but nothing *validates* that invariant or would catch it
if a future change broke it. This module makes that invariant explicit
and independently checkable, and gives every other named state
(``identity_candidate``, ``camera_disconnect``, ...) -- most of which
*do* already have a real signal source somewhere in this repo but no
shared vocabulary tying them together -- one shared vocabulary.

## Model: one ordered phase + independent health flags

The 19 states the brief names don't form one strict sequence -- some
(``rep_completed``) repeat many times within another
(``set_started``...``set_ended``), and three (``camera_disconnect``,
``ble_disconnect``, ``identity_degraded``) are conditions that can start
and end at any point in the sequence, not steps in it. Modeling all 19 as
one flat current-state enum would force an awkward choice every time a
disconnect happens mid-set about what "the" state even is. Instead:

- ``phase``: the ordered lifecycle position (``WorkoutPhase``) --
  ``WRISTBAND_ASSIGNED -> SESSION_STARTED -> MEMBER_DETECTED ->
  IDENTITY_CANDIDATE -> IDENTITY_CONFIRMED -> EXERCISE_CANDIDATE ->
  EXERCISE_CONFIRMED -> SET_STARTED -> SET_ENDED -> ... -> SESSION_ENDED
  -> WRISTBAND_RETURNED``, with ``REST_STARTED``/``REST_ENDED`` and
  repeated ``SET_STARTED``/``SET_ENDED`` cycles between
  ``EXERCISE_CONFIRMED`` and ``SESSION_ENDED``, and
  ``STATION_TRANSITION``/``CAMERA_HANDOFF`` as phase-preserving events
  that loop back to ``MEMBER_DETECTED``/``IDENTITY_CANDIDATE`` rather
  than needing their own slot in the main sequence (moving stations
  doesn't end the session or re-assign the band).
- ``rep_completed`` is not a phase at all -- it's an *event*, legal only
  while ``phase == SET_STARTED``, validated and counted by
  ``record_rep_completed`` without changing ``phase``.
- ``identity_degraded``/``identity_recovered``,
  ``camera_disconnect``/(recovered via a fresh ``camera_handoff`` or
  frame arrival), ``ble_disconnect``/(recovered via a fresh IMU sample)
  are independent boolean health flags, settable/clearable at any phase,
  surfaced via ``health`` -- see ``WorkoutHealth``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


class WorkoutPhase(Enum):
    WRISTBAND_ASSIGNED = "wristband_assigned"
    SESSION_STARTED = "session_started"
    MEMBER_DETECTED = "member_detected"
    IDENTITY_CANDIDATE = "identity_candidate"
    IDENTITY_CONFIRMED = "identity_confirmed"
    EXERCISE_CANDIDATE = "exercise_candidate"
    EXERCISE_CONFIRMED = "exercise_confirmed"
    SET_STARTED = "set_started"
    SET_ENDED = "set_ended"
    REST_STARTED = "rest_started"
    REST_ENDED = "rest_ended"
    SESSION_ENDED = "session_ended"
    WRISTBAND_RETURNED = "wristband_returned"


# Event-like states from the brief's list that aren't part of `phase`
# (see the module docstring): REP_COMPLETED is a per-set counted event
# (record_rep_completed); STATION_TRANSITION/CAMERA_HANDOFF are
# phase-preserving transition events (record_station_transition/
# record_camera_handoff); IDENTITY_DEGRADED/IDENTITY_RECOVERED,
# CAMERA_DISCONNECT, and BLE_DISCONNECT are WorkoutHealth flags.
# Exhaustive vocabulary check only -- see test_workout_state.py.
NON_PHASE_STATE_NAMES = {
    "rep_completed", "station_transition", "camera_handoff",
    "camera_disconnect", "ble_disconnect", "identity_degraded", "identity_recovered",
}

# Legal phase -> {legal next phases}. Deliberately a whitelist, not a
# blacklist -- an unanticipated transition is rejected by default (the
# same "unknown over incorrect" posture as everywhere else in this repo)
# rather than silently allowed until something explicitly forbids it.
_ALLOWED_TRANSITIONS: Dict[WorkoutPhase, Set[WorkoutPhase]] = {
    WorkoutPhase.WRISTBAND_ASSIGNED: {WorkoutPhase.SESSION_STARTED},
    WorkoutPhase.SESSION_STARTED: {WorkoutPhase.MEMBER_DETECTED},
    WorkoutPhase.MEMBER_DETECTED: {WorkoutPhase.IDENTITY_CANDIDATE},
    WorkoutPhase.IDENTITY_CANDIDATE: {WorkoutPhase.IDENTITY_CONFIRMED, WorkoutPhase.MEMBER_DETECTED},
    WorkoutPhase.IDENTITY_CONFIRMED: {WorkoutPhase.EXERCISE_CANDIDATE, WorkoutPhase.MEMBER_DETECTED},
    WorkoutPhase.EXERCISE_CANDIDATE: {WorkoutPhase.EXERCISE_CONFIRMED, WorkoutPhase.IDENTITY_CANDIDATE, WorkoutPhase.MEMBER_DETECTED},
    WorkoutPhase.EXERCISE_CONFIRMED: {WorkoutPhase.SET_STARTED, WorkoutPhase.SESSION_ENDED, WorkoutPhase.MEMBER_DETECTED},
    WorkoutPhase.SET_STARTED: {WorkoutPhase.SET_ENDED},
    WorkoutPhase.SET_ENDED: {
        WorkoutPhase.REST_STARTED, WorkoutPhase.SET_STARTED, WorkoutPhase.SESSION_ENDED, WorkoutPhase.MEMBER_DETECTED,
    },
    WorkoutPhase.REST_STARTED: {WorkoutPhase.REST_ENDED},
    WorkoutPhase.REST_ENDED: {
        WorkoutPhase.SET_STARTED, WorkoutPhase.EXERCISE_CANDIDATE, WorkoutPhase.SESSION_ENDED, WorkoutPhase.MEMBER_DETECTED,
    },
    WorkoutPhase.SESSION_ENDED: {WorkoutPhase.WRISTBAND_RETURNED},
    WorkoutPhase.WRISTBAND_RETURNED: set(),  # terminal -- a returned band needs a fresh WorkoutStateMachine
}

# record_station_transition's target -- MEMBER_DETECTED is reachable
# from every "an identity has been confirmed at least once" phase (see
# that method's docstring); listed once here so both the transition
# table above and record_station_transition's own guard stay obviously
# in sync rather than duplicating the phase list.
_STATION_TRANSITION_ELIGIBLE_PHASES: Set[WorkoutPhase] = {
    p for p, allowed in _ALLOWED_TRANSITIONS.items() if WorkoutPhase.MEMBER_DETECTED in allowed
} | {WorkoutPhase.MEMBER_DETECTED}


@dataclass
class WorkoutHealth:
    """Independent connectivity/confidence flags -- see the module
    docstring for why these aren't part of ``phase``."""

    identity_degraded: bool = False
    camera_connected: bool = True
    ble_connected: bool = True

    def to_dict(self) -> dict:
        return {
            "identity_degraded": self.identity_degraded,
            "camera_connected": self.camera_connected,
            "ble_connected": self.ble_connected,
        }


class WorkoutStateError(Exception):
    """Raised for an illegal/duplicate transition -- a caller decides
    whether to log-and-ignore (the expected handling for a late/
    duplicate packet -- see ``record_rep_completed``) or treat it as a
    real bug, this module only refuses to silently apply it."""


@dataclass
class _SetRecord:
    set_index: int
    rep_count: int = 0
    ended: bool = False


@dataclass
class WorkoutStateMachine:
    """One instance per currently-open wristband session (mirrors
    ``ClockSyncEstimator``/``WristbandPlacementTracker``'s per-session
    lifetime in ``irix.live.station_runner.StationSessionRunner``).
    Every ``wristband_id`` gets its own machine, and only one machine may
    exist per ``wristband_id`` at a time (enforced by whatever registry
    constructs these -- see ``StationSessionRunner``'s session dict
    pattern for the established way to do that; this class itself does
    not check for a duplicate instance since it has no visibility into
    sibling instances, only ``StationSessionRunner``'s dict-keyed-by-
    wristband_id does)."""

    wristband_id: str
    phase: WorkoutPhase = WorkoutPhase.WRISTBAND_ASSIGNED
    health: WorkoutHealth = field(default_factory=WorkoutHealth)
    history: List[WorkoutPhase] = field(default_factory=list)
    _sets: List[_SetRecord] = field(default_factory=list)
    _current_camera_id: Optional[str] = None
    _current_station_id: Optional[str] = None

    def transition(self, to_phase: WorkoutPhase, at_time: Optional[float] = None) -> None:
        """Move to ``to_phase``. Raises ``WorkoutStateError`` for any
        transition not in ``_ALLOWED_TRANSITIONS`` for the current
        phase -- including re-entering ``WRISTBAND_ASSIGNED``/
        ``SESSION_STARTED`` from anywhere past them (duplicate-session
        prevention: a second ``session_started`` for the same band
        without a ``session_ended``/``wristband_returned`` in between is
        always illegal here, never silently accepted as "just start
        over")."""
        allowed = _ALLOWED_TRANSITIONS.get(self.phase, set())
        if to_phase not in allowed:
            raise WorkoutStateError(
                f"band {self.wristband_id!r}: illegal transition {self.phase.value} -> {to_phase.value}"
            )
        self.history.append(self.phase)
        self.phase = to_phase
        if to_phase == WorkoutPhase.SET_STARTED:
            self._sets.append(_SetRecord(set_index=len(self._sets)))
        elif to_phase == WorkoutPhase.SET_ENDED and self._sets:
            self._sets[-1].ended = True

    def record_rep_completed(self, rep_index_in_set: int) -> None:
        """A rep event for the *current* set. ``rep_index_in_set`` is
        the caller's own 1-based count within this set (``RepSession``
        already tracks this via ``RepCounter``) -- used only to detect a
        duplicate/replayed delivery of the *same* rep (same index seen
        twice for a set still open) or a late packet for an already-
        ``SET_ENDED`` set (the concrete "late packets reopening completed
        sets" guard: this raises rather than mutating a closed
        ``_SetRecord``, and the caller is expected to log-and-drop, not
        retry into a new set). Monotonic, not required to be contiguous
        (a genuinely dropped/never-recovered rep index is still fine --
        this only rejects going *backward or repeating* on an open set,
        or landing on a set that's already closed)."""
        if self.phase != WorkoutPhase.SET_STARTED:
            raise WorkoutStateError(
                f"band {self.wristband_id!r}: rep_completed received while phase is "
                f"{self.phase.value}, not set_started -- late packet for an already-ended set, or "
                f"a rep before any set opened"
            )
        current_set = self._sets[-1]
        if rep_index_in_set <= current_set.rep_count:
            raise WorkoutStateError(
                f"band {self.wristband_id!r}: duplicate rep_completed (index {rep_index_in_set}, "
                f"set already at {current_set.rep_count})"
            )
        current_set.rep_count = rep_index_in_set

    def record_station_transition(self, to_station_id: str) -> None:
        """Moving stations doesn't end the session -- loops back to
        ``MEMBER_DETECTED``/``IDENTITY_CANDIDATE`` (re-resolve identity
        at the new station) without touching set/rep state, matching
        ``irix.topology.handoff``'s existing "handoff, not a new
        session" model. Only legal from a phase reachable after at least
        one ``MEMBER_DETECTED`` (can't transition stations for a session
        that never started)."""
        if self._current_station_id == to_station_id:
            return  # not actually a transition -- a duplicate/redundant signal, not an error
        if self.phase not in _STATION_TRANSITION_ELIGIBLE_PHASES:
            raise WorkoutStateError(
                f"band {self.wristband_id!r}: station_transition illegal from phase {self.phase.value}"
            )
        self._current_station_id = to_station_id
        if self.phase != WorkoutPhase.MEMBER_DETECTED:
            self.transition(WorkoutPhase.MEMBER_DETECTED)

    def record_camera_handoff(self, to_camera_id: str) -> None:
        """A new camera picking up the same member at the same station
        (overlapping FOVs, ``irix.live.zone_runner.MultiCameraZoneRunner``)
        -- prevents camera-overlap double counting by construction:
        exactly one camera is ever "current" for a member at a time here,
        so a caller that routes rep/set events only from the current
        camera (checking ``current_camera_id`` before accepting an event)
        can never double-count the same physical rep seen by two
        overlapping cameras at once. A no-op if already current (a
        repeated/redundant handoff signal, not an error)."""
        if self._current_camera_id == to_camera_id:
            return
        self._current_camera_id = to_camera_id

    @property
    def current_camera_id(self) -> Optional[str]:
        return self._current_camera_id

    @property
    def current_station_id(self) -> Optional[str]:
        return self._current_station_id

    def set_identity_degraded(self, degraded: bool) -> None:
        """``identity_degraded``/``identity_recovered`` from the brief's
        list, as a health flag rather than a phase transition -- an
        ambiguous/low-confidence ``irix.identity.resolution.
        IdentityResolution`` (Priority 5) sets this ``True`` without
        forcing the phase backward to ``IDENTITY_CANDIDATE`` (a
        transient camera occlusion degrading confidence for a frame or
        two shouldn't discard an otherwise-still-correct confirmed
        identity); a caller that wants the harder reset instead calls
        ``transition(WorkoutPhase.IDENTITY_CANDIDATE)`` explicitly."""
        self.health.identity_degraded = degraded

    def set_camera_connected(self, connected: bool) -> None:
        self.health.camera_connected = connected

    def set_ble_connected(self, connected: bool) -> None:
        self.health.ble_connected = connected

    @property
    def total_reps(self) -> int:
        return sum(s.rep_count for s in self._sets)

    @property
    def completed_set_count(self) -> int:
        return sum(1 for s in self._sets if s.ended)

    def force_end_session(self) -> None:
        """End the session from *any* non-terminal phase -- for an
        exogenous "presence lost" signal (a band stops producing any BLE
        reading anywhere for the gym-wide presence timeout, see
        ``irix.live.gym_runner.GymSessionRunner.close_stale_sessions``),
        not a pipeline-internal claim about what happened. Deliberately
        bypasses ``_ALLOWED_TRANSITIONS`` (unlike ``transition()``) --
        the whitelist encodes what a well-formed *event stream* may
        legally claim happened next; a member simply walking away
        mid-set, mid-identity-resolution, or mid-rest is a real thing
        that can happen from *any* phase, not a duplicate/late/malformed
        event to reject. Closes an in-progress set first (if any) so its
        ``_SetRecord`` is marked ``ended`` rather than left dangling.
        No-op if already ``SESSION_ENDED``/``WRISTBAND_RETURNED``.
        """
        if self.phase in (WorkoutPhase.SESSION_ENDED, WorkoutPhase.WRISTBAND_RETURNED):
            return
        if self.phase == WorkoutPhase.SET_STARTED and self._sets:
            self._sets[-1].ended = True
        self.history.append(self.phase)
        self.phase = WorkoutPhase.SESSION_ENDED

    def to_dict(self) -> dict:
        return {
            "wristband_id": self.wristband_id,
            "phase": self.phase.value,
            "health": self.health.to_dict(),
            "current_camera_id": self._current_camera_id,
            "current_station_id": self._current_station_id,
            "total_reps": self.total_reps,
        }
