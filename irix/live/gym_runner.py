"""Runs several stations together, so a member walking from one station
to another doesn't get tracked (and counted) by both cameras at once.

``StationSessionRunner`` (see that module) is correct for exactly one
isolated station: it resolves "who's present" from whatever BLE readings
its own ``ble_reader`` returns, with no idea any other station exists.
That's fine for one station, but wrong for a real multi-station gym floor
(the Section 6, 10-camera scenario ``irix.demo.run_gym_demo`` already
demonstrates with synthetic data): two adjacent stations' radios both
picking up a band mid-walk between them would, if each resolved presence
independently, both start a session for the same member -- exactly the
double-counting problem ``irix.topology.handoff.GymCoordinator`` already
solves, just never wired into anything live before this.

``GymSessionRunner`` is that wiring: it owns one ``GymCoordinator``
(topology-aware, hysteresis-based station resolution -- the same class
``run_gym_demo.py`` drives with synthetic BLE readings) and one
``CheckoutRegistry``, resolves presence *gym-wide* once per tick from a
single raw BLE reading source, and only tells each station's
``StationSessionRunner.tick()`` about the member ``GymCoordinator`` says
is actually authoritative there right now -- never two stations at once
for the same band.

Same-station crowding (two different checked-out members whose bands
both resolve to the *same* station, which RSSI proximity alone can't
disambiguate) is a separate problem from cross-station handoff, and is
handled one level down: ``_present_wristbands_at`` below returns *every*
currently-authoritative member at a station (not just one), and
``StationSessionRunner.tick()`` is what actually buffers poses/IMU and
calls ``irix.identity.motion_correlation.MotionCorrelationResolver`` to
sort out who's who once more than one band is present there at once. See
that module's docstring for the buffering/resolution details -- this
module's job stops at "who's authoritative at this station right now",
plural.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ..identity.ble_pairing import BLEReading
from ..identity.checkout import CheckoutRegistry
from ..pipeline.schema import CameraEvent
from ..pipeline.workout_state import WorkoutPhase, WorkoutStateMachine
from ..topology.handoff import GymCoordinator
from ..topology.registry import StationRegistry
from .station_runner import StationSessionRunner


class GymSessionRunner:
    def __init__(
        self,
        registry: StationRegistry,
        checkout_registry: CheckoutRegistry,
        station_runners: Dict[str, StationSessionRunner],
        ble_reader: Callable[[], List[BLEReading]],
        presence_timeout_s: float = 5.0,
        min_consecutive: int = 3,
        rssi_tie_margin: float = 3.0,
        on_gym_events: Optional[Callable[[List[CameraEvent]], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ):
        """``station_runners``: one already-constructed
        ``StationSessionRunner`` per station, keyed by ``station_id`` --
        each with its own ``frame_source``/``imu_stream_factory``/
        exercise/etc. Their own ``ble_reader``s are never called by this
        class (presence is resolved gym-wide here instead and pushed into
        each one's ``tick()`` directly) -- pass a harmless placeholder
        (e.g. ``lambda: []``) when constructing them for use here.

        ``ble_reader``: called once per gym-wide tick, returns *every*
        currently-visible ``BLEReading`` across every station and every
        band (``station_id`` and ``wristband_id`` both set on each) --
        the raw, gym-wide signal ``GymCoordinator``/``StationPairing``
        needs to resolve which station wins for each band. Contrast with
        a lone ``StationSessionRunner``'s ``ble_reader``, which only ever
        needs to answer for its own station.

        ``presence_timeout_s`` here is evaluated per *member* (per band),
        gym-wide -- a band that stops producing any reading at all,
        anywhere, for this long is considered to have left the floor
        (set down, gym exited, band removed), independent of which
        station it was last authoritative at.
        """
        self.registry = registry
        self.checkout_registry = checkout_registry
        self.station_runners = station_runners
        self.ble_reader = ble_reader
        self.presence_timeout_s = presence_timeout_s
        self.coordinator = GymCoordinator(
            registry, min_consecutive=min_consecutive, rssi_tie_margin=rssi_tie_margin
        )
        self._on_gym_events = on_gym_events or (lambda events: None)
        self._clock = clock or time.monotonic

        self._last_seen_ts: Dict[str, float] = {}  # member_id -> last tick any reading for their band appeared
        self._band_for_member: Dict[str, str] = {}  # member_id -> wristband_id (reverse of checkout, cached per-tick)
        # One WorkoutStateMachine per wristband for its whole gym visit
        # (Priority 6) -- this is the correct scope for session-level
        # phases (SESSION_STARTED...SESSION_ENDED) and cross-station
        # transitions specifically because this class, not any one
        # StationSessionRunner, is the thing that already knows when a
        # band first appears gym-wide and when GymCoordinator resolves an
        # actual cross-station handoff -- see the module docstring and
        # irix.pipeline.workout_state's own docstring for why a
        # per-station instance would have gotten this scope wrong.
        self._workout_states: Dict[str, WorkoutStateMachine] = {}

    def _workout_state_for(self, wristband_id: str) -> WorkoutStateMachine:
        """The wristband's WorkoutStateMachine, creating and driving a
        fresh one through WRISTBAND_ASSIGNED -> ... -> IDENTITY_CONFIRMED
        the first time this band is ever seen this visit -- BLE presence
        plus a successful ``CheckoutRegistry.resolve_member`` *is*
        confident identity resolution for the common single-candidate
        case (matches ``irix.identity.resolution.from_sole_candidate``'s
        same reasoning); the crowded-station case still resolves a
        specific detected *person* to this already-identity-confirmed
        member separately, one level down, in ``StationSessionRunner``."""
        machine = self._workout_states.get(wristband_id)
        if machine is not None:
            return machine
        machine = WorkoutStateMachine(wristband_id=wristband_id)
        machine.transition(WorkoutPhase.SESSION_STARTED)
        machine.transition(WorkoutPhase.MEMBER_DETECTED)
        machine.transition(WorkoutPhase.IDENTITY_CANDIDATE)
        machine.transition(WorkoutPhase.IDENTITY_CONFIRMED)
        self._workout_states[wristband_id] = machine
        return machine

    def _advance_workout_state_to_station(self, machine: WorkoutStateMachine, station_id: str) -> None:
        """Re-confirm identity and the (single, station-fixed) exercise
        for wherever this band is now authoritative -- called both for a
        brand-new session's first station and for every later
        cross-station handoff."""
        if machine.current_station_id != station_id:
            machine.record_station_transition(station_id)
        if machine.phase == WorkoutPhase.MEMBER_DETECTED:
            machine.transition(WorkoutPhase.IDENTITY_CANDIDATE)
            machine.transition(WorkoutPhase.IDENTITY_CONFIRMED)
        if machine.phase == WorkoutPhase.IDENTITY_CONFIRMED:
            runner = self.station_runners.get(station_id)
            if runner is not None:
                machine.transition(WorkoutPhase.EXERCISE_CANDIDATE)
                machine.transition(WorkoutPhase.EXERCISE_CONFIRMED)

    def _update_gym_wide_presence(self, readings: List[BLEReading], now: float) -> None:
        by_band: Dict[str, List[BLEReading]] = {}
        for r in readings:
            if r.wristband_id is not None:
                by_band.setdefault(r.wristband_id, []).append(r)

        for wristband_id, band_readings in by_band.items():
            if not self.checkout_registry.is_checked_out(wristband_id):
                continue  # a band nobody currently has checked out is never tracked
            member_id = self.checkout_registry.resolve_member(wristband_id)
            machine = self._workout_state_for(wristband_id)
            handoff_event = self.coordinator.update_member(member_id, band_readings, timestamp=now)
            if handoff_event is not None:
                self._on_gym_events([handoff_event])
            current_station = self.coordinator.current_station(member_id)
            if current_station is not None:
                self._advance_workout_state_to_station(machine, current_station)
            self._last_seen_ts[member_id] = now
            self._band_for_member[member_id] = wristband_id

    def record_wristband_returned(self, wristband_id: str, at_time: Optional[float] = None) -> bool:
        """Backend entry point for the actual physical wristband-return
        event (e.g. a caller's front-desk check-in flow, after calling
        ``CheckoutRegistry.check_in`` -- this class doesn't hook that
        registry call automatically, since a member returning a band and
        this system's gym-wide presence timeout are two different real-
        world signals that shouldn't be conflated; see ``close_stale_
        sessions``' docstring for the timeout side). Advances straight to
        ``SESSION_ENDED`` first if the band's session hadn't already been
        ended by a presence timeout. Returns whether a tracked session
        for this band existed at all."""
        machine = self._workout_states.get(wristband_id)
        if machine is None:
            return False
        if machine.phase not in (WorkoutPhase.SESSION_ENDED, WorkoutPhase.WRISTBAND_RETURNED):
            if machine.phase in (WorkoutPhase.SET_STARTED,):
                machine.transition(WorkoutPhase.SET_ENDED)
            machine.transition(WorkoutPhase.SESSION_ENDED)
        if machine.phase != WorkoutPhase.WRISTBAND_RETURNED:
            machine.transition(WorkoutPhase.WRISTBAND_RETURNED)
        del self._workout_states[wristband_id]
        return True

    def force_end_session(self, wristband_id: str) -> bool:
        """Explicit backend-triggered session end (Priority 11 -- e.g. a
        Studio operator ending a member's workout early) distinct from
        ``record_wristband_returned``: this only advances the
        `WorkoutStateMachine` to ``SESSION_ENDED`` and leaves it tracked
        (a member who's done training but hasn't walked to the front
        desk yet still has their band on) -- the actual physical hand-
        back is a separate, later event. Returns whether a tracked
        session for this band existed at all."""
        machine = self._workout_states.get(wristband_id)
        if machine is None:
            return False
        machine.force_end_session()
        return True

    def close_stale_sessions(self, now: Optional[float] = None) -> None:
        """A band that hasn't produced any BLE reading anywhere for
        ``presence_timeout_s`` is considered to have ended its *workout*
        (``WorkoutStateMachine.force_end_session`` -- see that method's
        docstring for why this is a bypass, not a validated transition)
        -- deliberately not the same event as ``record_wristband_
        returned``: a member who sets their phone/band down for a few
        minutes mid-session hasn't returned it to the front desk. Call
        this once per gym-wide tick (``run_forever`` does) or on
        whatever cadence a caller drives ticks at directly."""
        now = now if now is not None else self._clock()
        for member_id, last_seen in list(self._last_seen_ts.items()):
            if now - last_seen < self.presence_timeout_s:
                continue
            wristband_id = self._band_for_member.get(member_id)
            if wristband_id is None:
                continue
            machine = self._workout_states.get(wristband_id)
            if machine is None:
                continue
            machine.force_end_session()

    def _present_wristbands_at(self, station_id: str, now: float) -> List[str]:
        """Every band currently authoritative at ``station_id``, per
        ``GymCoordinator`` -- may be more than one (a crowded station),
        which is exactly the case ``StationSessionRunner.tick()`` needs
        to know about in order to trigger motion-correlation
        disambiguation instead of naively routing camera detections to a
        single member."""
        present = []
        for member_id in self.coordinator.active_members_at(station_id):
            last_seen = self._last_seen_ts.get(member_id)
            if last_seen is not None and (now - last_seen) < self.presence_timeout_s:
                wristband_id = self._band_for_member.get(member_id)
                if wristband_id is not None:
                    present.append(wristband_id)
        return present

    def run_forever(self, max_frames: Optional[int] = None) -> None:
        """Runs until every station's ``frame_source`` is exhausted (only
        happens in tests -- a real ``ReconnectingFrameSource`` yields
        forever). Each gym-wide tick: pull one frame from every station,
        resolve presence gym-wide from one BLE read, then drive each
        station's ``tick()`` with whichever member (if any)
        ``GymCoordinator`` says is authoritative there right now.

        Stations whose frame source has already run out (only possible
        with ``max_frames`` in a test) are simply skipped for the rest of
        the run rather than ending the whole gym-wide loop early -- a
        real deployment's cameras don't all fail in lockstep.
        """
        frame_iters = {
            station_id: runner.frame_source.frames(max_frames=max_frames)
            for station_id, runner in self.station_runners.items()
        }
        while frame_iters:
            frames = {}
            exhausted = []
            for station_id, it in frame_iters.items():
                try:
                    frames[station_id] = next(it)
                except StopIteration:
                    exhausted.append(station_id)
            for station_id in exhausted:
                del frame_iters[station_id]
            if not frames:
                break

            now = self._clock()
            self._update_gym_wide_presence(self.ble_reader(), now)
            self.close_stale_sessions(now)

            for station_id, frame in frames.items():
                present_wristband_ids = self._present_wristbands_at(station_id, now)
                self.station_runners[station_id].tick(frame, now, present_wristband_ids)

    def close(self) -> None:
        """Flush every station's in-progress session and release every
        frame source -- call on shutdown."""
        for runner in self.station_runners.values():
            runner.close()
