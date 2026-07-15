"""The backend interface a future IRIX Studio calls (Priority 11).

## Explicit product boundary

This repo does not build IRIX Studio -- no member-facing app, no staff
console UI, none of that exists here or should. What this module *is*:
the concrete, well-typed set of operations the founding brief names
Studio as eventually needing (assign wristband, return wristband, start
session, end session, query battery, query assignment, query wristband
status), backed by whatever real state already exists elsewhere in this
repo (``irix.identity.checkout.CheckoutRegistry``,
``irix.live.gym_runner.GymSessionRunner``,
``irix.live.station_runner.StationSessionRunner``), so that whenever a
real Studio backend gets built, it's calling into a repo that already
has correct, tested behavior for every one of these operations -- not
reinventing state tracking a second time or reaching into this repo's
internals directly.

## Why some operations map cleanly and others need a documented judgment call

``assign_wristband``/``return_wristband``/``query_assignment`` map
directly onto ``CheckoutRegistry`` -- that class already *is* the
assignment ledger.

``start_session``/``end_session`` are less direct: this repo's actual
session lifecycle is *observed* from BLE presence (``GymSessionRunner``
starts a ``WorkoutStateMachine`` the moment a checked-out band is first
seen, ends it after a presence timeout -- see that module), not
*commanded*. Real hardware doesn't wait for an app tap before a member's
presence is detected. So: ``start_session`` is a no-op confirmation for
an already-assigned, already-active band (there's nothing to "start" that
presence detection hasn't already started) and an error for a band with
no assignment at all -- Studio calling this is really asking "is this
member's session live," not commanding one into existence.
``end_session`` **is** a real, useful command distinct from
``return_wristband``: a Studio operator ending a member's workout early
(``GymSessionRunner.force_end_session``) without also processing the
physical hand-back of the band, which is a separate, later event
(``return_wristband``).

``query_battery`` is the one operation this repo genuinely cannot answer
today: no battery-voltage/level signal exists anywhere in this codebase
(neither the real ``LiveBLEIMUStream`` stub nor ``irix.wristband_sim``
model battery at all -- see ``docs/WRISTBAND_SYSTEM.md``). Reporting a
fabricated number would violate this repo's "unknown over incorrect"
principle stated everywhere else; this returns an explicit
``status: "unknown"`` with a reason instead. See ``docs/TODO.md``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..identity.checkout import CheckoutRegistry
from ..identity.placement import BandSide
from ..live.gym_runner import GymSessionRunner


@dataclass
class StudioAPIError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


class StudioBackendAPI:
    """One instance per gym deployment -- wraps whatever
    ``CheckoutRegistry``/``GymSessionRunner`` that deployment already
    constructed (see ``irix.config.gym_config`` for how those get built
    from a per-gym config file). ``gym_session_runner`` is optional: a
    caller exercising only checkout/assignment logic (e.g. a front-desk-
    only integration test) doesn't need a live gym loop running to call
    ``assign_wristband``/``return_wristband``/``query_assignment`` --
    only the session/placement/status operations that need real-time
    state require one, and raise ``StudioAPIError`` clearly if called
    without one rather than silently no-op-ing.
    """

    def __init__(self, checkout_registry: CheckoutRegistry, gym_session_runner: Optional[GymSessionRunner] = None):
        self.checkout_registry = checkout_registry
        self.gym_session_runner = gym_session_runner

    def _require_gym_session_runner(self, operation: str) -> GymSessionRunner:
        if self.gym_session_runner is None:
            raise StudioAPIError(f"{operation} requires a live GymSessionRunner, none was configured")
        return self.gym_session_runner

    def assign_wristband(self, wristband_id: str, member_id: str, at_time: float) -> Dict[str, Any]:
        """Front-desk checkout -- also, in this repo's model, the start
        of the member's whole gym visit (see the module docstring)."""
        session = self.checkout_registry.check_out(wristband_id, member_id, timestamp=at_time)
        return {"wristband_id": wristband_id, "member_id": member_id, "assigned_at": session.checked_out_at}

    def return_wristband(self, wristband_id: str, at_time: float) -> Dict[str, Any]:
        """Physical hand-back at the front desk -- checks the band back
        in (``CheckoutRegistry``) and, if a live ``GymSessionRunner`` is
        tracking this band's workout state, ends and forgets its
        ``WorkoutStateMachine`` too (``record_wristband_returned``,
        which itself force-ends the session first if it wasn't already).
        A band already checked in is not an error -- returns
        ``was_active: False`` instead of raising, since "return a band
        that's already returned" is a plausible, harmless double-call a
        real front-desk flow could make."""
        closed_session = self.checkout_registry.check_in(wristband_id, timestamp=at_time)
        if self.gym_session_runner is not None:
            self.gym_session_runner.record_wristband_returned(wristband_id, at_time=at_time)
        return {"wristband_id": wristband_id, "was_active": closed_session is not None}

    def start_session(self, wristband_id: str) -> Dict[str, Any]:
        """See the module docstring for why this is a confirmation, not
        a command, in this repo's presence-driven model. Raises
        ``StudioAPIError`` for a wristband with no current assignment at
        all -- there is no session to confirm."""
        member_id = self.checkout_registry.resolve_member(wristband_id)
        if member_id is None:
            raise StudioAPIError(f"{wristband_id!r} is not currently assigned to any member")
        active = False
        if self.gym_session_runner is not None:
            machine = self.gym_session_runner._workout_states.get(wristband_id)
            active = machine is not None
        return {"wristband_id": wristband_id, "member_id": member_id, "session_active": active}

    def end_session(self, wristband_id: str) -> Dict[str, Any]:
        """Ends this band's tracked workout (``GymSessionRunner.
        force_end_session``) without processing a physical return --
        see the module docstring for why these are different
        operations."""
        runner = self._require_gym_session_runner("end_session")
        ended = runner.force_end_session(wristband_id)
        return {"wristband_id": wristband_id, "session_was_active": ended}

    def query_battery(self, wristband_id: str) -> Dict[str, Any]:
        """See the module docstring's "why some operations..." section --
        this repo has no battery signal source at all yet. Always
        returns ``status: "unknown"``, never a fabricated level."""
        return {
            "wristband_id": wristband_id, "status": "unknown",
            "reason": "no battery-level signal source exists in this repo yet -- see docs/WRISTBAND_SYSTEM.md",
        }

    def query_assignment(self, wristband_id: str) -> Dict[str, Any]:
        member_id = self.checkout_registry.resolve_member(wristband_id)
        return {"wristband_id": wristband_id, "member_id": member_id, "is_checked_out": member_id is not None}

    def query_wristband_status(self, wristband_id: str) -> Dict[str, Any]:
        """Everything this repo can honestly say about one band right
        now, in one call -- assignment (``CheckoutRegistry``), current
        station (``GymCoordinator``, via the gym session runner),
        placement state (``irix.identity.placement``, via whichever
        ``StationSessionRunner`` currently has this band's session
        open), clock-sync confidence, and battery (always "unknown" --
        see ``query_battery``). Every sub-field that needs a live
        ``GymSessionRunner`` degrades to ``None``/``"unavailable"``
        rather than raising, since "what do we know about this band" is
        a reasonable question to ask even for an idle deployment with no
        gym loop running (unlike ``start_session``/``end_session``,
        which are genuinely meaningless without one)."""
        member_id = self.checkout_registry.resolve_member(wristband_id)
        status: Dict[str, Any] = {
            "wristband_id": wristband_id,
            "member_id": member_id,
            "is_checked_out": member_id is not None,
            "current_station_id": None,
            "workout_phase": None,
            "placement": None,
            "clock_sync": None,
            "battery": self.query_battery(wristband_id),
        }
        if self.gym_session_runner is None or member_id is None:
            return status

        machine = self.gym_session_runner._workout_states.get(wristband_id)
        if machine is not None:
            status["workout_phase"] = machine.phase.value
            status["health"] = machine.health.to_dict()

        station_id = self.gym_session_runner.coordinator.current_station(member_id)
        status["current_station_id"] = station_id
        if station_id is not None:
            runner = self.gym_session_runner.station_runners.get(station_id)
            if runner is not None:
                placement = runner.placement_status(wristband_id)
                if placement is not None:
                    status["placement"] = placement.to_dict()
                clock_sync = runner.clock_sync_status(wristband_id)
                if clock_sync is not None:
                    status["clock_sync"] = {
                        "offset_s": clock_sync.offset_s, "drift_ppm": clock_sync.drift_ppm,
                        "confidence": clock_sync.confidence, "n_observations": clock_sync.n_observations,
                    }
        return status

    def request_placement_change(self, wristband_id: str, to_side: str, at_time: float) -> Dict[str, Any]:
        """Priority 4's placement backend entry point, exposed at the
        Studio-facing layer -- delegates to whichever ``StationSessionRunner``
        currently has this band's session open (resolved via
        ``GymCoordinator``, same as ``query_wristband_status``)."""
        runner = self._require_gym_session_runner("request_placement_change")
        member_id = self.checkout_registry.resolve_member(wristband_id)
        if member_id is None:
            raise StudioAPIError(f"{wristband_id!r} is not currently assigned to any member")
        station_id = runner.coordinator.current_station(member_id)
        if station_id is None:
            raise StudioAPIError(f"{wristband_id!r} has no currently-active station")
        station_runner = runner.station_runners.get(station_id)
        if station_runner is None:
            raise StudioAPIError(f"no StationSessionRunner registered for station {station_id!r}")
        side = BandSide(to_side)
        changed = station_runner.request_wristband_placement_change(wristband_id, side, at_time=at_time)
        return {"wristband_id": wristband_id, "station_id": station_id, "requested": changed}
