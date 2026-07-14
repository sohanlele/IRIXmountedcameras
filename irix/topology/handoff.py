"""Per-member station tracking with hysteresis, plus the gym-wide
coordinator that gates which station's camera events are authoritative
for a member at any given moment -- the piece that keeps two adjacent
cameras from double-counting the same person during a brief overlap
while they walk between stations.

``irix.identity.ble_pairing.StationPairing.resolve()`` already picks the
single best station for one snapshot of RSSI readings. What it doesn't
do -- because it's stateless, by design, one resolution per call -- is
decide *when* a change in the resolved station should actually count as
the member having moved, versus RSSI noise flickering the snapshot
result back and forth near a boundary between two stations' range. That
flicker is a real, well-documented BLE RSSI failure mode indoors (see
``irix.identity.ble_pairing``'s own docstring on ~5-10m typical accuracy
in cluttered spaces), and naively emitting a handoff on every resolved-
station change would spam ``StationHandoffEvent``s and threaten to
double-route rep events mid-flicker.

Mirrors ``irix.pipeline.events.BandPlacementTracker``'s split: the event
shape (``StationHandoffEvent``) lives in ``irix.pipeline.schema``; this
module holds the stateful logic that decides *when* to emit one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..identity.ble_pairing import BLEReading, StationPairing
from ..pipeline.schema import StationHandoffEvent
from .registry import StationRegistry


class MemberStationTracker:
    """Tracks one member's currently-resolved station, requiring
    ``min_consecutive`` readings in a row favoring a *different* station
    before actually switching -- the hysteresis band that keeps RSSI
    jitter near a station boundary from flip-flopping the assignment.
    """

    def __init__(self, member_id: str, pairing: StationPairing, min_consecutive: int = 3):
        self.member_id = member_id
        self.pairing = pairing
        self.min_consecutive = min_consecutive
        self.current_station: Optional[str] = None
        self._candidate: Optional[str] = None
        self._candidate_streak = 0

    def update(
        self, readings: List[BLEReading], timestamp: float, registry: Optional[StationRegistry] = None
    ) -> Optional[StationHandoffEvent]:
        resolved = self.pairing.resolve(readings)
        if resolved is None:
            return None

        if self.current_station is None:
            # First-ever assignment for this member -- not a "handoff"
            # (nothing to hand off from), so no event, just silently
            # adopt it.
            self.current_station = resolved
            self._candidate = None
            self._candidate_streak = 0
            return None

        if resolved == self.current_station:
            self._candidate = None
            self._candidate_streak = 0
            return None

        if resolved == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = resolved
            self._candidate_streak = 1

        if self._candidate_streak < self.min_consecutive:
            return None

        from_station = self.current_station
        plausible = True
        if registry is not None:
            plausible = registry.is_adjacent(from_station, resolved)
        self.current_station = resolved
        self._candidate = None
        self._candidate_streak = 0
        return StationHandoffEvent(
            member_id=self.member_id, from_station=from_station, to_station=resolved,
            timestamp=timestamp, plausible_adjacency=plausible,
        )


class GymCoordinator:
    """Gym-wide layer tying the station registry to every member's
    handoff tracker, and answering the question a station's edge box
    needs answered before it pushes a camera-derived event for some
    member_id onto the pipeline: "is this station actually the one
    currently authoritative for this member, or is this an overlapping
    detection from an adjacent camera's field of view that shouldn't be
    double-counted?"
    """

    def __init__(self, registry: StationRegistry, min_consecutive: int = 3, rssi_tie_margin: float = 3.0):
        self.registry = registry
        self._min_consecutive = min_consecutive
        self._rssi_tie_margin = rssi_tie_margin
        self._trackers: Dict[str, MemberStationTracker] = {}

    def _tracker_for(self, member_id: str) -> MemberStationTracker:
        if member_id not in self._trackers:
            self._trackers[member_id] = MemberStationTracker(
                member_id, StationPairing(rssi_tie_margin=self._rssi_tie_margin),
                min_consecutive=self._min_consecutive,
            )
        return self._trackers[member_id]

    def update_member(
        self, member_id: str, readings: List[BLEReading], timestamp: float
    ) -> Optional[StationHandoffEvent]:
        return self._tracker_for(member_id).update(readings, timestamp, registry=self.registry)

    def is_authoritative(self, member_id: str, station_id: str) -> bool:
        """Gate for a station's edge box: should it actually push a
        camera-derived RepCompletedEvent/WeightConfirmedEvent for this
        member_id right now? False for every station except the one this
        member is currently resolved to -- the mechanism that prevents
        two adjacent cameras from both reporting reps for the same person
        mid-walk between stations."""
        tracker = self._trackers.get(member_id)
        return tracker is not None and tracker.current_station == station_id

    def current_station(self, member_id: str) -> Optional[str]:
        tracker = self._trackers.get(member_id)
        return tracker.current_station if tracker else None

    def active_members_at(self, station_id: str) -> List[str]:
        return [m for m, t in self._trackers.items() if t.current_station == station_id]
