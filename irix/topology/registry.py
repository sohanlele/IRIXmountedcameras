"""Static configuration of a gym's camera/station layout.

A 10-camera install isn't 10 independent copies of the single-station
demo -- each camera covers a fixed physical station, and the edge
pipeline (irix.pipeline) needs to know that layout to route events (which
zone does a station's LocalBuffer belong to, Section 6.3) and to let
irix.topology.handoff reason about "is this a plausible next station for
a member who just left station X" (adjacent zones) vs. an implausible
jump (likely a mis-resolved BLE reading, not a real handoff).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class StationInfo:
    station_id: str
    camera_id: str
    zone: str  # e.g. "free_weights", "machines", "platform" -- groups stations for Aggregator zone routing (Section 6.3)
    default_exercise: Optional[str] = None
    # station_ids a member could plausibly walk to next without an
    # implausible-jump flag getting raised (irix.topology.handoff). Two
    # stations on opposite ends of a large gym floor generally shouldn't
    # be adjacent even if BLE RSSI briefly suggests otherwise (multipath/
    # reflection is a known BLE RSSI failure mode in cluttered indoor
    # spaces -- see irix.identity.ble_pairing's docstring).
    adjacent_station_ids: List[str] = field(default_factory=list)


class StationRegistry:
    """Holds the fixed station/camera layout for one gym location."""

    def __init__(self, stations: Optional[List[StationInfo]] = None):
        self._by_id: Dict[str, StationInfo] = {s.station_id: s for s in (stations or [])}

    def register(self, station: StationInfo) -> None:
        self._by_id[station.station_id] = station

    def get(self, station_id: str) -> Optional[StationInfo]:
        return self._by_id.get(station_id)

    def all(self) -> List[StationInfo]:
        return list(self._by_id.values())

    def stations_in_zone(self, zone: str) -> List[StationInfo]:
        return [s for s in self._by_id.values() if s.zone == zone]

    def camera_for(self, station_id: str) -> Optional[str]:
        s = self.get(station_id)
        return s.camera_id if s else None

    def is_adjacent(self, from_station_id: str, to_station_id: str) -> bool:
        """True if ``to_station_id`` is a station a member at
        ``from_station_id`` could plausibly walk to directly. Unknown
        stations (not in the registry) are treated as *not* adjacent --
        conservative default, since an unregistered station_id is more
        likely a data error than a real 11th camera nobody registered."""
        s = self.get(from_station_id)
        if s is None:
            return False
        return to_station_id in s.adjacent_station_ids

    def __len__(self) -> int:
        return len(self._by_id)


def build_default_ten_station_gym() -> StationRegistry:
    """A concrete, reasonable 10-camera layout for demo/test purposes:
    a small free-weights section (squat racks, bench, deadlift platform),
    a dumbbell/curl area, and a machine row (leg press, hack squat) --
    covering every exercise irix.rep_counting.exercises currently
    configures. Adjacency is a simple layout: stations within the same
    physical row/section are adjacent to their immediate neighbors; the
    two zones connect through one shared aisle station on each side.
    """
    stations = [
        StationInfo("squat-1", "cam-1", "free_weights", "squat", adjacent_station_ids=["squat-2", "bench-1"]),
        StationInfo("squat-2", "cam-2", "free_weights", "squat", adjacent_station_ids=["squat-1", "deadlift-1"]),
        StationInfo("bench-1", "cam-3", "free_weights", "bench_press", adjacent_station_ids=["squat-1", "bench-2"]),
        StationInfo("bench-2", "cam-4", "free_weights", "bench_press", adjacent_station_ids=["bench-1", "deadlift-1"]),
        StationInfo("deadlift-1", "cam-5", "platform", "deadlift", adjacent_station_ids=["squat-2", "bench-2", "curl-1"]),
        StationInfo("curl-1", "cam-6", "dumbbell", "bicep_curl", adjacent_station_ids=["deadlift-1", "curl-2"]),
        StationInfo("curl-2", "cam-7", "dumbbell", "bicep_curl", adjacent_station_ids=["curl-1", "leg-press-1"]),
        StationInfo("leg-press-1", "cam-8", "machines", "leg_press", adjacent_station_ids=["curl-2", "leg-press-2"]),
        StationInfo("leg-press-2", "cam-9", "machines", "leg_press", adjacent_station_ids=["leg-press-1", "hack-squat-1"]),
        StationInfo("hack-squat-1", "cam-10", "machines", "hack_squat", adjacent_station_ids=["leg-press-2"]),
    ]
    return StationRegistry(stations)
