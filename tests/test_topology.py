"""Tests for irix.topology -- station registry + BLE-based handoff with hysteresis."""
from irix.identity.ble_pairing import BLEReading
from irix.topology.handoff import GymCoordinator, MemberStationTracker
from irix.topology.registry import StationInfo, StationRegistry, build_default_ten_station_gym
from irix.identity.ble_pairing import StationPairing


def test_default_ten_station_gym_has_ten_stations():
    registry = build_default_ten_station_gym()
    assert len(registry) == 10
    for station in registry.all():
        assert station.default_exercise is not None


def test_registry_adjacency_lookup():
    registry = build_default_ten_station_gym()
    assert registry.is_adjacent("squat-1", "squat-2")
    assert not registry.is_adjacent("squat-1", "hack-squat-1")


def test_registry_adjacency_unknown_station_is_not_adjacent():
    registry = StationRegistry([StationInfo("a", "cam-a", "zone", adjacent_station_ids=["b"])])
    assert not registry.is_adjacent("unknown", "a")


def test_first_assignment_is_not_a_handoff():
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=3)
    event = tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    assert event is None
    assert tracker.current_station == "squat-1"


def test_single_noisy_reading_does_not_trigger_handoff():
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=3)
    tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    event = tracker.update(
        [BLEReading("squat-1", -55.0, 1.0), BLEReading("squat-2", -53.0, 1.0)], timestamp=1.0,
    )
    assert event is None
    assert tracker.current_station == "squat-1"


def test_sustained_signal_triggers_handoff_after_min_consecutive():
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=3)
    tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    events = []
    for t in (1.0, 2.0, 3.0):
        events.append(tracker.update([BLEReading("squat-2", -45.0, t)], timestamp=t))
    assert events[0] is None
    assert events[1] is None
    assert events[2] is not None
    assert events[2].from_station == "squat-1"
    assert events[2].to_station == "squat-2"
    assert tracker.current_station == "squat-2"


def test_streak_resets_if_candidate_changes_mid_streak():
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=3)
    tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    tracker.update([BLEReading("squat-2", -45.0, 1.0)], timestamp=1.0)  # streak=1 toward squat-2
    tracker.update([BLEReading("curl-1", -45.0, 2.0)], timestamp=2.0)  # candidate changes -> streak resets to 1
    event = tracker.update([BLEReading("curl-1", -45.0, 3.0)], timestamp=3.0)  # streak=2, still < 3
    assert event is None
    assert tracker.current_station == "squat-1"


def test_handoff_event_flags_implausible_adjacency():
    registry = build_default_ten_station_gym()
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=2)
    tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    event = None
    for t in (1.0, 2.0):
        result = tracker.update([BLEReading("hack-squat-1", -45.0, t)], timestamp=t, registry=registry)
        if result is not None:
            event = result
    assert event is not None
    assert event.plausible_adjacency is False


def test_handoff_event_plausible_when_adjacent():
    registry = build_default_ten_station_gym()
    tracker = MemberStationTracker("m1", StationPairing(), min_consecutive=2)
    tracker.update([BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    event = None
    for t in (1.0, 2.0):
        result = tracker.update([BLEReading("squat-2", -45.0, t)], timestamp=t, registry=registry)
        if result is not None:
            event = result
    assert event is not None
    assert event.plausible_adjacency is True


def test_gym_coordinator_authoritative_gating():
    registry = build_default_ten_station_gym()
    coord = GymCoordinator(registry, min_consecutive=2)
    coord.update_member("m1", [BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    assert coord.is_authoritative("m1", "squat-1") is True
    assert coord.is_authoritative("m1", "squat-2") is False
    # Unknown member: not authoritative anywhere.
    assert coord.is_authoritative("m2", "squat-1") is False


def test_gym_coordinator_active_members_at_station():
    registry = build_default_ten_station_gym()
    coord = GymCoordinator(registry, min_consecutive=1)
    coord.update_member("m1", [BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    coord.update_member("m2", [BLEReading("squat-1", -48.0, 0.0)], timestamp=0.0)
    coord.update_member("m3", [BLEReading("curl-1", -48.0, 0.0)], timestamp=0.0)
    active = coord.active_members_at("squat-1")
    assert set(active) == {"m1", "m2"}


def test_gym_coordinator_tracks_multiple_members_independently():
    registry = build_default_ten_station_gym()
    coord = GymCoordinator(registry, min_consecutive=2)
    coord.update_member("m1", [BLEReading("squat-1", -50.0, 0.0)], timestamp=0.0)
    coord.update_member("m2", [BLEReading("curl-1", -50.0, 0.0)], timestamp=0.0)
    assert coord.current_station("m1") == "squat-1"
    assert coord.current_station("m2") == "curl-1"
