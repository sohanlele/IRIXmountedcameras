"""Unit tests for irix.live.disambiguation.CrowdedGroupDisambiguator in
isolation -- irix/live/test_station_runner.py already covers it wired
into a real StationSessionRunner end to end; these tests exercise the
buffering/reset/sticky-routing state machine directly, independent of
any camera/session plumbing, since irix.live.zone_runner will drive
several instances of this class at once and needs each one's behavior to
be well understood on its own.
"""
from __future__ import annotations

from irix.demo.mock_pose import synthetic_imu_stream, synthetic_wrist_motion_pose_stream
from irix.live.disambiguation import CrowdedGroupDisambiguator


def _member_resolver(mapping):
    return lambda wristband_id: mapping.get(wristband_id)


def test_single_candidate_group_resolves_and_routes_stickily():
    resolver = _member_resolver({"band-a": "member-a", "band-b": "member-b"})
    disambiguator = CrowdedGroupDisambiguator(disambiguation_window_frames=180)
    group = frozenset({"band-a", "band-b"})

    poses_a = synthetic_wrist_motion_pose_stream(n_frames=200, reps_per_second=0.6, phase=0.0, seed=1)
    poses_b = synthetic_wrist_motion_pose_stream(n_frames=200, reps_per_second=0.3, phase=0.9, seed=2)
    imu_a = synthetic_imu_stream(n_seconds=200 / 30.0, reps_per_second=0.6, phase=0.0, seed=3)
    imu_b = synthetic_imu_stream(n_seconds=200 / 30.0, reps_per_second=0.3, phase=0.9, seed=4)
    samples_per_tick = max(len(imu_a) // 200, 1)

    routed_ticks = []
    for i in range(200):
        now = i / 30.0
        people = [poses_a[i], poses_b[i]]
        polled = {
            "band-a": imu_a[i * samples_per_tick:(i + 1) * samples_per_tick],
            "band-b": imu_b[i * samples_per_tick:(i + 1) * samples_per_tick],
        }
        routed = disambiguator.route(now, group, people, polled, resolver)
        routed_ticks.append(routed)

    # Nothing routed during the 180-frame buffering window.
    assert all(r == {} for r in routed_ticks[:180])
    # Resolved and routed for the remaining frames.
    resolved_ticks = routed_ticks[180:]
    assert any(r for r in resolved_ticks)
    for r in resolved_ticks:
        if r:
            assert set(r.keys()) <= {"band-a", "band-b"}
    assert disambiguator.slot_assignment == {0: "band-a", 1: "band-b"}


def test_group_change_resets_and_starts_a_fresh_window():
    resolver = _member_resolver({"band-a": "member-a", "band-b": "member-b", "band-c": "member-c"})
    disambiguator = CrowdedGroupDisambiguator(disambiguation_window_frames=5)
    group_ab = frozenset({"band-a", "band-b"})

    for i in range(5):
        disambiguator.route(float(i), group_ab, [], {}, resolver)
    # A fresh group (different membership) must reset, not carry over
    # whatever partial buffer/resolution existed for the old group.
    group_bc = frozenset({"band-b", "band-c"})
    routed = disambiguator.route(5.0, group_bc, [], {}, resolver)
    assert routed == {}
    assert disambiguator.slot_assignment == {}


def test_reset_clears_all_state():
    resolver = _member_resolver({"band-a": "member-a"})
    disambiguator = CrowdedGroupDisambiguator(disambiguation_window_frames=2)
    group = frozenset({"band-a"})
    disambiguator.route(0.0, group, [], {}, resolver)
    disambiguator.reset()
    assert disambiguator.slot_assignment == {}
    assert disambiguator._pending_wristband_ids is None
    # After reset, the very next route() call starts fresh buffering
    # again for the same group rather than reusing stale state.
    routed = disambiguator.route(1.0, group, [], {}, resolver)
    assert routed == {}


def test_unresolvable_member_id_is_skipped_not_crashed_on():
    # resolve_member_id returning None for a candidate (e.g. checked-out
    # band whose account lookup somehow fails) shouldn't blow up
    # resolution -- that candidate just never gets a slot.
    resolver = _member_resolver({"band-a": "member-a"})  # band-b resolves to None
    disambiguator = CrowdedGroupDisambiguator(disambiguation_window_frames=3)
    group = frozenset({"band-a", "band-b"})
    for i in range(3):
        disambiguator.route(float(i), group, [], {}, resolver)
    # Resolution ran without raising; band-b (unresolvable) never gets a slot.
    assert "band-b" not in disambiguator.slot_assignment.values()
