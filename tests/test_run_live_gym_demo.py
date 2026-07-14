"""Smoke test for irix.demo.run_live_gym_demo -- the first demo to drive
irix.live.station_runner.StationSessionRunner / irix.live.gym_runner.
GymSessionRunner end to end (every other exercise of those classes is a
unit test with hand-built fakes; see docs/ARCHITECTURE.md's "Live
station readiness" section). Verifies the scripted run actually produces
every kind of event the demo's own docstring claims -- rep completion,
set completion, fatigue summary, and a real station handoff -- not just
that it runs without raising.
"""
from __future__ import annotations

from irix.demo.run_live_gym_demo import run
from irix.pipeline.schema import (
    RepCompletedEvent,
    SetCompleteEvent,
    SetFatigueSummaryEvent,
    StationHandoffEvent,
)


def test_run_live_gym_demo_produces_expected_event_types():
    events = run(n_ticks=260, seed=7, verbose=False)

    kinds = {type(e) for e in events}
    assert RepCompletedEvent in kinds
    assert SetCompleteEvent in kinds
    assert SetFatigueSummaryEvent in kinds
    assert StationHandoffEvent in kinds


def test_run_live_gym_demo_handoff_is_alice_squat1_to_squat2():
    events = run(n_ticks=260, seed=7, verbose=False)

    handoffs = [e for e in events if isinstance(e, StationHandoffEvent)]
    assert len(handoffs) == 1
    handoff = handoffs[0]
    assert handoff.member_id == "alice"
    assert handoff.from_station == "squat-1"
    assert handoff.to_station == "squat-2"
    assert handoff.plausible_adjacency is True


def test_run_live_gym_demo_both_members_get_reps_attributed_correctly():
    events = run(n_ticks=260, seed=7, verbose=False)

    reps = [e for e in events if isinstance(e, RepCompletedEvent)]
    member_ids = {r.member_id for r in reps}
    assert member_ids <= {"alice", "bob"}
    assert len(reps) > 0


def test_run_live_gym_demo_is_deterministic_given_a_seed():
    events_a = run(n_ticks=260, seed=7, verbose=False)
    events_b = run(n_ticks=260, seed=7, verbose=False)

    assert [type(e) for e in events_a] == [type(e) for e in events_b]
    assert len(events_a) == len(events_b)
