"""Smoke tests for irix.demo.run_gym_demo -- the multi-station (10-camera)
end-to-end scenario tying together topology handoff, rep fusion, fatigue
analysis, form scoring, and the weight-recognition geometry cross-check.
"""
from irix.demo.run_gym_demo import _demo_motion_correlation_disambiguation, main
from irix.pipeline.schema import (
    BandPlacementRequiredEvent,
    RepCompletedEvent,
    SetCompleteEvent,
    SetFatigueSummaryEvent,
    WeightConfirmedEvent,
)


def _rep_completed_events(events):
    return [e for e in events if isinstance(e, RepCompletedEvent)]


def test_run_gym_demo_end_to_end_no_errors():
    clouds = main()
    assert set(clouds.keys()) == {"squat-1", "curl-1", "leg-press-1"}
    for cloud in clouds.values():
        assert len(cloud.received) > 0


def test_squat_zone_has_two_sets_worth_of_events_and_fatigue_summaries():
    clouds = main()
    events = clouds["squat-1"].received
    set_events = [e for e in events if isinstance(e, SetCompleteEvent)]
    fatigue_events = [e for e in events if isinstance(e, SetFatigueSummaryEvent)]
    weight_events = [e for e in events if isinstance(e, WeightConfirmedEvent)]
    assert len(set_events) == 2
    assert len(fatigue_events) == 2
    assert len(weight_events) == 2

    # Fusion fields populated on both sets (a synthetic IMU stream is
    # always generated in _run_one_set).
    for e in set_events:
        assert e.fused_rep_count is not None
        assert e.rep_count_source is not None

    # Session fatigue tracker accumulates across the two sets.
    assert fatigue_events[0].completed_sets_this_session == 1
    assert fatigue_events[1].completed_sets_this_session == 2

    # Barbell tracking is wired in for squat (has a published velocity
    # anchor) -- fatigue analysis should run on the calibrated m/s tier,
    # not fall back to the joint-angular deg/s proxy.
    for e in fatigue_events:
        assert e.velocity_tier == "m_s"
    # The second set's synthetic bar-velocity stream decays more per rep
    # than the first (see run_gym_demo.main's velocity_decay_per_rep),
    # so it should show a real, higher velocity loss.
    assert fatigue_events[1].velocity_loss_pct > fatigue_events[0].velocity_loss_pct
    assert fatigue_events[1].velocity_loss_zone in ("VL10", "VL20", "VL30", "VL45")

    for e in _rep_completed_events(events):
        assert e.mean_velocity_m_s is not None
        assert e.estimated_rpe is not None

    # Weight geometry cross-check: one plausible, one flagged.
    consistent_flags = [e.geometry_consistent for e in weight_events]
    assert True in consistent_flags
    assert False in consistent_flags


def test_curl_zone_form_fault_injected_and_detected():
    clouds = main()
    events = clouds["curl-1"].received
    rep_events = [e for e in events if isinstance(e, RepCompletedEvent)]
    assert len(rep_events) >= 3
    # inject_form_fault="leaning_back" was used for this set -- every rep
    # after the first (which starts mid-motion, see irix.demo.mock_pose)
    # should show it caught.
    for e in rep_events[1:]:
        assert "leaning_back" in e.form_faults
        assert e.form_score is not None and e.form_score < 1.0


def test_leg_press_zone_shows_band_placement_and_imu_fallback_on_occlusion():
    clouds = main()
    events = clouds["leg-press-1"].received
    band_events = [e for e in events if isinstance(e, BandPlacementRequiredEvent)]
    set_events = [e for e in events if isinstance(e, SetCompleteEvent)]
    assert len(band_events) == 1
    assert band_events[0].to_placement == "ankle"

    assert len(set_events) == 1
    set_event = set_events[0]
    # Occlusion was injected for this set -- camera undercounts and its
    # tracking confidence is low, so fusion should have leaned on the IMU.
    assert set_event.rep_count_agreement is False
    assert set_event.rep_count_source == "imu_preferred_on_disagreement"
    assert set_event.fused_rep_count == set_event.imu_rep_count


def test_motion_correlation_disambiguation_resolves_correctly():
    # _demo_motion_correlation_disambiguation itself asserts correctness
    # internally (verbose=True path) -- call it directly here so a
    # regression trips a test failure with a real traceback, not just a
    # silently-wrong printout the next time someone runs the demo by hand.
    _demo_motion_correlation_disambiguation(verbose=True)
