from irix.demo.run_demo import run_mock
from irix.pipeline.schema import RepCompletedEvent, SetCompleteEvent


def test_run_mock_end_to_end():
    counter, cloud = run_mock(
        exercise_name="squat", member_id="m1", station_id="s1", n_frames=300, verbose=False
    )
    assert counter.rep_count > 0
    rep_events = [e for e in cloud.received if isinstance(e, RepCompletedEvent)]
    set_events = [e for e in cloud.received if isinstance(e, SetCompleteEvent)]
    assert len(rep_events) == counter.rep_count
    assert len(set_events) == 1
    assert set_events[0].total_reps == counter.rep_count


def test_run_mock_rep_events_carry_velocity_and_duration():
    counter, cloud = run_mock(
        exercise_name="squat", member_id="m1", station_id="s1", n_frames=300, verbose=False
    )
    rep_events = [e for e in cloud.received if isinstance(e, RepCompletedEvent)]
    for event in rep_events:
        assert event.duration_s is not None and event.duration_s > 0
        assert event.peak_velocity_deg_s is not None
        assert event.mean_velocity_deg_s is not None
        assert event.peak_velocity_deg_s >= event.mean_velocity_deg_s


def test_run_mock_emits_band_placement_event_for_ankle_exercise():
    counter, cloud = run_mock(
        exercise_name="leg_press", member_id="m1", station_id="s1", n_frames=300, verbose=False
    )
    from irix.pipeline.schema import BandPlacementRequiredEvent

    placement_events = [e for e in cloud.received if isinstance(e, BandPlacementRequiredEvent)]
    assert len(placement_events) == 1
    assert placement_events[0].to_placement == "ankle"


def test_run_mock_wrist_exercise_emits_no_band_placement_event():
    counter, cloud = run_mock(
        exercise_name="bicep_curl", member_id="m1", station_id="s1", n_frames=300, verbose=False
    )
    from irix.pipeline.schema import BandPlacementRequiredEvent

    placement_events = [e for e in cloud.received if isinstance(e, BandPlacementRequiredEvent)]
    assert len(placement_events) == 0
