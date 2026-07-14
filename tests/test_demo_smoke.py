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


def test_run_mock_with_barbell_tracking_populates_velocity_and_rpe():
    counter, cloud = run_mock(
        exercise_name="squat", member_id="m1", station_id="s1", n_frames=300, verbose=False,
        with_barbell_tracking=True,
    )
    rep_events = [e for e in cloud.received if isinstance(e, RepCompletedEvent)]
    assert len(rep_events) >= 3
    # First rep has nothing to compare against yet.
    assert rep_events[0].velocity_loss_pct is None
    assert rep_events[0].mean_velocity_m_s is not None
    assert rep_events[0].estimated_rpe is not None
    # Synthetic stream decays amplitude rep-over-rep -> increasing fatigue signal.
    losses = [e.velocity_loss_pct for e in rep_events[1:]]
    assert losses == sorted(losses)
    assert losses[-1] > 0


def test_run_mock_without_barbell_tracking_leaves_new_fields_none():
    counter, cloud = run_mock(
        exercise_name="squat", member_id="m1", station_id="s1", n_frames=300, verbose=False,
    )
    rep_events = [e for e in cloud.received if isinstance(e, RepCompletedEvent)]
    for event in rep_events:
        assert event.mean_velocity_m_s is None
        assert event.estimated_rpe is None
