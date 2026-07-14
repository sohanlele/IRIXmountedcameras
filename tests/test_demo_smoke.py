from irix.demo.run_demo import run_mock


def test_run_mock_end_to_end():
    counter, cloud = run_mock(
        exercise_name="squat", member_id="m1", station_id="s1", n_frames=300, verbose=False
    )
    assert counter.rep_count > 0
    assert len(cloud.received) == counter.rep_count
