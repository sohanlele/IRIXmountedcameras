from irix.rep_counting.exercises import SQUAT
from irix.rep_counting.state_machine import RepCounter
from irix.demo.mock_pose import synthetic_angle_stream


def test_counts_reps_from_synthetic_stream():
    counter = RepCounter(SQUAT)
    events = []
    for t, angle in synthetic_angle_stream(SQUAT, n_frames=300, fps=30.0, reps_per_second=0.5):
        event = counter.update(angle, timestamp=t)
        if event:
            events.append(event)
    # 300 frames @ 30fps = 10s, at 0.5 reps/sec -> ~5 reps
    assert 4 <= counter.rep_count <= 6
    assert len(events) == counter.rep_count
    assert events[0].rep_number == 1


def test_ignores_small_noise_near_threshold():
    counter = RepCounter(SQUAT)
    # Hover near the top without ever reaching bottom -- should count 0 reps.
    for angle in [165, 168, 166, 169, 167, 170, 168]:
        counter.update(angle)
    assert counter.rep_count == 0


def test_nan_angle_is_ignored():
    counter = RepCounter(SQUAT)
    assert counter.update(float("nan")) is None
    assert counter.rep_count == 0


def test_rep_events_carry_sane_velocity_and_duration():
    counter = RepCounter(SQUAT)
    events = []
    for t, angle in synthetic_angle_stream(SQUAT, n_frames=300, fps=30.0, reps_per_second=0.5):
        event = counter.update(angle, timestamp=t)
        if event:
            events.append(event)
    assert len(events) >= 4
    for event in events:
        assert event.peak_angular_velocity_deg_s is not None
        assert event.mean_angular_velocity_deg_s is not None
        # Peak speed within a rep is always >= the mean speed over that rep.
        assert event.peak_angular_velocity_deg_s >= event.mean_angular_velocity_deg_s > 0
        assert event.duration_s > 0


def test_first_rep_duration_is_not_wall_clock_garbage():
    # Regression test: duration_s used to be computed against
    # time.monotonic() captured at RepCounter construction, which produced
    # a huge/garbage value for the first rep whenever the caller's
    # timestamp convention didn't happen to match wall-clock monotonic
    # time (true of every synthetic/test/mock-demo timestamp stream).
    counter = RepCounter(SQUAT)
    events = []
    for t, angle in synthetic_angle_stream(SQUAT, n_frames=300, fps=30.0, reps_per_second=0.5):
        event = counter.update(angle, timestamp=t)
        if event:
            events.append(event)
    assert events[0].duration_s < 10  # first rep lands well under 10s in, not ~time.monotonic()


def test_faster_exercise_shows_higher_velocity_than_slower_one():
    from irix.rep_counting.exercises import BICEP_CURL

    squat_events = []
    counter = RepCounter(SQUAT)
    for t, angle in synthetic_angle_stream(SQUAT, n_frames=300, fps=30.0, reps_per_second=0.5):
        event = counter.update(angle, timestamp=t)
        if event:
            squat_events.append(event)

    fast_curl_events = []
    counter = RepCounter(BICEP_CURL)
    for t, angle in synthetic_angle_stream(BICEP_CURL, n_frames=300, fps=30.0, reps_per_second=1.5):
        event = counter.update(angle, timestamp=t)
        if event:
            fast_curl_events.append(event)

    assert fast_curl_events, "expected at least one completed rep at the faster tempo"
    # A 1.5 reps/sec tempo should read out a shorter inter-rep duration
    # than a 0.5 reps/sec tempo -- a sanity check that duration_s actually
    # reflects tempo rather than being some constant.
    assert fast_curl_events[0].duration_s < squat_events[0].duration_s
