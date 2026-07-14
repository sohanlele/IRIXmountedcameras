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
