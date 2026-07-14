import pytest

from irix.barbell.rpe import RPETracker, EXERCISE_1RM_VELOCITY_MS


def test_velocity_loss_none_on_first_rep():
    tracker = RPETracker("squat")
    assert tracker.velocity_loss_pct(0.8) is None  # first rep just sets the baseline


def test_velocity_loss_pct_relative_to_first_rep():
    tracker = RPETracker("squat")
    tracker.velocity_loss_pct(0.8)  # baseline
    loss = tracker.velocity_loss_pct(0.6)
    assert loss == pytest.approx(25.0)  # (0.8-0.6)/0.8 * 100


def test_velocity_loss_pct_negative_when_speeding_up():
    tracker = RPETracker("squat")
    tracker.velocity_loss_pct(0.5)
    loss = tracker.velocity_loss_pct(0.6)
    assert loss < 0  # got faster than the first rep -- "negative loss"


def test_estimate_rpe_at_published_1rm_anchor_is_near_ten():
    tracker = RPETracker("squat")
    rpe = tracker.estimate_rpe(EXERCISE_1RM_VELOCITY_MS["squat"])
    assert rpe == pytest.approx(10.0, abs=0.01)


def test_estimate_rpe_faster_rep_gives_lower_rpe():
    tracker = RPETracker("bench_press")
    fast = tracker.estimate_rpe(0.5)
    slow = tracker.estimate_rpe(0.12)
    assert fast < slow


def test_estimate_rpe_clamped_to_valid_range():
    tracker = RPETracker("deadlift")
    # absurdly fast rep shouldn't extrapolate below 1
    assert tracker.estimate_rpe(10.0) >= 1.0
    # absurdly slow rep shouldn't extrapolate above 10
    assert tracker.estimate_rpe(0.001) <= 10.0


def test_estimate_rpe_none_for_exercise_without_published_anchor():
    tracker = RPETracker("leg_press")
    assert tracker.estimate_rpe(0.3) is None


def test_estimate_bundles_both_signals():
    tracker = RPETracker("squat")
    first = tracker.estimate(0.8)
    assert first.velocity_loss_pct is None
    assert first.estimated_rpe is not None
    second = tracker.estimate(0.6)
    assert second.velocity_loss_pct == pytest.approx(25.0)


def test_reset_clears_baseline_for_new_set():
    tracker = RPETracker("squat")
    tracker.velocity_loss_pct(0.8)
    tracker.reset()
    assert tracker.velocity_loss_pct(0.5) is None  # baseline forgotten, this is a new "first rep"
