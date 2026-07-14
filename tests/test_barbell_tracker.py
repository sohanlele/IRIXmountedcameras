import pytest

from irix.barbell.calibration import calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from irix.barbell.tracker import BarPathTracker


def _calibration():
    # 180px = 450mm plate -> 0.4 px/mm -> 2.5mm/px -> 0.0025 m/px
    return calibrate_from_known_object(
        pixel_size=180.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id="s1"
    )


def test_constant_velocity_ascent_measured_correctly():
    cal = _calibration()
    tracker = BarPathTracker(cal)
    fps = 30
    v_true = 0.3  # m/s upward
    y0_px = 1000.0
    px_per_m = 1.0 / cal.pixels_to_m(1)
    for i in range(31):
        t = i / fps
        y_px = y0_px - v_true * t * px_per_m  # decreasing y = moving up
        tracker.push(t, y_px)

    result = tracker.velocity_for_window(0.0, 1.0)
    assert result.mean_velocity_m_s == pytest.approx(v_true, abs=1e-6)
    assert result.peak_velocity_m_s == pytest.approx(v_true, abs=1e-6)
    assert result.displacement_m == pytest.approx(v_true * 1.0, abs=1e-6)


def test_stationary_bar_gives_near_zero_velocity():
    cal = _calibration()
    tracker = BarPathTracker(cal)
    for i in range(10):
        tracker.push(i / 30.0, 1000.0)  # no movement
    result = tracker.velocity_for_window(0.0, 10 / 30.0)
    assert result.mean_velocity_m_s == pytest.approx(0.0, abs=1e-9)


def test_too_few_samples_returns_none_fields():
    cal = _calibration()
    tracker = BarPathTracker(cal)
    tracker.push(0.0, 1000.0)
    result = tracker.velocity_for_window(0.0, 1.0)
    assert result.mean_velocity_m_s is None
    assert result.peak_velocity_m_s is None
    assert result.displacement_m is None


def test_buffer_prunes_old_samples_beyond_max_buffer_s():
    cal = _calibration()
    tracker = BarPathTracker(cal, max_buffer_s=1.0)
    tracker.push(0.0, 1000.0)
    tracker.push(5.0, 1000.0)  # far beyond max_buffer_s after this push
    assert len(tracker._samples) == 1  # the t=0.0 sample should have been pruned


def test_reset_clears_buffer():
    cal = _calibration()
    tracker = BarPathTracker(cal)
    tracker.push(0.0, 1000.0)
    tracker.push(0.1, 990.0)
    tracker.reset()
    assert tracker._samples == []
