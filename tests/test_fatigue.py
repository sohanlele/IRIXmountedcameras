"""Tests for irix.fatigue -- set/session-level fatigue analysis."""
import pytest

from irix.fatigue.models import RepFatigueSample
from irix.fatigue.session_analysis import SessionFatigueTracker
from irix.fatigue.set_analysis import SetFatigueAnalyzer


def test_analyze_empty_reps_returns_none():
    assert SetFatigueAnalyzer().analyze("squat", []) is None


def test_prefers_m_s_tier_when_available():
    reps = [
        RepFatigueSample(1, mean_velocity_m_s=0.55, mean_velocity_deg_s=100.0),
        RepFatigueSample(2, mean_velocity_m_s=0.50, mean_velocity_deg_s=95.0),
    ]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.velocity_tier == "m_s"
    assert analysis.first_rep_velocity == 0.55


def test_falls_back_to_deg_s_tier_without_m_s():
    reps = [
        RepFatigueSample(1, mean_velocity_deg_s=100.0),
        RepFatigueSample(2, mean_velocity_deg_s=90.0),
    ]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.velocity_tier == "deg_s"
    assert analysis.first_rep_velocity == 100.0


def test_no_velocity_data_at_all_yields_none_tier():
    reps = [RepFatigueSample(1, duration_s=2.0), RepFatigueSample(2, duration_s=2.5)]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.velocity_tier == "none"
    assert analysis.velocity_loss_pct is None
    assert analysis.velocity_loss_zone is None
    # Tempo drift still computable from duration alone.
    assert analysis.tempo_drift_pct == 25.0


def test_velocity_loss_trend_and_zone_classification():
    # First rep 1.0 m/s, last rep 0.65 m/s -> 35% loss -> VL30 zone.
    reps = [
        RepFatigueSample(1, mean_velocity_m_s=1.0),
        RepFatigueSample(2, mean_velocity_m_s=0.85),
        RepFatigueSample(3, mean_velocity_m_s=0.65),
    ]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.velocity_loss_trend_pct == pytest.approx([0.0, 15.0, 35.0])
    assert analysis.velocity_loss_pct == 35.0
    assert analysis.velocity_loss_zone == "VL30"


def test_velocity_loss_below_vl10_has_no_zone():
    reps = [RepFatigueSample(1, mean_velocity_m_s=1.0), RepFatigueSample(2, mean_velocity_m_s=0.95)]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.velocity_loss_zone is None


def test_reps_are_sorted_by_rep_number_regardless_of_input_order():
    reps = [
        RepFatigueSample(3, mean_velocity_m_s=0.5),
        RepFatigueSample(1, mean_velocity_m_s=1.0),
        RepFatigueSample(2, mean_velocity_m_s=0.8),
    ]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.first_rep_velocity == 1.0
    assert analysis.last_rep_velocity == 0.5


def test_mean_form_score_and_most_common_fault():
    reps = [
        RepFatigueSample(1, mean_velocity_m_s=1.0, form_score=1.0, form_faults=[]),
        RepFatigueSample(2, mean_velocity_m_s=0.9, form_score=0.7, form_faults=["knee_valgus"]),
        RepFatigueSample(3, mean_velocity_m_s=0.8, form_score=0.6, form_faults=["knee_valgus", "insufficient_depth"]),
    ]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert abs(analysis.mean_form_score - (1.0 + 0.7 + 0.6) / 3) < 1e-9
    assert analysis.most_common_fault == "knee_valgus"


def test_no_form_data_leaves_mean_form_score_none():
    reps = [RepFatigueSample(1, mean_velocity_m_s=1.0), RepFatigueSample(2, mean_velocity_m_s=0.9)]
    analysis = SetFatigueAnalyzer().analyze("squat", reps)
    assert analysis.mean_form_score is None
    assert analysis.most_common_fault is None


def test_session_tracker_first_set_has_no_across_set_trend():
    analyzer = SetFatigueAnalyzer()
    reps = [RepFatigueSample(1, mean_velocity_m_s=1.0), RepFatigueSample(2, mean_velocity_m_s=0.9)]
    analysis = analyzer.analyze("squat", reps)
    tracker = SessionFatigueTracker()
    summary = tracker.add_set("m1", "squat", analysis)
    assert summary.completed_sets == 1
    assert summary.set_to_set_velocity_trend_pct == [0.0]


def test_session_tracker_detects_across_set_decline():
    analyzer = SetFatigueAnalyzer()
    tracker = SessionFatigueTracker()

    set1 = analyzer.analyze("squat", [
        RepFatigueSample(1, mean_velocity_m_s=1.0), RepFatigueSample(2, mean_velocity_m_s=0.95),
    ])
    tracker.add_set("m1", "squat", set1)

    # Second set opens noticeably slower than the first set opened.
    set2 = analyzer.analyze("squat", [
        RepFatigueSample(1, mean_velocity_m_s=0.8), RepFatigueSample(2, mean_velocity_m_s=0.75),
    ])
    summary = tracker.add_set("m1", "squat", set2)

    assert summary.completed_sets == 2
    assert summary.set_to_set_velocity_trend_pct[1] == pytest.approx(20.0)  # (1.0-0.8)/1.0*100
    assert summary.session_fatigue_index is not None
    assert summary.session_fatigue_index > 0


def test_session_tracker_keeps_member_exercise_histories_independent():
    analyzer = SetFatigueAnalyzer()
    tracker = SessionFatigueTracker()
    squat_set = analyzer.analyze("squat", [RepFatigueSample(1, mean_velocity_m_s=1.0)])
    curl_set = analyzer.analyze("bicep_curl", [RepFatigueSample(1, mean_velocity_m_s=0.5)])

    tracker.add_set("m1", "squat", squat_set)
    summary = tracker.add_set("m1", "bicep_curl", curl_set)
    assert summary.completed_sets == 1  # bicep_curl history for m1 is separate from squat's


def test_session_tracker_reset_clears_history():
    analyzer = SetFatigueAnalyzer()
    tracker = SessionFatigueTracker()
    s = analyzer.analyze("squat", [RepFatigueSample(1, mean_velocity_m_s=1.0)])
    tracker.add_set("m1", "squat", s)
    tracker.reset("m1", "squat")
    summary = tracker.add_set("m1", "squat", s)
    assert summary.completed_sets == 1


def test_rep_fatigue_sample_from_rep_completed_event():
    from irix.pipeline.schema import RepCompletedEvent

    event = RepCompletedEvent(
        member_id="m1", station_id="s1", exercise="squat", rep_count=3,
        duration_s=2.0, mean_velocity_m_s=0.5, mean_velocity_deg_s=110.0,
        form_score=0.9, form_faults=["knee_valgus"],
    )
    sample = RepFatigueSample.from_rep_completed_event(event)
    assert sample.rep_number == 3
    assert sample.mean_velocity_m_s == 0.5
    assert sample.form_faults == ["knee_valgus"]
