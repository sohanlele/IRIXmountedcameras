"""Sanity tests for irix.benchmark -- structural correctness (right
shape, JSON-serializable, sane value ranges), not performance assertions
(actual timing numbers are environment-dependent and shouldn't gate CI).
"""
from __future__ import annotations

import json

import pytest

from irix.benchmark.run_benchmarks import (
    benchmark_ble_disconnect_recovery,
    benchmark_camera_reconnect_schedule,
    benchmark_clock_sync,
    benchmark_ekf,
    benchmark_event_latency,
    benchmark_exercise_recognition,
    benchmark_identity_resolution_latency,
    benchmark_live_gym_pipeline,
    benchmark_packet_loss_impact,
    benchmark_pose_tracker,
    benchmark_rep_fusion,
    format_report,
    resource_usage,
    run_all,
)


def test_timing_benchmarks_return_positive_sane_values():
    for fn in (
        benchmark_pose_tracker, benchmark_exercise_recognition, benchmark_rep_fusion, benchmark_ekf,
        benchmark_clock_sync, benchmark_identity_resolution_latency, benchmark_event_latency,
    ):
        result = fn()
        assert result.mean_ms > 0
        assert result.p95_ms >= result.p50_ms >= 0
        assert result.fps > 0
        assert result.n_iterations > 0


def test_live_pipeline_benchmark_reports_positive_throughput():
    result = benchmark_live_gym_pipeline(n_ticks=60)
    assert result["n_ticks"] == 60
    assert result["elapsed_s"] > 0
    assert result["ticks_per_second"] > 0
    assert result["n_events_produced"] >= 0


def test_camera_reconnect_schedule_matches_exponential_backoff():
    result = benchmark_camera_reconnect_schedule(backoff_s=2.0, max_backoff_s=30.0, n_failures=6)
    assert result["backoff_schedule_s"] == [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert result["recovered_successfully"] is True
    assert result["n_reconnect_attempts_before_success"] == 6


def test_ble_disconnect_recovery_reports_correct_outage_vs_timeout():
    result = benchmark_ble_disconnect_recovery()
    assert result["outage_s"] > 0
    assert result["session_survives"] is True
    assert result["recovery_margin_s"] == result["presence_timeout_s"] - result["outage_s"]


def test_resource_usage_reports_positive_memory():
    usage = resource_usage()
    assert usage["peak_rss_mb"] > 0
    assert usage["user_cpu_s"] >= 0


def test_run_all_report_is_json_serializable_and_formattable():
    report = run_all()
    serialized = json.dumps(report)  # raises if anything isn't JSON-safe (e.g. a stray numpy type)
    assert len(serialized) > 0

    text = format_report(report)
    assert "IRIX benchmark report" in text
    assert "Full simulated live pipeline" in text


def test_identity_resolution_latency_reports_n_candidates_and_scales_with_them():
    small = benchmark_identity_resolution_latency(n_candidates=2)
    large = benchmark_identity_resolution_latency(n_candidates=4)
    assert small.mean_ms > 0 and large.mean_ms > 0
    assert "2 candidates" in small.name
    assert "4 candidates" in large.name


def test_event_latency_benchmark_reports_a_sane_per_frame_batch_time():
    result = benchmark_event_latency()
    assert result.mean_ms > 0
    assert result.n_iterations > 0
    assert "process_frame" in result.name


def test_packet_loss_impact_shows_completeness_degrading_with_loss_rate():
    result = benchmark_packet_loss_impact(loss_levels=(0.0, 0.2, 0.5))
    completeness_by_loss = {r["packet_loss_pct"]: r["imu_sample_completeness"] for r in result["results"]}
    assert completeness_by_loss[0.0] == pytest.approx(1.0, abs=0.02)
    # Completeness should be monotonically non-increasing as loss rises.
    assert completeness_by_loss[0.0] >= completeness_by_loss[0.2] >= completeness_by_loss[0.5]
    assert all(r["fused_rep_count"] is not None for r in result["results"])
