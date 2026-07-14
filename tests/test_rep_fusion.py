"""Tests for irix.fusion.rep_fusion.RepCountFusion."""
from irix.demo.mock_pose import synthetic_imu_stream
from irix.fusion.rep_fusion import RepCountFusion


def test_fuse_without_imu_samples_falls_back_to_camera_only():
    fusion = RepCountFusion()
    result = fusion.fuse(camera_count=8, camera_confidence=0.95, imu_samples=None)
    assert result.fused_count == 8
    assert result.source == "camera_only"
    assert result.imu_count is None
    assert result.agreement is True


def test_fuse_agreement_prefers_camera_count():
    fusion = RepCountFusion()
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=1)
    result = fusion.fuse(
        camera_count=8, camera_confidence=0.98, imu_samples=samples,
        camera_rep_durations=[2.0] * 8,
    )
    assert result.imu_count is not None
    assert result.agreement is True
    assert result.fused_count == 8
    assert result.source == "camera_imu_agreement"


def test_fuse_disagreement_prefers_higher_confidence_source():
    fusion = RepCountFusion()
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=1)
    # Camera badly undercounted (e.g. heavy occlusion) and has low
    # confidence -- fusion should lean on the IMU-derived count instead.
    result = fusion.fuse(
        camera_count=2, camera_confidence=0.15, imu_samples=samples,
        camera_rep_durations=[2.0, 2.0],
    )
    assert result.agreement is False
    assert result.source == "imu_preferred_on_disagreement"
    assert result.fused_count == result.imu_count


def test_fuse_disagreement_camera_confident_keeps_camera_count():
    fusion = RepCountFusion()
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=1)
    # Camera is highly confident (no occlusion) even though the IMU
    # algorithm's own confidence for this contrived call is lower --
    # fusion should trust the camera.
    result = fusion.fuse(
        camera_count=100, camera_confidence=0.99, imu_samples=samples,
        camera_rep_durations=[2.0] * 8,
    )
    assert result.agreement is False
    assert result.source == "camera_preferred_on_disagreement"
    assert result.fused_count == 100


def test_fuse_unusable_imu_signal_falls_back_to_camera_only():
    fusion = RepCountFusion()
    # Way too short/flat a signal for either IMU algorithm to find a period.
    samples = synthetic_imu_stream(n_seconds=0.5, reps_per_second=0.5, seed=1)
    result = fusion.fuse(camera_count=5, camera_confidence=0.9, imu_samples=samples)
    assert result.source == "camera_only"
    assert result.fused_count == 5
    assert result.imu_count is None


def test_period_bounds_derived_from_camera_durations_narrower_than_default():
    fusion = RepCountFusion(default_min_period=1.0, default_max_period=4.0)
    tight_bounds = fusion._period_bounds([2.0, 2.1, 1.9, 2.0])
    default_bounds = fusion._period_bounds([])
    assert tight_bounds != default_bounds
    assert tight_bounds[0] < 2.0 < tight_bounds[1]


def test_imu_peak_timestamps_are_populated_on_agreement():
    fusion = RepCountFusion()
    samples = synthetic_imu_stream(n_seconds=16.0, reps_per_second=0.5, seed=1)
    result = fusion.fuse(
        camera_count=8, camera_confidence=0.98, imu_samples=samples,
        camera_rep_durations=[2.0] * 8,
    )
    assert len(result.imu_peak_timestamps) == result.imu_count
    assert all(isinstance(t, float) for t in result.imu_peak_timestamps)


def test_sample_completeness_full_rate_stream_is_near_one():
    fusion = RepCountFusion(imu_sample_rate_hz=100.0)
    samples = synthetic_imu_stream(n_seconds=16.0, fs=100.0, reps_per_second=0.5, seed=1)
    completeness = fusion._sample_completeness(samples)
    assert completeness > 0.95


def test_sample_completeness_reflects_known_decimation_ratio():
    fusion = RepCountFusion(imu_sample_rate_hz=100.0)
    full = synthetic_imu_stream(n_seconds=16.0, fs=100.0, reps_per_second=0.5, seed=1)
    decimated = full[::5]  # keep 1 in 5 -- ~80% packet loss

    completeness = fusion._sample_completeness(decimated)

    assert 0.15 < completeness < 0.30


def test_sample_completeness_handles_single_sample_without_crashing():
    fusion = RepCountFusion()
    assert fusion._sample_completeness([synthetic_imu_stream(n_seconds=1.0, fs=100.0)[0]]) == 1.0


def test_heavy_packet_loss_discounts_effective_imu_confidence():
    fusion = RepCountFusion(imu_sample_rate_hz=100.0, completeness_floor=0.7)
    full = synthetic_imu_stream(n_seconds=16.0, fs=100.0, reps_per_second=0.5, seed=1)
    decimated = full[::8]  # ~87.5% packet loss -- well below completeness_floor

    full_result = fusion.fuse(camera_count=8, camera_confidence=0.5, imu_samples=full, camera_rep_durations=[2.0] * 8)
    decimated_result = fusion.fuse(
        camera_count=8, camera_confidence=0.5, imu_samples=decimated, camera_rep_durations=[2.0] * 8,
    )

    assert decimated_result.imu_sample_completeness < fusion.completeness_floor
    assert full_result.imu_sample_completeness > fusion.completeness_floor
    # The decimated stream's reported confidence should never exceed what
    # completeness alone allows through.
    assert decimated_result.imu_confidence <= decimated_result.imu_sample_completeness / fusion.completeness_floor + 1e-9


def test_ordinary_packet_loss_above_floor_is_not_discounted():
    """A light, realistic packet-loss rate (e.g. ~10%) shouldn't discount
    the fusion decision at all -- only meaningfully degraded streams
    below completeness_floor should."""
    fusion = RepCountFusion(imu_sample_rate_hz=100.0, completeness_floor=0.7)
    full = synthetic_imu_stream(n_seconds=16.0, fs=100.0, reps_per_second=0.5, seed=1)
    lightly_dropped = [s for i, s in enumerate(full) if i % 10 != 0]  # drop every 10th sample (~10% loss)

    result = fusion.fuse(camera_count=8, camera_confidence=0.5, imu_samples=lightly_dropped, camera_rep_durations=[2.0] * 8)

    assert result.imu_sample_completeness > fusion.completeness_floor
