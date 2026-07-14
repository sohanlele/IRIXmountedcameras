"""Integration test: irix.demo.run_upload against the real pretrained
pose model (not mocked) -- proves the whole upload-mode wiring actually
runs the real PoseEstimator end to end, the way
tests/test_pose_estimator_integration.py does for run_live. The detailed
business-logic assertions (set boundaries, IMU fusion, weight
recognition, barbell velocity) live in tests/test_run_upload_wiring.py
with a scripted pose stand-in instead, since they need precise control
over the joint-angle sequence that a real image doesn't give.

Skipped automatically if ultralytics/torch aren't installed (the 'pose'
extra).
"""
import pytest

ultralytics = pytest.importorskip("ultralytics", reason="requires 'pip install irix[pose]'")


def _zidane_image_path() -> str:
    from ultralytics.utils import ASSETS

    return str(ASSETS / "zidane.jpg")


def _write_test_video(path: str, n_frames: int = 15, fps: float = 10.0) -> None:
    import cv2

    frame = cv2.imread(_zidane_image_path())
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def test_run_upload_against_real_video_no_crash(tmp_path):
    from irix.demo.run_upload import run_upload

    video_path = str(tmp_path / "video.mp4")
    _write_test_video(video_path)

    events = run_upload(video_path, "squat", "test-member", "test-station", max_frames=15)
    # A static image repeated has no motion, so no reps/sets are expected
    # -- what this proves is that the real model runs frame-by-frame
    # through the whole upload pipeline (pose, rep counter, set-boundary
    # detector, band-placement tracker) without raising.
    assert isinstance(events, list)
    for e in events:
        assert e.to_dict()["event_type"] in {
            "rep_completed", "set_complete", "band_placement_required",
            "weight_confirmed", "set_fatigue_summary",
        }


def test_run_upload_with_imu_file_against_real_video_no_crash(tmp_path):
    """Same as above, but also exercises the real IMU-file-loading path
    (irix.fusion.imu_io) alongside the real pose model -- the two real
    inputs (video + IMU file) this whole entrypoint exists for."""
    from irix.demo.mock_pose import synthetic_imu_stream
    from irix.demo.run_upload import run_upload

    video_path = str(tmp_path / "video.mp4")
    _write_test_video(video_path)

    imu_path = tmp_path / "imu.csv"
    samples = synthetic_imu_stream(n_seconds=2.0, fs=50.0, reps_per_second=1.0)
    lines = ["timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"]
    for s in samples:
        lines.append(
            f"{s.timestamp},{s.accel[0]},{s.accel[1]},{s.accel[2]},{s.gyro[0]},{s.gyro[1]},{s.gyro[2]}"
        )
    imu_path.write_text("\n".join(lines))

    events = run_upload(
        video_path, "squat", "test-member", "test-station", imu_path=str(imu_path), max_frames=15,
    )
    assert isinstance(events, list)
