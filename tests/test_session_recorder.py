"""irix.recording.session_recorder -- Priority 8's session-recording
tooling (camera timing, IMU, clock sync, events, metadata) and its
matching loader."""
from __future__ import annotations

import os

import numpy as np
import pytest

from irix.fusion.clock_sync import ClockSyncEstimate
from irix.fusion.imu import IMUSample
from irix.pipeline.schema import RepCompletedEvent
from irix.recording.session_recorder import SessionRecorder, load_recorded_session


def _sample(t):
    return IMUSample(timestamp=t, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3))


def test_records_frames_events_imu_and_metadata_without_raw_pixels_by_default(tmp_path):
    out_dir = str(tmp_path / "session-1")
    recorder = SessionRecorder(
        output_dir=out_dir, station_id="squat-1", exercise_name="squat", member_id="alice",
    )
    recorder.record_frame(np.zeros((4, 4, 3), dtype=np.uint8), ts=0.0)
    recorder.record_frame(np.ones((4, 4, 3), dtype=np.uint8), ts=0.1)
    recorder.record_imu_samples([_sample(0.0), _sample(0.05)])
    recorder.record_events([
        RepCompletedEvent(member_id="alice", station_id="squat-1", exercise="squat", rep_count=1, timestamp=0.1)
    ])
    written = recorder.close()

    assert os.path.exists(written["metadata"])
    assert os.path.exists(written["frames"])
    assert os.path.exists(written["events"])
    assert os.path.exists(written["imu"])
    # Default is opt-out of raw pixel storage -- no frames/ directory at all.
    assert not os.path.isdir(os.path.join(out_dir, "frames"))

    session = load_recorded_session(out_dir)
    assert session.metadata["station_id"] == "squat-1"
    assert session.metadata["member_id"] == "alice"
    assert session.metadata["n_frames"] == 2
    assert len(session.frames) == 2
    assert session.frames[0].timestamp == 0.0
    assert len(session.imu_samples) == 2
    assert len(session.events) == 1
    assert session.events[0]["event_type"] == "rep_completed"
    assert session.events[0]["rep_count"] == 1


def test_save_raw_frames_opt_in_writes_actual_pixel_data(tmp_path):
    out_dir = str(tmp_path / "session-2")
    recorder = SessionRecorder(
        output_dir=out_dir, station_id="squat-1", exercise_name="squat", save_raw_frames=True,
    )
    frame = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    recorder.record_frame(frame, ts=0.0)
    recorder.close()

    frame_path = os.path.join(out_dir, "frames", "frame_0.npy")
    assert os.path.exists(frame_path)
    loaded = np.load(frame_path)
    assert np.array_equal(loaded, frame)


def test_a_session_with_no_imu_omits_the_imu_file_entirely(tmp_path):
    out_dir = str(tmp_path / "session-3")
    recorder = SessionRecorder(output_dir=out_dir, station_id="squat-1", exercise_name="squat")
    recorder.record_frame(np.zeros((2, 2, 3), dtype=np.uint8), ts=0.0)
    written = recorder.close()

    assert "imu" not in written
    assert not os.path.exists(os.path.join(out_dir, "imu.json"))
    session = load_recorded_session(out_dir)
    assert session.imu_samples == []


def test_clock_sync_snapshot_is_recorded_when_provided(tmp_path):
    out_dir = str(tmp_path / "session-4")
    recorder = SessionRecorder(output_dir=out_dir, station_id="squat-1", exercise_name="squat")
    recorder.record_clock_sync_snapshot(
        ClockSyncEstimate(offset_s=-0.35, drift_ppm=12.0, confidence=0.9, n_observations=3)
    )
    recorder.close()

    session = load_recorded_session(out_dir)
    assert session.metadata["clock_sync"]["offset_s"] == -0.35
    assert session.metadata["clock_sync"]["n_observations"] == 3


def test_loading_a_directory_with_no_metadata_json_raises():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            load_recorded_session(d)


def test_extra_metadata_is_merged_into_the_written_file(tmp_path):
    out_dir = str(tmp_path / "session-5")
    recorder = SessionRecorder(
        output_dir=out_dir, station_id="squat-1", exercise_name="squat",
        extra_metadata={"gym_id": "gym-42", "camera_id": "cam-1"},
    )
    recorder.close()
    session = load_recorded_session(out_dir)
    assert session.metadata["gym_id"] == "gym-42"
    assert session.metadata["camera_id"] == "cam-1"
