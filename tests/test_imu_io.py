import json

import pytest

from irix.fusion.imu import IMUSample
from irix.fusion.imu_io import load_imu_samples, save_imu_samples


def _write_csv(path, rows, header="timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z"):
    lines = [header] + [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines))


def test_loads_valid_csv_sorted_by_timestamp(tmp_path):
    path = tmp_path / "imu.csv"
    _write_csv(path, [
        (1.0, 0.1, 0.2, -9.8, 0.01, 0.02, 0.03),
        (0.0, 0.0, 0.0, -9.8, 0.0, 0.0, 0.0),
        (2.0, -0.1, 0.1, -9.9, 0.02, 0.01, 0.0),
    ])
    samples = load_imu_samples(str(path))
    assert [s.timestamp for s in samples] == [0.0, 1.0, 2.0]
    assert samples[1].accel.tolist() == [0.1, 0.2, -9.8]
    assert samples[1].gyro.tolist() == [0.01, 0.02, 0.03]


def test_loads_valid_json(tmp_path):
    path = tmp_path / "imu.json"
    payload = [
        {"timestamp": 0.0, "accel_x": 0.0, "accel_y": 0.0, "accel_z": -9.8,
         "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0},
        {"timestamp": 0.5, "accel_x": 1.0, "accel_y": 0.0, "accel_z": -9.7,
         "gyro_x": 0.1, "gyro_y": 0.0, "gyro_z": 0.0},
    ]
    path.write_text(json.dumps(payload))
    samples = load_imu_samples(str(path))
    assert len(samples) == 2
    assert samples[0].timestamp == 0.0
    assert samples[1].accel.tolist() == [1.0, 0.0, -9.7]


def test_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_imu_samples(str(tmp_path / "does_not_exist.csv"))


def test_unrecognized_extension_raises_value_error(tmp_path):
    path = tmp_path / "imu.txt"
    path.write_text("garbage")
    with pytest.raises(ValueError, match="unrecognized IMU file extension"):
        load_imu_samples(str(path))


def test_csv_missing_column_raises_value_error(tmp_path):
    path = tmp_path / "imu.csv"
    _write_csv(path, [(0.0, 0.0, 0.0, -9.8, 0.0, 0.0, 0.0)], header="timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y")
    with pytest.raises(ValueError, match="missing column"):
        load_imu_samples(str(path))


def test_csv_non_numeric_value_raises_value_error_with_row_number(tmp_path):
    path = tmp_path / "imu.csv"
    _write_csv(path, [
        (0.0, 0.0, 0.0, -9.8, 0.0, 0.0, 0.0),
        (1.0, "not-a-number", 0.0, -9.8, 0.0, 0.0, 0.0),
    ])
    with pytest.raises(ValueError, match="row 3"):
        load_imu_samples(str(path))


def test_json_missing_field_raises_value_error(tmp_path):
    path = tmp_path / "imu.json"
    path.write_text(json.dumps([{"timestamp": 0.0, "accel_x": 0.0}]))
    with pytest.raises(ValueError, match="missing field"):
        load_imu_samples(str(path))


def test_empty_file_raises_value_error(tmp_path):
    path = tmp_path / "imu.csv"
    _write_csv(path, [])
    with pytest.raises(ValueError, match="no IMU samples"):
        load_imu_samples(str(path))


def test_save_then_load_round_trips_through_csv(tmp_path):
    import numpy as np

    samples = [
        IMUSample(timestamp=0.0, accel=np.array([0.1, 0.2, -9.8]), gyro=np.array([0.01, 0.02, 0.03])),
        IMUSample(timestamp=0.5, accel=np.array([0.0, 0.0, -9.81]), gyro=np.array([0.0, 0.0, 0.0])),
    ]
    path = str(tmp_path / "roundtrip.csv")
    save_imu_samples(samples, path)
    reloaded = load_imu_samples(path)

    assert [s.timestamp for s in reloaded] == [0.0, 0.5]
    assert reloaded[0].accel.tolist() == pytest.approx([0.1, 0.2, -9.8])
    assert reloaded[0].gyro.tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_save_then_load_round_trips_through_json(tmp_path):
    import numpy as np

    samples = [IMUSample(timestamp=1.0, accel=np.array([1.0, 2.0, 3.0]), gyro=np.array([4.0, 5.0, 6.0]))]
    path = str(tmp_path / "roundtrip.json")
    save_imu_samples(samples, path)
    reloaded = load_imu_samples(path)

    assert reloaded[0].timestamp == 1.0
    assert reloaded[0].accel.tolist() == [1.0, 2.0, 3.0]


def test_save_with_unrecognized_extension_raises():
    import numpy as np

    samples = [IMUSample(timestamp=0.0, accel=np.zeros(3), gyro=np.zeros(3))]
    with pytest.raises(ValueError):
        save_imu_samples(samples, "/tmp/whatever.txt")
