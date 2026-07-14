import pytest

from irix.fusion.imu import IMUSample
from irix.fusion.imu_stream import LiveBLEIMUStream, RecordedIMUStream


def _sample(t):
    return IMUSample(timestamp=t, accel=[0.0, 0.0, -9.8], gyro=[0.0, 0.0, 0.0])


def test_recorded_stream_returns_everything_on_first_poll():
    samples = [_sample(0.0), _sample(0.1), _sample(0.2)]
    stream = RecordedIMUStream(samples)
    polled = stream.poll()
    assert [s.timestamp for s in polled] == [0.0, 0.1, 0.2]


def test_recorded_stream_returns_nothing_new_on_subsequent_polls():
    stream = RecordedIMUStream([_sample(0.0)])
    stream.poll()
    assert stream.poll() == []
    assert stream.poll() == []


def test_recorded_stream_empty_input_polls_empty():
    stream = RecordedIMUStream([])
    assert stream.poll() == []


def test_live_ble_imu_stream_is_a_documented_stub():
    stream = LiveBLEIMUStream(wristband_id="band-1")
    with pytest.raises(NotImplementedError, match="LiveBLEIMUStream is a sketch"):
        stream.poll()
