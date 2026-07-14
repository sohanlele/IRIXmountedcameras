from __future__ import annotations

import numpy as np
import pytest

from irix.wristband_sim.simulator import (
    SimulatedBLEGateway,
    SimulatedBLEIMUStream,
    SimulatedWristband,
)


def test_wristband_idle_advance_produces_gravity_plus_bias():
    band = SimulatedWristband("band-1", sample_rate_hz=100.0, accel_noise_std=0.0, gyro_noise_std=0.0, seed=0)
    band.set_motion("idle")

    samples = band.advance(dt=1.0)

    assert len(samples) == 100
    for s in samples:
        np.testing.assert_allclose(s.accel, np.array([0.0, 0.0, 9.80665]) + band.accel_bias)
        np.testing.assert_allclose(s.gyro, band.gyro_bias)


def test_wristband_reps_oscillates_vertical_accel():
    band = SimulatedWristband("band-1", sample_rate_hz=100.0, accel_noise_std=0.0, gyro_noise_std=0.0, seed=0)
    band.set_motion("reps", reps_per_second=0.5, amplitude=6.0)

    samples = band.advance(dt=4.0)  # two full rep cycles
    z = np.array([s.accel[2] for s in samples])

    # oscillates well above and below plain gravity+bias
    assert z.max() > 9.80665 + band.accel_bias[2] + 3.0
    assert z.min() < 9.80665 + band.accel_bias[2] - 3.0


def test_wristband_advance_is_monotonic_across_calls():
    band = SimulatedWristband("band-1", sample_rate_hz=50.0, seed=0)
    band.set_motion("idle")
    first = band.advance(dt=0.5)
    second = band.advance(dt=0.5)
    assert first[-1].timestamp < second[0].timestamp


def test_gateway_ble_reader_reports_only_bands_at_a_station():
    gateway = SimulatedBLEGateway(packet_loss_pct=0.0, seed=0)
    band = SimulatedWristband("band-1", seed=0)
    gateway.add_wristband(band)

    gateway.tick(now=0.0)
    assert gateway.ble_reader() == []

    gateway.move_to_station("band-1", "squat-1")
    gateway.tick(now=0.1)
    readings = gateway.ble_reader()

    assert len(readings) == 1
    assert readings[0].station_id == "squat-1"
    assert readings[0].wristband_id == "band-1"
    assert readings[0].rssi < 0  # dBm


def test_gateway_imu_stream_drains_samples_generated_since_last_tick():
    gateway = SimulatedBLEGateway(packet_loss_pct=0.0, seed=0)
    band = SimulatedWristband("band-1", sample_rate_hz=100.0, seed=0)
    band.set_motion("idle")
    gateway.add_wristband(band)
    gateway.move_to_station("band-1", "squat-1")

    stream = gateway.imu_stream_factory("band-1")
    assert isinstance(stream, SimulatedBLEIMUStream)

    assert stream.poll() == []  # nothing generated before the first tick

    gateway.tick(now=0.0)
    assert stream.poll() == []  # dt=0 on the very first tick -- no elapsed time yet

    gateway.tick(now=1.0)
    samples = stream.poll()
    assert len(samples) == 100

    assert stream.poll() == []  # already drained


def test_gateway_disconnect_drops_ble_and_imu_for_scheduled_ticks():
    gateway = SimulatedBLEGateway(packet_loss_pct=0.0, seed=0)
    band = SimulatedWristband("band-1", sample_rate_hz=100.0, seed=0)
    gateway.add_wristband(band)
    gateway.move_to_station("band-1", "squat-1")
    stream = gateway.imu_stream_factory("band-1")

    gateway.tick(now=0.0)
    stream.poll()

    gateway.disconnect("band-1", ticks=2)

    gateway.tick(now=1.0)
    assert gateway.ble_reader() == []
    assert stream.poll() == []

    gateway.tick(now=2.0)
    assert gateway.ble_reader() == []
    assert stream.poll() == []

    # disconnect window over -- back to normal
    gateway.tick(now=3.0)
    assert len(gateway.ble_reader()) == 1
    assert len(stream.poll()) == 100


def test_gateway_packet_loss_drops_some_readings_over_many_ticks():
    gateway = SimulatedBLEGateway(packet_loss_pct=0.5, seed=1)
    band = SimulatedWristband("band-1", seed=0)
    gateway.add_wristband(band)
    gateway.move_to_station("band-1", "squat-1")

    seen = 0
    for i in range(200):
        gateway.tick(now=i * 0.1)
        seen += len(gateway.ble_reader())

    assert 0 < seen < 200
    assert gateway.dropped_ble_readings > 0


def test_gateway_rejects_invalid_packet_loss_pct():
    with pytest.raises(ValueError):
        SimulatedBLEGateway(packet_loss_pct=1.5)


def test_gateway_move_to_station_none_removes_from_readings():
    gateway = SimulatedBLEGateway(packet_loss_pct=0.0, seed=0)
    band = SimulatedWristband("band-1", seed=0)
    gateway.add_wristband(band)
    gateway.move_to_station("band-1", "squat-1")
    gateway.tick(now=0.0)
    assert len(gateway.ble_reader()) == 1

    gateway.move_to_station("band-1", None)
    gateway.tick(now=0.1)
    assert gateway.ble_reader() == []
