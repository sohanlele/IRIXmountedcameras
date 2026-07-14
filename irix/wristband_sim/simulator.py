"""Simulated wristband + BLE gateway (see package docstring for why this
exists and where it stops).

Three pieces:

- ``SimulatedWristband`` -- one physical band, generating a continuous
  IMU sample stream at a fixed rate, following a settable "motion
  program" (idle / oscillating reps), with a constant, known-in-advance
  bias baked in so ``irix.wristband_sim.calibration`` has something real
  to recover in tests/demos.
- ``SimulatedBLEGateway`` -- owns N wristbands, each optionally "at" a
  station (or ``None``, out of gateway range -- e.g. still at the front
  desk). Ticked once per gym-wide loop iteration, it produces
  ``BLEReading``s (``irix.identity.ble_pairing``) with distance-appropriate
  RSSI + noise and a configurable per-reading packet-loss probability,
  and buffers each present band's IMU samples for ``SimulatedBLEIMUStream``
  to drain -- the exact shapes ``irix.live.gym_runner.GymSessionRunner``'s
  ``ble_reader`` and ``irix.live.station_runner.StationSessionRunner``'s
  ``imu_stream_factory`` constructor args expect.
- ``SimulatedBLEIMUStream`` -- the ``irix.fusion.imu_stream.IMUStream``
  implementation backed by a gateway's per-wristband buffer.

``disconnect()`` schedules a scripted total dropout (no BLE reading, no
IMU samples) for a band, for exercising FINAL GOAL's "recover from BLE
disconnects" against the real live pipeline -- see
``irix.demo.run_live_gym_demo`` for a full scripted run including one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..fusion.imu import IMUSample
from ..identity.ble_pairing import BLEReading

DEFAULT_RSSI_AT_STATION_DBM = -55.0
DEFAULT_RSSI_NOISE_STD = 2.0
DEFAULT_ACCEL_BIAS = (0.05, -0.03, 0.08)  # m/s^2, arbitrary but fixed -- calibration.py's target
DEFAULT_GYRO_BIAS = (0.01, -0.005, 0.02)  # rad/s


class SimulatedWristband:
    """One simulated physical wristband.

    ``station_id`` is the "ground truth" of where the member wearing this
    band physically is right now (``None`` = out of every station's BLE
    range) -- set directly by a demo/test script to simulate someone
    walking, the same role a real gateway would infer purely from RSSI in
    production. The simulator is allowed to know ground truth; the
    pipeline it feeds is not -- ``SimulatedBLEGateway.ble_reader()`` only
    ever exposes RSSI + noise, never ``station_id`` directly, so the
    resolution logic being exercised (``irix.identity.ble_pairing``,
    ``irix.topology.handoff.GymCoordinator``) is the same code a real
    deployment runs.
    """

    def __init__(
        self,
        wristband_id: str,
        sample_rate_hz: float = 100.0,
        accel_noise_std: float = 0.15,
        gyro_noise_std: float = 0.02,
        accel_bias=DEFAULT_ACCEL_BIAS,
        gyro_bias=DEFAULT_GYRO_BIAS,
        clock_drift_ppm: float = 0.0,
        seed: int = 0,
    ):
        self.wristband_id = wristband_id
        self.sample_rate_hz = sample_rate_hz
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        self.accel_bias = np.asarray(accel_bias, dtype=float)
        self.gyro_bias = np.asarray(gyro_bias, dtype=float)
        # This band's onboard crystal runs at (1 + clock_drift_ppm/1e6) times
        # the gateway's "true" tick rate -- the same free-running-oscillator
        # behavior a real wristband has (see irix.fusion.clock_sync's module
        # docstring for the BLE spec drift numbers this should be set from:
        # up to +/-20 ppm main clock, up to +/-250 ppm sleep clock). Both the
        # *number* of samples generated per real-world dt and each sample's
        # own timestamp advance at this drifted rate, so a wristband's
        # reported timestamps genuinely diverge from true elapsed time over
        # a session -- exactly what irix.fusion.clock_sync exists to detect
        # and correct for.
        self.clock_drift_ppm = clock_drift_ppm
        self.station_id: Optional[str] = None

        self._rng = np.random.default_rng(seed)
        self._t = 0.0
        self._motion = "idle"
        self._reps_per_second = 0.5
        self._rep_amplitude = 6.0
        self._phase = 0.0

    def set_motion(
        self,
        motion: str,
        reps_per_second: float = 0.5,
        amplitude: float = 6.0,
        phase: float = 0.0,
    ) -> None:
        """``"idle"``: gravity + bias + noise only, no oscillation
        (what a band on someone's wrist between sets, or sitting in a
        charging cradle, looks like -- and what ``calibration.
        calibrate_stationary`` should be run against). ``"reps"``:
        vertical accel oscillating on top of gravity, same generative
        model ``irix.demo.mock_pose.synthetic_imu_stream`` uses (minus
        the deliberate bias this class also injects), so a consumer
        downstream sees consistent statistics regardless of whether
        samples came from a batch file, ``mock_pose``, or this
        simulator."""
        if motion not in ("idle", "reps"):
            raise ValueError(f"motion must be 'idle' or 'reps', got {motion!r}")
        self._motion = motion
        self._reps_per_second = reps_per_second
        self._rep_amplitude = amplitude
        self._phase = phase

    def advance(self, dt: float) -> List[IMUSample]:
        """Generate however many samples fall in the next ``dt`` seconds
        at this band's sample rate. Called by ``SimulatedBLEGateway.tick``
        once per gateway tick (``dt`` = wall-clock time since the last
        tick) rather than each band free-running its own clock, so every
        band in a gateway stays in lockstep with one tick loop."""
        # The wristband's own onboard clock advances by dt*(1+drift) while
        # true (gateway/reference) time advances by dt -- so, at this
        # band's *nominal* sample_rate_hz (a fixed property of its own
        # clock), it produces slightly more or fewer samples per true-time
        # tick than an undrifted band would, and each sample's own
        # timestamp is spaced at 1/sample_rate_hz in the band's own
        # (drifted) clock -- not the gateway's.
        internal_dt = dt * (1.0 + self.clock_drift_ppm / 1e6)
        n = max(0, int(round(internal_dt * self.sample_rate_hz)))
        samples = []
        for _ in range(n):
            self._t += 1.0 / self.sample_rate_hz
            gravity = np.array([0.0, 0.0, GRAVITY_M_S2])
            if self._motion == "reps":
                vertical = self._rep_amplitude * math.sin(
                    2 * math.pi * self._reps_per_second * self._t + self._phase
                )
            else:
                vertical = 0.0
            true_accel = gravity + np.array([0.0, 0.0, vertical])
            accel = true_accel + self.accel_bias + self._rng.normal(0, self.accel_noise_std, 3)
            gyro = self.gyro_bias + self._rng.normal(0, self.gyro_noise_std, 3)
            samples.append(IMUSample(timestamp=self._t, accel=accel, gyro=gyro))
        return samples


GRAVITY_M_S2 = 9.80665


class SimulatedBLEIMUStream:
    """``irix.fusion.imu_stream.IMUStream`` implementation backed by a
    ``SimulatedBLEGateway``'s per-wristband buffer -- the software-only
    stand-in for ``LiveBLEIMUStream`` (see that class's docstring for why
    it stays unimplemented) that lets ``RepSession``/
    ``StationSessionRunner`` be driven by something that behaves like a
    live BLE connection (samples genuinely arrive incrementally across
    ``poll()`` calls, can go quiet during a simulated disconnect) without
    needing real hardware."""

    def __init__(self, gateway: "SimulatedBLEGateway", wristband_id: str):
        self._gateway = gateway
        self._wristband_id = wristband_id

    def poll(self) -> List[IMUSample]:
        return self._gateway._drain(self._wristband_id)


class SimulatedBLEGateway:
    """Simulates a gym's BLE receiver infrastructure for N wristbands.

    ``packet_loss_pct`` (0-1) is applied independently to each band's BLE
    advertisement *and* each band's IMU packet on every tick -- a real
    radio genuinely drops some fraction of both due to interference/
    timing, and the two are transmitted separately (advertisement vs. a
    notify characteristic), so a lost IMU packet doesn't imply a lost
    presence reading in the same tick or vice versa.
    """

    def __init__(
        self,
        packet_loss_pct: float = 0.0,
        rssi_at_station_dbm: float = DEFAULT_RSSI_AT_STATION_DBM,
        rssi_noise_std: float = DEFAULT_RSSI_NOISE_STD,
        seed: int = 0,
    ):
        if not 0.0 <= packet_loss_pct <= 1.0:
            raise ValueError(f"packet_loss_pct must be in [0, 1], got {packet_loss_pct}")
        self.packet_loss_pct = packet_loss_pct
        self.rssi_at_station_dbm = rssi_at_station_dbm
        self.rssi_noise_std = rssi_noise_std
        self._rng = np.random.default_rng(seed)

        self._wristbands: Dict[str, SimulatedWristband] = {}
        self._buffers: Dict[str, List[IMUSample]] = {}
        self._disconnect_ticks_remaining: Dict[str, int] = {}
        self._last_tick_time: Optional[float] = None

        # Counters a demo/test can inspect to confirm loss/disconnect
        # actually happened (rather than trusting the probability alone).
        self.dropped_ble_readings = 0
        self.dropped_imu_packets = 0

        # Wristbands that were mid-disconnect *this* tick -- computed once
        # in tick() and reused by ble_reader() so a band scheduled for
        # exactly N disconnected ticks is actually skipped by both BLE
        # readings and IMU generation for all N ticks (checking the
        # countdown dict directly in ble_reader() after tick() has already
        # decremented it would under-count the last disconnected tick).
        self._disconnected_this_tick: set = set()

    def add_wristband(self, wristband: SimulatedWristband) -> None:
        self._wristbands[wristband.wristband_id] = wristband
        self._buffers[wristband.wristband_id] = []

    def move_to_station(self, wristband_id: str, station_id: Optional[str]) -> None:
        """Simulate the member wearing this band physically walking to
        ``station_id`` (or leaving every station's range, if ``None``)."""
        self._wristbands[wristband_id].station_id = station_id

    def disconnect(self, wristband_id: str, ticks: int) -> None:
        """Schedule the next ``ticks`` gateway ticks to drop this band
        entirely -- no BLE reading, no IMU samples -- simulating a radio
        dropout rather than the member leaving (``move_to_station`` is
        the latter). A short disconnect (fewer ticks than a
        ``StationSessionRunner``'s ``presence_timeout_s`` grace period)
        should be survived by the session with zero data loss beyond the
        gap itself; a longer one legitimately closes it, same as real
        hardware."""
        self._disconnect_ticks_remaining[wristband_id] = ticks

    def tick(self, now: float) -> None:
        """Advance every wristband's IMU generator and refill this
        tick's BLE/IMU buffers. Call once per gym-wide loop iteration,
        before ``ble_reader()``/any ``IMUStream.poll()`` for this tick."""
        dt = 0.0 if self._last_tick_time is None else max(0.0, now - self._last_tick_time)
        self._last_tick_time = now
        self._disconnected_this_tick = set()
        for wristband_id, band in self._wristbands.items():
            remaining = self._disconnect_ticks_remaining.get(wristband_id, 0)
            if remaining > 0:
                self._disconnect_ticks_remaining[wristband_id] = remaining - 1
                self._disconnected_this_tick.add(wristband_id)
                continue  # total dropout this tick -- no samples generated or buffered
            samples = band.advance(dt)
            if samples and self._rng.random() < self.packet_loss_pct:
                self.dropped_imu_packets += 1
                continue
            self._buffers.setdefault(wristband_id, []).extend(samples)

    def ble_reader(self) -> List[BLEReading]:
        """The callable ``GymSessionRunner``/``StationSessionRunner``'s
        ``ble_reader`` constructor arg expects: every currently-visible
        reading, one per wristband that's both at a station and not
        mid-disconnect this tick, with RSSI + Gaussian noise standing in
        for real signal strength."""
        readings = []
        now = self._last_tick_time or 0.0
        for wristband_id, band in self._wristbands.items():
            if band.station_id is None:
                continue
            if wristband_id in self._disconnected_this_tick:
                continue
            if self._rng.random() < self.packet_loss_pct:
                self.dropped_ble_readings += 1
                continue
            rssi = self.rssi_at_station_dbm + self._rng.normal(0, self.rssi_noise_std)
            readings.append(
                BLEReading(
                    station_id=band.station_id,
                    rssi=rssi,
                    timestamp=now,
                    recent_motion=True,
                    wristband_id=wristband_id,
                )
            )
        return readings

    def imu_stream_factory(self, wristband_id: str) -> SimulatedBLEIMUStream:
        return SimulatedBLEIMUStream(self, wristband_id)

    def _drain(self, wristband_id: str) -> List[IMUSample]:
        samples = self._buffers.get(wristband_id, [])
        self._buffers[wristband_id] = []
        return samples
