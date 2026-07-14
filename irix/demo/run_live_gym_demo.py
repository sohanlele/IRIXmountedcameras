"""End-to-end demo of the *live* pipeline: two simulated wristbands, a
BLE gateway with packet loss and a scripted disconnect, two stations, a
station handoff, and structured events delivered to a mock backend --
all without a camera, a real wristband, or any model weights.

Every other demo in this package (``run_demo.py``, ``run_gym_demo.py``,
``run_upload.py``) either drives ``RepCounter``/fusion code directly
against a synthetic stream, or replays an already-finished file.
``irix.live.station_runner.StationSessionRunner`` and ``irix.live.
gym_runner.GymSessionRunner`` -- the actual 24/7-station orchestration
this repo has real, tested code for (see docs/ARCHITECTURE.md's "Live
station readiness" section) -- had no demo exercising them end to end
before this one; each was only ever driven by unit tests with hand-built
fakes. This wires the real thing together:

- ``irix.wristband_sim.simulator.SimulatedBLEGateway`` -- two wristbands
  broadcasting BLE presence + IMU samples, standing in for real
  hardware exactly at the seam ``irix.fusion.imu_stream.LiveBLEIMUStream``
  and a station's ``ble_reader`` are documented to eventually fill.
- ``irix.demo.synthetic_live.SyntheticFrameSource`` /
  ``SyntheticPoseEstimator`` -- so the same tick loop a real 24/7 station
  runs (pull a frame, estimate a pose, feed the pipeline) runs here too,
  just without a camera.
- ``irix.identity.checkout.CheckoutRegistry`` -- both bands are checked
  out to real member ids up front, the front-desk step a live station
  requires before it will attribute anything to an account.
- ``irix.pipeline.edge_buffer.LocalBuffer`` / ``irix.pipeline.aggregator.
  Aggregator`` / ``irix.pipeline.cloud_sync.InMemoryCloudSync`` -- the
  mock backend every event this run produces gets delivered to.

Scripted over the run: Alice (band-101) starts a squat set at squat-1;
partway through, her band suffers a short BLE disconnect (shorter than
``presence_timeout_s``) and recovers with no session loss; Bob
(band-102) then walks up to squat-2 and does a set of his own; Bob
leaves; Alice then walks from squat-1 to squat-2, producing a real
``StationHandoffEvent`` through ``irix.topology.handoff.GymCoordinator``.

    python -m irix.demo.run_live_gym_demo
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Callable, Dict, List, Optional

from ..fusion.imu import IMUSample
from ..identity.checkout import CheckoutRegistry
from ..live.gym_runner import GymSessionRunner
from ..live.station_runner import StationSessionRunner
from ..pipeline.aggregator import Aggregator
from ..pipeline.cloud_sync import InMemoryCloudSync
from ..pipeline.edge_buffer import LocalBuffer
from ..pipeline.schema import CameraEvent
from ..rep_counting.exercises import EXERCISES
from ..topology.registry import StationInfo, StationRegistry
from ..wristband_sim.calibration import calibrate_stationary
from ..wristband_sim.simulator import SimulatedBLEGateway, SimulatedWristband
from .synthetic_live import SyntheticFrameSource, SyntheticPoseEstimator


class _ScriptedClock:
    """Advances by a fixed ``dt`` each call and remembers the current
    tick index/time, so a scripted BLE reader (below) can trigger
    movement/disconnect events at specific ticks in lockstep with
    whatever ``GymSessionRunner.run_forever`` does with the same clock."""

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        self.now = 0.0
        self.tick_index = -1

    def __call__(self) -> float:
        self.now += self.dt
        self.tick_index += 1
        return self.now


def _build_registry() -> StationRegistry:
    return StationRegistry(
        [
            StationInfo(
                station_id="squat-1", camera_id="cam-1", zone="free_weights",
                default_exercise="squat", adjacent_station_ids=["squat-2"],
            ),
            StationInfo(
                station_id="squat-2", camera_id="cam-2", zone="free_weights",
                default_exercise="squat", adjacent_station_ids=["squat-1"],
            ),
        ]
    )


def _make_scripted_ble_reader(gateway: SimulatedBLEGateway, clock: _ScriptedClock, script: Dict[int, Callable]):
    """Wraps ``gateway.ble_reader`` so that, each tick, it first runs
    whatever scripted action (wristband movement, a disconnect) is due at
    this tick and ticks the gateway's IMU generators forward -- then
    returns this tick's BLE readings. ``GymSessionRunner.run_forever``
    calls ``self._clock()`` once per tick immediately before calling this
    callable, so ``clock.tick_index``/``clock.now`` are already current."""

    def _ble_reader():
        action = script.get(clock.tick_index)
        if action is not None:
            action(gateway)
        gateway.tick(clock.now)
        return gateway.ble_reader()

    return _ble_reader


def run(n_ticks: int = 260, seed: int = 7, verbose: bool = True) -> List[CameraEvent]:
    registry = _build_registry()
    checkout = CheckoutRegistry()
    checkout.check_out("band-101", member_id="alice", timestamp=0.0)
    checkout.check_out("band-102", member_id="bob", timestamp=0.0)

    gateway = SimulatedBLEGateway(packet_loss_pct=0.05, seed=seed)
    band_alice = SimulatedWristband("band-101", seed=seed + 1)
    band_alice.set_motion("reps", reps_per_second=0.5)
    band_bob = SimulatedWristband("band-102", seed=seed + 2)
    band_bob.set_motion("reps", reps_per_second=0.6, phase=1.0)
    gateway.add_wristband(band_alice)
    gateway.add_wristband(band_bob)
    gateway.move_to_station("band-101", "squat-1")  # Alice already at squat-1 when the run starts

    events: List[CameraEvent] = []
    free_weights_buffer = LocalBuffer()
    aggregator = Aggregator(InMemoryCloudSync())
    aggregator.register_zone("free_weights", free_weights_buffer)

    def on_events(new_events: List[CameraEvent]) -> None:
        events.extend(new_events)
        for e in new_events:
            free_weights_buffer.push(e)

    clock = _ScriptedClock(dt=1.0 / 30.0)
    presence_timeout_s = 1.0  # 30 ticks at 30fps -- long enough to survive the scripted disconnect below

    station_runners = {
        "squat-1": StationSessionRunner(
            station_id="squat-1", exercise_name="squat", checkout_registry=checkout,
            frame_source=SyntheticFrameSource(), ble_reader=lambda: [],
            imu_stream_factory=gateway.imu_stream_factory,
            pose_estimator=SyntheticPoseEstimator(EXERCISES["squat"], reps_per_second=0.5),
            presence_timeout_s=presence_timeout_s, on_events=on_events, clock=clock,
        ),
        "squat-2": StationSessionRunner(
            station_id="squat-2", exercise_name="squat", checkout_registry=checkout,
            frame_source=SyntheticFrameSource(), ble_reader=lambda: [],
            imu_stream_factory=gateway.imu_stream_factory,
            pose_estimator=SyntheticPoseEstimator(EXERCISES["squat"], reps_per_second=0.5),
            presence_timeout_s=presence_timeout_s, on_events=on_events, clock=clock,
        ),
    }

    # Scripted timeline (tick index -> action taken on the gateway
    # immediately before that tick's BLE read):
    #   90:  Bob walks up to squat-2 and starts a set
    #   150: Alice's band suffers a 10-tick (~0.3s) BLE disconnect --
    #        shorter than presence_timeout_s, so her session should
    #        survive with just a data gap, not a close+reopen.
    #   180: Bob finishes and walks away (out of gateway range)
    #   210: Alice walks from squat-1 to squat-2 -- a real station handoff
    script: Dict[int, Callable] = {
        90: lambda gw: gw.move_to_station("band-102", "squat-2"),
        150: lambda gw: gw.disconnect("band-101", ticks=10),
        180: lambda gw: gw.move_to_station("band-102", None),
        210: lambda gw: gw.move_to_station("band-101", "squat-2"),
    }
    ble_reader = _make_scripted_ble_reader(gateway, clock, script)

    gym = GymSessionRunner(
        registry=registry, checkout_registry=checkout, station_runners=station_runners,
        ble_reader=ble_reader, presence_timeout_s=presence_timeout_s,
        on_gym_events=on_events, clock=clock,
    )
    gym.run_forever(max_frames=n_ticks)
    gym.close()

    aggregator.sync()

    if verbose:
        _print_summary(events, gateway)

    return events


def _print_summary(events: List[CameraEvent], gateway: SimulatedBLEGateway) -> None:
    counts = Counter(e.__class__.__name__ for e in events)
    print("Simulated live gym run -- event counts:")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
    print(
        f"BLE gateway: {gateway.dropped_ble_readings} presence readings dropped, "
        f"{gateway.dropped_imu_packets} IMU packets dropped (packet_loss_pct="
        f"{gateway.packet_loss_pct})"
    )

    handoffs = [e for e in events if e.__class__.__name__ == "StationHandoffEvent"]
    for h in handoffs:
        print(f"  handoff: {h.member_id} {h.from_station} -> {h.to_station} (plausible={h.plausible_adjacency})")

    # Demonstrate calibration against a short idle sample batch from a
    # freshly-created band with a known bias -- shows calibrate_stationary
    # actually recovering it, not just that it runs without error.
    calib_band = SimulatedWristband("calib-demo", seed=99)
    calib_band.set_motion("idle")
    stationary_samples: List[IMUSample] = calib_band.advance(dt=2.0)  # 2s @ 100Hz = 200 samples
    calibration = calibrate_stationary(stationary_samples)
    print(
        "Calibration check (calib-demo, 2s stationary): "
        f"recovered accel_bias={calibration.accel_bias.round(3).tolist()} "
        f"(true={list(calib_band.accel_bias.round(3))}), "
        f"gyro_bias={calibration.gyro_bias.round(3).tolist()} "
        f"(true={list(calib_band.gyro_bias.round(3))})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=260, help="Number of gym-wide ticks to run (30 ticks/s).")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(n_ticks=args.ticks, seed=args.seed)


if __name__ == "__main__":
    main()
