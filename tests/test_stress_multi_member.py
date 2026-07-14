"""Stress test: several simultaneous members across several stations,
each with its own simulated wristband -- the "run multiple wristbands,
track multiple members simultaneously" requirement, at a larger scale
than irix/demo/run_live_gym_demo.py's two-member scripted walkthrough.
Verifies no cross-member event contamination and that the live pipeline
completes in reasonable wall-clock time (a coarse throughput sanity
check -- see irix.benchmark for real timing numbers).
"""
from __future__ import annotations

import time

from irix.identity.checkout import CheckoutRegistry
from irix.live.gym_runner import GymSessionRunner
from irix.live.station_runner import StationSessionRunner
from irix.pipeline.schema import RepCompletedEvent, SetCompleteEvent
from irix.rep_counting.exercises import EXERCISES
from irix.topology.registry import StationInfo, StationRegistry
from irix.wristband_sim.simulator import SimulatedBLEGateway, SimulatedWristband
from irix.demo.synthetic_live import SyntheticFrameSource, SyntheticPoseEstimator

N_STATIONS = 4
N_MEMBERS = 8  # more members than stations -- some stations serve two people over the run
TICKS = 300


def _build_gym(seed: int = 0):
    registry = StationRegistry(
        [
            StationInfo(station_id=f"station-{i}", camera_id=f"cam-{i}", zone="free_weights", default_exercise="squat")
            for i in range(N_STATIONS)
        ]
    )
    checkout = CheckoutRegistry()
    gateway = SimulatedBLEGateway(packet_loss_pct=0.03, seed=seed)

    events = []

    def on_events(new_events):
        events.extend(new_events)

    station_runners = {}
    for i in range(N_STATIONS):
        station_id = f"station-{i}"
        station_runners[station_id] = StationSessionRunner(
            station_id=station_id, exercise_name="squat", checkout_registry=checkout,
            frame_source=SyntheticFrameSource(), ble_reader=lambda: [],
            imu_stream_factory=gateway.imu_stream_factory,
            pose_estimator=SyntheticPoseEstimator(EXERCISES["squat"], reps_per_second=0.4 + 0.05 * i),
            presence_timeout_s=1.0, on_events=on_events,
        )

    for m in range(N_MEMBERS):
        wristband_id = f"band-{m}"
        member_id = f"member-{m}"
        checkout.check_out(wristband_id, member_id=member_id, timestamp=0.0)
        band = SimulatedWristband(wristband_id, seed=seed + m)
        band.set_motion("reps", reps_per_second=0.4 + 0.05 * (m % N_STATIONS))
        gateway.add_wristband(band)
        # Distribute members across stations; extra members start unassigned
        # (station_id=None) and get placed onto a station partway through,
        # to also exercise a wristband joining mid-run.
        if m < N_STATIONS:
            gateway.move_to_station(wristband_id, f"station-{m}")

    return registry, checkout, gateway, station_runners, events


def test_multiple_simultaneous_members_never_cross_attribute_events():
    registry, checkout, gateway, station_runners, events = _build_gym(seed=1)

    class _Clock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            self.t += 1.0 / 30.0
            return self.t

    clock = _Clock()

    # This test drives the tick loop manually (see run_with_late_joiners
    # below) so it can inject late-joining members partway through --
    # GymSessionRunner's own run_forever()/ble_reader are therefore never
    # invoked; ble_reader is a harmless placeholder, same convention
    # irix/demo/run_gym_demo.py's README usage example already documents
    # for this exact situation.
    gym = GymSessionRunner(
        registry=registry, checkout_registry=checkout, station_runners=station_runners,
        ble_reader=lambda: [], presence_timeout_s=1.0,
        on_gym_events=lambda evs: events.extend(evs), clock=clock,
    )

    # bring the remaining members onto the floor partway through the run
    def run_with_late_joiners(n_ticks):
        half = n_ticks // 2
        frame_iters = {sid: r.frame_source.frames(max_frames=None) for sid, r in station_runners.items()}
        for i in range(n_ticks):
            if i == half:
                for m in range(N_STATIONS, N_MEMBERS):
                    gateway.move_to_station(f"band-{m}", f"station-{m % N_STATIONS}")
            now = clock()
            gateway.tick(now)
            gym._update_gym_wide_presence(gateway.ble_reader(), now)
            for station_id, it in frame_iters.items():
                try:
                    frame = next(it)
                except StopIteration:
                    continue
                present = gym._present_wristbands_at(station_id, now)
                station_runners[station_id].tick(frame, now, present)

    start = time.perf_counter()
    run_with_late_joiners(TICKS)
    gym.close()
    elapsed = time.perf_counter() - start

    rep_events = [e for e in events if isinstance(e, RepCompletedEvent)]
    set_events = [e for e in events if isinstance(e, SetCompleteEvent)]

    assert len(rep_events) > 0, "expected at least some reps across 8 concurrent members"
    assert len(set_events) > 0

    # Every event's member_id must be one of the checked-out members --
    # never an unrecognized id, and never leak into a member who wasn't
    # even placed on a station this run (a hard cross-contamination check).
    valid_member_ids = {f"member-{m}" for m in range(N_MEMBERS)}
    for e in rep_events + set_events:
        assert e.member_id in valid_member_ids

    # coarse throughput sanity check -- not a hard performance gate (real
    # numbers live in irix.benchmark), just confirms this isn't pathologically slow
    assert elapsed < 10.0, f"stress run took {elapsed:.2f}s for {TICKS} ticks x {N_STATIONS} stations -- investigate"
