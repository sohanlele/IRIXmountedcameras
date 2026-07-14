"""Multi-station gym demo (Section 6: a real 10-camera deployment).

``run_demo.py``'s ``--mock-pose``/``--source`` flow demonstrates one
camera watching one lifter through one set -- useful for exercising each
subsystem in isolation, but it doesn't show what only shows up once
there's more than one camera and more than one member on the floor at
once. This does:

- ``irix.topology``: each member's authoritative station (BLE RSSI +
  hysteresis) gates which station's camera events actually get pushed to
  the pipeline, so a member walking past an adjacent camera on the way to
  their next exercise doesn't get double-counted by both.
- ``irix.fusion.rep_fusion``: each set's camera rep count is reconciled
  against a synthetic wristband IMU stream into one authoritative count,
  with the fallback direction (trust the IMU more) demonstrated
  explicitly for a heavily-occluded set.
- ``irix.fatigue``: each set is scored (velocity loss %, VL-zone, tempo
  drift, form trend), and a running ``SessionFatigueTracker`` shows the
  cross-set trend building up across a member's 2nd/3rd set of the same
  exercise -- the "context for the app's AI" this whole file is building
  toward.
- ``irix.weight_recognition.plate_geometry_check``: a synthetic VLM
  weight read is sanity-checked against a synthetic barbell-detector
  plate count from the same frame.

Deterministic (seeded) synthetic data throughout -- no camera, no model
weights, same "runs anywhere, no hardware needed" property as
``run_demo.py --mock-pose``.

    python -m irix.demo.run_gym_demo
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..barbell.detector import FreeWeightClass, FreeWeightDetection
from ..fatigue.models import RepFatigueSample
from ..fatigue.session_analysis import SessionFatigueTracker
from ..fatigue.set_analysis import SetFatigueAnalyzer
from ..form.scoring import FormScorer
from ..fusion.rep_fusion import RepCountFusion
from ..identity.ble_pairing import BLEReading
from ..pipeline.aggregator import Aggregator
from ..pipeline.cloud_sync import InMemoryCloudSync
from ..pipeline.edge_buffer import LocalBuffer
from ..pipeline.events import BandPlacementTracker
from ..pipeline.schema import (
    RepCompletedEvent,
    SetCompleteEvent,
    SetFatigueSummaryEvent,
    WeightConfirmedEvent,
)
from ..rep_counting.exercises import EXERCISES
from ..rep_counting.state_machine import RepCounter
from ..topology.handoff import GymCoordinator
from ..topology.registry import build_default_ten_station_gym
from ..weight_recognition.plate_geometry_check import check_plate_geometry
from .mock_pose import synthetic_angle_stream, synthetic_imu_stream, synthetic_pose_stream


def _run_one_set(
    exercise_name: str,
    member_id: str,
    station_id: str,
    buffer: LocalBuffer,
    n_frames: int = 300,
    fps: float = 30.0,
    reps_per_second: float = 0.5,
    seed: int = 0,
    occlusion: bool = False,
    inject_form_fault: Optional[str] = None,
    session_tracker: Optional[SessionFatigueTracker] = None,
    verbose: bool = True,
) -> SetCompleteEvent:
    """Runs one full set at one station: camera rep counting (+ optional
    pose stream for form scoring), a synthetic wristband IMU stream for
    fusion, and set-level fatigue analysis -- everything
    ``run_demo.run_mock`` does, plus the fusion/fatigue pieces that only
    make sense at full-set granularity (see irix.fusion.rep_fusion's and
    irix.fatigue's module docstrings for why: both operate on a set as
    the unit, not a single rep).

    ``occlusion=True`` simulates most camera frames losing tracking
    (angle -> NaN) partway through the set, to demonstrate
    RepCountFusion falling back toward the IMU count.
    """
    exercise = EXERCISES[exercise_name]
    counter = RepCounter(exercise)
    form_scorer = FormScorer()
    fatigue_samples: List[RepFatigueSample] = []

    pose_stream = None
    if exercise_name in ("squat", "leg_press", "hack_squat", "bicep_curl"):
        pose_stream = synthetic_pose_stream(
            exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second,
            inject_fault=inject_form_fault,
        )

    for i, (t, angle) in enumerate(
        synthetic_angle_stream(exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second)
    ):
        pose = None
        if pose_stream is not None:
            _, _, pose = next(pose_stream)
        frame_angle = angle
        if occlusion and n_frames // 3 <= i < 2 * n_frames // 3:
            # Simulate the camera losing the lifter for the middle third
            # of the set (someone walked through frame, bad lighting,
            # whatever) -- RepCounter.update()'s NaN guard drops these
            # frames from both rep counting and tracking_confidence.
            frame_angle = float("nan")
        rep_event = counter.update(frame_angle, timestamp=t, pose=pose)
        if rep_event:
            camera_event = RepCompletedEvent(
                member_id=member_id, station_id=station_id, exercise=exercise_name,
                rep_count=rep_event.rep_number, duration_s=rep_event.duration_s,
                peak_velocity_deg_s=rep_event.peak_angular_velocity_deg_s,
                mean_velocity_deg_s=rep_event.mean_angular_velocity_deg_s,
            )
            if pose_stream is not None:
                assessment = form_scorer.score_rep(exercise_name, rep_event.poses)
                if assessment is not None:
                    camera_event.form_score = assessment.score
                    camera_event.form_faults = assessment.faults
            buffer.push(camera_event)
            fatigue_samples.append(RepFatigueSample.from_rep_completed_event(camera_event))
            if verbose:
                print(f"  [{t:6.2f}s] rep {camera_event.rep_count} {camera_event.to_dict()}")

    imu_samples = synthetic_imu_stream(
        n_seconds=n_frames / fps, reps_per_second=reps_per_second, seed=seed,
    )
    fusion = RepCountFusion()
    durations = [s.duration_s for s in fatigue_samples if s.duration_s]
    fusion_result = fusion.fuse(
        camera_count=counter.rep_count, camera_confidence=counter.tracking_confidence,
        imu_samples=imu_samples, camera_rep_durations=durations,
    )
    if verbose:
        print(f"  [fusion] camera_confidence={counter.tracking_confidence:.2f} {fusion_result.to_dict()}")

    set_event = SetCompleteEvent(
        member_id=member_id, station_id=station_id, exercise=exercise_name, total_reps=counter.rep_count,
        imu_rep_count=fusion_result.imu_count, fused_rep_count=fusion_result.fused_count,
        rep_count_agreement=fusion_result.agreement, rep_count_source=fusion_result.source,
    )
    buffer.push(set_event)

    set_analysis = SetFatigueAnalyzer().analyze(exercise_name, fatigue_samples)
    if set_analysis is not None:
        set_to_set = None
        session_index = None
        completed_sets = 1
        if session_tracker is not None:
            summary = session_tracker.add_set(member_id, exercise_name, set_analysis)
            set_to_set = summary.set_to_set_velocity_trend_pct[-1]
            session_index = summary.session_fatigue_index
            completed_sets = summary.completed_sets
        fatigue_event = SetFatigueSummaryEvent(
            member_id=member_id, station_id=station_id, exercise=exercise_name,
            rep_count=set_analysis.rep_count, velocity_tier=set_analysis.velocity_tier,
            velocity_loss_pct=set_analysis.velocity_loss_pct, velocity_loss_zone=set_analysis.velocity_loss_zone,
            tempo_drift_pct=set_analysis.tempo_drift_pct, mean_form_score=set_analysis.mean_form_score,
            most_common_fault=set_analysis.most_common_fault,
            set_to_set_velocity_trend_pct=set_to_set, session_fatigue_index=session_index,
            completed_sets_this_session=completed_sets,
        )
        buffer.push(fatigue_event)
        if verbose:
            print(f"  [fatigue] {fatigue_event.to_dict()}")

    return set_event


def _demo_weight_geometry_check(buffer: LocalBuffer, member_id: str, station_id: str, verbose: bool = True) -> None:
    """A synthetic VLM weight read, sanity-checked against a synthetic
    barbell-detector plate count in the same frame -- see
    irix.weight_recognition.plate_geometry_check's module docstring for
    why this is a corroborating check, not a second independent reading
    method."""

    def plate(cx: float) -> FreeWeightDetection:
        return FreeWeightDetection(FreeWeightClass.PLATE, (cx, 500.0), (cx - 100, 450.0, cx + 100, 550.0), 0.9)

    # 4 plates visible (2/side) -- consistent with a 100kg VLM read
    # (20kg bar + 2x(25+15)kg), inconsistent with a badly misread 180kg.
    detections = [plate(100), plate(200), plate(1200), plate(1300)]

    good_read = check_plate_geometry(read_weight_kg=100.0, detections=detections)
    good_event = WeightConfirmedEvent(
        member_id=member_id, station_id=station_id, exercise="squat", weight_kg=100.0, confidence=0.93,
        geometry_consistent=good_read.consistent, geometry_check_reason=good_read.reason,
    )
    buffer.push(good_event)
    if verbose:
        print(f"  [weight, plausible VLM read] {good_event.to_dict()}")

    bad_read = check_plate_geometry(read_weight_kg=180.0, detections=detections)
    bad_event = WeightConfirmedEvent(
        member_id=member_id, station_id=station_id, exercise="squat", weight_kg=180.0, confidence=0.88,
        geometry_consistent=bad_read.consistent, geometry_check_reason=bad_read.reason,
    )
    buffer.push(bad_event)
    if verbose:
        print(f"  [weight, implausible VLM read -- flagged] {bad_event.to_dict()}")


def _demo_station_handoff_and_dedup(verbose: bool = True) -> GymCoordinator:
    """Walks a member from squat-1 to squat-2 (registered adjacent
    stations), through a burst of BLE jitter that shouldn't trigger a
    handoff, then a sustained signal that should -- and shows
    ``is_authoritative`` gating a spurious detection from a station the
    member isn't actually at."""
    registry = build_default_ten_station_gym()
    coordinator = GymCoordinator(registry, min_consecutive=3)
    member = "member-alice"

    for t in (0.0, 1.0, 2.0):
        coordinator.update_member(member, [BLEReading("squat-1", -50.0, t)], timestamp=t)
    if verbose:
        print(f"  [topology] alice settled at {coordinator.current_station(member)}")

    # RSSI jitter near the boundary -- one noisy reading favoring squat-2,
    # should NOT cause a handoff.
    jitter_event = coordinator.update_member(
        member, [BLEReading("squat-1", -55.0, 3.0), BLEReading("squat-2", -53.0, 3.0)], timestamp=3.0,
    )
    assert jitter_event is None, "single noisy reading should not trigger a handoff"

    # Meanwhile: squat-2's camera (adjacent, briefly sees alice in its
    # periphery) would normally be tempted to report a rep for her too --
    # is_authoritative correctly says no, since she's still resolved to
    # squat-1.
    spurious_authoritative = coordinator.is_authoritative(member, "squat-2")
    if verbose:
        print(f"  [topology] squat-2 authoritative for alice mid-jitter? {spurious_authoritative} (correctly False)")

    # Sustained signal -- 3 consecutive readings favoring squat-2 -- this
    # IS a real handoff. The event fires as soon as the streak crosses
    # min_consecutive, which may be before the loop's last iteration (any
    # reading after that just confirms squat-2, resolved == current
    # station by then, so returns None again) -- keep the first non-None
    # result, don't just take whatever the last call happened to return.
    handoff_event = None
    for t in (4.0, 5.0, 6.0):
        result = coordinator.update_member(member, [BLEReading("squat-2", -45.0, t)], timestamp=t)
        if result is not None:
            handoff_event = result
    if verbose:
        print(f"  [topology] handoff event: {handoff_event}")
        print(f"  [topology] squat-1 authoritative now? {coordinator.is_authoritative(member, 'squat-1')} (correctly False)")
        print(f"  [topology] squat-2 authoritative now? {coordinator.is_authoritative(member, 'squat-2')} (correctly True)")

    return coordinator


def main() -> Dict[str, InMemoryCloudSync]:
    print("=== Station topology + BLE handoff / anti-double-count demo ===")
    _demo_station_handoff_and_dedup()

    print("\n=== Member 'alice': 2 squat sets at squat-1 (session fatigue building) ===")
    cloud_squat = InMemoryCloudSync()
    buffer_squat = LocalBuffer()
    agg_squat = Aggregator(cloud_sync=cloud_squat)
    agg_squat.register_zone("zone-squat-1", buffer_squat)
    alice_session = SessionFatigueTracker()
    _run_one_set(
        "squat", "member-alice", "squat-1", buffer_squat, seed=1,
        session_tracker=alice_session, verbose=True,
    )
    print("  -- rest --")
    _run_one_set(
        "squat", "member-alice", "squat-1", buffer_squat, seed=2, reps_per_second=0.55,
        session_tracker=alice_session, verbose=True,
    )

    print("\n=== Member 'alice': weight-recognition geometry cross-check ===")
    _demo_weight_geometry_check(buffer_squat, "member-alice", "squat-1")

    print("\n=== Member 'alice' hands off to curl-1: bicep curl set with an injected form fault ===")
    cloud_curl = InMemoryCloudSync()
    buffer_curl = LocalBuffer()
    agg_curl = Aggregator(cloud_sync=cloud_curl)
    agg_curl.register_zone("zone-curl-1", buffer_curl)
    _run_one_set(
        "bicep_curl", "member-alice", "curl-1", buffer_curl, seed=3,
        inject_form_fault="leaning_back", session_tracker=alice_session, verbose=True,
    )

    print("\n=== Member 'bob': leg press with heavy camera occlusion -- fusion falls back to IMU ===")
    cloud_legpress = InMemoryCloudSync()
    buffer_legpress = LocalBuffer()
    agg_legpress = Aggregator(cloud_sync=cloud_legpress)
    agg_legpress.register_zone("zone-leg-press-1", buffer_legpress)
    band_tracker = BandPlacementTracker(member_id="member-bob")
    placement_event = band_tracker.event_for(EXERCISES["leg_press"])
    if placement_event:
        buffer_legpress.push(placement_event)
        print(f"  [band placement] {placement_event.to_dict()}")
    _run_one_set(
        "leg_press", "member-bob", "leg-press-1", buffer_legpress, seed=4,
        occlusion=True, verbose=True,
    )

    print("\nSynced events per zone:")
    for name, agg, cloud in (
        ("squat-1", agg_squat, cloud_squat), ("curl-1", agg_curl, cloud_curl),
        ("leg-press-1", agg_legpress, cloud_legpress),
    ):
        synced = agg.sync()
        print(f"  {name}: {synced} events synced ({len(cloud.received)} total in cloud)")

    return {"squat-1": cloud_squat, "curl-1": cloud_curl, "leg-press-1": cloud_legpress}


if __name__ == "__main__":
    main()
