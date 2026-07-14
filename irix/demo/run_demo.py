"""End-to-end demo entrypoint.

Pipeline: frame source -> PoseEstimator -> joint angle -> RepCounter ->
CoachingTrigger -> TTSEngine, with each completed rep also pushed into the
edge pipeline (LocalBuffer -> Aggregator -> CloudSync) as a
DerivedMetricsEvent.

Two modes:
  --mock-pose   Synthetic joint-angle stream (no camera, no model weights,
                no display needed). Good for smoke-testing the pipeline.
  --source N|path  Real webcam index or video file, run through
                PoseEstimator (requires the 'pose' extra / ultralytics).

--with-imu-crosscheck (mock mode only) additionally runs a synthetic
wristband IMU stream through RecoFitCounter/ULiftCounter (Section 4.6/5.3)
alongside the camera-based joint-angle counter, and prints both counts.
This is what "two independent signals" looks like end-to-end: in a real
deployment the two would feed the EKF (irix.fusion.ekf) rather than just
being printed side by side.

Example:
    python -m irix.demo.run_demo --mock-pose --exercise squat
    python -m irix.demo.run_demo --mock-pose --exercise squat --with-imu-crosscheck
    python -m irix.demo.run_demo --source 0 --exercise bicep_curl
"""
from __future__ import annotations

import argparse
import sys
import time

from ..coaching.triggers import CoachingTrigger
from ..coaching.tts_engine import NullTTSEngine
from ..pipeline.aggregator import Aggregator
from ..pipeline.cloud_sync import InMemoryCloudSync
from ..pipeline.edge_buffer import LocalBuffer
from ..pipeline.schema import DerivedMetricsEvent
from ..pose.geometry import joint_angle
from ..rep_counting.exercises import EXERCISES
from ..rep_counting.state_machine import RepCounter


def _run_imu_crosscheck(reps_per_second: float, n_seconds: float, verbose: bool) -> None:
    """Generate a synthetic wristband IMU stream at the same tempo as the
    mock camera stream and run both IMU-only counters against it, purely
    to demonstrate the cross-check (Section 4.6: "two independent
    anchors" -- here, two independent *counters* on one wristband
    signal). See irix/fusion/imu_rep_counting.py for provenance."""
    from ..fusion.imu_rep_counting import RecoFitCounter, ULiftCounter
    from tests.test_imu_rep_counting import synthetic_imu_stream

    samples = synthetic_imu_stream(n_seconds=n_seconds, reps_per_second=reps_per_second)
    recofit = RecoFitCounter(min_period=1.0 / reps_per_second * 0.5, max_period=1.0 / reps_per_second * 2.0)
    ulift = ULiftCounter()
    r1 = recofit.count(samples)
    r2 = ulift.count(samples)
    if verbose:
        print(
            f"[wristband IMU crosscheck] RecoFit: {r1.count} reps (confidence {r1.confidence:.2f}) | "
            f"uLift: {r2.count} reps (confidence {r2.confidence:.2f})"
        )


def run_mock(
    exercise_name: str,
    member_id: str,
    station_id: str,
    n_frames: int,
    verbose: bool = True,
    with_imu_crosscheck: bool = False,
):
    from .mock_pose import synthetic_angle_stream

    exercise = EXERCISES[exercise_name]
    counter = RepCounter(exercise)
    coach = CoachingTrigger()
    tts = NullTTSEngine()
    buffer = LocalBuffer()
    cloud = InMemoryCloudSync()
    aggregator = Aggregator(cloud_sync=cloud)
    aggregator.register_zone("zone-mock", buffer)

    fps = 30.0
    reps_per_second = 0.5
    for t, angle in synthetic_angle_stream(exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second):
        event = counter.update(angle, timestamp=t)
        if event:
            line = coach.on_rep(event)
            tts.speak(line)
            buffer.push(
                DerivedMetricsEvent(
                    member_id=member_id,
                    station_id=station_id,
                    exercise=exercise_name,
                    rep_count=event.rep_number,
                )
            )
            if verbose:
                print(f"[{t:6.2f}s] {line}")

    synced = aggregator.sync()
    if verbose:
        print(f"Total reps (camera): {counter.rep_count}. Synced {synced} events to cloud.")

    if with_imu_crosscheck:
        _run_imu_crosscheck(reps_per_second=reps_per_second, n_seconds=n_frames / fps, verbose=verbose)

    return counter, cloud


def run_live(source: str, exercise_name: str, member_id: str, station_id: str):
    import cv2

    from ..pose.estimator import PoseEstimator

    exercise = EXERCISES[exercise_name]
    counter = RepCounter(exercise)
    coach = CoachingTrigger()
    tts = NullTTSEngine()
    buffer = LocalBuffer()
    cloud = InMemoryCloudSync()
    aggregator = Aggregator(cloud_sync=cloud)
    aggregator.register_zone("zone-live", buffer)
    estimator = PoseEstimator()

    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"Could not open source: {source}", file=sys.stderr)
        sys.exit(1)

    a_name, v_name, c_name = exercise.joint_triplet
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            people = estimator.estimate(frame)
            if people:
                person = people[0]
                a, v, c = person.xy(a_name), person.xy(v_name), person.xy(c_name)
                if a is not None and v is not None and c is not None:
                    angle = joint_angle(a, v, c)
                    event = counter.update(angle, timestamp=time.monotonic())
                    if event:
                        line = coach.on_rep(event)
                        tts.speak(line)
                        buffer.push(
                            DerivedMetricsEvent(
                                member_id=member_id,
                                station_id=station_id,
                                exercise=exercise_name,
                                rep_count=event.rep_number,
                            )
                        )
                        print(line)
                    cv2.putText(
                        frame, f"Reps: {counter.rep_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                    )
            cv2.imshow("IRIX demo", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        aggregator.sync()
        print(f"Total reps: {counter.rep_count}")


def main():
    parser = argparse.ArgumentParser(description="IRIX rep-tracking demo")
    parser.add_argument("--mock-pose", action="store_true", help="Use a synthetic angle stream, no camera required.")
    parser.add_argument("--source", default="0", help="Webcam index or video file path (ignored with --mock-pose).")
    parser.add_argument("--exercise", default="squat", choices=list(EXERCISES.keys()))
    parser.add_argument("--member-id", default="demo-member")
    parser.add_argument("--station-id", default="demo-station")
    parser.add_argument("--frames", type=int, default=300, help="Frame count for --mock-pose.")
    parser.add_argument(
        "--with-imu-crosscheck", action="store_true",
        help="(--mock-pose only) also run a synthetic wristband IMU stream through RecoFit/uLift.",
    )
    args = parser.parse_args()

    if args.mock_pose:
        run_mock(
            args.exercise, args.member_id, args.station_id, args.frames,
            with_imu_crosscheck=args.with_imu_crosscheck,
        )
    else:
        run_live(args.source, args.exercise, args.member_id, args.station_id)


if __name__ == "__main__":
    main()
