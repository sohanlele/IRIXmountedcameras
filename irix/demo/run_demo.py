"""End-to-end demo entrypoint.

Pipeline: frame source -> PoseEstimator -> joint angle -> RepCounter ->
structured CameraEvent -> edge pipeline (LocalBuffer -> Aggregator ->
CloudSync). This repo's job stops at "here's what happened on the gym
floor" -- turning that into spoken instructions / UI is
jeffreyjy/irix-mvp-app's job (its `agents/` layer + iOS frontend), not
this one. The demo just prints the events that would be sent onward.

Two modes:
  --mock-pose   Synthetic joint-angle stream (no camera, no model weights,
                no display needed). Good for smoke-testing the pipeline.
  --source N|path  Real webcam index or video file, run through
                PoseEstimator (requires the 'pose' extra / ultralytics).

--with-imu-crosscheck (mock mode only) additionally runs a synthetic
wristband IMU stream through RecoFitCounter/ULiftCounter (Section 4.6/5.3)
alongside the camera-based joint-angle counter, and prints both counts.

--with-barbell-tracking (mock mode only, exercises with a published
velocity anchor -- squat/bench_press/deadlift) additionally runs a
synthetic barbell-pixel stream through irix.barbell's calibration ->
BarPathTracker -> RPETracker, populating each RepCompletedEvent's
calibrated m/s velocity, velocity-loss %, and estimated RPE fields
alongside the always-present joint-angle deg/s proxy. The synthetic
stream decays in amplitude rep-over-rep so velocity_loss_pct has
something nonzero to show.

Example:
    python -m irix.demo.run_demo --mock-pose --exercise squat
    python -m irix.demo.run_demo --mock-pose --exercise squat --with-imu-crosscheck
    python -m irix.demo.run_demo --mock-pose --exercise squat --with-barbell-tracking
    python -m irix.demo.run_demo --mock-pose --exercise leg_press
    python -m irix.demo.run_demo --source 0 --exercise bicep_curl

--with-form-scoring (mock mode only, squat/bicep_curl -- the two
exercises the synthetic pose generator supports) additionally runs a
synthetic full-body keypoint stream through irix.form.scoring.FormScorer,
populating each RepCompletedEvent's form_score/form_faults fields.
--inject-form-fault lets the mock stream deliberately produce bad form
(knee_valgus / leaning_back / elbow_drift) so the demo can show a fault
actually getting caught, rather than only ever showing clean reps. In
live mode (--source), form scoring runs automatically whenever the real
PoseEstimator returns a confident pose, no flag needed -- see run_live.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from ..barbell.calibration import calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from ..barbell.rpe import RPETracker
from ..barbell.tracker import BarPathTracker
from ..pipeline.aggregator import Aggregator
from ..pipeline.cloud_sync import InMemoryCloudSync
from ..pipeline.edge_buffer import LocalBuffer
from ..pipeline.events import BandPlacementTracker
from ..pipeline.schema import RepCompletedEvent, SetCompleteEvent
from ..form.scoring import FormScorer
from ..pose.geometry import joint_angle
from ..pose.estimator import PersonPose
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
    with_barbell_tracking: bool = False,
    with_form_scoring: bool = False,
    inject_form_fault: Optional[str] = None,
):
    from .mock_pose import synthetic_angle_stream, synthetic_bar_pixel_stream, synthetic_pose_stream

    exercise = EXERCISES[exercise_name]
    counter = RepCounter(exercise)
    buffer = LocalBuffer()
    cloud = InMemoryCloudSync()
    aggregator = Aggregator(cloud_sync=cloud)
    aggregator.register_zone("zone-mock", buffer)

    # Band placement is decided station/exercise-side; the event just
    # tells the app something changed, the app decides how to say it.
    band_tracker = BandPlacementTracker(member_id=member_id)
    placement_event = band_tracker.event_for(exercise)
    if placement_event:
        buffer.push(placement_event)
        if verbose:
            print(f"[event] {placement_event.to_dict()}")

    fps = 30.0
    reps_per_second = 0.5

    bar_tracker = None
    rpe_tracker = None
    bar_stream = None
    if with_barbell_tracking:
        # Self-calibrate from a plate of known diameter (Section: cameras-
        # only install constraint) rather than a painted marker -- see
        # irix/barbell/calibration.py. The 180px figure is a stand-in for
        # whatever irix.barbell.detector.FreeWeightDetector would measure
        # from a real frame.
        calibration = calibrate_from_known_object(
            pixel_size=180.0, real_world_size_mm=COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id=station_id
        )
        bar_tracker = BarPathTracker(calibration)
        rpe_tracker = RPETracker(exercise_name)
        bar_stream = synthetic_bar_pixel_stream(
            n_frames=n_frames, fps=fps, reps_per_second=reps_per_second,
            amplitude_px=90.0, velocity_decay_per_rep=0.08,
        )

    form_scorer = FormScorer() if with_form_scoring else None
    pose_stream = None
    if with_form_scoring:
        pose_stream = synthetic_pose_stream(
            exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second,
            inject_fault=inject_form_fault,
        )

    angle_stream = synthetic_angle_stream(exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second)
    for t, angle in angle_stream:
        pose = None
        if pose_stream is not None:
            # synthetic_pose_stream computes its own angle from the same
            # tempo/exercise, in lockstep with angle_stream -- discard its
            # angle and reuse the one from synthetic_angle_stream so the
            # rep-counting angle source stays single-sourced regardless of
            # which optional streams are enabled.
            _, _, pose = next(pose_stream)
        if bar_tracker is not None:
            _, y_px = next(bar_stream)
            bar_tracker.push(t, y_px)

        rep_event = counter.update(angle, timestamp=t, pose=pose)
        if rep_event:
            camera_event = RepCompletedEvent(
                member_id=member_id,
                station_id=station_id,
                exercise=exercise_name,
                rep_count=rep_event.rep_number,
                duration_s=rep_event.duration_s,
                peak_velocity_deg_s=rep_event.peak_angular_velocity_deg_s,
                mean_velocity_deg_s=rep_event.mean_angular_velocity_deg_s,
            )
            if bar_tracker is not None and rep_event.concentric_start_timestamp is not None:
                bar_velocity = bar_tracker.velocity_for_window(rep_event.concentric_start_timestamp, rep_event.timestamp)
                camera_event.peak_velocity_m_s = bar_velocity.peak_velocity_m_s
                camera_event.mean_velocity_m_s = bar_velocity.mean_velocity_m_s
                if bar_velocity.mean_velocity_m_s is not None:
                    rpe = rpe_tracker.estimate(bar_velocity.mean_velocity_m_s)
                    camera_event.velocity_loss_pct = rpe.velocity_loss_pct
                    camera_event.estimated_rpe = rpe.estimated_rpe
            if form_scorer is not None:
                assessment = form_scorer.score_rep(exercise_name, rep_event.poses)
                if assessment is not None:
                    camera_event.form_score = assessment.score
                    camera_event.form_faults = assessment.faults
            buffer.push(camera_event)
            if verbose:
                print(f"[{t:6.2f}s] [event] {camera_event.to_dict()}")

    if counter.rep_count:
        buffer.push(
            SetCompleteEvent(
                member_id=member_id,
                station_id=station_id,
                exercise=exercise_name,
                total_reps=counter.rep_count,
            )
        )

    synced = aggregator.sync()
    if verbose:
        print(f"Total reps (camera): {counter.rep_count}. Synced {synced} events.")

    if with_imu_crosscheck:
        _run_imu_crosscheck(reps_per_second=reps_per_second, n_seconds=n_frames / fps, verbose=verbose)

    return counter, cloud


def run_live(source: str, exercise_name: str, member_id: str, station_id: str):
    import cv2

    from ..pose.estimator import PoseEstimator

    exercise = EXERCISES[exercise_name]
    counter = RepCounter(exercise)
    buffer = LocalBuffer()
    cloud = InMemoryCloudSync()
    aggregator = Aggregator(cloud_sync=cloud)
    aggregator.register_zone("zone-live", buffer)
    estimator = PoseEstimator()
    # Form scoring runs "for free" in live mode -- RepCounter already
    # buffers whatever PersonPose is passed into update(), and the real
    # PoseEstimator gives us a full-body pose every frame (unlike the
    # mock-pose synthetic angle stream, which needs the separate
    # synthetic_pose_stream + --with-form-scoring opt-in above).
    form_scorer = FormScorer()

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
                    rep_event = counter.update(angle, timestamp=time.monotonic(), pose=person)
                    if rep_event:
                        camera_event = RepCompletedEvent(
                            member_id=member_id,
                            station_id=station_id,
                            exercise=exercise_name,
                            rep_count=rep_event.rep_number,
                            duration_s=rep_event.duration_s,
                            peak_velocity_deg_s=rep_event.peak_angular_velocity_deg_s,
                            mean_velocity_deg_s=rep_event.mean_angular_velocity_deg_s,
                        )
                        assessment = form_scorer.score_rep(exercise_name, rep_event.poses)
                        if assessment is not None:
                            camera_event.form_score = assessment.score
                            camera_event.form_faults = assessment.faults
                        buffer.push(camera_event)
                        print(f"[event] {camera_event.to_dict()}")
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
    parser.add_argument(
        "--with-barbell-tracking", action="store_true",
        help="(--mock-pose only) also run a synthetic barbell-pixel stream through irix.barbell for "
             "calibrated m/s velocity, velocity-loss %%, and estimated RPE.",
    )
    parser.add_argument(
        "--with-form-scoring", action="store_true",
        help="(--mock-pose only, squat/bicep_curl) also run a synthetic full-body pose stream through "
             "irix.form.scoring.FormScorer, populating form_score/form_faults.",
    )
    parser.add_argument(
        "--inject-form-fault", default=None, choices=["knee_valgus", "leaning_back", "elbow_drift"],
        help="(--with-form-scoring only) deliberately inject bad form into the synthetic pose stream "
             "so the demo shows a fault actually getting caught.",
    )
    args = parser.parse_args()

    if args.mock_pose:
        run_mock(
            args.exercise, args.member_id, args.station_id, args.frames,
            with_imu_crosscheck=args.with_imu_crosscheck,
            with_barbell_tracking=args.with_barbell_tracking,
            with_form_scoring=args.with_form_scoring,
            inject_form_fault=args.inject_form_fault,
        )
    else:
        run_live(args.source, args.exercise, args.member_id, args.station_id)


if __name__ == "__main__":
    main()
