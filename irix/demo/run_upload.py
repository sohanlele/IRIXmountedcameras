"""Offline "upload mode": a recorded video file + (optionally) a recorded
wristband IMU file in, the full ``CameraEvent`` JSON stream out.

Every other entrypoint in this package is either fully synthetic
(``run_demo.py --mock-pose``, ``run_gym_demo.py``) or a live/webcam loop
that only ever wires pose -> joint angle -> ``RepCounter`` -> ``FormScorer``
(``run_demo.py --source``/``run_live``) -- none of them take a real,
already-recorded video and a real, already-recorded wristband export and
run the *rest* of this repo's real modules (IMU fusion, weight
recognition, fatigue analysis, set-boundary detection) against them. This
is that entrypoint.

What's real and always on, given just a video file:
    pose (``PoseEstimator``) -> joint angle -> ``RepCounter`` -> per-rep
    ``FormScorer`` -> a ``RestGapSetBoundaryDetector`` segmenting the
    continuous rep stream into sets (nothing hand-scripts set length here,
    unlike the mock demos) -> ``SetFatigueAnalyzer``/
    ``SessionFatigueTracker`` summarizing each closed set.

What's real and turns on if you also supply the matching input:
    - ``imu_path``: a real wristband recording (see ``irix.fusion.imu_io``
      for the file format), sliced per set and reconciled against the
      camera's count via ``RepCountFusion`` -- populates
      ``SetCompleteEvent.fused_rep_count``.
    - ``vlm_backend``: a ``VLMBackend`` (e.g. ``GeminiVLMBackend`` with a
      real API key you supply -- none is bundled here) -- periodic frames
      get read by ``VisionPlateClassifier``, populating
      ``WeightConfirmedEvent`` and each rep's ``weight_kg``.
    - ``barbell_detector``: a ``FreeWeightDetector`` pointed at a real
      trained barbell/plate checkpoint (none is bundled -- see
      ``docs/ARCHITECTURE.md``'s "Model weights" section for why) -- once
      supplied, this unlocks calibrated bar velocity (``irix.barbell``),
      which upgrades each rep from the deg/s joint-angle velocity proxy to
      real m/s velocity + RPE (``irix.barbell.rpe``), and lets
      ``WeightConfirmedEvent`` get an independent geometry sanity check
      (``irix.weight_recognition.plate_geometry_check``). Left off by
      default (``None``) rather than guessed at, same reasoning as
      ``FreeWeightDetector`` itself staying a documented stub.

One correctness note worth being explicit about: this processes a
*recorded* video file, not a live camera, so frame timestamps are derived
from the video's own frame index / fps (``frame_index / fps``), not
wall-clock processing time. ``run_live`` (the webcam/live-demo path) uses
``time.monotonic()`` instead, which is correct there because a live
camera's frame arrival time *is* wall-clock time -- but for an uploaded
file, processing speed (which varies with inference latency, CPU load)
has nothing to do with the footage's actual timeline, and using wall-clock
time here would silently misalign a real IMU file's timestamps against
the video's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from ..barbell.calibration import calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from ..barbell.detector import FreeWeightClass, FreeWeightDetection, FreeWeightDetector
from ..barbell.rpe import RPETracker
from ..barbell.tracker import BarPathTracker
from ..fatigue.models import RepFatigueSample
from ..fatigue.session_analysis import SessionFatigueTracker
from ..fatigue.set_analysis import SetFatigueAnalyzer
from ..fusion.imu_io import load_imu_samples
from ..fusion.rep_fusion import RepCountFusion
from ..pipeline.events import BandPlacementTracker, RestGapSetBoundaryDetector
from ..pipeline.schema import (
    CameraEvent,
    RepCompletedEvent,
    SetCompleteEvent,
    SetFatigueSummaryEvent,
    WeightConfirmedEvent,
)
from ..pose.geometry import joint_angle
from ..form.scoring import FormScorer
from ..rep_counting.exercises import EXERCISES
from ..rep_counting.state_machine import RepCounter
from ..weight_recognition.plate_geometry_check import check_plate_geometry
from ..weight_recognition.vision_classifier import VisionPlateClassifier
from ..weight_recognition.vlm_backend import VLMBackend


def run_upload(
    video_path: str,
    exercise_name: str,
    member_id: str,
    station_id: str,
    imu_path: Optional[str] = None,
    vlm_backend: Optional[VLMBackend] = None,
    weight_check_every_n_frames: int = 30,
    barbell_detector: Optional[FreeWeightDetector] = None,
    rest_gap_s: float = 20.0,
    max_frames: Optional[int] = None,
) -> List[CameraEvent]:
    """Run the full real pipeline against an uploaded video (and,
    optionally, an uploaded wristband IMU recording), returning every
    ``CameraEvent`` produced, in chronological order -- the structured
    payload ``irix-mvp-app``'s AI needs (see ``irix/pipeline/schema.py``'s
    module docstring for the API-contract framing).

    Raises ``ValueError`` if ``exercise_name`` isn't a known exercise or
    ``video_path`` can't be opened.
    """
    import cv2

    from ..pose.estimator import PoseEstimator

    if exercise_name not in EXERCISES:
        raise ValueError(f"Unknown exercise {exercise_name!r} -- choices: {sorted(EXERCISES)}")
    exercise = EXERCISES[exercise_name]

    imu_samples = load_imu_samples(imu_path) if imu_path else []

    counter = RepCounter(exercise)
    form_scorer = FormScorer()
    estimator = PoseEstimator()
    band_tracker = BandPlacementTracker(member_id)
    boundary_detector = RestGapSetBoundaryDetector(rest_gap_s=rest_gap_s)
    rep_fusion = RepCountFusion()
    set_fatigue_analyzer = SetFatigueAnalyzer()
    session_fatigue_tracker = SessionFatigueTracker()
    weight_classifier = VisionPlateClassifier(vlm_backend) if vlm_backend is not None else None
    rpe_tracker = RPETracker(exercise_name) if barbell_detector is not None else None
    bar_tracker: Optional[BarPathTracker] = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # some containers/codecs (e.g. certain synthetic/test mp4s) don't report fps

    events: List[CameraEvent] = []

    band_event = band_tracker.event_for(exercise)
    if band_event is not None:
        events.append(band_event)

    set_reps: List[RepFatigueSample] = []
    set_start_ts: Optional[float] = None
    prev_rep_ts: Optional[float] = None
    current_weight_kg: Optional[float] = None

    def _close_set(end_ts: float) -> None:
        nonlocal set_reps, set_start_ts
        if not set_reps:
            return
        camera_count = len(set_reps)
        window_imu = (
            [s for s in imu_samples if set_start_ts <= s.timestamp <= end_ts]
            if imu_samples and set_start_ts is not None
            else []
        )
        fused = rep_fusion.fuse(
            camera_count=camera_count,
            camera_confidence=counter.tracking_confidence,
            imu_samples=window_imu,
            camera_rep_durations=[r.duration_s for r in set_reps if r.duration_s],
        )
        events.append(
            SetCompleteEvent(
                member_id=member_id,
                station_id=station_id,
                exercise=exercise_name,
                total_reps=camera_count,
                imu_rep_count=fused.imu_count,
                fused_rep_count=fused.fused_count,
                rep_count_agreement=fused.agreement,
                rep_count_source=fused.source,
            )
        )
        analysis = set_fatigue_analyzer.analyze(exercise_name, set_reps)
        if analysis is not None:
            summary = session_fatigue_tracker.add_set(member_id, exercise_name, analysis)
            trend = summary.set_to_set_velocity_trend_pct
            events.append(
                SetFatigueSummaryEvent(
                    member_id=member_id,
                    station_id=station_id,
                    exercise=exercise_name,
                    rep_count=analysis.rep_count,
                    velocity_tier=analysis.velocity_tier,
                    velocity_loss_pct=analysis.velocity_loss_pct,
                    velocity_loss_zone=analysis.velocity_loss_zone,
                    tempo_drift_pct=analysis.tempo_drift_pct,
                    mean_form_score=analysis.mean_form_score,
                    most_common_fault=analysis.most_common_fault,
                    set_to_set_velocity_trend_pct=trend[-1] if trend else None,
                    session_fatigue_index=summary.session_fatigue_index,
                    completed_sets_this_session=summary.completed_sets,
                )
            )
        if rpe_tracker is not None:
            rpe_tracker.reset()
        set_reps = []
        set_start_ts = None

    a_name, v_name, c_name = exercise.joint_triplet
    frame_index = 0
    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_index / fps
            frame_index += 1

            # -- weight recognition (periodic, real VLM call if configured) --
            if weight_classifier is not None and frame_index % weight_check_every_n_frames == 0:
                reading = weight_classifier.read_frame(frame)
                if reading is not None:
                    geometry_consistent = None
                    geometry_reason = None
                    if barbell_detector is not None:
                        detections = barbell_detector.detect(frame)
                        gcheck = check_plate_geometry(reading.value, detections)
                        geometry_consistent = gcheck.consistent
                        geometry_reason = gcheck.reason
                    current_weight_kg = reading.value
                    events.append(
                        WeightConfirmedEvent(
                            member_id=member_id,
                            station_id=station_id,
                            exercise=exercise_name,
                            weight_kg=reading.value,
                            confidence=reading.confidence,
                            geometry_consistent=geometry_consistent,
                            geometry_check_reason=geometry_reason,
                        )
                    )

            # -- barbell centroid tracking (self-calibrates from the first
            # detected plate, then feeds BarPathTracker every frame) --
            if barbell_detector is not None:
                detections: List[FreeWeightDetection] = barbell_detector.detect(frame)
                if bar_tracker is None:
                    plate = FreeWeightDetector.largest_plate(detections)
                    if plate is not None:
                        calibration = calibrate_from_known_object(
                            plate.pixel_diameter, COMPETITION_BUMPER_PLATE_DIAMETER_MM, station_id
                        )
                        bar_tracker = BarPathTracker(calibration)
                if bar_tracker is not None:
                    bar = next((d for d in detections if d.class_label == FreeWeightClass.BARBELL), None)
                    if bar is not None:
                        bar_tracker.push(ts, bar.centroid_px[1])

            # -- pose -> joint angle -> rep counting -> form scoring --
            people = estimator.estimate(frame)
            if not people:
                continue
            person = people[0]
            a, v, c = person.xy(a_name), person.xy(v_name), person.xy(c_name)
            if a is None or v is None or c is None:
                continue
            angle = joint_angle(a, v, c)
            rep_event = counter.update(angle, timestamp=ts, pose=person)
            if rep_event is None:
                continue

            if boundary_detector.observe(rep_event.timestamp) and prev_rep_ts is not None:
                _close_set(end_ts=prev_rep_ts)
            if set_start_ts is None:
                set_start_ts = rep_event.concentric_start_timestamp or rep_event.timestamp

            camera_event = RepCompletedEvent(
                member_id=member_id,
                station_id=station_id,
                exercise=exercise_name,
                rep_count=rep_event.rep_number,
                duration_s=rep_event.duration_s,
                peak_velocity_deg_s=rep_event.peak_angular_velocity_deg_s,
                mean_velocity_deg_s=rep_event.mean_angular_velocity_deg_s,
                weight_kg=current_weight_kg,
            )
            assessment = form_scorer.score_rep(exercise_name, rep_event.poses)
            if assessment is not None:
                camera_event.form_score = assessment.score
                camera_event.form_faults = assessment.faults

            if bar_tracker is not None and rep_event.concentric_start_timestamp is not None:
                bar_velocity = bar_tracker.velocity_for_window(
                    rep_event.concentric_start_timestamp, rep_event.timestamp
                )
                if bar_velocity.mean_velocity_m_s is not None:
                    camera_event.peak_velocity_m_s = bar_velocity.peak_velocity_m_s
                    camera_event.mean_velocity_m_s = bar_velocity.mean_velocity_m_s
                    if rpe_tracker is not None:
                        rpe_estimate = rpe_tracker.estimate(bar_velocity.mean_velocity_m_s)
                        camera_event.estimated_rpe = rpe_estimate.estimated_rpe
                        camera_event.velocity_loss_pct = rpe_estimate.velocity_loss_pct

            events.append(camera_event)
            set_reps.append(RepFatigueSample.from_rep_completed_event(camera_event))
            prev_rep_ts = rep_event.timestamp
    finally:
        cap.release()
        if set_reps:
            _close_set(end_ts=prev_rep_ts if prev_rep_ts is not None else 0.0)

    return events


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full IRIX pipeline against an uploaded video + (optionally) wristband IMU file."
    )
    parser.add_argument("--video", required=True, help="Path to a recorded video file.")
    parser.add_argument("--exercise", default="squat", choices=list(EXERCISES.keys()))
    parser.add_argument("--member-id", default="upload-member")
    parser.add_argument("--station-id", default="upload-station")
    parser.add_argument("--imu", default=None, help="Path to a recorded wristband IMU export (.csv or .json).")
    parser.add_argument(
        "--gemini-api-key", default=None,
        help="Gemini API key for weight recognition (GeminiVLMBackend). Not connected/bundled by default -- "
             "no weight recognition runs unless you pass this.",
    )
    parser.add_argument("--gemini-model", default="gemini-2.5-flash-lite")
    parser.add_argument(
        "--weight-check-every-n-frames", type=int, default=30,
        help="How often to call the VLM for a weight read (only relevant with --gemini-api-key).",
    )
    parser.add_argument(
        "--barbell-model", default=None,
        help="Path to a real trained barbell/plate detector checkpoint (FreeWeightDetector). None is bundled "
             "with this repo -- see docs/ARCHITECTURE.md's 'Model weights' section. Without this, reps fall "
             "back to the deg/s joint-angle velocity proxy instead of calibrated m/s + RPE.",
    )
    parser.add_argument("--rest-gap-s", type=float, default=20.0, help="Rest gap (seconds) that closes a set.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (mainly for testing).")
    parser.add_argument("--out", default=None, help="Write the JSON event list here instead of stdout.")
    args = parser.parse_args()

    vlm_backend = None
    if args.gemini_api_key:
        from ..weight_recognition.vlm_backend import GeminiVLMBackend

        vlm_backend = GeminiVLMBackend(api_key=args.gemini_api_key, model=args.gemini_model)

    barbell_detector = FreeWeightDetector(model_path=args.barbell_model) if args.barbell_model else None

    started = time.monotonic()
    events = run_upload(
        video_path=args.video,
        exercise_name=args.exercise,
        member_id=args.member_id,
        station_id=args.station_id,
        imu_path=args.imu,
        vlm_backend=vlm_backend,
        weight_check_every_n_frames=args.weight_check_every_n_frames,
        barbell_detector=barbell_detector,
        rest_gap_s=args.rest_gap_s,
        max_frames=args.max_frames,
    )
    elapsed = time.monotonic() - started

    payload = [e.to_dict() for e in events]
    output = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output)
        print(f"Wrote {len(payload)} events to {args.out} ({elapsed:.1f}s processing time)", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
