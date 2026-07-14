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

from ..barbell.detector import FreeWeightDetector
from ..fusion.imu_io import load_imu_samples
from ..pipeline.rep_session import RepSession
from ..pipeline.schema import CameraEvent
from ..rep_counting.exercises import EXERCISES
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

    A thin driver now: the actual per-member logic (rep counting, form
    scoring, weight recognition, barbell velocity/RPE, set-boundary
    detection, fatigue analysis) lives in ``irix.pipeline.rep_session.
    RepSession``, shared with the live station runner (``irix.live.
    station_runner``) -- this function's job is just "read frames from a
    video file, feed them to one RepSession, load an IMU file upfront if
    given."

    Raises ``ValueError`` if ``exercise_name`` isn't a known exercise or
    ``video_path`` can't be opened.
    """
    import cv2

    from ..pose.estimator import PoseEstimator

    session = RepSession(
        exercise_name=exercise_name,
        member_id=member_id,
        station_id=station_id,
        vlm_backend=vlm_backend,
        weight_check_every_n_frames=weight_check_every_n_frames,
        barbell_detector=barbell_detector,
        rest_gap_s=rest_gap_s,
    )
    if imu_path:
        session.add_imu_samples(load_imu_samples(imu_path))

    estimator = PoseEstimator()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # some containers/codecs (e.g. certain synthetic/test mp4s) don't report fps

    events: List[CameraEvent] = list(session.initial_events)
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

            people = estimator.estimate(frame)
            person = people[0] if people else None
            events.extend(session.process_frame(frame, ts, person))
    finally:
        cap.release()
        events.extend(session.close())

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
