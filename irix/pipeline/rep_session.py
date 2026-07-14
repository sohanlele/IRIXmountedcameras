"""One member's ongoing tracked state at one station: rep counting, form
scoring, weight recognition, barbell velocity/RPE, set-boundary
detection, and fatigue analysis, all in one place.

This is exactly the logic ``irix.demo.run_upload`` used to run inline,
against a whole pre-recorded video, for one member, start to finish.
Factored out here so ``irix.live.station_runner.StationSessionRunner``
can reuse the *exact same* per-member logic for a live 24/7 station,
where the difference isn't the logic -- it's the lifecycle around it:
``run_upload`` constructs one ``RepSession`` and feeds it frames until a
video file ends; a live station runner constructs a fresh ``RepSession``
whenever a checked-out member's wristband is newly detected present at
the station, feeds it frames (and, once available, live IMU samples) for
as long as they stay present, and calls ``close()`` when they step away
-- then does it again for the next member who shows up. Neither has to
duplicate the actual event-construction logic, which is the part worth
keeping in exactly one place.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..barbell.calibration import calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from ..barbell.detector import FreeWeightClass, FreeWeightDetection, FreeWeightDetector
from ..barbell.rpe import RPETracker
from ..barbell.tracker import BarPathTracker
from ..fatigue.models import RepFatigueSample
from ..fatigue.session_analysis import SessionFatigueTracker
from ..fatigue.set_analysis import SetFatigueAnalyzer
from ..fusion.imu import IMUSample
from ..fusion.rep_fusion import RepCountFusion
from ..form.scoring import FormScorer
from ..pose.estimator import PersonPose
from ..pose.geometry import joint_angle
from ..rep_counting.exercises import EXERCISES
from ..rep_counting.state_machine import RepCounter
from ..weight_recognition.plate_geometry_check import check_plate_geometry
from ..weight_recognition.vision_classifier import VisionPlateClassifier
from ..weight_recognition.vlm_backend import VLMBackend
from .events import BandPlacementTracker, RestGapSetBoundaryDetector
from .schema import (
    CameraEvent,
    RepCompletedEvent,
    SetCompleteEvent,
    SetFatigueSummaryEvent,
    WeightConfirmedEvent,
)


class RepSession:
    def __init__(
        self,
        exercise_name: str,
        member_id: str,
        station_id: str,
        vlm_backend: Optional[VLMBackend] = None,
        weight_check_every_n_frames: int = 30,
        barbell_detector: Optional[FreeWeightDetector] = None,
        rest_gap_s: float = 20.0,
    ):
        if exercise_name not in EXERCISES:
            raise ValueError(f"Unknown exercise {exercise_name!r} -- choices: {sorted(EXERCISES)}")
        self.exercise = EXERCISES[exercise_name]
        self.exercise_name = exercise_name
        self.member_id = member_id
        self.station_id = station_id
        self.weight_check_every_n_frames = weight_check_every_n_frames
        self.barbell_detector = barbell_detector

        self.counter = RepCounter(self.exercise)
        self.form_scorer = FormScorer()
        self.band_tracker = BandPlacementTracker(member_id)
        self.boundary_detector = RestGapSetBoundaryDetector(rest_gap_s=rest_gap_s)
        self.rep_fusion = RepCountFusion()
        self.set_fatigue_analyzer = SetFatigueAnalyzer()
        self.session_fatigue_tracker = SessionFatigueTracker()
        self.weight_classifier = VisionPlateClassifier(vlm_backend) if vlm_backend is not None else None
        self.rpe_tracker = RPETracker(exercise_name) if barbell_detector is not None else None
        self.bar_tracker: Optional[BarPathTracker] = None

        self._imu_samples: List[IMUSample] = []
        self._set_reps: List[RepFatigueSample] = []
        self._set_start_ts: Optional[float] = None
        self._prev_rep_ts: Optional[float] = None
        self._current_weight_kg: Optional[float] = None
        self._frame_count = 0

        self.initial_events: List[CameraEvent] = []
        band_event = self.band_tracker.event_for(self.exercise)
        if band_event is not None:
            self.initial_events.append(band_event)

    def add_imu_samples(self, samples: List[IMUSample]) -> None:
        """Feed newly-available IMU samples in -- called once with a
        whole loaded file (offline, see ``irix.demo.run_upload``) or
        repeatedly with whatever a live ``IMUStream.poll()`` returns each
        tick (see ``irix.live.station_runner``). Either way, samples just
        accumulate here and get sliced to the right set's time window
        when that set closes -- the two callers don't need any different
        logic downstream of this."""
        self._imu_samples.extend(samples)

    def process_frame(self, frame: np.ndarray, ts: float, person: Optional[PersonPose]) -> List[CameraEvent]:
        """Feed one frame in (plus that frame's resolved pose for this
        member, if any -- a live caller resolves which detected person in
        a multi-person frame is this member elsewhere; this class stays
        agnostic of that). Returns any events this frame produced (often
        none)."""
        events: List[CameraEvent] = []
        self._frame_count += 1

        # -- weight recognition (periodic, real VLM call if configured) --
        if self.weight_classifier is not None and self._frame_count % self.weight_check_every_n_frames == 0:
            reading = self.weight_classifier.read_frame(frame)
            if reading is not None:
                geometry_consistent = None
                geometry_reason = None
                if self.barbell_detector is not None:
                    detections = self.barbell_detector.detect(frame)
                    gcheck = check_plate_geometry(reading.value, detections)
                    geometry_consistent = gcheck.consistent
                    geometry_reason = gcheck.reason
                self._current_weight_kg = reading.value
                events.append(
                    WeightConfirmedEvent(
                        member_id=self.member_id,
                        station_id=self.station_id,
                        exercise=self.exercise_name,
                        weight_kg=reading.value,
                        confidence=reading.confidence,
                        geometry_consistent=geometry_consistent,
                        geometry_check_reason=geometry_reason,
                    )
                )

        # -- barbell centroid tracking (self-calibrates from the first
        # detected plate, then feeds BarPathTracker every frame) --
        if self.barbell_detector is not None:
            detections: List[FreeWeightDetection] = self.barbell_detector.detect(frame)
            if self.bar_tracker is None:
                plate = FreeWeightDetector.largest_plate(detections)
                if plate is not None:
                    calibration = calibrate_from_known_object(
                        plate.pixel_diameter, COMPETITION_BUMPER_PLATE_DIAMETER_MM, self.station_id
                    )
                    self.bar_tracker = BarPathTracker(calibration)
            if self.bar_tracker is not None:
                bar = next((d for d in detections if d.class_label == FreeWeightClass.BARBELL), None)
                if bar is not None:
                    self.bar_tracker.push(ts, bar.centroid_px[1])

        # -- pose -> joint angle -> rep counting -> form scoring --
        if person is None:
            return events
        a_name, v_name, c_name = self.exercise.joint_triplet
        a, v, c = person.xy(a_name), person.xy(v_name), person.xy(c_name)
        if a is None or v is None or c is None:
            return events
        angle = joint_angle(a, v, c)
        rep_event = self.counter.update(angle, timestamp=ts, pose=person)
        if rep_event is None:
            return events

        if self.boundary_detector.observe(rep_event.timestamp) and self._prev_rep_ts is not None:
            events.extend(self._close_set(end_ts=self._prev_rep_ts))
        if self._set_start_ts is None:
            self._set_start_ts = rep_event.concentric_start_timestamp or rep_event.timestamp

        camera_event = RepCompletedEvent(
            member_id=self.member_id,
            station_id=self.station_id,
            exercise=self.exercise_name,
            rep_count=rep_event.rep_number,
            duration_s=rep_event.duration_s,
            peak_velocity_deg_s=rep_event.peak_angular_velocity_deg_s,
            mean_velocity_deg_s=rep_event.mean_angular_velocity_deg_s,
            weight_kg=self._current_weight_kg,
        )
        assessment = self.form_scorer.score_rep(self.exercise_name, rep_event.poses)
        if assessment is not None:
            camera_event.form_score = assessment.score
            camera_event.form_faults = assessment.faults

        if self.bar_tracker is not None and rep_event.concentric_start_timestamp is not None:
            bar_velocity = self.bar_tracker.velocity_for_window(
                rep_event.concentric_start_timestamp, rep_event.timestamp
            )
            if bar_velocity.mean_velocity_m_s is not None:
                camera_event.peak_velocity_m_s = bar_velocity.peak_velocity_m_s
                camera_event.mean_velocity_m_s = bar_velocity.mean_velocity_m_s
                if self.rpe_tracker is not None:
                    rpe_estimate = self.rpe_tracker.estimate(bar_velocity.mean_velocity_m_s)
                    camera_event.estimated_rpe = rpe_estimate.estimated_rpe
                    camera_event.velocity_loss_pct = rpe_estimate.velocity_loss_pct

        events.append(camera_event)
        self._set_reps.append(RepFatigueSample.from_rep_completed_event(camera_event))
        self._prev_rep_ts = rep_event.timestamp
        return events

    def close(self, end_ts: Optional[float] = None) -> List[CameraEvent]:
        """Flush any set still in progress -- call when this member's
        session at the station is ending (a video file ran out, or,
        live, they've stepped away / checked their band back in).
        Harmless to call with no set in progress (returns an empty
        list)."""
        if not self._set_reps:
            return []
        resolved_end_ts = end_ts if end_ts is not None else (self._prev_rep_ts or 0.0)
        return self._close_set(end_ts=resolved_end_ts)

    def _close_set(self, end_ts: float) -> List[CameraEvent]:
        events: List[CameraEvent] = []
        if not self._set_reps:
            return events
        camera_count = len(self._set_reps)
        window_imu = (
            [s for s in self._imu_samples if self._set_start_ts <= s.timestamp <= end_ts]
            if self._imu_samples and self._set_start_ts is not None
            else []
        )
        fused = self.rep_fusion.fuse(
            camera_count=camera_count,
            camera_confidence=self.counter.tracking_confidence,
            imu_samples=window_imu,
            camera_rep_durations=[r.duration_s for r in self._set_reps if r.duration_s],
        )
        events.append(
            SetCompleteEvent(
                member_id=self.member_id,
                station_id=self.station_id,
                exercise=self.exercise_name,
                total_reps=camera_count,
                imu_rep_count=fused.imu_count,
                fused_rep_count=fused.fused_count,
                rep_count_agreement=fused.agreement,
                rep_count_source=fused.source,
            )
        )
        analysis = self.set_fatigue_analyzer.analyze(self.exercise_name, self._set_reps)
        if analysis is not None:
            summary = self.session_fatigue_tracker.add_set(self.member_id, self.exercise_name, analysis)
            trend = summary.set_to_set_velocity_trend_pct
            events.append(
                SetFatigueSummaryEvent(
                    member_id=self.member_id,
                    station_id=self.station_id,
                    exercise=self.exercise_name,
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
        if self.rpe_tracker is not None:
            self.rpe_tracker.reset()
        self._set_reps = []
        self._set_start_ts = None
        return events
