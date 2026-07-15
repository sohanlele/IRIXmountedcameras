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

from typing import Dict, List, Optional

import numpy as np

from ..barbell.calibration import CameraCalibration, calibrate_from_known_object, COMPETITION_BUMPER_PLATE_DIAMETER_MM
from ..barbell.detector import FreeWeightClass, FreeWeightDetection, FreeWeightDetector
from ..barbell.rpe import RPETracker
from ..barbell.tracker import BarPathTracker
from ..fatigue.models import RepFatigueSample
from ..fatigue.session_analysis import SessionFatigueTracker
from ..fatigue.set_analysis import SetFatigueAnalyzer
from ..fusion.clock_sync import ClockSyncEstimator, apply_clock_sync
from ..identity.placement import WristbandPlacementTracker, limb_type_of
from ..fusion.imu import IMUSample
from ..fusion.rep_fusion import RepCountFusion
from ..form.scoring import FormScorer
from ..pose.estimator import PersonPose
from ..pose.geometry import joint_angle
from ..rep_counting.exercises import EXERCISES
from ..rep_counting.state_machine import RepCounter
from ..weight_recognition.plate_color_check import (
    detect_color_plates, estimate_load_from_color_plates, MENS_OLYMPIC_BARBELL_WEIGHT_KG,
)
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
        camera_tilt_deg: float = 0.0,
        camera_tilt_deg_by_camera: Optional[Dict[str, float]] = None,
        start_ts: Optional[float] = None,
        clock_sync_estimator: Optional[ClockSyncEstimator] = None,
        placement_tracker: Optional[WristbandPlacementTracker] = None,
        bar_weight_kg: float = MENS_OLYMPIC_BARBELL_WEIGHT_KG,
    ):
        """``camera_tilt_deg``: the tilt correction to use for whichever
        camera's frames arrive without a more specific entry in
        ``camera_tilt_deg_by_camera`` -- the only one that matters for
        the common single-camera case (``irix.live.station_runner.
        StationSessionRunner``, ``irix.demo.run_upload``), where every
        frame implicitly comes from "the" camera (``process_frame``'s
        ``camera_id`` stays ``None`` throughout).

        ``camera_tilt_deg_by_camera``: per-``camera_id`` tilt overrides
        for a member tracked across more than one camera at once (``irix.
        live.zone_runner.MultiCameraZoneRunner``) -- different physical
        cameras covering the same zone are very plausibly mounted at
        different angles, so one shared tilt value wouldn't be correct
        for all of them. See ``process_frame``'s ``camera_id`` parameter
        and this class's bar-tracking block for how this feeds into
        per-camera self-calibration.

        ``start_ts``: the timestamp this session actually started at
        (e.g. the live caller's ``now``, or an uploaded file's frame-0
        timestamp) -- used only for ``initial_events``' timestamp
        (currently just a possible ``BandPlacementRequiredEvent``).
        Defaults to ``None``, which leaves that event's timestamp at its
        dataclass default (wall-clock ``time.monotonic()`` at
        construction) for backward compatibility with any existing
        caller that doesn't supply one; a caller that cares about
        deterministic replay (``irix.demo.run_live_gym_demo``,
        ``irix.live.station_runner``) should supply it.

        ``clock_sync_estimator``: shared ``irix.fusion.clock_sync.
        ClockSyncEstimator`` this session corrects incoming IMU samples
        against (Phase 3 default production behavior -- see
        ``irix.live.station_runner.StationSessionRunner``, which
        constructs one per session by default). Every ``add_imu_samples``
        call applies the estimator's *current* best offset before
        samples are stored. ``None`` (default) disables this -- no
        correction is applied, unchanged from pre-Phase-3 behavior.

        This class deliberately does NOT auto-populate the estimator's
        observations from its own camera-rep-vs-IMU-peak timestamps. That
        was tried during development and reverted: a camera's
        rep-*completion* timestamp and an IMU counter's acceleration-
        *peak* timestamp mark different phases of the same physical rep
        (e.g. top-of-lift vs. peak concentric acceleration), so pairing
        them conflates a fixed phase offset with actual clock drift and
        silently biases the estimate -- worse than not auto-calibrating
        at all. Populate the estimator from a source that pairs directly
        comparable signals instead, e.g. ``irix.fusion.clock_sync.
        estimate_offset_via_cross_correlation`` against camera-derived
        joint-angle velocity and raw wrist accel magnitude over the same
        window (see that function's tests for a validated example), run
        as an explicit calibration step rather than inferred per-set.

        ``placement_tracker``: shared ``irix.identity.placement.
        WristbandPlacementTracker`` for this member's band (Phase 3
        default production behavior -- see ``irix.live.station_runner.
        StationSessionRunner``, which constructs one per session by
        default). While it reports ``paused`` (mid placement-change, see
        that module's state machine docstring) or its current side's
        limb type doesn't match this exercise's own ``ExerciseConfig.
        band_placement`` requirement, incoming IMU samples are held back
        from fusion entirely rather than trusted -- the "never reuse
        wrist thresholds for ankle data or vice versa" rule made
        concrete: an exercise needing ankle data gets camera-only
        behavior (the existing, already-tested graceful degradation
        path) for as long as the band's confirmed placement doesn't
        actually match, rather than silently fusing mismatched-limb
        samples. ``None`` (default) disables this -- every sample is
        trusted immediately, unchanged from pre-Phase-3 behavior.

        ``bar_weight_kg``: this station's actual bar weight (Priority 7's
        "equipment metadata" -- a women's bar, technique bar, or fixed
        machine-arm equivalent is *not* always the men's Olympic 20kg
        default ``irix.weight_recognition.plate_color_check.
        estimate_load_from_color_plates`` otherwise silently assumes).
        Defaults to that same 20kg constant for backward compatibility
        with every existing caller that doesn't know or care.
        """
        if exercise_name not in EXERCISES:
            raise ValueError(f"Unknown exercise {exercise_name!r} -- choices: {sorted(EXERCISES)}")
        self.exercise = EXERCISES[exercise_name]
        self.exercise_name = exercise_name
        self.member_id = member_id
        self.station_id = station_id
        self.weight_check_every_n_frames = weight_check_every_n_frames
        self.barbell_detector = barbell_detector
        self.camera_tilt_deg = camera_tilt_deg
        self.camera_tilt_deg_by_camera = camera_tilt_deg_by_camera or {}
        self.bar_weight_kg = bar_weight_kg

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
        # One CameraCalibration per camera_id (None is the key for the
        # common single-camera case), each self-calibrated independently
        # the first time *that* camera's frame shows a detected plate --
        # see process_frame's bar-tracking block. Different physical
        # cameras have different actual px-per-mm scales even for the
        # same plate, so a calibration derived from camera A's pixels
        # must never be applied to camera B's.
        self._bar_calibrations: Dict[Optional[str], CameraCalibration] = {}

        self.clock_sync_estimator = clock_sync_estimator
        self.placement_tracker = placement_tracker

        self._imu_samples: List[IMUSample] = []
        self._set_reps: List[RepFatigueSample] = []
        self._set_start_ts: Optional[float] = None
        self._prev_rep_ts: Optional[float] = None
        self._current_weight_kg: Optional[float] = None
        self._frame_count = 0

        self.initial_events: List[CameraEvent] = []
        band_event = self.band_tracker.event_for(self.exercise, timestamp=start_ts)
        if band_event is not None:
            self.initial_events.append(band_event)

    def _camera_tilt_for(self, camera_id: Optional[str]) -> float:
        """The tilt-correction angle to use when self-calibrating a frame
        that came from ``camera_id`` -- ``camera_tilt_deg_by_camera``'s
        entry for that camera_id if one was given, else the single
        shared ``camera_tilt_deg`` (which is exactly right for the
        common single-camera case, where ``camera_id`` stays ``None``
        throughout)."""
        return self.camera_tilt_deg_by_camera.get(camera_id, self.camera_tilt_deg)

    def add_imu_samples(self, samples: List[IMUSample]) -> None:
        """Feed newly-available IMU samples in -- called once with a
        whole loaded file (offline, see ``irix.demo.run_upload``) or
        repeatedly with whatever a live ``IMUStream.poll()`` returns each
        tick (see ``irix.live.station_runner``). Either way, samples just
        accumulate here and get sliced to the right set's time window
        when that set closes -- the two callers don't need any different
        logic downstream of this.

        If ``clock_sync_estimator`` was supplied, every incoming batch is
        corrected against its *current* best offset estimate before being
        stored (a no-op, offset 0.0, until the first set closes and
        produces an observation -- see ``_close_set``).

        If ``placement_tracker`` was supplied, every batch is first fed
        to it (so it can detect settling/calibration progress even while
        its own samples aren't yet trustworthy for fusion -- see that
        class's ``feed_samples``); the batch is then stored only if the
        tracker reports both un-paused *and* currently at a side whose
        limb type matches this exercise's ``band_placement`` requirement.
        Otherwise the batch is dropped entirely for fusion purposes (not
        buffered for "later" -- once a placement change genuinely
        happens, whatever was captured mid-change or on the wrong limb
        was never a usable signal for this exercise in the first place;
        see ``irix.identity.placement``'s module docstring)."""
        if not samples:
            return
        if self.placement_tracker is not None:
            self.placement_tracker.feed_samples(samples)
            if self.placement_tracker.paused:
                return
            if self.placement_tracker.limb_type != self.exercise.band_placement:
                return
        if self.clock_sync_estimator is not None:
            estimate = self.clock_sync_estimator.estimate()
            if estimate.n_observations > 0:
                samples = apply_clock_sync(samples, estimate)
        self._imu_samples.extend(samples)

    def process_frame(
        self,
        frame: np.ndarray,
        ts: float,
        person: Optional[PersonPose],
        camera_id: Optional[str] = None,
    ) -> List[CameraEvent]:
        """Feed one frame in (plus that frame's resolved pose for this
        member, if any -- a live caller resolves which detected person in
        a multi-person frame is this member elsewhere; this class stays
        agnostic of that). Returns any events this frame produced (often
        none).

        ``camera_id`` identifies which physical camera this frame came
        from -- ``None`` (the default) for the common single-camera case.
        A member tracked across more than one overlapping-FOV camera at
        once (``irix.live.zone_runner.MultiCameraZoneRunner``) will call
        this once per camera per tick with each camera's own id, so that
        the bar-tracking block below self-calibrates and reads pixels
        separately per camera rather than misapplying one camera's
        px-per-mm scale to another's frames."""
        events: List[CameraEvent] = []
        self._frame_count += 1

        # -- weight recognition (periodic; color-plate check always runs
        # since it needs no model/API key, VLM only if configured) --
        if self._frame_count % self.weight_check_every_n_frames == 0:
            color_estimate = estimate_load_from_color_plates(detect_color_plates(frame), bar_weight_kg=self.bar_weight_kg)
            vlm_reading = self.weight_classifier.read_frame(frame) if self.weight_classifier is not None else None

            if vlm_reading is not None:
                # VLM is the primary read when configured; color-plate
                # detection becomes a cross-check alongside the existing
                # geometry one, same "independent corroborating signal"
                # role plate_color_check.py's own docstring describes.
                geometry_consistent = None
                geometry_reason = None
                if self.barbell_detector is not None:
                    detections = self.barbell_detector.detect(frame)
                    gcheck = check_plate_geometry(vlm_reading.value, detections)
                    geometry_consistent = gcheck.consistent
                    geometry_reason = gcheck.reason
                color_consistent = None
                color_reason = None
                if color_estimate.total_weight_kg is not None:
                    color_consistent = abs(color_estimate.total_weight_kg - vlm_reading.value) < 1e-6 or                         abs(color_estimate.total_weight_kg - vlm_reading.value) <= 0.05 * vlm_reading.value
                    color_reason = (
                        f"color_plate_read={color_estimate.total_weight_kg}kg vs vlm_read={vlm_reading.value}kg"
                    )
                self._current_weight_kg = vlm_reading.value
                events.append(
                    WeightConfirmedEvent(
                        member_id=self.member_id,
                        station_id=self.station_id,
                        exercise=self.exercise_name,
                        weight_kg=vlm_reading.value,
                        confidence=vlm_reading.confidence,
                        geometry_consistent=geometry_consistent,
                        geometry_check_reason=geometry_reason,
                        color_check_consistent=color_consistent,
                        color_check_reason=color_reason,
                        method="vlm",
                        timestamp=ts,  # this frame's own timestamp, not wall-clock
                    )
                )
            elif color_estimate.total_weight_kg is not None and color_estimate.confidence >= 0.5:
                # No VLM configured (no API key / not wired for this
                # station) -- color-coded-bumper-plate detection is a
                # complete, zero-training method on its own for
                # equipment that follows the IWF color standard. Never
                # fabricated: estimate_load_from_color_plates already
                # returns total_weight_kg=None for anything it can't
                # confidently, symmetrically pair (see that function's
                # docstring), so reaching here means a real, confident
                # read -- not a guess.
                geometry_consistent = None
                geometry_reason = None
                if self.barbell_detector is not None:
                    detections = self.barbell_detector.detect(frame)
                    gcheck = check_plate_geometry(color_estimate.total_weight_kg, detections)
                    geometry_consistent = gcheck.consistent
                    geometry_reason = gcheck.reason
                self._current_weight_kg = color_estimate.total_weight_kg
                events.append(
                    WeightConfirmedEvent(
                        member_id=self.member_id,
                        station_id=self.station_id,
                        exercise=self.exercise_name,
                        weight_kg=color_estimate.total_weight_kg,
                        confidence=color_estimate.confidence,
                        geometry_consistent=geometry_consistent,
                        geometry_check_reason=geometry_reason,
                        color_check_consistent=True,
                        color_check_reason=color_estimate.reason,
                        method="color_plate",
                        timestamp=ts,  # this frame's own timestamp, not wall-clock
                    )
                )

        # -- barbell centroid tracking (self-calibrates from the first
        # detected plate, per camera_id, then feeds BarPathTracker every
        # frame using *that camera's own* calibration -- see
        # process_frame's camera_id docstring and BarPathTracker.push's
        # docstring for why one continuous tracker/buffer still works
        # correctly across a camera switch even though calibration is
        # per-camera) --
        if self.barbell_detector is not None:
            detections: List[FreeWeightDetection] = self.barbell_detector.detect(frame)
            calibration = self._bar_calibrations.get(camera_id)
            if calibration is None:
                plate = FreeWeightDetector.largest_plate(detections)
                if plate is not None:
                    calibration = calibrate_from_known_object(
                        plate.pixel_diameter, COMPETITION_BUMPER_PLATE_DIAMETER_MM, self.station_id,
                        camera_tilt_deg=self._camera_tilt_for(camera_id),
                    )
                    self._bar_calibrations[camera_id] = calibration
                    if self.bar_tracker is None:
                        self.bar_tracker = BarPathTracker(calibration)
            if self.bar_tracker is not None and calibration is not None:
                bar = next((d for d in detections if d.class_label == FreeWeightClass.BARBELL), None)
                if bar is not None:
                    self.bar_tracker.push(ts, bar.centroid_px[1], calibration=calibration)

        # -- pose -> joint angle -> rep counting -> form scoring --
        if person is None:
            return events
        a_name, v_name, c_name = self.exercise.joint_triplet
        # Prefer a triangulated 3D joint angle when this pose came from
        # multi-view fusion (irix.pose.multiview.triangulate_pose, wired
        # in via MultiCameraZoneRunner) and all 3 needed keypoints
        # triangulated this tick -- a 3D angle isn't subject to any one
        # camera's foreshortening/self-occlusion, so it's strictly more
        # accurate than a single 2D view's angle whenever it's
        # available. Falls back to the ordinary 2D angle otherwise
        # (always true for the single-camera case, and for multi-view
        # ticks where fewer than 2 cameras covered one of these 3
        # specific keypoints) -- see PersonPose.xyz's docstring.
        a, v, c = person.xyz(a_name), person.xyz(v_name), person.xyz(c_name)
        if a is None or v is None or c is None:
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
            # rep_event.timestamp, not the dataclass's wall-clock default --
            # this is the actual detected-rep time (deterministic under replay
            # for run_upload's frame_index/fps timestamps or a live caller's
            # injected clock), see docs/VALIDATION.md on deterministic replay.
            timestamp=rep_event.timestamp,
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
                timestamp=end_ts,  # the set's actual close time, not wall-clock
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
                    timestamp=end_ts,  # same set-close time as SetCompleteEvent, not wall-clock
                )
            )
        if self.rpe_tracker is not None:
            self.rpe_tracker.reset()
        self._set_reps = []
        self._set_start_ts = None
        return events
