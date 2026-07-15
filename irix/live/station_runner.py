"""Ties together everything a real, continuously-running station needs
that a single ``RepSession`` doesn't provide on its own: knowing *whose*
session this even is, when one starts and ends, and -- since more than
one checked-out member can legitimately be at the same station at once
(a shared bench, a crowded curl rack) -- *which* detected person is
which member.

The pieces this composes, none of which talk to each other anywhere else
in this repo:

- ``irix.identity.checkout.CheckoutRegistry`` -- resolves a BLE-observed
  wristband id to the account it's currently checked out to. Without
  this, "member_id" is just a string a caller has to already know; a real
  station only ever learns a band's BLE id, not an account id.
- ``irix.identity.ble_pairing.BLEReading`` (now carrying
  ``wristband_id``) -- who's currently near this station's radio.
- ``irix.live.camera_source.ReconnectingFrameSource`` -- frames that keep
  coming even if the camera drops and reconnects.
- ``irix.fusion.imu_stream.IMUStream`` -- live samples for whichever
  band(s) are currently tracked, once a session is active.
- ``irix.live.disambiguation.CrowdedGroupDisambiguator`` (wraps
  ``irix.identity.motion_correlation.MotionCorrelationResolver`` -- the
  same class ``irix.topology.handoff.GymCoordinator.
  disambiguate_by_motion`` delegates to in the synthetic demo) -- when
  *more than one* checked-out band is present at once, resolves which
  camera-detected skeleton belongs to which band by correlating each
  candidate's wristband IMU signal against each detected person's wrist
  motion. This class owns exactly one ``CrowdedGroupDisambiguator``
  instance (one camera, one detection source); ``irix.live.zone_runner.
  MultiCameraZoneRunner`` is the multi-camera generalization, one
  instance per camera sharing the same candidate group.
- ``irix.pipeline.rep_session.RepSession`` -- the actual per-member
  pipeline (already shared with ``irix.demo.run_upload``), now one
  instance per *currently-present* band, not just one for the whole
  station.

What "session" means here: a checked-out band showing up starts one (a
fresh ``RepSession``, and a request for that band's live ``IMUStream``);
a band no longer showing up for ``presence_timeout_s`` ends it (flush
whatever set was in progress, release the ``RepSession``). This mirrors
the "run_upload waits for a set to naturally show a gap, closes it" logic
from ``RestGapSetBoundaryDetector``, just one level up -- that class
decides when a *set* ends within an active session; this class decides
when the *session itself* ends.

**The crowded-station case.** If exactly one band is present, routing is
trivial: whatever person the camera detects is that member. If *more*
than one checked-out band is present at once, ``irix.identity.
ble_pairing``'s RSSI proximity alone can't say which detected skeleton is
which -- see ``irix.identity.motion_correlation``'s module docstring for
why and how. This class buffers a short window of (detected-person-slot,
pose) and (wristband, raw IMU samples) data, calls
``MotionCorrelationResolver`` once the window is full, and then routes
each detected person to their resolved session for as long as the same
group of bands stays present. Two things worth being explicit about:

1. While a window is buffering (or for any slot that never resolves
   confidently), frames for the still-ambiguous group aren't attributed
   to anyone -- reps genuinely happening during that short window are
   missed rather than guessed at. The window is short
   (``disambiguation_window_frames`` frames, a few seconds at a typical
   camera fps) and this only affects the *crowded* case, not the common
   single-lifter one.
2. Routing assumes a detected person's position in ``PoseEstimator.
   estimate()``'s returned list stays consistent for the duration of one
   buffering window -- ``PoseEstimator`` doesn't run persistent
   cross-frame object tracking (``PersonPose.track_id`` is just that
   frame's list index, not a stable id -- see ``irix/pose/estimator.py``).
   Reasonable for a short window with a static camera; not a guarantee
   over a long session, which is why re-resolution happens fresh every
   time the present-band group actually changes rather than trusting one
   resolution indefinitely.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from ..barbell.calibration import MENS_OLYMPIC_BARBELL_WEIGHT_KG
from ..barbell.detector import FreeWeightDetector
from ..fusion.clock_sync import ClockSyncEstimator, apply_clock_sync
from ..fusion.imu import IMUSample
from ..fusion.imu_stream import IMUStream
from ..identity.ble_pairing import BLEReading
from ..identity.checkout import CheckoutRegistry
from ..identity.motion_correlation import MotionCorrelationResolver
from ..identity.placement import BandSide, WristbandPlacementTracker
from ..pipeline.rep_session import RepSession
from ..pipeline.schema import BandPlacementConfirmedEvent, CameraEvent, TrackingLostEvent, TrackingRecoveredEvent
from ..pose.calibration import CalibrationProfile
from ..recording.session_recorder import SessionRecorder
from ..weight_recognition.vlm_backend import VLMBackend
from .disambiguation import CrowdedGroupDisambiguator


class StationSessionRunner:
    def __init__(
        self,
        station_id: str,
        exercise_name: str,
        checkout_registry: CheckoutRegistry,
        frame_source,  # irix.live.camera_source.ReconnectingFrameSource, or anything with a .frames() generator
        ble_reader: Callable[[], List[BLEReading]],
        imu_stream_factory: Optional[Callable[[str], IMUStream]] = None,
        pose_estimator=None,
        presence_timeout_s: float = 5.0,
        vlm_backend: Optional[VLMBackend] = None,
        weight_check_every_n_frames: int = 30,
        barbell_detector: Optional[FreeWeightDetector] = None,
        rest_gap_s: float = 20.0,
        bar_weight_kg: float = MENS_OLYMPIC_BARBELL_WEIGHT_KG,
        on_events: Optional[Callable[[List[CameraEvent]], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        motion_resolver: Optional[MotionCorrelationResolver] = None,
        disambiguation_window_frames: int = 60,
        calibration_profile: Optional[CalibrationProfile] = None,
        tracking_lost_after_frames: int = 15,
        session_recorder: Optional[SessionRecorder] = None,
    ):
        """``ble_reader``: called once per frame tick (in ``run_forever``
        only -- ``tick()`` called directly, e.g. by ``irix.live.
        gym_runner.GymSessionRunner``, bypasses this), returns whatever
        ``BLEReading``s (with ``wristband_id`` set) this station's radio
        currently sees -- real hardware/firmware detail, injected here as
        a callable so this class doesn't need to know how.

        ``imu_stream_factory``: given a wristband id, returns an
        ``IMUStream`` for it (real deployment: a ``LiveBLEIMUStream``
        once that exists; tests: a fake). ``None`` means run camera-only,
        same as calling ``irix.demo.run_upload`` without ``imu_path``.

        ``on_events``: called with each frame's newly-produced events
        (often an empty list) -- the hook a real deployment would use to
        push events to ``irix.pipeline.aggregator.Aggregator`` /
        ``CloudSync`` as they happen, rather than collecting everything
        in memory for an unbounded 24/7 run the way ``run_upload``
        collects a whole (bounded, pre-recorded) video's events into one
        list.

        ``clock``: defaults to ``time.monotonic``, the correct choice for
        a real run (see ``run_forever``'s docstring). Injectable so tests
        can drive ``presence_timeout_s`` deterministically instead of
        depending on real wall-clock time elapsing during a fast test
        loop.

        ``motion_resolver``/``disambiguation_window_frames``: how a
        crowded station (2+ checked-out bands present at once) gets
        disambiguated -- see the module docstring. Irrelevant whenever at
        most one band is present, which is the common case.

        ``calibration_profile``: this station's install-time ``irix.
        pose.calibration.CalibrationProfile`` (checkerboard intrinsics,
        see that module), applied to undistort every frame before it
        reaches ``pose_estimator`` -- the "Camera Calibration" stage of
        the authoritative pipeline (Camera Streams -> Pose Estimation ->
        Tracking -> Camera Calibration -> ...). ``None`` (default) skips
        undistortion, unchanged from pre-Phase-3 behavior -- correct for
        any camera whose lens distortion hasn't been calibrated yet
        rather than silently running uncorrected pose estimation through
        a calibration step that doesn't exist for it.

        ``tracking_lost_after_frames``: how many *consecutive* ticks with
        nobody detected (the common single-candidate path only -- see
        ``TrackingLostEvent``'s own docstring for the crowded-station
        gap) before a ``TrackingLostEvent`` fires, paired with a
        ``TrackingRecoveredEvent`` once someone is detected again.
        Deliberately a streak, not a single missed frame -- an ordinary
        detector miss on one frame (motion blur, brief self-occlusion)
        shouldn't itself be reported as a tracking-loss incident.

        ``session_recorder``: an ``irix.recording.session_recorder.
        SessionRecorder`` (Priority 8) to feed every frame/IMU
        batch/event this station produces into, for later deterministic
        replay or algorithm comparison. Opt-in and ``None`` by default --
        recording (especially raw video, itself further opt-in on the
        recorder's own ``save_raw_frames``) is a deliberate choice a
        deployment makes for a specific station/time window, not default
        production behavior. One recorder covers this station's whole
        run, spanning however many different members' sessions come and
        go -- every recorded event/sample is already member_id-tagged
        (see each event/IMU sample's own fields), so slicing a single
        member's activity back out happens at analysis time, not by
        needing a fresh recorder per member session.
        """
        self.station_id = station_id
        self.exercise_name = exercise_name
        self.checkout_registry = checkout_registry
        self.frame_source = frame_source
        self.ble_reader = ble_reader
        self.imu_stream_factory = imu_stream_factory
        self.pose_estimator = pose_estimator
        self.calibration_profile = calibration_profile
        self.tracking_lost_after_frames = tracking_lost_after_frames
        self.session_recorder = session_recorder
        self.presence_timeout_s = presence_timeout_s
        self._clock = clock or time.monotonic
        self._session_kwargs = dict(
            vlm_backend=vlm_backend,
            weight_check_every_n_frames=weight_check_every_n_frames,
            barbell_detector=barbell_detector,
            rest_gap_s=rest_gap_s,
            bar_weight_kg=bar_weight_kg,
        )
        self._on_events = on_events or (lambda events: None)
        self._disambiguator = CrowdedGroupDisambiguator(
            motion_resolver=motion_resolver, disambiguation_window_frames=disambiguation_window_frames,
        )

        self._sessions: Dict[str, RepSession] = {}
        self._imu_streams: Dict[str, Optional[IMUStream]] = {}
        self._last_seen: Dict[str, float] = {}
        self._consecutive_missed_frames: Dict[str, int] = {}
        self._tracking_lost: Dict[str, bool] = {}
        self._tracking_lost_since: Dict[str, float] = {}
        # One ClockSyncEstimator per currently-open session (Phase 3
        # default production behavior) -- correction is applied to every
        # add_imu_samples() call from the moment it exists, but it starts
        # with zero observations (a no-op) until something calls
        # calibrate_wristband_clock() for that band. See that method's
        # docstring for why this repo doesn't auto-derive observations
        # from RepSession's own per-set rep timestamps.
        self._clock_sync_estimators: Dict[str, ClockSyncEstimator] = {}
        # One WristbandPlacementTracker per currently-open session (Phase
        # 3 default production behavior) -- see irix.identity.placement's
        # module docstring for the full state machine. Starts assuming
        # LEFT_WRIST (that class's own default) until something calls
        # request_wristband_placement_change() for that band.
        self._placement_trackers: Dict[str, WristbandPlacementTracker] = {}

    def _ensure_estimator(self):
        if self.pose_estimator is None:
            # Default production path only -- a caller that injects its
            # own pose_estimator (every existing test/demo that needs
            # deterministic synthetic output) keeps getting exactly what
            # it passed in, untouched. Real deployments get persistent
            # track_id (irix.pose.tracker.PoseTracker, ByteTrack-derived)
            # by default, not as an opt-in extra step -- see
            # irix/pose/tracker.py's module docstring for why that
            # matters for a crowded, multi-person station.
            from ..pose.estimator import PoseEstimator
            from ..pose.tracker import TrackedPoseEstimator

            self.pose_estimator = TrackedPoseEstimator(PoseEstimator())
        return self.pose_estimator

    def _resolve_present_wristbands(self) -> List[str]:
        """Every band, among this tick's local BLE readings, that's
        actually checked out to an account right now -- an unchecked-out
        band (someone wearing a band from a different gym / a past visit
        that never got returned) is never tracked, same as a real front
        desk wouldn't start a session for a band it doesn't currently
        have handed out. May return more than one -- see the module
        docstring for how that's handled."""
        readings = self.ble_reader()
        return list({
            r.wristband_id for r in readings
            if r.wristband_id is not None and self.checkout_registry.is_checked_out(r.wristband_id)
        })

    def _start_session(self, wristband_id: str, now: float) -> None:
        member_id = self.checkout_registry.resolve_member(wristband_id)
        assert member_id is not None  # guaranteed by callers checking is_checked_out first
        clock_sync_estimator = ClockSyncEstimator()
        placement_tracker = WristbandPlacementTracker(wristband_id)
        session = RepSession(
            exercise_name=self.exercise_name,
            member_id=member_id,
            station_id=self.station_id,
            start_ts=now,
            clock_sync_estimator=clock_sync_estimator,
            placement_tracker=placement_tracker,
            **self._session_kwargs,
        )
        self._emit(session.initial_events)
        self._sessions[wristband_id] = session
        self._clock_sync_estimators[wristband_id] = clock_sync_estimator
        self._placement_trackers[wristband_id] = placement_tracker
        self._imu_streams[wristband_id] = self.imu_stream_factory(wristband_id) if self.imu_stream_factory else None

    def _end_session(self, wristband_id: str, end_ts: float) -> None:
        session = self._sessions.pop(wristband_id, None)
        self._imu_streams.pop(wristband_id, None)
        self._last_seen.pop(wristband_id, None)
        self._clock_sync_estimators.pop(wristband_id, None)
        self._placement_trackers.pop(wristband_id, None)
        self._consecutive_missed_frames.pop(wristband_id, None)
        self._tracking_lost.pop(wristband_id, None)
        self._tracking_lost_since.pop(wristband_id, None)
        if session is not None:
            self._emit(session.close(end_ts=end_ts))
        # A session ending always coincides with a present-set change,
        # which the disambiguator's own route() already detects and
        # resets for on its next call -- resetting here too is a harmless,
        # slightly-earlier-than-strictly-necessary reset, not a behavior
        # change (see CrowdedGroupDisambiguator.reset()'s docstring).
        self._disambiguator.reset()

    def calibrate_wristband_clock(
        self, wristband_id: str, offset_s: float, confidence: float, at_time: Optional[float] = None,
    ) -> bool:
        """Record a clock-offset observation for a currently-open
        session's wristband, so its ``RepSession``'s ``add_imu_samples``
        starts correcting incoming samples against it. Returns whether a
        session for this band was actually open to receive it.

        Deliberately a thin pass-through to ``ClockSyncEstimator.
        add_observation`` rather than something this class derives on its
        own: this repo does NOT auto-derive clock-sync observations from
        camera-rep-vs-IMU-peak timestamp pairing (see ``irix.pipeline.
        rep_session.RepSession``'s ``__init__`` docstring and ``irix.
        fusion.clock_sync.estimate_offset_from_paired_events``'s
        docstring for the phase-offset bug that approach has). The
        caller is responsible for computing a trustworthy ``(offset_s,
        confidence)`` pair from directly-comparable signals -- e.g.
        ``irix.fusion.clock_sync.estimate_offset_via_cross_correlation``
        against camera-tracked wrist-keypoint vertical velocity and raw
        wristband vertical accel over the same window (see that
        function's tests; wiring an automatic per-tick caller for this
        is tracked in ``docs/TODO.md``, not yet built)."""
        estimator = self._clock_sync_estimators.get(wristband_id)
        if estimator is None:
            return False
        estimator.add_observation(
            at_time=at_time if at_time is not None else self._clock(), offset_s=offset_s, confidence=confidence,
        )
        return True

    def _emit(self, events: List[CameraEvent]) -> None:
        """Every event this station produces flows through here --
        pushed to ``on_events`` (unchanged, still the primary hook) and,
        if a ``session_recorder`` is configured, also logged there
        (Priority 8). A thin wrapper specifically so every existing
        `self._on_events(...)` call site didn't need its own separate
        recorder call bolted on next to it."""
        self._on_events(events)
        if self.session_recorder is not None and events:
            self.session_recorder.record_events(events)

    def _track_tracking_loss(self, wristband_id: str, person_detected: bool, now: float) -> List[CameraEvent]:
        """Streak-based TrackingLostEvent/TrackingRecoveredEvent (see
        __init__'s ``tracking_lost_after_frames`` docstring) for the
        common single-candidate path. member_id is resolved fresh here
        (not cached) since it never changes for an open session and this
        avoids threading it through every caller."""
        events: List[CameraEvent] = []
        if person_detected:
            self._consecutive_missed_frames[wristband_id] = 0
            if self._tracking_lost.get(wristband_id):
                self._tracking_lost[wristband_id] = False
                since = self._tracking_lost_since.pop(wristband_id, now)
                session = self._sessions.get(wristband_id)
                member_id = session.member_id if session is not None else wristband_id
                events.append(
                    TrackingRecoveredEvent(
                        member_id=member_id, station_id=self.station_id,
                        gap_duration_s=max(0.0, now - since), timestamp=now,
                    )
                )
            return events
        missed = self._consecutive_missed_frames.get(wristband_id, 0) + 1
        self._consecutive_missed_frames[wristband_id] = missed
        if missed >= self.tracking_lost_after_frames and not self._tracking_lost.get(wristband_id):
            self._tracking_lost[wristband_id] = True
            self._tracking_lost_since[wristband_id] = now
            session = self._sessions.get(wristband_id)
            member_id = session.member_id if session is not None else wristband_id
            events.append(
                TrackingLostEvent(
                    member_id=member_id, station_id=self.station_id,
                    consecutive_missed_frames=missed, timestamp=now,
                )
            )
        return events

    def request_wristband_placement_change(
        self, wristband_id: str, to_side: BandSide, at_time: Optional[float] = None,
    ) -> bool:
        """Backend entry point for "this member's band has been moved"
        (Priority 4/Section 5.2) -- called by a future IRIX app or
        front-desk console once a member has been instructed to move
        their band and has done so (this repo deliberately does not
        build that app -- see ``irix.identity.placement``'s module
        docstring). Returns whether a session for this band was open to
        receive the request.

        From this call until the tracker confirms the new side (see
        ``WristbandPlacementTracker``'s state machine), that band's
        ``RepSession.add_imu_samples`` drops every incoming batch rather
        than fusing it -- both the settling/fastening-motion period and
        any exercise this member is doing in the meantime that needs a
        limb type the band isn't confirmed at yet."""
        tracker = self._placement_trackers.get(wristband_id)
        if tracker is None:
            return False
        tracker.request_change(to_side, at_time=at_time if at_time is not None else self._clock())
        return True

    def tick(self, frame, now: float, present_wristband_ids: List[str]) -> None:
        """One frame's worth of work, given an *already-resolved* list of
        wristbands present at this station right now (0, 1, or more).
        ``run_forever`` below resolves that locally, once per frame, via
        ``self.ble_reader()`` -- correct for a single isolated station.
        ``irix.live.gym_runner.GymSessionRunner`` instead resolves
        presence gym-wide (cross-station handoff hysteresis) and calls
        this directly, once per station per gym-wide tick, bypassing this
        station's own ``ble_reader`` entirely.

        Routing (single-vs-ambiguous) is decided from *this tick's*
        ``present_wristband_ids``, not from how many ``RepSession``s
        happen to still be open. A session that just stopped being
        reported keeps running through its ``presence_timeout_s`` grace
        period (tolerating a brief radio dropout, same as before this
        class supported more than one session at once) but simply
        doesn't receive this tick's detected person -- so an ordinary
        one-after-another handoff at a station (someone leaves, someone
        else steps up right after) still routes unambiguously to whoever
        is actually seen *this* tick, and only genuine same-tick
        multi-presence (two+ bands reported at once) triggers the
        buffering/disambiguation path below.
        """
        present_set = set(present_wristband_ids)

        for wristband_id in list(self._sessions.keys()):
            if wristband_id in present_set:
                continue
            last_seen = self._last_seen.get(wristband_id)
            if last_seen is not None and (now - last_seen) >= self.presence_timeout_s:
                self._end_session(wristband_id, end_ts=last_seen)

        for wristband_id in present_wristband_ids:
            self._last_seen[wristband_id] = now
            if wristband_id not in self._sessions:
                self._start_session(wristband_id, now)

        if not self._sessions:
            return

        if self.calibration_profile is not None:
            frame = self.calibration_profile.undistort_frame(frame)
        if self.session_recorder is not None:
            self.session_recorder.record_frame(frame, now)
        estimator = self._ensure_estimator()
        people = estimator.estimate(frame)

        # IMU fusion keeps running for every open session, including one
        # lingering in its grace period -- fusion shouldn't drop samples
        # just because a session isn't receiving a routed pose this tick.
        polled: Dict[str, List[IMUSample]] = {}
        for wristband_id, imu_stream in self._imu_streams.items():
            samples = imu_stream.poll() if imu_stream is not None else []
            polled[wristband_id] = samples
            if self.session_recorder is not None and samples:
                self.session_recorder.record_imu_samples(samples)
            tracker = self._placement_trackers.get(wristband_id)
            side_before = tracker.current_side if tracker is not None else None
            self._sessions[wristband_id].add_imu_samples(samples)
            # add_imu_samples() itself drives the placement tracker (see
            # RepSession's docstring) -- detect a same-tick STABLE
            # confirmation here so it becomes a real, observable event
            # rather than a change only visible by polling tracker state.
            if tracker is not None and tracker.current_side != side_before:
                self._emit([
                    BandPlacementConfirmedEvent(
                        wristband_id=wristband_id,
                        from_side=side_before.value if side_before is not None else "unknown",
                        to_side=tracker.current_side.value,
                        timestamp=now,
                    )
                ])

        if len(present_set) <= 1:
            self._disambiguator.reset()
            if present_set:
                wristband_id = next(iter(present_set))
                session = self._sessions.get(wristband_id)
                if session is not None:
                    person = people[0] if people else None
                    self._emit(self._track_tracking_loss(wristband_id, person is not None, now))
                    self._emit(session.process_frame(frame, now, person))
            return

        # Identity/motion-correlation disambiguation (irix.identity.
        # motion_correlation) needs clock-synchronized, genuinely-body-
        # motion IMU samples to correlate meaningfully against camera
        # keypoint motion -- Priority 5's "fuse ... clock synchronization"
        # requirement made concrete: apply each band's current best clock
        # offset (same correction RepSession.add_imu_samples applies for
        # fusion) before handing samples to the disambiguator, and
        # withhold a band's samples entirely while its placement tracker
        # reports mid-change (fastening/carrying motion is not this
        # member's body motion signal -- see irix.identity.placement).
        synced_polled: Dict[str, List[IMUSample]] = {}
        for wristband_id, samples in polled.items():
            tracker = self._placement_trackers.get(wristband_id)
            if tracker is not None and tracker.paused:
                synced_polled[wristband_id] = []
                continue
            estimator = self._clock_sync_estimators.get(wristband_id)
            if estimator is not None and samples:
                sync_estimate = estimator.estimate()
                if sync_estimate.n_observations > 0:
                    samples = apply_clock_sync(samples, sync_estimate)
            synced_polled[wristband_id] = samples

        routed = self._disambiguator.route(
            now, frozenset(present_set), people, synced_polled, self.checkout_registry.resolve_member,
        )
        for wristband_id, pose in routed.items():
            session = self._sessions.get(wristband_id)
            if session is not None:
                self._emit(session.process_frame(frame, now, pose))

    def run_forever(self, max_frames: Optional[int] = None) -> None:
        """Runs until ``frame_source`` stops yielding (only happens in
        tests, via ``max_frames`` -- a real ``ReconnectingFrameSource``
        yields forever). Each frame: resolve BLE presence locally (see
        ``_resolve_present_wristbands``) and hand off to ``tick()``.
        Runtime timestamps come from ``time.monotonic()``, the correct
        clock here (unlike ``irix.demo.run_upload``'s ``frame_index /
        fps``) since this is genuinely live: frame arrival time *is*
        wall-clock time for a live camera, and BLE presence/timeout logic
        has to be measured against the same clock the radio readings
        themselves arrive on.
        """
        for frame in self.frame_source.frames(max_frames=max_frames):
            now = self._clock()
            present_wristband_ids = self._resolve_present_wristbands()
            self.tick(frame, now, present_wristband_ids)

    def close(self) -> None:
        """Flush every active session and release the frame source --
        call on shutdown."""
        for wristband_id in list(self._sessions.keys()):
            last_seen = self._last_seen.get(wristband_id)
            self._end_session(wristband_id, end_ts=last_seen if last_seen is not None else self._clock())
        close_fn = getattr(self.frame_source, "close", None)
        if close_fn is not None:
            close_fn()
