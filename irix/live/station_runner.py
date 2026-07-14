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

from ..barbell.detector import FreeWeightDetector
from ..fusion.imu import IMUSample
from ..fusion.imu_stream import IMUStream
from ..identity.ble_pairing import BLEReading
from ..identity.checkout import CheckoutRegistry
from ..identity.motion_correlation import MotionCorrelationResolver
from ..pipeline.rep_session import RepSession
from ..pipeline.schema import CameraEvent
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
        on_events: Optional[Callable[[List[CameraEvent]], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        motion_resolver: Optional[MotionCorrelationResolver] = None,
        disambiguation_window_frames: int = 60,
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
        """
        self.station_id = station_id
        self.exercise_name = exercise_name
        self.checkout_registry = checkout_registry
        self.frame_source = frame_source
        self.ble_reader = ble_reader
        self.imu_stream_factory = imu_stream_factory
        self.pose_estimator = pose_estimator
        self.presence_timeout_s = presence_timeout_s
        self._clock = clock or time.monotonic
        self._session_kwargs = dict(
            vlm_backend=vlm_backend,
            weight_check_every_n_frames=weight_check_every_n_frames,
            barbell_detector=barbell_detector,
            rest_gap_s=rest_gap_s,
        )
        self._on_events = on_events or (lambda events: None)
        self._disambiguator = CrowdedGroupDisambiguator(
            motion_resolver=motion_resolver, disambiguation_window_frames=disambiguation_window_frames,
        )

        self._sessions: Dict[str, RepSession] = {}
        self._imu_streams: Dict[str, Optional[IMUStream]] = {}
        self._last_seen: Dict[str, float] = {}

    def _ensure_estimator(self):
        if self.pose_estimator is None:
            from ..pose.estimator import PoseEstimator

            self.pose_estimator = PoseEstimator()
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
        session = RepSession(
            exercise_name=self.exercise_name,
            member_id=member_id,
            station_id=self.station_id,
            start_ts=now,
            **self._session_kwargs,
        )
        self._on_events(session.initial_events)
        self._sessions[wristband_id] = session
        self._imu_streams[wristband_id] = self.imu_stream_factory(wristband_id) if self.imu_stream_factory else None

    def _end_session(self, wristband_id: str, end_ts: float) -> None:
        session = self._sessions.pop(wristband_id, None)
        self._imu_streams.pop(wristband_id, None)
        self._last_seen.pop(wristband_id, None)
        if session is not None:
            self._on_events(session.close(end_ts=end_ts))
        # A session ending always coincides with a present-set change,
        # which the disambiguator's own route() already detects and
        # resets for on its next call -- resetting here too is a harmless,
        # slightly-earlier-than-strictly-necessary reset, not a behavior
        # change (see CrowdedGroupDisambiguator.reset()'s docstring).
        self._disambiguator.reset()

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

        estimator = self._ensure_estimator()
        people = estimator.estimate(frame)

        # IMU fusion keeps running for every open session, including one
        # lingering in its grace period -- fusion shouldn't drop samples
        # just because a session isn't receiving a routed pose this tick.
        polled: Dict[str, List[IMUSample]] = {}
        for wristband_id, imu_stream in self._imu_streams.items():
            samples = imu_stream.poll() if imu_stream is not None else []
            polled[wristband_id] = samples
            self._sessions[wristband_id].add_imu_samples(samples)

        if len(present_set) <= 1:
            self._disambiguator.reset()
            if present_set:
                wristband_id = next(iter(present_set))
                session = self._sessions.get(wristband_id)
                if session is not None:
                    person = people[0] if people else None
                    self._on_events(session.process_frame(frame, now, person))
            return

        routed = self._disambiguator.route(
            now, frozenset(present_set), people, polled, self.checkout_registry.resolve_member,
        )
        for wristband_id, pose in routed.items():
            session = self._sessions.get(wristband_id)
            if session is not None:
                self._on_events(session.process_frame(frame, now, pose))

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
