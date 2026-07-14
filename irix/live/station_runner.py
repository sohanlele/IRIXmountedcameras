"""Ties together everything a real, continuously-running station needs
that a single ``RepSession`` doesn't provide on its own: knowing *whose*
session this even is, and *when* one starts and ends.

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
- ``irix.fusion.imu_stream.IMUStream`` -- live samples for whichever band
  is the currently-tracked member's, once a session is active.
- ``irix.pipeline.rep_session.RepSession`` -- the actual per-member
  pipeline (already shared with ``irix.demo.run_upload``).

What "session" means here: a checked-out band showing up in this
station's BLE readings starts one (a fresh ``RepSession``, and a request
for that band's live ``IMUStream``); the band no longer showing up for
``presence_timeout_s`` ends it (flush whatever set was in progress,
release the ``RepSession``). This mirrors the "run_upload waits for a set
to naturally show a gap, closes it" logic from ``RestGapSetBoundaryDetector``,
just one level up -- that class decides when a *set* ends within an
active session; this class decides when the *session itself* ends.

One thing this deliberately does NOT decide: which single wristband
"wins" when several are visible to one station's radio at once (a
plausible real scenario -- someone walking past, or spotting). The v1
heuristic here is the simplest defensible one (strongest RSSI among
*checked-out* bands), same spirit as ``StationPairing``'s own v1
heuristic for the station-selection problem -- not claimed to be
more rigorous than that.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

from ..barbell.detector import FreeWeightDetector
from ..fusion.imu_stream import IMUStream
from ..identity.ble_pairing import BLEReading
from ..identity.checkout import CheckoutRegistry
from ..pipeline.rep_session import RepSession
from ..pipeline.schema import CameraEvent
from ..weight_recognition.vlm_backend import VLMBackend


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
    ):
        """``ble_reader``: called once per frame tick, returns whatever
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

        self._active_wristband_id: Optional[str] = None
        self._active_session: Optional[RepSession] = None
        self._active_imu_stream: Optional[IMUStream] = None
        self._last_seen_ts: Optional[float] = None

    def _ensure_estimator(self):
        if self.pose_estimator is None:
            from ..pose.estimator import PoseEstimator

            self.pose_estimator = PoseEstimator()
        return self.pose_estimator

    def _resolve_present_wristband(self) -> Optional[str]:
        """Among this tick's BLE readings, the strongest-RSSI band that's
        actually checked out to an account right now -- an unchecked-out
        band (someone walked in wearing a band from a different gym /
        past visit that never got returned) is never tracked, same as a
        real front desk wouldn't start a session for a band it doesn't
        currently have handed out."""
        readings = self.ble_reader()
        checked_out = [
            r for r in readings
            if r.wristband_id is not None and self.checkout_registry.is_checked_out(r.wristband_id)
        ]
        if not checked_out:
            return None
        return max(checked_out, key=lambda r: r.rssi).wristband_id

    def _start_session(self, wristband_id: str) -> None:
        member_id = self.checkout_registry.resolve_member(wristband_id)
        assert member_id is not None  # guaranteed by _resolve_present_wristband's is_checked_out filter
        session = RepSession(
            exercise_name=self.exercise_name,
            member_id=member_id,
            station_id=self.station_id,
            **self._session_kwargs,
        )
        self._on_events(session.initial_events)
        self._active_wristband_id = wristband_id
        self._active_session = session
        self._active_imu_stream = self.imu_stream_factory(wristband_id) if self.imu_stream_factory else None

    def _end_session(self, end_ts: float) -> None:
        if self._active_session is not None:
            self._on_events(self._active_session.close(end_ts=end_ts))
        self._active_wristband_id = None
        self._active_session = None
        self._active_imu_stream = None
        self._last_seen_ts = None

    def tick(self, frame, now: float, present_wristband_id: Optional[str]) -> None:
        """One frame's worth of work, given an *already-resolved*
        "which wristband is present at this station right now" (or
        ``None``). ``run_forever`` below resolves that locally, once per
        frame, via ``self.ble_reader()`` -- correct for a single isolated
        station. ``irix.live.gym_runner.GymSessionRunner`` instead
        resolves presence gym-wide (cross-station handoff hysteresis via
        ``irix.topology.handoff.GymCoordinator``, so a station doesn't
        just trust its own local RSSI snapshot) and calls this directly,
        once per station per gym-wide tick, bypassing this station's own
        ``ble_reader``/``_resolve_present_wristband`` entirely. Either
        way, this method is the actual per-tick state machine: start a
        session on a new presence, end one on absence past
        ``presence_timeout_s`` (or immediately if a *different*
        checked-out band preempts it), and feed the frame through if a
        session is active.
        """
        estimator = self._ensure_estimator()

        if present_wristband_id is not None:
            if self._active_wristband_id is not None and present_wristband_id != self._active_wristband_id:
                # someone else showed up while a session was active --
                # end the previous one first rather than silently
                # attributing their reps to whoever was here before.
                self._end_session(end_ts=self._last_seen_ts or now)
            if self._active_session is None:
                self._start_session(present_wristband_id)
            self._last_seen_ts = now
        elif self._active_session is not None and self._last_seen_ts is not None:
            if now - self._last_seen_ts >= self.presence_timeout_s:
                self._end_session(end_ts=self._last_seen_ts)

        if self._active_session is None:
            return

        if self._active_imu_stream is not None:
            self._active_session.add_imu_samples(self._active_imu_stream.poll())

        people = estimator.estimate(frame)
        person = people[0] if people else None
        self._on_events(self._active_session.process_frame(frame, now, person))

    def run_forever(self, max_frames: Optional[int] = None) -> None:
        """Runs until ``frame_source`` stops yielding (only happens in
        tests, via ``max_frames`` -- a real ``ReconnectingFrameSource``
        yields forever). Each frame: resolve BLE presence locally (see
        ``_resolve_present_wristband``) and hand off to ``tick()``.
        Runtime timestamps come from ``time.monotonic()``, the correct
        clock here (unlike ``irix.demo.run_upload``'s ``frame_index /
        fps``) since this is genuinely live: frame arrival time *is*
        wall-clock time for a live camera, and BLE presence/timeout logic
        has to be measured against the same clock the radio readings
        themselves arrive on.
        """
        for frame in self.frame_source.frames(max_frames=max_frames):
            now = self._clock()
            present_wristband_id = self._resolve_present_wristband()
            self.tick(frame, now, present_wristband_id)

    def close(self) -> None:
        """Flush the active session (if any) and release the frame
        source -- call on shutdown."""
        if self._active_session is not None:
            self._on_events(self._active_session.close(end_ts=self._last_seen_ts or self._clock()))
            self._active_session = None
            self._active_wristband_id = None
            self._active_imu_stream = None
        close_fn = getattr(self.frame_source, "close", None)
        if close_fn is not None:
            close_fn()
