"""Runs several cameras with *overlapping* fields of view over one shared
physical area (e.g. a free-weights section covered by an array of
cameras) -- a different topology from ``irix.live.station_runner.
StationSessionRunner``, which assumes exactly one camera per station and
already handles multiple co-located members at *that one camera* via
``irix.live.disambiguation.CrowdedGroupDisambiguator``.

**Why a dense multi-camera zone needs a different orchestrator, not just
a bigger ``StationSessionRunner``.** Two things break the single-camera
model once several cameras genuinely overlap the same space:

1. The same physical person can be visible in more than one camera's
   frame at once -- a single "detected people" list per tick (what
   ``StationSessionRunner``/``CrowdedGroupDisambiguator`` assume) no
   longer makes sense; there are ``N`` separate detected-people lists
   per tick, one per camera, potentially containing redundant views of
   the same people.
2. Which cameras currently see whom shifts tick to tick as people move
   and occlude each other from a given angle. ``CrowdedGroupDisambiguator``
   requires a *stable* detected-person count across its whole buffering
   window (any tick where the count doesn't match gets silently dropped
   -- see that module's docstring) -- pooling every camera's detections
   into one combined list would make that count fluctuate constantly and
   the buffering effectively never fill.

**Design: no cross-camera pixel-level person matching.** This doesn't
attempt to solve "is detection #2 in camera A's frame the same physical
person as detection #0 in camera B's frame" by appearance or geometry --
that's the general multi-camera re-identification problem sports-
analytics systems (e.g. SkillCorner) solve with jersey-number recognition
and pose-guided embeddings, deliberately not the approach this repo takes
anywhere else (see ``irix.identity.motion_correlation``'s own docstring
on why wristband-based identity was chosen over vision-only re-ID).
Instead, **each camera in the zone runs its own independent
``CrowdedGroupDisambiguator``, all sharing the same zone-wide candidate
wristband group.** The wristband IMU signal is what ties multiple
cameras' views of one person together, for free: if camera A's slot 2
and camera B's slot 0 both correlate best with wristband X's IMU, they're
the same physical person -- neither camera ever needs to know the other
exists, or that the other even detected someone this tick. This also
means a person only visible to one camera this tick (occluded from
another camera's angle) still gets tracked correctly, since each camera
independently attempts disambiguation against whatever it currently sees.

**Avoiding double-counting when 2+ cameras agree.** If more than one
camera resolves a pose for the *same* member in the same tick (a
legitimately overlapping view), exactly one of them is fed into that
member's ``RepSession`` that tick -- picked by a fixed camera-priority
order (the first camera, in the order this runner was given its cameras,
that has a routed pose for that member wins that tick). Never more than
one ``process_frame`` call per member per tick, which would otherwise
risk double-counting a rep.

**Bar-path calibration is per-camera-aware.** ``RepSession`` self-
calibrates a separate px-per-mm scale independently for *each*
``camera_id`` that has fed it a frame with a detected plate (keyed in
``RepSession._bar_calibrations``), and ``BarPathTracker.push()`` takes an
explicit per-call ``calibration`` override -- so when the per-tick camera
routing below hands a different member's set off to a different physical
camera mid-set, that camera's own calibration (self-calibrated the first
time *that* camera saw a plate, not reused from whichever camera saw one
first) is what gets applied to its pixels. Every sample already pushed
stays in real-world meters regardless of which calibration produced it,
so one continuous ``BarPathTracker``/velocity-window query still spans a
camera switch correctly -- see ``irix.barbell.tracker.BarPathTracker.
push``'s and ``irix.pipeline.rep_session.RepSession.process_frame``'s
docstrings for the mechanics. Joint-angle-based rep counting was never
affected by this either way (joint angles are relative measurements
between a frame's own keypoints, not dependent on any absolute px-per-mm
calibration).

**Optional multi-view 3D pose fusion.** When ``camera_projections`` is
given (a geometric camera calibration per ``camera_id`` -- see
``irix.pose.multiview``'s module docstring), this runner triangulates a
3D pose from every camera that currently resolves a pose for a given
member in a tick (not just the priority-winner used for weight/barbell
detection), via ``irix.pose.multiview.triangulate_pose``. That 3D pose is
what actually gets fed into ``RepSession.process_frame`` -- ``RepSession``
prefers a triangulated 3D joint angle over a single camera's 2D one
whenever all 3 of an exercise's needed keypoints triangulated that tick
(see its own docstring), so rep counting becomes immune to any one
camera's foreshortening or self-occlusion of the joint that matters, not
just tolerant of losing sight of the person entirely. This is strictly
opt-in: without ``camera_projections`` (the default), this runner behaves
exactly as before -- single 2D pose, one camera per tick.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..barbell.detector import FreeWeightDetector
from ..fusion.imu import IMUSample
from ..fusion.imu_stream import IMUStream
from ..identity.ble_pairing import BLEReading
from ..identity.checkout import CheckoutRegistry
from ..identity.motion_correlation import MotionCorrelationResolver
from ..pipeline.rep_session import RepSession
from ..pipeline.schema import CameraEvent
from ..pose.estimator import PersonPose
from ..pose.multiview import CameraProjection, triangulate_pose
from ..weight_recognition.vlm_backend import VLMBackend
from .disambiguation import CrowdedGroupDisambiguator


class ZoneCamera:
    """One camera's feed within a ``MultiCameraZoneRunner`` -- a frame
    source plus (optionally) its own ``PoseEstimator`` instance. Several
    ``ZoneCamera``s share one zone-wide member/session set; each gets its
    own ``CrowdedGroupDisambiguator`` (constructed by the owning
    ``MultiCameraZoneRunner``, not here) since disambiguation buffering
    state is inherently per-detection-source -- see the module docstring.
    """

    def __init__(self, camera_id: str, frame_source, pose_estimator=None):
        self.camera_id = camera_id
        self.frame_source = frame_source
        self.pose_estimator = pose_estimator
        # Assigned by MultiCameraZoneRunner.__init__ -- present here only
        # so type checkers/callers see the attribute exists.
        self.disambiguator: Optional[CrowdedGroupDisambiguator] = None


class MultiCameraZoneRunner:
    def __init__(
        self,
        zone_id: str,
        exercise_name: str,
        checkout_registry: CheckoutRegistry,
        cameras: List[ZoneCamera],
        ble_reader: Callable[[], List[BLEReading]],
        imu_stream_factory: Optional[Callable[[str], IMUStream]] = None,
        presence_timeout_s: float = 5.0,
        vlm_backend: Optional[VLMBackend] = None,
        weight_check_every_n_frames: int = 30,
        barbell_detector: Optional[FreeWeightDetector] = None,
        rest_gap_s: float = 20.0,
        on_events: Optional[Callable[[List[CameraEvent]], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        motion_resolver: Optional[MotionCorrelationResolver] = None,
        disambiguation_window_frames: int = 60,
        camera_tilt_deg_by_camera: Optional[Dict[str, float]] = None,
        camera_projections: Optional[Dict[str, CameraProjection]] = None,
    ):
        """``cameras``: every camera covering this zone, in a fixed
        priority order -- when 2+ cameras resolve a pose for the same
        member in the same tick, the earliest one in this list wins (see
        module docstring). Order shouldn't matter for correctness, only
        for which camera's frame ends up feeding ``RepSession.
        process_frame`` (and therefore weight/barbell detection) on a
        tick where more than one camera could have.

        ``ble_reader``: called once per zone-wide tick (in ``run_forever``
        only), returns every currently-visible ``BLEReading`` for this
        zone's radio(s) -- same shape as ``StationSessionRunner``'s own
        ``ble_reader``, just zone-wide instead of per-station.

        ``camera_tilt_deg_by_camera``: optional ``camera_id -> tilt_deg``
        map, forwarded verbatim into every ``RepSession`` this runner
        creates (see ``RepSession.__init__``'s docstring). Since a zone's
        cameras are physically distinct mountings, they can plausibly
        have different actual tilt angles relative to the bar's vertical
        travel plane -- this lets each one get its own correction rather
        than sharing one value across the whole zone.

        ``camera_projections``: optional ``camera_id -> CameraProjection``
        geometric calibration map -- when given, enables multi-view 3D
        pose triangulation (see module docstring's "Optional multi-view
        3D pose fusion" section and ``irix.pose.multiview``). Omitted
        (the default), this runner's behavior is unchanged from before
        multi-view fusion existed.

        Every other parameter mirrors ``StationSessionRunner``'s
        constructor exactly -- see that class's docstring for the ones
        not re-explained here.
        """
        if not cameras:
            raise ValueError("MultiCameraZoneRunner needs at least one camera")
        self.zone_id = zone_id
        self.exercise_name = exercise_name
        self.checkout_registry = checkout_registry
        self.cameras = cameras
        self.ble_reader = ble_reader
        self.imu_stream_factory = imu_stream_factory
        self.presence_timeout_s = presence_timeout_s
        self.camera_projections = camera_projections or {}
        self._clock = clock or time.monotonic
        self._session_kwargs = dict(
            vlm_backend=vlm_backend,
            weight_check_every_n_frames=weight_check_every_n_frames,
            barbell_detector=barbell_detector,
            rest_gap_s=rest_gap_s,
            camera_tilt_deg_by_camera=camera_tilt_deg_by_camera,
        )
        self._on_events = on_events or (lambda events: None)
        for camera in self.cameras:
            camera.disambiguator = CrowdedGroupDisambiguator(
                motion_resolver=motion_resolver, disambiguation_window_frames=disambiguation_window_frames,
            )

        self._sessions: Dict[str, RepSession] = {}
        self._imu_streams: Dict[str, Optional[IMUStream]] = {}
        self._last_seen: Dict[str, float] = {}

    def _ensure_estimator(self, camera: ZoneCamera):
        if camera.pose_estimator is None:
            from ..pose.estimator import PoseEstimator

            camera.pose_estimator = PoseEstimator()
        return camera.pose_estimator

    def _resolve_present_wristbands(self) -> List[str]:
        readings = self.ble_reader()
        return list({
            r.wristband_id for r in readings
            if r.wristband_id is not None and self.checkout_registry.is_checked_out(r.wristband_id)
        })

    def _start_session(self, wristband_id: str) -> None:
        member_id = self.checkout_registry.resolve_member(wristband_id)
        assert member_id is not None  # guaranteed by callers checking is_checked_out first
        session = RepSession(
            exercise_name=self.exercise_name,
            member_id=member_id,
            station_id=self.zone_id,
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
        for camera in self.cameras:
            camera.disambiguator.reset()

    def tick(self, frames: Dict[str, np.ndarray], now: float, present_wristband_ids: List[str]) -> None:
        """One zone-wide tick. ``frames`` maps ``camera_id -> that
        camera's frame this tick`` -- a camera missing from the dict is
        treated as having produced nothing this tick (e.g. a dropped
        connection or a slower frame rate than the others) and simply
        contributes no detections, same as any single camera returning
        an empty people list.
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
                self._start_session(wristband_id)

        if not self._sessions:
            return

        # IMU fusion keeps running for every open session regardless of
        # camera routing -- same reasoning as StationSessionRunner.
        polled: Dict[str, List[IMUSample]] = {}
        for wristband_id, imu_stream in self._imu_streams.items():
            samples = imu_stream.poll() if imu_stream is not None else []
            polled[wristband_id] = samples
            self._sessions[wristband_id].add_imu_samples(samples)

        if len(present_set) <= 1:
            for camera in self.cameras:
                camera.disambiguator.reset()
            if not present_set:
                return
            wristband_id = next(iter(present_set))
            session = self._sessions.get(wristband_id)
            if session is None:
                return
            # Only one member in the whole zone -- no identity ambiguity
            # possible. Without camera_projections (the common case), this
            # stays exactly the pre-multi-view-fusion behavior: take the
            # first camera (priority order) that currently sees anyone at
            # all, feed only that one, same double-counting guard as the
            # ambiguous path below. With camera_projections configured,
            # every camera that currently sees this lone member is polled
            # (not just the first) so their poses can be triangulated into
            # one 3D pose -- see module docstring's "Optional multi-view
            # 3D pose fusion" section.
            if not self.camera_projections:
                for camera in self.cameras:
                    frame = frames.get(camera.camera_id)
                    if frame is None:
                        continue
                    people = self._ensure_estimator(camera).estimate(frame)
                    if people:
                        self._on_events(session.process_frame(frame, now, people[0], camera_id=camera.camera_id))
                        break
                return

            poses_by_camera: Dict[str, PersonPose] = {}
            first_frame: Optional[np.ndarray] = None
            first_camera_id: Optional[str] = None
            for camera in self.cameras:
                frame = frames.get(camera.camera_id)
                if frame is None:
                    continue
                people = self._ensure_estimator(camera).estimate(frame)
                if people:
                    poses_by_camera[camera.camera_id] = people[0]
                    if first_frame is None:
                        first_frame, first_camera_id = frame, camera.camera_id
            if not poses_by_camera:
                return
            fused = triangulate_pose(poses_by_camera, self.camera_projections)
            pose = fused if fused is not None else poses_by_camera[first_camera_id]
            self._on_events(session.process_frame(first_frame, now, pose, camera_id=first_camera_id))
            return

        candidate_ids = frozenset(present_set)
        # (camera_id, frame, pose) rather than just (frame, pose) -- the
        # camera_id has to survive into the feed loop below so each
        # member's RepSession.process_frame call knows which camera this
        # tick's pose came from, and therefore which camera's own
        # calibration to self-calibrate/read bar-path pixels against (see
        # module docstring's "Bar-path calibration is per-camera-aware"
        # section).
        routed_this_tick: Dict[str, Tuple[str, np.ndarray, PersonPose]] = {}
        # camera_id -> pose for every camera that resolved *this* member
        # this tick, not just the priority-winner above -- feeds
        # triangulate_pose below when camera_projections is configured,
        # so a member seen by 2+ cameras this tick can get a fused 3D
        # pose even though only one (camera_id, frame) pair is used for
        # weight/barbell detection.
        all_routed_this_tick: Dict[str, Dict[str, PersonPose]] = {}
        for camera in self.cameras:
            frame = frames.get(camera.camera_id)
            if frame is None:
                continue
            people = self._ensure_estimator(camera).estimate(frame)
            routed = camera.disambiguator.route(
                now, candidate_ids, people, polled, self.checkout_registry.resolve_member,
            )
            for wristband_id, pose in routed.items():
                if wristband_id not in routed_this_tick:
                    routed_this_tick[wristband_id] = (camera.camera_id, frame, pose)
                all_routed_this_tick.setdefault(wristband_id, {})[camera.camera_id] = pose

        for wristband_id, (camera_id, frame, pose) in routed_this_tick.items():
            session = self._sessions.get(wristband_id)
            if session is None:
                continue
            fused_pose = pose
            if self.camera_projections and len(all_routed_this_tick.get(wristband_id, {})) >= 2:
                fused = triangulate_pose(all_routed_this_tick[wristband_id], self.camera_projections)
                if fused is not None:
                    fused_pose = fused
            self._on_events(session.process_frame(frame, now, fused_pose, camera_id=camera_id))

    def run_forever(self, max_frames: Optional[int] = None) -> None:
        """Runs until every camera's ``frame_source`` is exhausted (only
        happens in tests -- a real ``ReconnectingFrameSource`` yields
        forever). Each zone-wide tick: pull one frame from every camera,
        resolve zone-wide presence from one BLE read, then drive
        ``tick()``.

        Cameras whose frame source has already run out (only possible
        with ``max_frames`` in a test) are simply skipped for the rest of
        the run rather than ending the whole zone-wide loop early -- same
        pattern as ``irix.live.gym_runner.GymSessionRunner.run_forever``,
        since a real deployment's cameras don't all fail in lockstep.
        """
        frame_iters = {
            camera.camera_id: camera.frame_source.frames(max_frames=max_frames) for camera in self.cameras
        }
        while frame_iters:
            frames: Dict[str, np.ndarray] = {}
            exhausted = []
            for camera_id, it in frame_iters.items():
                try:
                    frames[camera_id] = next(it)
                except StopIteration:
                    exhausted.append(camera_id)
            for camera_id in exhausted:
                del frame_iters[camera_id]
            if not frames:
                break

            now = self._clock()
            present_wristband_ids = self._resolve_present_wristbands()
            self.tick(frames, now, present_wristband_ids)

    def close(self) -> None:
        """Flush every active session and release every camera's frame
        source -- call on shutdown."""
        for wristband_id in list(self._sessions.keys()):
            last_seen = self._last_seen.get(wristband_id)
            self._end_session(wristband_id, end_ts=last_seen if last_seen is not None else self._clock())
        for camera in self.cameras:
            close_fn = getattr(camera.frame_source, "close", None)
            if close_fn is not None:
                close_fn()
