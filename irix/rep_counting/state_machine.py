"""Joint-angle rep-counting state machine (Section 4.2).

Reps are counted from joint-angle state machines rather than raw frame
classification: track the angle of the relevant joint across frames, and
count a rep on each full transition through the concentric/eccentric range
defined for that exercise. This mirrors the logic underlying existing
vision fitness-coaching products and is far more robust than trying to
classify "a rep happened" from raw pixels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from ..pose.estimator import PersonPose
from .exercises import ExerciseConfig

# Soft cap on how many poses accumulate in a rep's pose buffer -- guards
# against unbounded growth if a caller keeps calling update() with a pose
# while the lifter is idle at the top for a long time between sets
# (handled explicitly below too; this is just a backstop).
_MAX_BUFFERED_POSES = 600


class RepState(Enum):
    TOP = auto()
    DESCENDING = auto()
    BOTTOM = auto()
    ASCENDING = auto()


@dataclass
class RepEvent:
    rep_number: int
    exercise: str
    timestamp: float
    duration_s: float  # time since the previous rep completed (or session start)
    # Joint-angular velocity (deg/s) over the concentric (bottom -> top)
    # phase of this rep -- a rep-speed *proxy*, not a calibrated linear
    # bar velocity. That needs Section 4.5's barbell centroid tracking +
    # per-station camera calibration, which isn't built yet (see
    # irix.rep_counting.exercises and docs/ARCHITECTURE.md). None if too
    # few samples were captured during the concentric phase to estimate
    # (e.g. a very fast rep with a low-fps camera).
    peak_angular_velocity_deg_s: Optional[float] = None
    mean_angular_velocity_deg_s: Optional[float] = None
    # Timestamp of the first sample in this rep's concentric phase (when
    # the angle first entered the "bottom" zone). Lets a caller precisely
    # window an independent signal -- e.g. irix.barbell.tracker.BarPathTracker
    # -- against the exact same phase this angular-velocity estimate used,
    # rather than approximating it from duration_s (which spans the whole
    # previous-rep-to-this-rep gap, not just the concentric phase).
    concentric_start_timestamp: Optional[float] = None
    # Every PersonPose sample seen from (just after) the previous rep's
    # completion through this rep's completion -- the full eccentric +
    # concentric cycle, not just the concentric window above. Feeds
    # irix.form.scoring.FormScorer. None if the caller never passed a
    # ``pose`` into update() (e.g. IMU-only or angle-only callers, or the
    # PoseEstimator wasn't confident enough to yield keypoints that frame).
    poses: Optional[List[PersonPose]] = None


def _phase_velocity(samples: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    """(peak, mean) of |d(angle)/dt| across consecutive samples, deg/s."""
    if len(samples) < 2:
        return None, None
    speeds = []
    for (t0, a0), (t1, a1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt > 0:
            speeds.append(abs(a1 - a0) / dt)
    if not speeds:
        return None, None
    return max(speeds), sum(speeds) / len(speeds)


class RepCounter:
    """Tracks one exercising person's rep count from a stream of joint angles.

    A rep is counted the moment the angle reaches the "top" zone, provided
    the angle passed through the "bottom" zone at some point since the last
    rep was counted (a full concentric/eccentric traversal). Hysteresis
    bands around each threshold prevent sensor/pose noise from triggering
    spurious counts near a boundary.

    While the angle is in the bottom-to-top (concentric) phase, every
    sample is buffered so the completed ``RepEvent`` can report peak/mean
    angular velocity alongside the rep count and inter-rep timing -- this
    is what feeds fatigue trend tracking (e.g. velocity loss across a set)
    on the irix-mvp-app side; this repo only supplies the numbers, not the
    fatigue judgment itself.

    If callers also pass a ``pose`` into ``update()``, the full rep cycle's
    poses are buffered separately (see ``RepEvent.poses``) for
    ``irix.form.scoring.FormScorer`` to score after the fact -- this class
    stays agnostic of form scoring itself, it just carries the data.
    """

    def __init__(self, exercise: ExerciseConfig):
        self.exercise = exercise
        self.state = RepState.TOP
        self.rep_count = 0
        self._last_rep_time: Optional[float] = None
        # Lazily set from the first timestamp seen in update(), rather than
        # time.monotonic() at construction: callers (tests, the mock demo,
        # a real edge pipeline) each have their own timestamp convention,
        # and comparing against wall-clock-at-construction produced a
        # garbage duration_s on the very first rep whenever that
        # convention didn't happen to match wall-clock monotonic time.
        self._session_start: Optional[float] = None
        self._reached_bottom = False
        self._concentric_samples: List[Tuple[float, float]] = []
        self._pose_samples: List[PersonPose] = []

    def _descending_is_decreasing(self) -> bool:
        """True if moving from top -> bottom means the angle is decreasing."""
        return self.exercise.top_angle > self.exercise.bottom_angle

    def update(
        self, angle: float, timestamp: Optional[float] = None, pose: Optional[PersonPose] = None
    ) -> Optional[RepEvent]:
        """Feed the latest joint angle in. Returns a RepEvent iff a rep just completed."""
        if angle != angle:  # NaN guard (occlusion / missed keypoint)
            return None
        ts = timestamp if timestamp is not None else time.monotonic()
        if self._session_start is None:
            self._session_start = ts
        cfg = self.exercise
        h = cfg.hysteresis
        descending_is_decreasing = self._descending_is_decreasing()

        if pose is not None and len(self._pose_samples) < _MAX_BUFFERED_POSES:
            self._pose_samples.append(pose)

        at_bottom = (angle <= cfg.bottom_angle + h) if descending_is_decreasing else (angle >= cfg.bottom_angle - h)
        at_top = (angle >= cfg.top_angle - h) if descending_is_decreasing else (angle <= cfg.top_angle + h)

        if at_bottom:
            self._reached_bottom = True
            self.state = RepState.BOTTOM
            # (Re)start the concentric-phase sample window from the bottom.
            self._concentric_samples = [(ts, angle)]
            return None

        if at_top:
            if self._reached_bottom:
                self._concentric_samples.append((ts, angle))
                concentric_start = self._concentric_samples[0][0]
                peak_v, mean_v = _phase_velocity(self._concentric_samples)
                self._concentric_samples = []
                poses = self._pose_samples if self._pose_samples else None
                self._pose_samples = []

                self.rep_count += 1
                duration = ts - (self._last_rep_time or self._session_start)
                self._last_rep_time = ts
                self._reached_bottom = False
                self.state = RepState.TOP
                return RepEvent(
                    rep_number=self.rep_count,
                    exercise=cfg.name,
                    timestamp=ts,
                    duration_s=duration,
                    peak_angular_velocity_deg_s=peak_v,
                    mean_angular_velocity_deg_s=mean_v,
                    concentric_start_timestamp=concentric_start,
                    poses=poses,
                )
            # Idle at the top (no bottom reached since the last rep, or
            # ever): nothing to report, and no reason to keep accumulating
            # poses indefinitely while someone just stands there between
            # sets.
            self.state = RepState.TOP
            self._pose_samples = []
            return None

        # Between thresholds: keep buffering concentric-phase samples if
        # we're past the bottom (ascending toward top). Also track
        # direction loosely for observability -- neither affects counting
        # correctness, which relies solely on _reached_bottom.
        if self._reached_bottom:
            self._concentric_samples.append((ts, angle))
        self.state = RepState.DESCENDING if self.state in (RepState.TOP, RepState.DESCENDING) else RepState.ASCENDING
        return None
