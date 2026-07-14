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
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from .exercises import ExerciseConfig


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


class RepCounter:
    """Tracks one exercising person's rep count from a stream of joint angles.

    A rep is counted the moment the angle reaches the "top" zone, provided
    the angle passed through the "bottom" zone at some point since the last
    rep was counted (a full concentric/eccentric traversal). Hysteresis
    bands around each threshold prevent sensor/pose noise from triggering
    spurious counts near a boundary.
    """

    def __init__(self, exercise: ExerciseConfig):
        self.exercise = exercise
        self.state = RepState.TOP
        self.rep_count = 0
        self._last_rep_time: Optional[float] = None
        self._session_start = time.monotonic()
        self._reached_bottom = False

    def _descending_is_decreasing(self) -> bool:
        """True if moving from top -> bottom means the angle is decreasing."""
        return self.exercise.top_angle > self.exercise.bottom_angle

    def update(self, angle: float, timestamp: Optional[float] = None) -> Optional[RepEvent]:
        """Feed the latest joint angle in. Returns a RepEvent iff a rep just completed."""
        if angle != angle:  # NaN guard (occlusion / missed keypoint)
            return None
        ts = timestamp if timestamp is not None else time.monotonic()
        cfg = self.exercise
        h = cfg.hysteresis
        descending_is_decreasing = self._descending_is_decreasing()

        at_bottom = (angle <= cfg.bottom_angle + h) if descending_is_decreasing else (angle >= cfg.bottom_angle - h)
        at_top = (angle >= cfg.top_angle - h) if descending_is_decreasing else (angle <= cfg.top_angle + h)

        if at_bottom:
            self._reached_bottom = True
            self.state = RepState.BOTTOM
            return None

        if at_top:
            if self._reached_bottom:
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
                )
            self.state = RepState.TOP
            return None

        # Between thresholds: track direction loosely for observability only
        # (does not affect counting correctness, which relies solely on
        # _reached_bottom).
        self.state = RepState.DESCENDING if self.state in (RepState.TOP, RepState.DESCENDING) else RepState.ASCENDING
        return None
