"""Reusable synthetic frame/pose sources for exercising the *live*
runners (``irix.live.station_runner.StationSessionRunner`` /
``irix.live.gym_runner.GymSessionRunner``) end to end without a camera --
the same "no hardware needed" property ``irix.demo.mock_pose`` already
gives ``run_demo.py``/``run_gym_demo.py``, extended to the tick-loop live
path neither of those exercises (both drive ``RepCounter`` etc. directly;
see ``irix.demo.run_live_gym_demo`` for the live-path demo this enables).
"""
from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

from ..pose.estimator import PersonPose
from ..rep_counting.exercises import ExerciseConfig
from .mock_pose import synthetic_pose_stream


class SyntheticFrameSource:
    """Drop-in for ``irix.live.camera_source.ReconnectingFrameSource``'s
    ``.frames()``/``.close()`` interface, backed by nothing but a frame
    counter. ``StationSessionRunner.tick()`` only ever passes its
    ``frame`` argument through to ``PoseEstimator.estimate(frame)``, so a
    real image array is only required if whatever estimator is paired
    with this source actually looks at pixel content -- paired with
    ``SyntheticPoseEstimator`` below, it doesn't, so a cheap placeholder
    array is enough."""

    def __init__(self):
        self._closed = False

    def frames(self, max_frames: Optional[int] = None) -> Iterator[np.ndarray]:
        i = 0
        while max_frames is None or i < max_frames:
            if self._closed:
                return
            yield np.zeros((2, 2, 3), dtype=np.uint8)
            i += 1

    def close(self) -> None:
        self._closed = True


class SyntheticPoseEstimator:
    """Ignores whatever frame it's given and instead returns the next
    pose off a pre-generated ``irix.demo.mock_pose.synthetic_pose_stream``
    sequence -- so a live ``StationSessionRunner`` sees the same
    geometrically-self-consistent synthetic body motion
    ``run_demo.py``/``run_gym_demo.py`` already use, driven through the
    tick-based live path instead of being fed straight to ``RepCounter``.

    Loops back to the start of the sequence once exhausted -- a 24/7
    station keeps ticking long after any fixed-length demo sequence would
    naturally end.
    """

    def __init__(
        self,
        exercise: ExerciseConfig,
        n_frames: int = 300,
        fps: float = 30.0,
        reps_per_second: float = 0.5,
    ):
        self._poses: List[PersonPose] = [
            pose
            for _, _, pose in synthetic_pose_stream(
                exercise, n_frames=n_frames, fps=fps, reps_per_second=reps_per_second
            )
        ]
        self._i = 0

    def estimate(self, frame) -> List[PersonPose]:
        if not self._poses:
            return []
        pose = self._poses[self._i % len(self._poses)]
        self._i += 1
        return [pose]
