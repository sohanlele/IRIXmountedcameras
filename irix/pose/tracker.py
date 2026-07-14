"""Intra-camera multi-person tracking (ByteTrack-lite).

``PoseEstimator.estimate()`` sets ``PersonPose.track_id`` to that frame's
list index (see its own source) -- not a stable identity across frames.
Every caller that needs short-window person-identity stability
(``irix.live.disambiguation.CrowdedGroupDisambiguator``,
``irix.identity.motion_correlation``) currently works around this by
indexing detected people by list position within one short buffering
window instead, a real, documented limitation (see
``irix.live.station_runner``'s module docstring and
``docs/TRACKING.md``). This module fixes the underlying gap: a real,
tested tracker that assigns persistent ``track_id``s across frames within
one camera, so any future/optional consumer gets actual identity
continuity instead of a per-frame list index.

Design, grounded in the tracking-by-detection literature rather than
invented from scratch:

- **SORT** (Bewley et al., 2016, arXiv:1602.00763) -- constant-velocity
  Kalman filter over each track's bounding box, Hungarian-matched to new
  detections by IoU each frame. This module's ``_KalmanBoxTrack`` is
  that filter (state ``[cx, cy, w, h, vx, vy]``).
- **ByteTrack** (Zhang et al., ECCV 2022, arXiv:2110.06864) -- the one
  change this module actually adopts over plain SORT: don't discard
  low-confidence detections, match them in a *second* association pass
  against tracks the high-confidence pass left unmatched. A person
  mid-occlusion (partially blocked by a barbell, another lifter, or a
  rack) typically produces a lower-confidence detection just before
  their track would otherwise be lost entirely -- ByteTrack's two-stage
  association recovers exactly that case instead of dropping the track
  and assigning a new id once the person re-emerges clearly.
- **BoT-SORT** (Aharon et al., 2022, arXiv:2206.14651) -- adds a deep
  ReID appearance embedding and camera-motion compensation on top of
  ByteTrack. Deliberately **not** adopted here: BoT-SORT's ReID
  extension exists for crowded, high-throughput scenes (pedestrian
  tracking at stadium/street scale) where IoU alone confuses similarly-
  positioned people; a gym station's camera sees 1-4 people at a time
  with a static, non-panning camera (no camera-motion compensation
  needed either), where IoU + short-horizon motion prediction is
  sufficient and a deep embedding network would be pure inference-cost
  overhead with no accuracy benefit at this scale.

No external tracking dependency is pulled in -- the Kalman filter is
implemented directly against the small, fixed state size above (no need
for `filterpy` or similar), and Hungarian assignment uses
`scipy.optimize.linear_sum_assignment`, already a hard dependency of
this repo.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .estimator import PersonPose


def _bbox_of(person: PersonPose) -> Optional[np.ndarray]:
    """``person.bbox`` if the estimator/caller set one; otherwise the
    tight bounding box of whichever keypoints were actually detected
    (confidence > 0) -- covers synthetic poses (e.g.
    ``irix.demo.mock_pose``) that may not always populate ``bbox``."""
    if person.bbox is not None:
        return np.asarray(person.bbox, dtype=float)
    xs = [kp.x for kp in person.keypoints if kp.confidence > 0]
    ys = [kp.y for kp in person.keypoints if kp.confidence > 0]
    if not xs:
        return None
    return np.array([min(xs), min(ys), max(xs), max(ys)])


def _mean_confidence(person: PersonPose) -> float:
    if not person.keypoints:
        return 0.0
    return float(np.mean([kp.confidence for kp in person.keypoints]))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _KalmanBoxTrack:
    """One tracked person: constant-velocity Kalman filter state
    ``[cx, cy, w, h, vx, vy]`` (SORT's state model)."""

    def __init__(self, track_id: int, bbox: np.ndarray, now: float):
        self.track_id = track_id
        cx, cy, w, h = self._to_cxcywh(bbox)
        self.state = np.array([cx, cy, w, h, 0.0, 0.0])
        self.P = np.eye(6) * 10.0
        self.last_update_time = now
        self.hits = 1
        self.time_since_update = 0
        self.last_person: Optional[PersonPose] = None

    @staticmethod
    def _to_cxcywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, 1.0), max(y2 - y1, 1.0)

    @staticmethod
    def _to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self, now: float) -> np.ndarray:
        """Advance the constant-velocity model to ``now`` and return the
        predicted bbox -- used both for this tick's IoU association and
        as the track's position while it's unmatched (bridges a short
        occlusion instead of freezing/vanishing)."""
        dt = max(0.0, now - self.last_update_time)
        F = np.eye(6)
        F[0, 4] = dt
        F[1, 5] = dt
        self.state = F @ self.state
        Q = np.eye(6) * 1.0
        self.P = F @ self.P @ F.T + Q
        cx, cy, w, h, _, _ = self.state
        return self._to_xyxy(cx, cy, max(w, 1.0), max(h, 1.0))

    def update(self, bbox: np.ndarray, now: float, person: PersonPose) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox)
        z = np.array([cx, cy, w, h])
        H = np.zeros((4, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 3] = 1.0
        R = np.eye(4) * 1.0
        y = z - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        self.last_update_time = now
        self.hits += 1
        self.time_since_update = 0
        self.last_person = person

    def mark_missed(self) -> None:
        self.time_since_update += 1


class PoseTracker:
    """Assigns persistent ``track_id``s to a per-frame ``PersonPose``
    list from any pose source (real ``PoseEstimator`` or a synthetic
    one). See module docstring for the ByteTrack-derived two-stage
    association this implements.

    ``max_age``: how many consecutive unmatched ticks a track survives
    on predicted motion alone before being dropped -- the occlusion-
    recovery budget. ``min_hits``: how many updates a track needs before
    it's returned to the caller, filtering out one-frame false-positive
    detections; default 1 is appropriate at gym-station scale (1-4
    people, not a crowded scene where spurious detections are common).
    """

    def __init__(
        self,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
    ):
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: List[_KalmanBoxTrack] = []
        self._next_track_id = 1

    def _associate(
        self,
        detections: List[Tuple[PersonPose, np.ndarray, float]],
        tracks: List[_KalmanBoxTrack],
        predicted: Dict[int, np.ndarray],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not detections or not tracks:
            return [], list(range(len(detections))), list(range(len(tracks)))
        iou_matrix = np.zeros((len(detections), len(tracks)))
        for di, (_, bbox, _) in enumerate(detections):
            for ti, t in enumerate(tracks):
                iou_matrix[di, ti] = _iou(bbox, predicted[t.track_id])
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches, matched_d, matched_t = [], set(), set()
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                matched_d.add(r)
                matched_t.add(c)
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
        unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
        return matches, unmatched_d, unmatched_t

    def update(self, people: List[PersonPose], now: float) -> List[PersonPose]:
        """One frame's detections in, the same people back out with
        ``track_id`` overwritten to a persistent id (people that
        haven't reached ``min_hits`` yet are dropped from the returned
        list, though their track keeps accumulating internally)."""
        detections = []
        for person in people:
            bbox = _bbox_of(person)
            if bbox is not None:
                detections.append((person, bbox, _mean_confidence(person)))

        high = [d for d in detections if d[2] >= self.high_conf_threshold]
        low = [d for d in detections if self.low_conf_threshold <= d[2] < self.high_conf_threshold]

        predicted = {t.track_id: t.predict(now) for t in self._tracks}
        assigned: Dict[int, PersonPose] = {}

        # Stage 1: high-confidence detections against every current track.
        active_tracks = list(self._tracks)
        matches1, unmatched_d1, unmatched_t1 = self._associate(high, active_tracks, predicted)
        for di, ti in matches1:
            person, bbox, _ = high[di]
            track = active_tracks[ti]
            track.update(bbox, now, person)
            assigned[track.track_id] = person

        # Stage 2: low-confidence detections against whatever's still
        # unmatched -- ByteTrack's occlusion-recovery pass (see module
        # docstring).
        remaining_tracks = [active_tracks[i] for i in unmatched_t1]
        matches2, _unmatched_d2, unmatched_t2 = self._associate(low, remaining_tracks, predicted)
        for di, ti in matches2:
            person, bbox, _ = low[di]
            track = remaining_tracks[ti]
            track.update(bbox, now, person)
            assigned[track.track_id] = person

        # Unmatched high-confidence detections start new tracks -- only
        # high-confidence ones, so a single noisy low-confidence blip
        # can't spawn a spurious identity.
        for di in unmatched_d1:
            person, bbox, _ = high[di]
            track = _KalmanBoxTrack(self._next_track_id, bbox, now)
            self._next_track_id += 1
            track.last_person = person
            self._tracks.append(track)
            assigned[track.track_id] = person

        for ti in unmatched_t2:
            remaining_tracks[ti].mark_missed()

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]
        tracks_by_id = {t.track_id: t for t in self._tracks}

        results = []
        for track_id, person in assigned.items():
            track = tracks_by_id.get(track_id)
            if track is None or track.hits < self.min_hits:
                continue
            person.track_id = track_id
            results.append(person)
        return results

    def reset(self) -> None:
        self._tracks = []
        self._next_track_id = 1


class TrackedPoseEstimator:
    """Drop-in wrapper: anything with ``.estimate(frame) -> List[PersonPose]``
    (a real ``PoseEstimator`` or a synthetic one) plus a ``PoseTracker``,
    exposing the same ``.estimate(frame)`` interface every caller in this
    repo already expects (``irix.live.station_runner.StationSessionRunner``'s
    ``pose_estimator`` constructor arg, ``run_demo.py --source``) -- so
    adding persistent tracking to any pipeline is a one-line change with
    no call-site rewrite.

    Uses its own clock (defaults to ``time.monotonic``) purely for the
    Kalman filter's motion-prediction ``dt`` -- independent of, and not
    required to match, whatever clock a session/BLE layer uses elsewhere,
    since bbox motion prediction only needs elapsed time between
    consecutive ``estimate()`` calls, not wall-clock alignment with
    anything else.
    """

    def __init__(self, pose_estimator, tracker: Optional[PoseTracker] = None, clock=None):
        self.pose_estimator = pose_estimator
        self.tracker = tracker or PoseTracker()
        import time

        self._clock = clock or time.monotonic

    def estimate(self, frame) -> List[PersonPose]:
        people = self.pose_estimator.estimate(frame)
        now = self._clock()
        return self.tracker.update(people, now)
