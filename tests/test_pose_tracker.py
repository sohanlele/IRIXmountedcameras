from __future__ import annotations

import numpy as np
import pytest

from irix.pose.estimator import Keypoint, PersonPose
from irix.pose.tracker import PoseTracker, TrackedPoseEstimator, _iou


def _person(bbox, confidence=0.9, n_kp=17):
    keypoints = [Keypoint(x=(bbox[0] + bbox[2]) / 2, y=(bbox[1] + bbox[3]) / 2, confidence=confidence) for _ in range(n_kp)]
    return PersonPose(keypoints=keypoints, bbox=bbox)


def test_iou_identical_boxes_is_one():
    box = np.array([0.0, 0.0, 10.0, 10.0])
    assert _iou(box, box) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([100.0, 100.0, 110.0, 110.0])
    assert _iou(a, b) == 0.0


def test_tracker_assigns_stable_id_to_a_slowly_moving_person():
    tracker = PoseTracker()
    ids = []
    for i in range(10):
        bbox = (i * 2.0, 0.0, 100.0 + i * 2.0, 200.0)
        result = tracker.update([_person(bbox)], now=i * 0.1)
        assert len(result) == 1
        ids.append(result[0].track_id)

    assert len(set(ids)) == 1  # same track_id every frame


def test_tracker_assigns_different_ids_to_two_far_apart_people():
    tracker = PoseTracker()
    p1 = _person((0.0, 0.0, 100.0, 200.0))
    p2 = _person((500.0, 0.0, 600.0, 200.0))

    result = tracker.update([p1, p2], now=0.0)

    assert len(result) == 2
    assert result[0].track_id != result[1].track_id


def test_tracker_survives_a_brief_low_confidence_dip_without_new_id():
    """ByteTrack's core behavior: a low-confidence detection (partial
    occlusion) should still match the existing track rather than being
    discarded and forcing a new id once confidence recovers."""
    tracker = PoseTracker(high_conf_threshold=0.5, low_conf_threshold=0.1)

    bbox = (0.0, 0.0, 100.0, 200.0)
    first = tracker.update([_person(bbox, confidence=0.9)], now=0.0)
    original_id = first[0].track_id

    # occluded frame: same position, low confidence
    occluded = tracker.update([_person(bbox, confidence=0.2)], now=0.1)
    assert len(occluded) == 1
    assert occluded[0].track_id == original_id

    recovered = tracker.update([_person(bbox, confidence=0.9)], now=0.2)
    assert recovered[0].track_id == original_id


def test_tracker_drops_stale_track_after_max_age():
    tracker = PoseTracker(max_age=2)
    bbox = (0.0, 0.0, 100.0, 200.0)
    tracker.update([_person(bbox)], now=0.0)
    assert len(tracker._tracks) == 1

    # 3 ticks with nobody detected -- exceeds max_age=2
    tracker.update([], now=0.1)
    tracker.update([], now=0.2)
    tracker.update([], now=0.3)

    assert len(tracker._tracks) == 0


def test_tracker_survives_one_missed_frame_then_rematches_same_id():
    tracker = PoseTracker(max_age=5)
    bbox = (0.0, 0.0, 100.0, 200.0)
    first = tracker.update([_person(bbox)], now=0.0)
    original_id = first[0].track_id

    tracker.update([], now=0.1)  # missed frame (occlusion), no detections

    reappeared = tracker.update([_person(bbox)], now=0.2)
    assert reappeared[0].track_id == original_id


def test_tracker_reset_clears_state_and_restarts_ids():
    tracker = PoseTracker()
    first = tracker.update([_person((0.0, 0.0, 100.0, 200.0))], now=0.0)
    tracker.reset()
    second = tracker.update([_person((0.0, 0.0, 100.0, 200.0))], now=0.0)
    assert first[0].track_id == second[0].track_id  # both restart from id 1


def test_bbox_fallback_from_keypoints_when_bbox_is_none():
    from irix.pose.tracker import _bbox_of

    keypoints = [Keypoint(x=10.0, y=20.0, confidence=0.9), Keypoint(x=30.0, y=40.0, confidence=0.8)]
    person = PersonPose(keypoints=keypoints, bbox=None)

    bbox = _bbox_of(person)
    np.testing.assert_allclose(bbox, [10.0, 20.0, 30.0, 40.0])


class _ScriptedEstimator:
    def __init__(self, boxes):
        self._boxes = boxes
        self._i = 0

    def estimate(self, frame):
        if self._i >= len(self._boxes):
            return []
        bbox = self._boxes[self._i]
        self._i += 1
        return [_person(bbox)] if bbox is not None else []


def test_tracked_pose_estimator_is_a_drop_in_estimate_interface():
    clock = iter([0.0, 0.1, 0.2])
    estimator = TrackedPoseEstimator(
        _ScriptedEstimator([(0.0, 0.0, 100.0, 200.0), (2.0, 0.0, 102.0, 200.0), (4.0, 0.0, 104.0, 200.0)]),
        clock=lambda: next(clock),
    )

    ids = [estimator.estimate(frame=None)[0].track_id for _ in range(3)]
    assert len(set(ids)) == 1
