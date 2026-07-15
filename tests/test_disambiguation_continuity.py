"""irix.live.disambiguation.CrowdedGroupDisambiguator's "previous
confirmed identity" continuity (Priority 5): a resolved {person_index:
wristband_id} mapping should be remembered and threaded into the next
resolution (as a member_id-keyed prior) as soon as the buffer refills and
resolves again -- even across a reset() (session churn happens often;
discarding the prior on every one of those would make it useless, see
irix.live.disambiguation's docstring)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from irix.fusion.imu import IMUSample
from irix.live.disambiguation import CrowdedGroupDisambiguator
from irix.pose.estimator import PersonPose


class _RecordingResolver:
    """Duck-typed stand-in for irix.identity.motion_correlation.
    MotionCorrelationResolver: always resolves slot 0 -> "alice", slot 1
    -> "bob" (deterministic, ignores the actual signal), but records every
    prior_slot_assignment it was called with so the test can assert on
    what CrowdedGroupDisambiguator actually threaded through."""

    def __init__(self):
        self.calls: List[Optional[Dict[int, str]]] = []

    def resolve(self, candidate_imu_streams, detected_people_poses, pose_fps, prior_slot_assignment=None):
        self.calls.append(dict(prior_slot_assignment) if prior_slot_assignment else prior_slot_assignment)
        from irix.identity.motion_correlation import MotionCorrelationMatch

        results = []
        fixed = {0: "alice", 1: "bob"}
        for i in range(len(detected_people_poses)):
            member_id = fixed.get(i)
            results.append(
                MotionCorrelationMatch(person_index=i, member_id=member_id, correlation=0.9, confidence=0.9)
                if member_id is not None else None
            )
        return results


def _blank_pose():
    return PersonPose(keypoints=[])


def _resolve_registry():
    return {"band-alice": "alice", "band-bob": "bob"}.get


def test_a_resolved_assignment_is_offered_as_a_prior_on_the_next_resolution():
    resolver = _RecordingResolver()
    disambiguator = CrowdedGroupDisambiguator(motion_resolver=resolver, disambiguation_window_frames=3)
    group = frozenset({"band-alice", "band-bob"})

    for i in range(3):
        disambiguator.route(
            now=float(i), candidate_wristband_ids=group, people=[_blank_pose(), _blank_pose()],
            polled_imu={"band-alice": [], "band-bob": []}, resolve_member_id=_resolve_registry(),
        )
    assert len(resolver.calls) == 1
    assert resolver.calls[0] == {}  # first-ever resolution: no prior yet
    assert disambiguator.slot_assignment == {0: "band-alice", 1: "band-bob"}

    # A group change (e.g. a third member briefly detected) forces a
    # fresh buffer -- but the prior from the resolution above should
    # still be offered once this new buffer fills and resolves.
    bigger_group = frozenset({"band-alice", "band-bob", "band-carol"})
    for i in range(3, 6):
        disambiguator.route(
            now=float(i), candidate_wristband_ids=bigger_group,
            people=[_blank_pose(), _blank_pose()],
            polled_imu={"band-alice": [], "band-bob": [], "band-carol": []},
            resolve_member_id={"band-alice": "alice", "band-bob": "bob", "band-carol": "carol"}.get,
        )
    assert len(resolver.calls) == 2
    assert resolver.calls[1] == {0: "alice", 1: "bob"}


def test_reset_does_not_erase_the_continuity_prior():
    """reset() clears the active buffer (session churn, e.g. a member
    stepping away) but deliberately keeps the last resolved mapping
    around -- see the module docstring for why discarding it here would
    defeat the point."""
    resolver = _RecordingResolver()
    disambiguator = CrowdedGroupDisambiguator(motion_resolver=resolver, disambiguation_window_frames=2)
    group = frozenset({"band-alice", "band-bob"})

    for i in range(2):
        disambiguator.route(
            now=float(i), candidate_wristband_ids=group, people=[_blank_pose(), _blank_pose()],
            polled_imu={"band-alice": [], "band-bob": []}, resolve_member_id=_resolve_registry(),
        )
    assert disambiguator.slot_assignment == {0: "band-alice", 1: "band-bob"}

    disambiguator.reset()
    assert disambiguator.slot_assignment == {}

    for i in range(2, 4):
        disambiguator.route(
            now=float(i), candidate_wristband_ids=group, people=[_blank_pose(), _blank_pose()],
            polled_imu={"band-alice": [], "band-bob": []}, resolve_member_id=_resolve_registry(),
        )
    assert len(resolver.calls) == 2
    assert resolver.calls[1] == {0: "alice", 1: "bob"}
