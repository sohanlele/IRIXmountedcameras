"""irix.identity.resolution.IdentityResolution -- the shared identity +
confidence + ambiguity + evidence shape for both the trivial
(sole-candidate) and motion-correlated resolution paths (Priority 5)."""
from __future__ import annotations

from irix.identity.motion_correlation import MotionCorrelationMatch
from irix.identity.resolution import (
    AMBIGUOUS_BELOW_CONFIDENCE,
    from_motion_correlation_match,
    from_sole_candidate,
)


def test_sole_candidate_resolution_is_confident_and_unambiguous():
    resolution = from_sole_candidate("band-1", "member-alice")
    assert resolution.member_id == "member-alice"
    assert resolution.wristband_id == "band-1"
    assert resolution.confidence == 1.0
    assert resolution.ambiguous is False
    assert resolution.evidence["method"] == "sole_candidate_present"
    assert resolution.to_dict()["member_id"] == "member-alice"


def test_confident_motion_correlation_match_carries_evidence():
    match = MotionCorrelationMatch(person_index=0, member_id="member-bob", correlation=0.9, confidence=0.8)
    resolution = from_motion_correlation_match("band-2", match, had_prior=True)
    assert resolution.member_id == "member-bob"
    assert resolution.confidence == 0.8
    assert resolution.ambiguous is False
    assert resolution.evidence == {
        "method": "motion_correlation", "correlation": 0.9, "person_index": 0, "had_prior": True,
    }


def test_low_confidence_motion_correlation_match_is_reported_ambiguous():
    match = MotionCorrelationMatch(
        person_index=1, member_id="member-carol", correlation=0.2, confidence=AMBIGUOUS_BELOW_CONFIDENCE - 0.01,
    )
    resolution = from_motion_correlation_match("band-3", match)
    assert resolution.member_id == "member-carol"  # still reported -- ambiguous flags low trust, doesn't hide the value
    assert resolution.ambiguous is True


def test_no_match_is_ambiguous_with_no_fabricated_member_id():
    resolution = from_motion_correlation_match("band-4", None, had_prior=False)
    assert resolution.member_id is None
    assert resolution.confidence == 0.0
    assert resolution.ambiguous is True
    assert resolution.evidence["reason"] == "no_confident_match"
    assert resolution.evidence["had_prior"] is False
