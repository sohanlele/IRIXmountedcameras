"""``IdentityResolution`` -- the one output shape Priority 5 asks for:
identity + confidence + ambiguity + supporting evidence, produced the
same way regardless of *how* a station arrived at it (BLE-only trivial
case, or full motion-correlation disambiguation).

## Why a shared shape

Before this module, "who is this detected person" had two different,
incompatible answers depending on the code path: the common single-
candidate case (``irix.live.station_runner.StationSessionRunner.tick``'s
``len(present_set) <= 1`` branch) just used the sole present wristband's
member_id directly, with no confidence/evidence attached at all; the
crowded case (``irix.identity.motion_correlation.MotionCorrelationResolver``)
produced a richer ``MotionCorrelationMatch`` with correlation/confidence,
but only for a station with more than one candidate. Neither path
distinguished "resolved with real evidence" from "resolved because there
was only one option" -- both looked the same to a downstream consumer.
``IdentityResolution`` gives both paths one shape, so ``irix.live.
station_runner``, a future workout state machine (Section 5.6's
``identity_candidate``/``identity_confirmed``/``identity_degraded``
states), and any offline analysis of a recorded session can all reason
about identity confidence uniformly, instead of the crowded case being
the only one with evidence at all.

## Fused signals

Per the founding brief's explicit list -- camera trajectory, IMU motion,
timing, clock synchronization, station occupancy, camera zones, previous
confirmed identity, BLE context, motion onset -- each already has a real
home *before* this module (this is a deliberate consolidation, not a
rewrite): BLE context + station occupancy is ``irix.identity.
ble_pairing``/``CheckoutRegistry.resolve_member`` (a station's candidate
group *is* current occupancy); camera trajectory + IMU motion +
timing + previous confirmed identity is ``irix.identity.
motion_correlation.MotionCorrelationResolver`` (Phase 3: now also
carrying a continuity prior, see that module) fed clock-synchronized
IMU by ``StationSessionRunner.tick`` (Phase 3, see that module's
``synced_polled``); camera zones is which ``StationSessionRunner``
instance (one per camera/zone) is asking at all. "Motion onset" (a
just-arrived member's first detected movement, as a corroborating signal
distinct from steady-state periodic correlation) is not yet a separate
input anywhere in this repo -- noted as a real, not-yet-built gap rather
than silently treated as covered; see ``docs/TODO.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .motion_correlation import MotionCorrelationMatch

# Below this, a resolution is reported ambiguous rather than confident --
# matches irix.identity.motion_correlation.MotionCorrelationResolver's
# own min_confidence_margin*2 gate (that resolver already returns None
# instead of a low-confidence guess, so in practice a motion-correlated
# IdentityResolution's confidence is always either 0.0 (ambiguous, see
# from_no_match) or >= that gate); kept as an explicit, independent
# threshold here so a future resolution source with its own confidence
# scale still gets classified consistently.
AMBIGUOUS_BELOW_CONFIDENCE = 0.3


@dataclass
class IdentityResolution:
    """One detected person's resolved (or not) identity for one tick/
    window, with the evidence behind that resolution -- never just a
    bare member_id string past this point in the pipeline."""

    wristband_id: Optional[str]
    member_id: Optional[str]
    confidence: float
    ambiguous: bool
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "wristband_id": self.wristband_id,
            "member_id": self.member_id,
            "confidence": self.confidence,
            "ambiguous": self.ambiguous,
            "evidence": self.evidence,
        }


def from_sole_candidate(wristband_id: str, member_id: str) -> IdentityResolution:
    """The common, trivial case: exactly one checked-out band is present
    at this station/camera this tick, so whatever person the camera
    detects is unambiguously that member -- no motion correlation
    needed. Still confidence-scored and evidenced (not just a bare
    id) so a downstream consumer treats this the same way as a
    motion-correlated result, and so this resolution can be reported and
    later audited alongside crowded-station ones."""
    return IdentityResolution(
        wristband_id=wristband_id, member_id=member_id, confidence=1.0, ambiguous=False,
        evidence={"method": "sole_candidate_present"},
    )


def from_motion_correlation_match(
    wristband_id: str, match: Optional[MotionCorrelationMatch], had_prior: bool = False,
) -> IdentityResolution:
    """Wrap a ``MotionCorrelationResolver`` result (or lack of one) into
    the shared shape. ``match is None`` means the resolver itself
    couldn't confidently pick a candidate (too close to call, or no
    usable signal) -- reported as ambiguous with the actual reason
    surfaced in ``evidence``, never silently dropped.

    ``had_prior``: whether a previous-confirmed-identity continuity hint
    (``irix.live.disambiguation.CrowdedGroupDisambiguator``'s
    ``_last_slot_assignment``) was available for this slot at all this
    resolution -- surfaced as evidence regardless of whether it actually
    changed the outcome, so an auditor can distinguish "resolved purely
    from this window's motion" from "resolved with continuity support."
    """
    if match is None:
        return IdentityResolution(
            wristband_id=wristband_id, member_id=None, confidence=0.0, ambiguous=True,
            evidence={"method": "motion_correlation", "reason": "no_confident_match", "had_prior": had_prior},
        )
    return IdentityResolution(
        wristband_id=wristband_id, member_id=match.member_id, confidence=match.confidence,
        ambiguous=match.confidence < AMBIGUOUS_BELOW_CONFIDENCE,
        evidence={
            "method": "motion_correlation", "correlation": match.correlation,
            "person_index": match.person_index, "had_prior": had_prior,
        },
    )
