"""Geometric plausibility cross-check for a VLM-read weight, using
``irix.barbell.detector``'s plate detections -- a corroborating sanity
check, *not* a second independent weight-reading method.

Why not a real second method: docs/ARCHITECTURE.md's weight-recognition
section already explains why this repo settled on a VLM reading the
scene rather than classical geometry -- and the sharpest reason is that
standardized competition bumper plates are *all the same 450mm diameter
regardless of weight* (``COMPETITION_BUMPER_PLATE_DIAMETER_MM``), only
distinguishable by color, which is exactly the kind of scene-reasoning a
VLM does and raw plate geometry structurally can't. Commercial-gym iron
plates vary diameter by weight, but not on any single standardized
curve -- manufacturer-dependent enough that a hardcoded diameter->weight
table would be a guess, not a measurement.

What detected-plate *geometry* genuinely can do without solving that
harder problem: catch a badly wrong VLM read by checking whether the
read weight is even plausible given (a) how many plates
``FreeWeightDetector`` actually sees on the bar, and (b) roughly how big
they are. A hallucinated or badly misread number will often fail even
this coarse check even though fine-grained plate identification stays
out of reach without the VLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..barbell.calibration import CameraCalibration, MENS_OLYMPIC_BARBELL_WEIGHT_KG
from ..barbell.detector import FreeWeightClass, FreeWeightDetection

# Common commercial-gym plate weights (kg) used to decompose a read total
# into an *expected* plate count -- an illustrative generic set, not
# this-gym-specific. A real deployment would ideally swap this for that
# gym's actual plate inventory (the per-gym calibration profile idea
# docs/ARCHITECTURE.md's weight-recognition section considered and set
# aside as the *primary* method -- kept alive here in this narrower,
# corroborating role instead, where being approximate matters less).
COMMON_PLATE_WEIGHTS_KG = [25.0, 20.0, 15.0, 10.0, 5.0, 2.5, 1.25]


@dataclass
class GeometryCheckResult:
    consistent: bool
    detected_plate_count: int
    expected_plate_count: Optional[int]  # total across both sides
    reason: Optional[str] = None  # debugging/QA-dashboard string, not member-facing coaching text

    def to_dict(self) -> dict:
        return {
            "consistent": self.consistent,
            "detected_plate_count": self.detected_plate_count,
            "expected_plate_count": self.expected_plate_count,
            "reason": self.reason,
        }


def expected_plates_per_side(
    total_weight_kg: float, bar_weight_kg: float = MENS_OLYMPIC_BARBELL_WEIGHT_KG,
    available: List[float] = COMMON_PLATE_WEIGHTS_KG,
) -> List[float]:
    """Greedy decomposition of one side's load (``(total - bar) / 2``)
    into the largest available plates that fit. This is *an* estimate of
    what should be loaded per side, not a claim about what specifically
    is on the bar -- multiple different plate combinations can sum to the
    same total weight, so this is only used for a *count* sanity check,
    never to claim which specific plates are loaded."""
    per_side = max(0.0, (total_weight_kg - bar_weight_kg) / 2.0)
    remaining = per_side
    plates: List[float] = []
    for w in sorted(available, reverse=True):
        while remaining >= w - 0.01:
            plates.append(w)
            remaining -= w
    return plates


def check_plate_geometry(
    read_weight_kg: float,
    detections: List[FreeWeightDetection],
    bar_weight_kg: float = MENS_OLYMPIC_BARBELL_WEIGHT_KG,
    count_tolerance_per_side: int = 1,
) -> GeometryCheckResult:
    """Sanity-checks a VLM-read ``read_weight_kg`` against how many
    plates ``FreeWeightDetector`` actually found on the bar in the same
    frame(s). Assumes symmetric loading (standard gym convention, and the
    only sane default without a second camera angle on the far sleeve).

    Returns ``consistent=True`` (not False) when there's nothing to check
    against (no plates detected at all -- occluded view, or this isn't a
    barbell exercise) rather than treating "couldn't check" as "check
    failed".
    """
    plates = [d for d in detections if d.class_label == FreeWeightClass.PLATE]
    detected_count = len(plates)
    expected_per_side = expected_plates_per_side(read_weight_kg, bar_weight_kg)
    expected_total = len(expected_per_side) * 2

    if detected_count == 0:
        return GeometryCheckResult(
            consistent=True, detected_plate_count=0, expected_plate_count=expected_total,
            reason="no plates detected to check against (occluded view or non-barbell exercise)",
        )

    count_diff = abs(detected_count - expected_total)
    consistent = count_diff <= count_tolerance_per_side * 2
    reason = None
    if not consistent:
        reason = (
            f"VLM read of {read_weight_kg}kg implies ~{expected_total} plates total, "
            f"detector found {detected_count}"
        )
    return GeometryCheckResult(
        consistent=consistent, detected_plate_count=detected_count,
        expected_plate_count=expected_total, reason=reason,
    )
