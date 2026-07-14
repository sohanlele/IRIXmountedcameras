"""Per-exercise fault-rule registry and score aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..pose.estimator import PersonPose
from ..rep_counting.exercises import EXERCISES, ExerciseConfig
from .rules import (
    FormFault,
    check_elbow_drift,
    check_hip_shoulder_rise,
    check_knee_valgus,
    check_squat_depth,
    check_torso_lean,
)

RuleFn = Callable[[List[PersonPose], Optional[ExerciseConfig]], Optional[FormFault]]

# Which fault checks apply to which exercise. Machine leg exercises
# (leg_press, hack_squat) reuse the squat checks -- same hip-knee-ankle
# joint triplet, same depth/valgus failure modes. bench_press has no
# entry yet: a meaningful bench fault check (elbow flare relative to the
# bar, not just the body) needs either a second camera angle or the
# barbell-tracking data from irix.barbell wired in, which this scaffold
# doesn't do yet -- form_score stays None for bench_press rather than
# reporting something not actually being measured.
FORM_RULES: Dict[str, List[RuleFn]] = {
    "squat": [check_squat_depth, check_knee_valgus],
    "leg_press": [check_squat_depth, check_knee_valgus],
    "hack_squat": [check_squat_depth, check_knee_valgus],
    "bicep_curl": [check_torso_lean, check_elbow_drift],
    "deadlift": [check_hip_shoulder_rise],
}

# How much one fault knocks off a perfect 1.0 score, scaled by its
# severity. Not a calibrated weighting (no ground-truth-labeled dataset
# to fit it against, unlike e.g. irix.barbell.rpe's velocity-loss-to-RPE
# mapping, which is drawn from published regression data) -- a
# deliberately simple, transparent heuristic: one clearly-flagged fault
# should visibly move the score, two shouldn't necessarily zero it out.
_FAULT_PENALTY_WEIGHT = 0.4


@dataclass
class FormAssessment:
    score: float  # 0-1, 1.0 = no faults detected
    faults: List[str] = field(default_factory=list)  # fault codes, e.g. ["knee_valgus"]


class FormScorer:
    """Scores one completed rep's buffered poses against its exercise's
    registered fault checks (see ``FORM_RULES``)."""

    def score_rep(self, exercise_name: str, poses: Optional[List[PersonPose]]) -> Optional[FormAssessment]:
        rules = FORM_RULES.get(exercise_name)
        if not rules or not poses:
            return None
        cfg = EXERCISES.get(exercise_name)
        faults: List[FormFault] = []
        for rule in rules:
            result = rule(poses, cfg)
            if result is not None:
                faults.append(result)
        if not faults:
            return FormAssessment(score=1.0, faults=[])
        penalty = sum(f.severity * _FAULT_PENALTY_WEIGHT for f in faults)
        score = max(0.0, min(1.0, 1.0 - penalty))
        return FormAssessment(score=score, faults=[f.code for f in faults])
