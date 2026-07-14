"""Zero-training exercise classifier -- see package docstring for the
full design rationale (why not a trained sequence model, and the
squat/leg_press/hack_squat structural ambiguity this handles honestly).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.signal import find_peaks

from ..pose.estimator import PersonPose
from ..pose.geometry import joint_angle
from ..rep_counting.exercises import EXERCISES, ExerciseConfig

MIN_VALID_FRAMES = 10
DEFAULT_MIN_SCORE = 0.35
DEFAULT_AMBIGUITY_MARGIN = 0.08


@dataclass
class ExerciseCandidateScore:
    """One candidate exercise's score against the observed pose window --
    every candidate is reported (not just the winner) so a caller can see
    *why* a decision was or wasn't made, and so tests/observability don't
    have to trust a single opaque number."""

    exercise: str
    range_of_motion_deg: float
    coverage: float  # 0-1, how well observed ROM matches this exercise's configured angle range
    n_cycles: int
    regularity: float  # 0-1, higher = more evenly-spaced cycles
    activity: float  # 0-1, motion-energy gate (0 for a stationary person)
    score: float  # combined 0-1


@dataclass
class ExerciseRecognitionResult:
    exercise: Optional[str]  # None = unknown
    confidence: float  # 0-1
    candidates: List[ExerciseCandidateScore] = field(default_factory=list)
    reason: Optional[str] = None  # set when exercise is None: "no_motion" | "ambiguous_with:a,b" | "no_candidates"


def _extract_trajectory(poses: List[PersonPose], config: ExerciseConfig) -> np.ndarray:
    a_name, b_name, c_name = config.joint_triplet
    angles = []
    for pose in poses:
        a, b, c = pose.get(a_name), pose.get(b_name), pose.get(c_name)
        if a is None or b is None or c is None or min(a.confidence, b.confidence, c.confidence) <= 0:
            angles.append(np.nan)
            continue
        angles.append(joint_angle(pose.xy(a_name), pose.xy(b_name), pose.xy(c_name)))
    return np.asarray(angles, dtype=float)


def _score_candidate(config: ExerciseConfig, trajectory: np.ndarray) -> ExerciseCandidateScore:
    valid = trajectory[~np.isnan(trajectory)]
    if len(valid) < MIN_VALID_FRAMES:
        return ExerciseCandidateScore(config.name, 0.0, 0.0, 0, 0.0, 0.0, 0.0)

    observed_lo, observed_hi = float(valid.min()), float(valid.max())
    rom = observed_hi - observed_lo

    cfg_lo, cfg_hi = sorted((config.bottom_angle, config.top_angle))
    cfg_range = max(cfg_hi - cfg_lo, 1e-6)

    # Coverage: overlap between the observed [lo, hi] band and this
    # exercise's configured band, as a fraction of the configured band --
    # 1.0 if the member swept through (at least) the full configured
    # range, less if they only moved through part of it or moved outside
    # it (e.g. a curl's elbow angle sweeping through a squat's knee-angle
    # range by coincidence would still be penalized here since the two
    # configured bands don't overlap at all).
    overlap = max(0.0, min(observed_hi, cfg_hi) - max(observed_lo, cfg_lo))
    coverage = float(np.clip(overlap / cfg_range, 0.0, 1.0))

    # Activity gate: near-zero for a standing-still (or barely-moving)
    # person -- rules out "coincidentally within range but not actually
    # exercising" (e.g. resting between sets). hysteresis*2 is a
    # reasonable noise floor -- RepCounter itself uses `hysteresis` as
    # its own double-count guard band, so real rep motion should clear
    # comfortably more than that.
    activity = float(np.clip(rom / (config.hysteresis * 2.0), 0.0, 1.0))

    # Cycle count + regularity via peak detection on both the raw and
    # inverted signal (peaks = tops, troughs = bottoms of each rep).
    # prominence tied to this exercise's configured range so a candidate
    # with a wide angle range doesn't get spurious peaks from small
    # noise wiggles counted as reps.
    prominence = max(cfg_range * 0.15, 3.0)
    peak_idx, _ = find_peaks(valid, prominence=prominence)
    trough_idx, _ = find_peaks(-valid, prominence=prominence)
    extrema = np.sort(np.concatenate([peak_idx, trough_idx]))
    n_cycles = max(0, len(extrema) - 1)

    if n_cycles >= 2:
        intervals = np.diff(extrema).astype(float)
        cv = float(np.std(intervals) / np.mean(intervals)) if np.mean(intervals) > 0 else 1.0
        regularity = float(np.clip(1.0 - cv, 0.0, 1.0))
    elif n_cycles == 1:
        regularity = 0.5  # one half-cycle seen -- plausible but not yet confirmed regular
    else:
        regularity = 0.0

    # Geometric mean: any one weak factor pulls the combined score down
    # rather than a strong factor in one dimension masking a genuinely
    # weak match in another (e.g. high coverage from a slow, one-off
    # movement that never repeats shouldn't score as well as a real set).
    factors = [coverage, activity, max(regularity, 0.15 if n_cycles >= 1 else 0.0)]
    score = float(np.prod(factors) ** (1.0 / len(factors))) if all(f >= 0 for f in factors) else 0.0

    return ExerciseCandidateScore(
        exercise=config.name, range_of_motion_deg=rom, coverage=coverage,
        n_cycles=n_cycles, regularity=regularity, activity=activity, score=score,
    )


def recognize_exercise(
    poses: List[PersonPose],
    candidates: Optional[List[ExerciseConfig]] = None,
    min_score: float = DEFAULT_MIN_SCORE,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> ExerciseRecognitionResult:
    """Classify which of ``candidates`` (default: every exercise in
    ``irix.rep_counting.exercises.EXERCISES``) the pose window in
    ``poses`` looks like, or ``None`` ("unknown") if nothing scores well
    enough, or if the top candidates are too close to call apart --see
    module/package docstrings for why that second case is a real,
    structural limitation (e.g. squat vs. leg_press vs. hack_squat) and
    not a bug to be tuned away.

    ``poses`` should be a short rolling window (a few seconds at camera
    frame rate is enough to see 1+ rep cycles for a real set; too short a
    window under-counts cycles and lowers ``regularity``, too long mixes
    across a rest period and lowers ``activity``/``coverage``).
    """
    exercise_configs = candidates if candidates is not None else list(EXERCISES.values())
    if not exercise_configs:
        return ExerciseRecognitionResult(exercise=None, confidence=0.0, reason="no_candidates")

    scores = [_score_candidate(cfg, _extract_trajectory(poses, cfg)) for cfg in exercise_configs]
    scores.sort(key=lambda s: s.score, reverse=True)

    if scores[0].score < min_score:
        reason = "no_motion" if scores[0].activity < 0.2 else "no_confident_match"
        return ExerciseRecognitionResult(exercise=None, confidence=scores[0].score, candidates=scores, reason=reason)

    tied = [s for s in scores if scores[0].score - s.score <= ambiguity_margin]
    if len(tied) > 1:
        tied_names = ",".join(s.exercise for s in tied)
        return ExerciseRecognitionResult(
            exercise=None, confidence=scores[0].score, candidates=scores,
            reason=f"ambiguous_with:{tied_names}",
        )

    return ExerciseRecognitionResult(exercise=scores[0].exercise, confidence=scores[0].score, candidates=scores)
