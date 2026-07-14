"""Exercise recognition: which configured exercise (if any) the currently
observed body motion actually matches.

## Why this exists

Nothing elsewhere in this repo classifies exercise from vision --
``irix.rep_counting`` requires a station's ``exercise_name`` to already
be known (``StationInfo.default_exercise`` or an explicit
``run_upload.py --exercise`` flag), and simply counts reps against that
exercise's pre-configured joint-angle range
(``irix.rep_counting.exercises.ExerciseConfig``). That's a real,
previously undocumented-as-a-gap limitation against the founding brief's
"recognize exercises" requirement (see ``docs/ROADMAP.md``/
``docs/IMPLEMENTATION_STATUS.md``) -- this package closes it.

## Why this approach, not a trained sequence model

Modern exercise/action recognition (ST-GCN -- Yan et al., AAAI 2018,
arXiv:1801.07455; temporal-attention-augmented GCN variants, e.g.
arXiv:2010.12221; video/pose transformers) all require a labeled
training set of pose sequences per exercise class. The one public
dataset built for exactly this camera+IMU combination -- MM-Fit
(Stromback, Huang & Radu, IMWUT 2020,
https://mmfit.github.io/, https://github.com/KDMStromback/mm-fit) --
is a real, non-proprietary resource and the correct target to train
against eventually, but it is multiple GB of synchronized RGB-D video +
IMU data, meant to be downloaded and trained against on real GPU
hardware -- not something to fetch and train in this sandboxed,
GPU-less environment. Shipping a randomly-initialized or
undertrained "deep model" here would be worse than being honest about
the gap: it would produce confident-looking wrong answers, the one
thing every identity/exercise/weight decision in this repo is
specifically designed to avoid (see ``docs/PRODUCT_SPEC.md``'s
non-goals -- "unknown is always preferred to a guessed/incorrect"
output).

Instead, this module implements the strongest defensible **zero-training**
baseline: for each candidate ``ExerciseConfig`` already defined in
``irix.rep_counting.exercises``, compute that exercise's own joint-angle
trajectory (the same joint triplet ``RepCounter`` would use) over a
short pose window, and score how well it actually looks like *that*
exercise being performed -- range-of-motion coverage against the
exercise's configured angle range, cycle count/regularity (via
``scipy.signal.find_peaks``, not hand-rolled peak detection), and a
motion-energy gate that rejects a standing-still person outright. This
is real-time, pose-sequence-classification-adjacent HAR grounded in the
same "deterministic finite-state" family the current rep counter already
uses, extended to *which* joint's motion looks legitimate rather than
just *whether* one already-chosen joint crossed a threshold -- see
``docs/RESEARCH_LOG.md`` for the wider prior-art survey (real-time
pose-based exercise recognition with deterministic methods is an
established, not novel, approach at this scale).

**A structural limitation, stated honestly rather than hidden:**
``squat``, ``leg_press``, and ``hack_squat`` all use the same
hip-knee-ankle joint triplet with overlapping angle ranges (see
``irix.rep_counting.exercises``) -- they are not distinguishable from
joint-angle trajectory shape alone, no matter how sophisticated the
scoring. Disambiguating them needs information this module doesn't have
(which station/equipment the member is at, torso-lean/seat-back context,
or a real trained model with more discriminative features). Rather than
guess, ``recognize_exercise`` detects this exact case (near-tied top
candidates) and returns ``exercise=None`` ("unknown") with every tied
candidate reported, instead of confidently picking one -- consistent
with every other identity-adjacent decision in this repo.

**Extending to a new exercise requires zero classifier code changes**:
add an ``ExerciseConfig`` to ``irix.rep_counting.exercises.EXERCISES``
(joint triplet + angle range) and it is automatically a scoring
candidate here.
"""
from .classifier import (
    ExerciseCandidateScore,
    ExerciseRecognitionResult,
    recognize_exercise,
)

__all__ = ["ExerciseCandidateScore", "ExerciseRecognitionResult", "recognize_exercise"]
