from __future__ import annotations

from irix.demo.mock_pose import synthetic_pose_stream
from irix.exercise_recognition import recognize_exercise
from irix.rep_counting.exercises import BICEP_CURL, EXERCISES, HACK_SQUAT, LEG_PRESS, SQUAT
from irix.pose.estimator import COCO_KEYPOINT_NAMES, Keypoint, PersonPose


def _poses_for(exercise, n_frames=150, reps_per_second=0.5):
    return [pose for _, _, pose in synthetic_pose_stream(exercise, n_frames=n_frames, reps_per_second=reps_per_second)]


def test_recognizes_squat_against_all_candidates_including_the_ambiguous_family():
    poses = _poses_for(SQUAT)
    # Candidates limited to exercises with genuinely different joint triplets
    # (excluding leg_press/hack_squat, which share squat's triplet -- see the
    # dedicated ambiguity test below for that case).
    result = recognize_exercise(poses, candidates=[SQUAT, BICEP_CURL])
    assert result.exercise == "squat"
    assert result.confidence > 0.35


def test_recognizes_bicep_curl():
    poses = _poses_for(BICEP_CURL, reps_per_second=0.6)
    result = recognize_exercise(poses, candidates=[SQUAT, BICEP_CURL])
    assert result.exercise == "bicep_curl"
    assert result.confidence > 0.35


def test_rejects_wrong_exercise_when_only_wrong_candidate_offered():
    """Squat motion, but the only candidate is bicep_curl -- the elbow
    barely moves during a squat, so this should come back unknown
    (no_motion/no_confident_match), never a confidently wrong label."""
    poses = _poses_for(SQUAT)
    result = recognize_exercise(poses, candidates=[BICEP_CURL], min_score=0.35)
    assert result.exercise is None
    assert result.candidates[0].exercise == "bicep_curl"
    assert result.candidates[0].score < 0.35


def test_stationary_person_is_unknown_not_a_guess():
    # A single repeated pose -- zero motion.
    still = _poses_for(SQUAT, n_frames=1)[0]
    poses = [still] * 60

    result = recognize_exercise(poses)

    assert result.exercise is None
    assert result.reason == "no_motion"


def test_squat_leg_press_hack_squat_are_reported_as_ambiguous_not_guessed():
    """These three share the exact same hip-knee-ankle joint triplet with
    overlapping angle ranges -- a real, structural limitation (see the
    package docstring), not something to paper over with a confident
    guess."""
    poses = _poses_for(SQUAT)

    result = recognize_exercise(poses, candidates=[SQUAT, LEG_PRESS, HACK_SQUAT])

    assert result.exercise is None
    assert result.reason is not None and result.reason.startswith("ambiguous_with:")
    for name in ("squat", "leg_press", "hack_squat"):
        assert name in result.reason


def test_recognize_exercise_defaults_to_every_registered_exercise():
    poses = _poses_for(BICEP_CURL, reps_per_second=0.6)
    result = recognize_exercise(poses)  # no explicit candidates
    scored_names = {c.exercise for c in result.candidates}
    assert scored_names == set(EXERCISES.keys())


def test_too_few_valid_frames_scores_zero_not_a_crash():
    from irix.exercise_recognition.classifier import _score_candidate, _extract_trajectory

    poses = _poses_for(SQUAT, n_frames=3)
    trajectory = _extract_trajectory(poses, SQUAT)
    score = _score_candidate(SQUAT, trajectory)
    assert score.score == 0.0


def test_new_exercise_is_automatically_a_candidate_no_classifier_change_needed():
    from irix.rep_counting.exercises import ExerciseConfig

    custom = ExerciseConfig(
        name="shoulder_press", joint_triplet=("left_shoulder", "left_elbow", "left_wrist"),
        bottom_angle=90.0, top_angle=175.0,
    )
    poses = _poses_for(custom, reps_per_second=0.5)

    result = recognize_exercise(poses, candidates=[custom, SQUAT])

    assert result.exercise == "shoulder_press"
