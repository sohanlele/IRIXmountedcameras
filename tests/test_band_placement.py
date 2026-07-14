from irix.coaching.triggers import BandPlacementCoach
from irix.rep_counting.exercises import BandPlacement, BICEP_CURL, LEG_PRESS, HACK_SQUAT, SQUAT


def test_no_prompt_when_staying_on_wrist():
    coach = BandPlacementCoach()  # defaults to wrist
    assert coach.prompt_for(SQUAT) is None
    assert coach.prompt_for(BICEP_CURL) is None
    assert coach.current_placement == BandPlacement.WRIST


def test_prompts_once_when_moving_to_ankle():
    coach = BandPlacementCoach()
    prompt = coach.prompt_for(LEG_PRESS)
    assert prompt is not None
    assert "ankle" in prompt
    assert coach.current_placement == BandPlacement.ANKLE
    # Next machine-leg exercise, still on the ankle -- no repeat prompt.
    assert coach.prompt_for(HACK_SQUAT) is None


def test_prompts_again_when_moving_back_to_wrist():
    coach = BandPlacementCoach()
    coach.prompt_for(LEG_PRESS)
    prompt = coach.prompt_for(BICEP_CURL)
    assert prompt is not None
    assert "wrist" in prompt
    assert coach.current_placement == BandPlacement.WRIST


def test_exercise_configs_have_expected_placement():
    assert SQUAT.band_placement == BandPlacement.WRIST
    assert BICEP_CURL.band_placement == BandPlacement.WRIST
    assert LEG_PRESS.band_placement == BandPlacement.ANKLE
    assert HACK_SQUAT.band_placement == BandPlacement.ANKLE
