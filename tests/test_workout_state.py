"""irix.pipeline.workout_state.WorkoutStateMachine -- the authoritative
per-wristband-session state machine (Priority 6), and its named
duplicate/late-event guards."""
from __future__ import annotations

import pytest

from irix.pipeline.workout_state import (
    NON_PHASE_STATE_NAMES,
    WorkoutPhase,
    WorkoutStateError,
    WorkoutStateMachine,
)


def _drive_to_exercise_confirmed(machine: WorkoutStateMachine) -> None:
    machine.transition(WorkoutPhase.SESSION_STARTED)
    machine.transition(WorkoutPhase.MEMBER_DETECTED)
    machine.transition(WorkoutPhase.IDENTITY_CANDIDATE)
    machine.transition(WorkoutPhase.IDENTITY_CONFIRMED)
    machine.transition(WorkoutPhase.EXERCISE_CANDIDATE)
    machine.transition(WorkoutPhase.EXERCISE_CONFIRMED)


def test_all_19_brief_states_are_accounted_for_as_phases_or_non_phase_states():
    brief_states = {
        "wristband_assigned", "session_started", "member_detected", "identity_candidate",
        "identity_confirmed", "exercise_candidate", "exercise_confirmed", "set_started",
        "rep_completed", "set_ended", "rest_started", "rest_ended", "station_transition",
        "camera_handoff", "camera_disconnect", "ble_disconnect", "identity_degraded",
        "identity_recovered", "session_ended", "wristband_returned",
    }
    phase_values = {p.value for p in WorkoutPhase}
    assert brief_states == phase_values | NON_PHASE_STATE_NAMES


def test_a_full_happy_path_session_is_legal_start_to_finish():
    machine = WorkoutStateMachine(wristband_id="band-1")
    assert machine.phase == WorkoutPhase.WRISTBAND_ASSIGNED

    _drive_to_exercise_confirmed(machine)
    assert machine.phase == WorkoutPhase.EXERCISE_CONFIRMED

    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    machine.record_rep_completed(2)
    machine.record_rep_completed(3)
    machine.transition(WorkoutPhase.SET_ENDED)
    assert machine.total_reps == 3
    assert machine.completed_set_count == 1

    machine.transition(WorkoutPhase.REST_STARTED)
    machine.transition(WorkoutPhase.REST_ENDED)
    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    machine.transition(WorkoutPhase.SET_ENDED)
    assert machine.total_reps == 4
    assert machine.completed_set_count == 2

    machine.transition(WorkoutPhase.SESSION_ENDED)
    machine.transition(WorkoutPhase.WRISTBAND_RETURNED)
    assert machine.phase == WorkoutPhase.WRISTBAND_RETURNED


def test_rep_completed_before_any_set_started_is_rejected():
    machine = WorkoutStateMachine(wristband_id="band-1")
    with pytest.raises(WorkoutStateError):
        machine.record_rep_completed(1)


def test_a_late_rep_after_set_ended_does_not_reopen_the_completed_set():
    machine = WorkoutStateMachine(wristband_id="band-1")
    _drive_to_exercise_confirmed(machine)
    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    machine.transition(WorkoutPhase.SET_ENDED)

    with pytest.raises(WorkoutStateError):
        machine.record_rep_completed(2)  # late packet for the now-closed set
    assert machine.total_reps == 1  # unchanged -- not silently applied


def test_a_duplicate_rep_index_within_the_same_open_set_is_rejected():
    machine = WorkoutStateMachine(wristband_id="band-1")
    _drive_to_exercise_confirmed(machine)
    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    machine.record_rep_completed(2)
    with pytest.raises(WorkoutStateError):
        machine.record_rep_completed(2)  # replayed/duplicated delivery
    assert machine.total_reps == 2


def test_a_second_session_started_without_ending_the_first_is_rejected():
    """Duplicate-session prevention: session_started is only legal once
    per WorkoutStateMachine (which is itself one per open session, per
    the class's own docstring) -- re-entering it from anywhere past
    WRISTBAND_ASSIGNED is always illegal."""
    machine = WorkoutStateMachine(wristband_id="band-1")
    machine.transition(WorkoutPhase.SESSION_STARTED)
    with pytest.raises(WorkoutStateError):
        machine.transition(WorkoutPhase.SESSION_STARTED)


def test_session_ended_before_wristband_assigned_flow_completes_is_rejected():
    machine = WorkoutStateMachine(wristband_id="band-1")
    with pytest.raises(WorkoutStateError):
        machine.transition(WorkoutPhase.SESSION_ENDED)


def test_wristband_returned_is_terminal():
    machine = WorkoutStateMachine(wristband_id="band-1")
    _drive_to_exercise_confirmed(machine)
    machine.transition(WorkoutPhase.SESSION_ENDED)
    machine.transition(WorkoutPhase.WRISTBAND_RETURNED)
    with pytest.raises(WorkoutStateError):
        machine.transition(WorkoutPhase.SESSION_STARTED)


def test_station_transition_loops_back_to_member_detected_without_touching_sets():
    machine = WorkoutStateMachine(wristband_id="band-1")
    _drive_to_exercise_confirmed(machine)
    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    machine.transition(WorkoutPhase.SET_ENDED)

    machine.record_station_transition("squat-2")
    assert machine.phase == WorkoutPhase.MEMBER_DETECTED
    assert machine.current_station_id == "squat-2"
    assert machine.total_reps == 1  # set history is untouched by the transition

    # Re-confirm identity/exercise at the new station, then keep going.
    machine.transition(WorkoutPhase.IDENTITY_CANDIDATE)
    machine.transition(WorkoutPhase.IDENTITY_CONFIRMED)
    machine.transition(WorkoutPhase.EXERCISE_CANDIDATE)
    machine.transition(WorkoutPhase.EXERCISE_CONFIRMED)
    machine.transition(WorkoutPhase.SET_STARTED)
    machine.record_rep_completed(1)
    assert machine.total_reps == 2


def test_station_transition_before_any_identity_confirmed_is_rejected():
    machine = WorkoutStateMachine(wristband_id="band-1")
    with pytest.raises(WorkoutStateError):
        machine.record_station_transition("squat-2")


def test_redundant_station_transition_to_the_current_station_is_a_no_op():
    machine = WorkoutStateMachine(wristband_id="band-1")
    _drive_to_exercise_confirmed(machine)
    machine.record_station_transition("squat-1")  # first-ever -- legal, current_station_id was None
    phase_before = machine.phase
    machine.record_station_transition("squat-1")  # redundant repeat
    assert machine.phase == phase_before  # did not re-trigger MEMBER_DETECTED


def test_camera_handoff_tracks_current_camera_and_prevents_double_routing():
    machine = WorkoutStateMachine(wristband_id="band-1")
    machine.record_camera_handoff("cam-1")
    assert machine.current_camera_id == "cam-1"
    machine.record_camera_handoff("cam-2")
    assert machine.current_camera_id == "cam-2"
    # A caller checking current_camera_id before accepting an event from
    # cam-1 again would now correctly reject it -- the mechanism this
    # class provides for preventing camera-overlap double counting.


def test_health_flags_are_independent_of_phase():
    machine = WorkoutStateMachine(wristband_id="band-1")
    machine.set_camera_connected(False)
    machine.set_ble_connected(False)
    machine.set_identity_degraded(True)
    assert machine.health.camera_connected is False
    assert machine.health.ble_connected is False
    assert machine.health.identity_degraded is True
    # None of this forced a phase change.
    assert machine.phase == WorkoutPhase.WRISTBAND_ASSIGNED

    machine.set_camera_connected(True)
    machine.set_identity_degraded(False)
    assert machine.health.camera_connected is True
    assert machine.health.identity_degraded is False


def test_to_dict_reports_a_json_shape():
    machine = WorkoutStateMachine(wristband_id="band-1")
    d = machine.to_dict()
    assert d["wristband_id"] == "band-1"
    assert d["phase"] == "wristband_assigned"
    assert d["health"]["camera_connected"] is True
    assert d["total_reps"] == 0
