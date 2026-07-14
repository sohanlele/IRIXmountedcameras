"""Coaching trigger logic (Section 7).

Turns a rep event (Section 4.2) into coaching speech text, entirely on the
local edge node -- no live cloud round-trip mid-set, consistent with the
audio-coaching latency budget in Section 7 and the personalization data
flow in Section 5.4 (profile pulled down to the edge node ahead of time).
"""
from __future__ import annotations

from typing import Optional

from ..rep_counting.exercises import BandPlacement, ExerciseConfig
from ..rep_counting.state_machine import RepEvent


class CoachingTrigger:
    def __init__(self, target_reps: Optional[int] = None):
        self.target_reps = target_reps

    def on_rep(self, event: RepEvent) -> str:
        """Return the coaching line to speak for a completed rep."""
        if self.target_reps:
            return f"Rep {event.rep_number} of {self.target_reps}."
        return f"Rep {event.rep_number}."

    def on_set_complete(self, exercise: str, total_reps: int) -> str:
        return f"Set complete. {total_reps} reps of {exercise.replace('_', ' ')}."


class BandPlacementCoach:
    """Tracks where the member's IMU band is currently worn across a
    session and emits a spoken reposition prompt only when the next
    exercise needs it moved (Section 5 / ``ExerciseConfig.band_placement``).

    Stateful by design: most sessions stay on the wrist the whole time
    (curls, presses, deadlift, squat), so this should stay silent except
    right before/after a machine-leg exercise like leg press or hack
    squat. Re-prompting every exercise regardless of whether the band
    actually needs to move would be exactly the kind of unnecessary
    friction the wristband is supposed to avoid.
    """

    def __init__(self, initial: BandPlacement = BandPlacement.WRIST):
        self._current = initial

    @property
    def current_placement(self) -> BandPlacement:
        return self._current

    def prompt_for(self, exercise: ExerciseConfig) -> Optional[str]:
        """Call once when transitioning into `exercise`. Returns a spoken
        instruction if the band needs to move, else None."""
        if exercise.band_placement == self._current:
            return None
        prompt = f"Move your IRIX band to your {exercise.band_placement.value} for {exercise.name.replace('_', ' ')}."
        self._current = exercise.band_placement
        return prompt
