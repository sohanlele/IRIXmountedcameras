"""Stateful event producers -- logic that decides *when* to emit a
structured event, as opposed to schema.py's plain data shapes.

``BandPlacementTracker`` replaces the old irix.coaching.BandPlacementCoach:
same "only fire on an actual change" logic, but it now produces a
``BandPlacementRequiredEvent`` for the app to turn into an instruction,
rather than generating spoken text itself.
"""
from __future__ import annotations

from typing import Optional

from ..rep_counting.exercises import BandPlacement, ExerciseConfig
from .schema import BandPlacementRequiredEvent


class BandPlacementTracker:
    """Tracks where the member's IMU band is currently worn across a
    session and emits a ``BandPlacementRequiredEvent`` only when the next
    exercise needs it moved (Section 5 / ``ExerciseConfig.band_placement``).

    Most sessions never leave the wrist (curls, presses, deadlift,
    squat), so this stays silent except right around a machine-leg
    exercise like leg press or hack squat -- re-emitting on every
    exercise regardless of whether the band actually needs to move would
    just be noise for the app to filter back out.
    """

    def __init__(self, member_id: str, initial: BandPlacement = BandPlacement.WRIST):
        self.member_id = member_id
        self._current = initial

    @property
    def current_placement(self) -> BandPlacement:
        return self._current

    def event_for(self, exercise: ExerciseConfig) -> Optional[BandPlacementRequiredEvent]:
        """Call once when transitioning into `exercise`. Returns an event
        if the band needs to move, else None."""
        if exercise.band_placement == self._current:
            return None
        event = BandPlacementRequiredEvent(
            member_id=self.member_id,
            exercise=exercise.name,
            from_placement=self._current.value,
            to_placement=exercise.band_placement.value,
        )
        self._current = exercise.band_placement
        return event
