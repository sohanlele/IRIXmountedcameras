"""Coaching trigger logic (Section 7).

Turns a rep event (Section 4.2) into coaching speech text, entirely on the
local edge node -- no live cloud round-trip mid-set, consistent with the
audio-coaching latency budget in Section 7 and the personalization data
flow in Section 5.4 (profile pulled down to the edge node ahead of time).
"""
from __future__ import annotations

from typing import Optional

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
