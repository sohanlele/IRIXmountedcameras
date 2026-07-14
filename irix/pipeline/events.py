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

    def event_for(
        self, exercise: ExerciseConfig, timestamp: Optional[float] = None,
    ) -> Optional[BandPlacementRequiredEvent]:
        """Call once when transitioning into `exercise`. Returns an event
        if the band needs to move, else None.

        ``timestamp``: the actual transition time, when the caller has
        one (e.g. a session's ``start_ts``) -- passed through to the
        event instead of leaving it at the dataclass's wall-clock
        default, so a deterministic-replay run doesn't leak real
        wall-clock time into an otherwise fully-reproducible event
        stream. ``None`` (default) preserves the old wall-clock-default
        behavior for any existing caller that doesn't have a timestamp
        to give."""
        if exercise.band_placement == self._current:
            return None
        kwargs = dict(
            member_id=self.member_id,
            exercise=exercise.name,
            from_placement=self._current.value,
            to_placement=exercise.band_placement.value,
        )
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        event = BandPlacementRequiredEvent(**kwargs)
        self._current = exercise.band_placement
        return event


class RestGapSetBoundaryDetector:
    """Detects a set boundary in a continuous stream of completed reps
    from real footage -- the piece the mock demos never needed, since
    they script a fixed number of reps and then hand-construct
    ``SetCompleteEvent`` at the end (see ``irix.demo.run_demo``/
    ``run_gym_demo``). A real uploaded video has no such script: nothing
    tells the pipeline in advance where one set ends and the next begins,
    so ``irix.demo.run_upload`` needs to infer it from the rep stream
    itself.

    The heuristic: if the gap since the previous completed rep exceeds
    ``rest_gap_s``, the previous set is considered already closed. This
    is deliberately simple -- a fixed threshold can't distinguish one
    unusually slow rep from the start of a rest period -- but it is the
    standard approach in wearable/video activity-segmentation work for
    exactly this problem (an inter-event-gap / "epoch" threshold), and a
    generous default (20s) comfortably separates within-set inter-rep
    gaps (a few seconds, even at a slow tempo) from between-set rest
    (typically 60s+, though configurable down for supersets or
    back-to-back circuit work where rest is short).
    """

    def __init__(self, rest_gap_s: float = 20.0):
        self.rest_gap_s = rest_gap_s
        self._last_rep_ts: Optional[float] = None

    def observe(self, rep_timestamp: float) -> bool:
        """Call once per completed rep, in timestamp order. Returns True
        if this rep follows a gap long enough that the *previous* set
        should be treated as already closed -- the caller is responsible
        for finalizing the previous set (using the previous rep's own
        timestamp as its end) before folding this rep into a new one.
        Always False for the first rep observed (nothing to compare
        against yet)."""
        is_boundary = self._last_rep_ts is not None and (rep_timestamp - self._last_rep_ts) >= self.rest_gap_s
        self._last_rep_ts = rep_timestamp
        return is_boundary

    def reset(self) -> None:
        self._last_rep_ts = None
