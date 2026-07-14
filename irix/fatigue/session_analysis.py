"""Tracks fatigue across multiple sets of the same exercise for one
member within a session -- the cross-set trend a single SetFatigueAnalysis
can't see on its own (e.g. a member opening their 3rd set already 15%
slower than their 1st set opened, before that 3rd set has shown any
within-set velocity loss of its own)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import SessionFatigueSummary, SetFatigueAnalysis

_SessionKey = Tuple[str, str]  # (member_id, exercise)


class SessionFatigueTracker:
    """Stateful across a session (construct one per active gym session,
    or keep one long-lived instance keyed by member -- the tracker itself
    partitions by (member_id, exercise) internally, so one instance can
    serve every station)."""

    def __init__(self, within_set_weight: float = 0.6, across_set_weight: float = 0.4):
        self._history: Dict[_SessionKey, List[SetFatigueAnalysis]] = {}
        self._within_set_weight = within_set_weight
        self._across_set_weight = across_set_weight

    def add_set(self, member_id: str, exercise: str, analysis: SetFatigueAnalysis) -> SessionFatigueSummary:
        key = (member_id, exercise)
        sets = self._history.setdefault(key, [])
        sets.append(analysis)

        trend = self._set_to_set_velocity_trend(sets)
        index = self._session_fatigue_index(sets, trend)

        return SessionFatigueSummary(
            member_id=member_id,
            exercise=exercise,
            completed_sets=len(sets),
            set_analyses=list(sets),
            set_to_set_velocity_trend_pct=trend,
            session_fatigue_index=index,
        )

    def reset(self, member_id: str, exercise: Optional[str] = None) -> None:
        """Clear history -- call at the start of a new session (or when a
        member is confirmed to have moved on to a genuinely new exercise
        that shouldn't be compared against an old one)."""
        if exercise is None:
            for key in list(self._history):
                if key[0] == member_id:
                    del self._history[key]
        else:
            self._history.pop((member_id, exercise), None)

    def _set_to_set_velocity_trend(self, sets: List[SetFatigueAnalysis]) -> List[Optional[float]]:
        baseline = next((s.first_rep_velocity for s in sets if s.first_rep_velocity is not None), None)
        if baseline is None or baseline <= 0:
            return [None for _ in sets]
        trend = []
        for s in sets:
            if s.first_rep_velocity is None:
                trend.append(None)
            else:
                trend.append((baseline - s.first_rep_velocity) / baseline * 100.0)
        return trend

    def _session_fatigue_index(
        self, sets: List[SetFatigueAnalysis], trend: List[Optional[float]]
    ) -> Optional[float]:
        """0-1 heuristic: 0.6x the most recent set's own velocity loss +
        0.4x how much slower this set opened vs. the session's first set
        (both %, clamped to [0, 100] and scaled to [0, 1] before
        weighting). Explicitly a coarse heuristic for the app to use as
        *one* input, not a validated composite fatigue score -- there's
        no published formula this repo is aware of for combining
        within-set and across-set velocity loss into one index, unlike
        velocity_loss_pct itself (which does have direct literature
        support, see irix.barbell.rpe). None if there's no velocity
        signal to compute it from at all."""
        latest = sets[-1]
        within = latest.velocity_loss_pct
        across = trend[-1] if trend else None
        if within is None and across is None:
            return None
        within_norm = max(0.0, min(1.0, (within or 0.0) / 100.0))
        across_norm = max(0.0, min(1.0, (across or 0.0) / 100.0))
        if within is None:
            return across_norm
        if across is None:
            return within_norm
        return self._within_set_weight * within_norm + self._across_set_weight * across_norm
