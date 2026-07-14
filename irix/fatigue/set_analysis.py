"""Aggregates one completed set's per-rep samples into a SetFatigueAnalysis."""
from __future__ import annotations

from collections import Counter
from typing import List, Optional

from .models import RepFatigueSample, SetFatigueAnalysis

# Descending so the first (largest) threshold a set's loss meets or
# exceeds wins -- matches how VL10/VL20/VL30/VL45 are used as named zones
# in the VBT literature cited in irix.barbell.rpe (a set with 32% loss is
# "in VL30", not double-counted as also being in VL10).
_VELOCITY_LOSS_ZONES = [(45.0, "VL45"), (30.0, "VL30"), (20.0, "VL20"), (10.0, "VL10")]


class SetFatigueAnalyzer:
    """Stateless -- call ``analyze()`` once per completed set with every
    rep sample from that set, in any order (it sorts by ``rep_number``).
    """

    def analyze(self, exercise: str, reps: List[RepFatigueSample]) -> Optional[SetFatigueAnalysis]:
        if not reps:
            return None
        reps = sorted(reps, key=lambda r: r.rep_number)

        tier = self._pick_tier(reps)
        velocities = [self._velocity(r, tier) for r in reps]
        loss_trend = self._loss_trend(velocities)
        first_v = next((v for v in velocities if v is not None), None)
        last_v = next((v for v in reversed(velocities) if v is not None), None)
        vl_pct = loss_trend[-1] if loss_trend and loss_trend[-1] is not None else None

        form_scores = [r.form_score for r in reps]
        valid_scores = [s for s in form_scores if s is not None]
        mean_form = sum(valid_scores) / len(valid_scores) if valid_scores else None

        all_faults = [f for r in reps for f in r.form_faults]
        most_common = Counter(all_faults).most_common(1)[0][0] if all_faults else None

        return SetFatigueAnalysis(
            exercise=exercise,
            rep_count=len(reps),
            velocity_tier=tier,
            first_rep_velocity=first_v,
            last_rep_velocity=last_v,
            velocity_loss_pct=vl_pct,
            velocity_loss_trend_pct=loss_trend,
            velocity_loss_zone=self._zone_for(vl_pct),
            tempo_drift_pct=self._tempo_drift(reps),
            mean_form_score=mean_form,
            form_score_trend=form_scores,
            most_common_fault=most_common,
        )

    def _pick_tier(self, reps: List[RepFatigueSample]) -> str:
        if any(r.mean_velocity_m_s is not None for r in reps):
            return "m_s"
        if any(r.mean_velocity_deg_s is not None for r in reps):
            return "deg_s"
        return "none"

    def _velocity(self, rep: RepFatigueSample, tier: str) -> Optional[float]:
        if tier == "m_s":
            return rep.mean_velocity_m_s
        if tier == "deg_s":
            return rep.mean_velocity_deg_s
        return None

    def _loss_trend(self, velocities: List[Optional[float]]) -> List[Optional[float]]:
        first = next((v for v in velocities if v is not None), None)
        if first is None or first <= 0:
            return [None for _ in velocities]
        trend = []
        for v in velocities:
            if v is None:
                trend.append(None)
            else:
                trend.append((first - v) / first * 100.0)
        return trend

    def _zone_for(self, vl_pct: Optional[float]) -> Optional[str]:
        if vl_pct is None:
            return None
        for threshold, zone in _VELOCITY_LOSS_ZONES:
            if vl_pct >= threshold:
                return zone
        return None

    def _tempo_drift(self, reps: List[RepFatigueSample]) -> Optional[float]:
        durations = [r.duration_s for r in reps if r.duration_s is not None and r.duration_s > 0]
        if len(durations) < 2:
            return None
        first, last = durations[0], durations[-1]
        return (last - first) / first * 100.0
