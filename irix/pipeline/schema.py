"""Derived-metrics event schema (Section 6.3 / 8.2).

Camera -> zone edge box (pose + object detection + rep logic, all local)
-> local buffer -> aggregator -> cloud (derived metrics, member profile
sync only). Raw video never persists beyond the local debug buffer, and no
field here carries video or a statutorily-defined biometric identifier
(Section 8.1) -- only rep counts, form scores, and a wristband-assigned
member id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DerivedMetricsEvent:
    member_id: str        # wristband-assigned id, not a biometric identifier
    station_id: str
    exercise: str
    rep_count: int
    form_score: Optional[float] = None   # 0-1, None if not yet scored
    weight_kg: Optional[float] = None
    timestamp: float = field(default_factory=lambda: __import__("time").monotonic())

    def to_dict(self) -> dict:
        return {
            "member_id": self.member_id,
            "station_id": self.station_id,
            "exercise": self.exercise,
            "rep_count": self.rep_count,
            "form_score": self.form_score,
            "weight_kg": self.weight_kg,
            "timestamp": self.timestamp,
        }
