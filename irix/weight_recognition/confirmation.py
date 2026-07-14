"""N-of-M confirmation windowing for weight extraction.

Generalized from the confirm_n/confirm_window + consistent_field +
validator pattern in jeffreyjy/IrixDemo's backend/guidance/spec.py
(``extraction_state``), which that system uses to turn a noisy per-frame
VLM read of a dumbbell's printed weight into a trustworthy value: require
several consecutive reads above a confidence threshold, all agreeing on
the same (validated) value, before accepting it. That system reads the
value with a cloud VLM (Gemini); IRIX's design doc explicitly avoids a
live cloud round-trip mid-set (Section 7), so this module keeps the
*windowing/consistency logic* -- which is backend-agnostic -- and applies
it to whatever local classifier produces a (value, confidence) reading:
``PlateQRReader`` (v1, already near-100% accurate so this matters less)
or ``VisionPlateClassifier`` (v2, where a single noisy frame is exactly
the "gaze-scan vs. stable read" problem this pattern solves).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


def validate_weight_kg(v: float, step_kg: float = 1.25, lo_kg: float = 1.25, hi_kg: float = 100.0) -> Optional[float]:
    """Range + grid validator for a plate/dumbbell weight reading (kg).

    Snaps to the nearest ``step_kg`` increment (commercial plates come in
    fixed steps) and rejects anything outside [lo_kg, hi_kg]. Analogous to
    IrixDemo's ``validate_weight_lbs`` (5 lb grid, [5, 200] lb range) --
    kg increments differ by gym/region, so the step is a parameter here
    rather than hardcoded.
    """
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    snapped = round(fv / step_kg) * step_kg
    return snapped if lo_kg <= snapped <= hi_kg else None


@dataclass
class ConfirmedReading(Generic[T]):
    value: T
    confidence: float


class ExtractionConfirmer(Generic[T]):
    """Accepts a value only once ``confirm_n`` of the last ``confirm_window``
    readings clear ``confidence_threshold``, pass ``validator``, and all
    agree on the same (validated) value -- rejecting a "gaze scan" where
    successive frames give different, individually-plausible answers.

    Usage::

        confirmer = ExtractionConfirmer(validator=validate_weight_kg)
        for value, confidence in stream_of_reads:
            result = confirmer.push(value, confidence)
            if result is not None:
                break  # result.value is the confirmed weight_kg
    """

    def __init__(
        self,
        validator: Optional[Callable[[T], Optional[T]]] = None,
        confidence_threshold: float = 0.85,
        confirm_n: int = 3,
        confirm_window: int = 3,
    ):
        self.validator = validator
        self.confidence_threshold = confidence_threshold
        self.confirm_n = confirm_n
        self._window: Deque[Tuple[T, float]] = deque(maxlen=confirm_window)

    def push(self, value: Optional[T], confidence: float) -> Optional[ConfirmedReading]:
        """Feed one (value, confidence) reading in. Returns a
        ConfirmedReading iff the window now satisfies confirm_n-of-window
        agreement; otherwise None (caller should keep reading)."""
        if value is not None and self.validator is not None:
            value = self.validator(value)
        if value is not None and confidence > self.confidence_threshold:
            self._window.append((value, confidence))
        else:
            self._window.clear()  # a bad frame breaks the consistency run

        if len(self._window) < self.confirm_n:
            return None
        recent = list(self._window)[-self.confirm_n:]
        values = {v for v, _ in recent}
        if len(values) != 1:
            return None
        agreed_value = recent[0][0]
        avg_confidence = sum(c for _, c in recent) / len(recent)
        return ConfirmedReading(value=agreed_value, confidence=avg_confidence)

    def reset(self) -> None:
        self._window.clear()
