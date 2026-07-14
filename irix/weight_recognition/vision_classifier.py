"""Pure-vision plate/load classification -- the only viable weight-reading
path given two hard constraints: nothing can be added to gym equipment
(no plate stickers -- that's an environment edit, not a camera install),
and printed plate numbers aren't legible from the Section 3.1 mounted-
camera geometry (3-4m back, 30-45 deg off-axis, chest height). Both rule
out this module's earlier QR-sticker (v1) and OCR-on-printed-numbers
ideas; see ``qr_reader.py`` for the QR approach, kept as reference but not
deployable under these constraints.

Rather than a hand-built classical-CV pipeline calibrated per gym against
that gym's specific plate colors/diameters (a real option, but one that
needs a manual per-plate-type setup step and doesn't generalize to
equipment the calibration never saw), this follows jeffreyjy/IrixDemo's
approach: ask a vision-language model to read the scene directly. A VLM
generalizes to whatever plates a given gym actually has without a
calibration step, the same way it generalized to reading a printed number
in their first-person case. See ``vlm_backend.py`` for why the backend
here defaults to a local/on-device model rather than their cloud Gemini
call, and ``confirmation.py`` for the N-of-M read-confirmation logic
(ported from the same source) that keeps a single noisy VLM read from
being trusted outright.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .confirmation import ConfirmedReading, ExtractionConfirmer, validate_weight_kg
from .vlm_backend import VLMBackend

# Mirrors the STANDARD_PROMPT_PREFIX pattern in jeffreyjy/IrixDemo's
# guidance/spec.py: every prompt gets the same camera-context opener so
# the model knows what kind of shot it's reasoning about. IRIX's mounted
# cameras are fixed, off-axis, third-person -- the opposite framing from
# their first-person glasses case, so the opener differs accordingly.
_PROMPT_PREFIX = (
    "This image is from a fixed gym camera mounted 3-4 meters from a "
    "free-weight station, at roughly 30-45 degrees off-axis and chest "
    "height. "
)

_LOAD_READ_PROMPT = _PROMPT_PREFIX + (
    "Identify every weight plate visible on the barbell or dumbbell at "
    "this station (there may be plates on both sleeves of a barbell). "
    "Use plate color, relative size, and any legible markings together -- "
    "don't rely on markings alone, they are often not legible at this "
    "distance and angle. Estimate the combined load in kilograms across "
    "all visible plates (excluding the bar itself). "
    "JSON: {plates_visible: boolean, total_weight_kg: number, "
    "confidence: number 0..1}."
)

_LOAD_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "plates_visible": {"type": "boolean"},
        "total_weight_kg": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["plates_visible", "confidence"],
}


class VisionPlateClassifier:
    """VLM-based total-load reader for a station's mounted camera.

    A single frame's read is noisy (bad angle, partial occlusion, a
    lifter's hand in the way), so this wraps an ``ExtractionConfirmer``:
    call ``read_frame`` once per incoming frame during setup, and only
    trust the result once ``confirm_n`` consecutive high-confidence reads
    agree (after ``validate_weight_kg`` snaps/range-checks the value).
    """

    def __init__(
        self,
        backend: VLMBackend,
        confirm_n: int = 3,
        confirm_window: int = 3,
        confidence_threshold: float = 0.8,
        weight_step_kg: float = 1.25,
    ):
        self.backend = backend
        self._confirmer = ExtractionConfirmer(
            validator=lambda v: validate_weight_kg(v, step_kg=weight_step_kg),
            confirm_n=confirm_n,
            confirm_window=confirm_window,
            confidence_threshold=confidence_threshold,
        )

    def read_frame(self, frame: np.ndarray) -> Optional[ConfirmedReading]:
        """Feed one frame in. Returns a ConfirmedReading once enough
        consistent reads have accumulated; otherwise None (caller should
        keep feeding frames, same as the confirm_n/confirm_window pattern
        in jeffreyjy/IrixDemo's extraction_state)."""
        result = self.backend.query(frame, _LOAD_READ_PROMPT, _LOAD_READ_SCHEMA)
        if not result.get("plates_visible"):
            self._confirmer.reset()
            return None
        value = result.get("total_weight_kg")
        confidence = float(result.get("confidence", 0.0) or 0.0)
        return self._confirmer.push(value, confidence)

    def reset(self) -> None:
        self._confirmer.reset()
