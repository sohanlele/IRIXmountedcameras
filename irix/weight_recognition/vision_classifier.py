"""Pure vision plate classification -- v2 weight recognition (Section 4.4).

A model trained on plate color, diameter, and text to infer load without
any added hardware on the plates themselves. Removes the sticker
dependency but needs a larger training set and controlled lighting to be
reliable -- explicitly scoped in the design doc as a v2 target once the
sticker-based version (``PlateQRReader``) has proven the rest of the
pipeline. Not implemented in this scaffold.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class VisionPlateClassifier:
    """Placeholder for a future vision-only plate weight classifier (v2)."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def read_total_weight(self, frame: np.ndarray) -> Optional[float]:
        raise NotImplementedError(
            "Vision-only plate classification is a v2 target (Section 4.4) "
            "and is not implemented in this scaffold. Use PlateQRReader for v1."
        )
