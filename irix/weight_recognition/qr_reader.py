"""QR/barcode plate-sticker reader -- kept for reference, not deployable.

This was the original v1 weight-recognition plan (Section 4.4): a cheap,
durable sticker on each plate that the camera reads directly. Near-100%
accurate and trivial to compute, but it requires adding something to
every plate in the gym -- which is an environment edit, and IRIX's
install constraint is camera-only (nothing else about the gym floor can
change). ``VisionPlateClassifier`` (vision_classifier.py) is the actual
deployable path. This module is kept as a reference implementation and
because the confirm/consistency pattern it doesn't need (QR reads are
already near-deterministic) is a useful contrast against the VLM path,
which does need it. ``pyzbar`` is an optional dependency (see
pyproject.toml `[qr]` extra).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Example sticker-code -> plate weight (kg) registry. In a real deployment
# this would be seeded per-gym during install/calibration (Section 6/11).
PLATE_REGISTRY: Dict[str, float] = {
    "IRIX-PLATE-1.25": 1.25,
    "IRIX-PLATE-2.5": 2.5,
    "IRIX-PLATE-5": 5.0,
    "IRIX-PLATE-10": 10.0,
    "IRIX-PLATE-15": 15.0,
    "IRIX-PLATE-20": 20.0,
    "IRIX-PLATE-25": 25.0,
}


class PlateQRReader:
    """Decodes plate-sticker QR/barcodes in a frame and sums recognized load."""

    def __init__(self, registry: Optional[Dict[str, float]] = None):
        self.registry = registry or PLATE_REGISTRY

    def _decode(self, frame: np.ndarray) -> List[str]:
        try:
            from pyzbar.pyzbar import decode
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "pyzbar is required for plate QR reading. Install the 'qr' extra: pip install irix[qr]"
            ) from exc
        return [d.data.decode("utf-8") for d in decode(frame)]

    def read_total_weight(self, frame: np.ndarray) -> Optional[float]:
        """Sum the weight of every recognized plate sticker visible in the frame.

        Returns None if no recognized stickers were found (caller should fall
        back to the vision-only classifier, Section 4.4 v2).
        """
        codes = self._decode(frame)
        weights = [self.registry[c] for c in codes if c in self.registry]
        if not weights:
            return None
        return sum(weights)
