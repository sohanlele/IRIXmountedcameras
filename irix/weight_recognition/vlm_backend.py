"""Pluggable vision-language-model backend for plate/load reading.

Jeffrey's system (jeffreyjy/IrixDemo) resolves the "every gym has
different plates" problem by not trying to solve it with a lookup table
at all: it asks a VLM to read the scene directly (color, shape, printed
text, context), so it generalizes to equipment it's never seen calibrated
against. That's the right call for IRIX too, now that neither plate
stickers (an environment edit) nor OCR on printed numbers (illegible from
the Section 3.1 mounted-camera angle/distance) are on the table -- a
per-gym color/diameter calibration profile was the classical-CV
alternative, but it re-introduces a manual setup step per plate type per
gym, which a VLM avoids by construction.

Where this diverges from his implementation: his backend is a cloud
Gemini call per frame, which conflicts with this design's own privacy
stance (Section 8: raw video never leaves the building). ``VLMBackend``
is therefore a protocol with two implementations:

- ``LocalVLMBackend`` (default/recommended): an on-device open-source VLM
  (e.g. Moondream, a quantized Qwen2-VL/LLaVA served via Ollama or
  llama.cpp) running on the zone edge box -- the same Jetson hardware
  already budgeted in Section 6. Frames never leave the building.
- ``GeminiVLMBackend``: mirrors jeffreyjy/IrixDemo's actual approach
  (cloud Gemini, structured JSON output) for parity/demo use. Included
  because it's a real, validated option -- but it trades the privacy
  property above for zero on-device model-serving setup, so it's opt-in,
  not the default.

Both are thin sketches (no bundled model weights or API wiring beyond a
lazy import), same pattern as ``PoseEstimator``'s optional ultralytics
dependency -- the interesting logic is ``VisionPlateClassifier`` and
``ExtractionConfirmer``, not the backend plumbing.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol

import numpy as np


class VLMBackend(Protocol):
    def query(self, frame: np.ndarray, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run one VLM call against a frame, return the parsed JSON response."""
        ...


class LocalVLMBackend:
    """On-device VLM served locally on the zone edge box (default backend).

    Sketch of an integration against an Ollama-style local HTTP server
    (``POST /api/generate`` with a JSON-constrained response), which is
    how small open-source VLMs (Moondream, LLaVA, Qwen2-VL) are commonly
    self-hosted. Swap in whatever local serving stack the edge box
    actually runs -- the important property is that ``endpoint_url``
    stays on localhost / the zone LAN, never a public API.
    """

    def __init__(self, endpoint_url: str = "http://localhost:11434/api/generate", model: str = "moondream"):
        self.endpoint_url = endpoint_url
        self.model = model

    def query(self, frame: np.ndarray, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(
            "LocalVLMBackend is a sketch: point endpoint_url at a real local "
            "VLM server (e.g. `ollama serve` with a vision model pulled) and "
            "implement the HTTP call + JSON-schema-constrained decoding here. "
            "Kept unimplemented rather than guessed at, since the right call "
            "shape depends on which local serving stack the edge box runs."
        )


class GeminiVLMBackend:
    """Cloud Gemini backend -- mirrors jeffreyjy/IrixDemo's actual approach.

    Structured output via ``google-genai``'s ``response_schema``, same
    shape as their ``backend/guidance/spec.py``. Opt-in only: using this
    means camera frames leave the building on every read, which conflicts
    with Section 8's data-minimization stance. Reasonable for a demo where
    that trade-off is acceptable; not the recommended default for a
    deployed pilot.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _load_client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "google-genai is required for GeminiVLMBackend. "
                    "Install it explicitly: pip install google-genai"
                ) from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def query(self, frame: np.ndarray, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        import json

        import cv2

        client = self._load_client()
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:  # pragma: no cover
            raise ValueError("Failed to encode frame as JPEG")
        response = client.models.generate_content(
            model=self.model,
            contents=[
                {"mime_type": "image/jpeg", "data": encoded.tobytes()},
                prompt,
            ],
            config={"response_mime_type": "application/json", "response_schema": schema},
        )
        return json.loads(response.text)


class FakeVLMBackend:
    """Test/demo backend: returns a pre-scripted sequence of responses.

    Not a real model call -- used by tests to exercise
    ``VisionPlateClassifier``'s confirmation-windowing logic without a
    live local or cloud VLM available.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def query(self, frame: np.ndarray, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        if self._i >= len(self._responses):
            return {"plates_visible": False, "confidence": 0.0}
        r = self._responses[self._i]
        self._i += 1
        return r
