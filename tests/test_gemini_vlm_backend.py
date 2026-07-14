"""Tests for GeminiVLMBackend.query() against the real google-genai SDK.

These verify the request is constructed correctly and the response is
parsed correctly WITHOUT making any live network call and WITHOUT a real
API key: ``_load_client`` is patched to return a fake client (same seam
pattern as ``test_weight_recognition.py``'s ``PlateQRReader._decode``
patch), so ``genai.Client`` itself is never invoked with the fake key,
and ``client.models.generate_content`` is a ``MagicMock`` we inspect and
control the return value of. ``google-genai`` must be importable for
these tests to exercise the real ``types.Part``/``types`` objects (that's
what makes this a genuine check of the SDK call shape, not just a mock of
our own code) -- gated by ``pytest.importorskip`` so the base suite
doesn't require it.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

genai = pytest.importorskip("google.genai", reason="requires 'pip install irix[vlm]'")

from irix.weight_recognition.vlm_backend import GeminiVLMBackend  # noqa: E402


def _fake_frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def _backend_with_fake_client(response_text: str) -> tuple[GeminiVLMBackend, MagicMock]:
    backend = GeminiVLMBackend(api_key="not-a-real-key")
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = MagicMock(text=response_text)
    backend._load_client = MagicMock(return_value=fake_client)
    return backend, fake_client


def test_query_never_touches_real_client_construction():
    """_load_client is never called with the real genai.Client -- confirms
    no live call/credential use happens in this test."""
    backend, fake_client = _backend_with_fake_client(json.dumps({"plates_visible": True, "confidence": 0.9}))
    with patch("google.genai.Client") as real_client_ctor:
        backend.query(_fake_frame(), "prompt", {"type": "object"})
        real_client_ctor.assert_not_called()


def test_query_passes_a_real_types_part_for_the_image():
    from google.genai import types

    backend, fake_client = _backend_with_fake_client(json.dumps({"plates_visible": True, "confidence": 0.9}))
    backend.query(_fake_frame(), "look at this", {"type": "object"})

    _, kwargs = fake_client.models.generate_content.call_args
    contents = kwargs["contents"]
    assert len(contents) == 2
    assert isinstance(contents[0], types.Part)
    assert contents[1] == "look at this"


def test_query_requests_json_schema_via_response_json_schema_not_response_schema():
    """_LOAD_READ_SCHEMA (vision_classifier.py) is lowercase standard JSON
    Schema. The google-genai SDK only accepts that shape under
    response_json_schema -- response_schema expects a Pydantic model or
    Gemini's own uppercase-typed schema dialect. Pinning this down is the
    whole point of this fix; regressing to response_schema would silently
    send a schema shape the API doesn't parse correctly for a plain dict."""
    schema = {
        "type": "object",
        "properties": {"plates_visible": {"type": "boolean"}, "confidence": {"type": "number"}},
        "required": ["plates_visible", "confidence"],
    }
    backend, fake_client = _backend_with_fake_client(json.dumps({"plates_visible": True, "confidence": 0.9}))
    backend.query(_fake_frame(), "prompt", schema)

    _, kwargs = fake_client.models.generate_content.call_args
    config = kwargs["config"]
    assert config["response_json_schema"] == schema
    assert "response_schema" not in config
    assert config["response_mime_type"] == "application/json"


def test_query_parses_response_text_as_json():
    backend, _ = _backend_with_fake_client(
        json.dumps({"plates_visible": True, "total_weight_kg": 42.5, "confidence": 0.95})
    )
    result = backend.query(_fake_frame(), "prompt", {"type": "object"})
    assert result == {"plates_visible": True, "total_weight_kg": 42.5, "confidence": 0.95}


def test_query_uses_the_configured_model():
    backend, fake_client = _backend_with_fake_client(json.dumps({"plates_visible": False, "confidence": 0.0}))
    backend.model = "gemini-2.5-flash-lite"
    backend.query(_fake_frame(), "prompt", {"type": "object"})

    _, kwargs = fake_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash-lite"
