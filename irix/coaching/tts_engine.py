"""On-device text-to-speech (Section 7.1).

Running detection, rep logic, and TTS locally at the zone edge box (rather
than round-tripping to a cloud TTS API) is what makes a sub-second
feedback loop realistic. Several production-grade offline engines already
run comfortably on CPU-only edge hardware -- Piper, Kokoro, Kitten TTS --
any of which fit the zone-box compute budgeted in Section 6.

``NullTTSEngine`` is the default/test implementation (no audio, just
records what would have been spoken). ``PiperTTSEngine`` sketches the
real integration point against a local Piper install.
"""
from __future__ import annotations

import subprocess
from typing import List, Protocol


class TTSEngine(Protocol):
    def speak(self, text: str) -> None: ...


class NullTTSEngine:
    """No-op engine for tests/demo runs without audio hardware."""

    def __init__(self):
        self.spoken: List[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class PiperTTSEngine:
    """Sketch of a real local TTS integration using Piper (Section 7.1).

    Piper synthesizes speech in real time on CPU alone (benchmarked on a
    Raspberry Pi 5 in the design doc's sources), which fits the zone-box
    compute already budgeted in Section 6. Not wired to a real Piper binary
    in this scaffold -- requires a local piper install + voice model.
    """

    def __init__(self, piper_binary: str = "piper", voice_model: str = "en_US-lessac-medium.onnx"):
        self.piper_binary = piper_binary
        self.voice_model = voice_model

    def speak(self, text: str) -> None:
        # Real integration: pipe text into piper, play resulting audio.
        # Left as a subprocess sketch rather than executed here.
        subprocess.run(
            [self.piper_binary, "--model", self.voice_model, "--output-raw"],
            input=text.encode("utf-8"),
            check=False,
        )
