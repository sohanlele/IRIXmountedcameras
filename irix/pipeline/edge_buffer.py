"""Zone edge-box local buffer (Section 6.3).

A short rolling window of structured camera events, local to one zone's
edge box. Mirrors the design doc's data-minimization stance: this buffer
only ever holds derived events (never raw video/frames), and old entries
are dropped once the buffer is full or once the aggregator has pulled
them.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List

from .schema import CameraEvent


class LocalBuffer:
    def __init__(self, maxlen: int = 500):
        self._buffer: Deque[CameraEvent] = deque(maxlen=maxlen)

    def push(self, event: CameraEvent) -> None:
        self._buffer.append(event)

    def drain(self) -> List[CameraEvent]:
        """Pop and return everything currently buffered (aggregator pull)."""
        drained = list(self._buffer)
        self._buffer.clear()
        return drained

    def __len__(self) -> int:
        return len(self._buffer)
