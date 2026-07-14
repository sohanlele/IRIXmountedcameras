"""Zone edge-box local buffer (Section 6.3).

A short rolling window of derived events, local to one zone's edge box.
Mirrors the design doc's data-minimization stance: this buffer only ever
holds derived metrics (never raw video/frames), and old entries are
dropped once the buffer is full or once the aggregator has pulled them.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List

from .schema import DerivedMetricsEvent


class LocalBuffer:
    def __init__(self, maxlen: int = 500):
        self._buffer: Deque[DerivedMetricsEvent] = deque(maxlen=maxlen)

    def push(self, event: DerivedMetricsEvent) -> None:
        self._buffer.append(event)

    def drain(self) -> List[DerivedMetricsEvent]:
        """Pop and return everything currently buffered (aggregator pull)."""
        drained = list(self._buffer)
        self._buffer.clear()
        return drained

    def __len__(self) -> int:
        return len(self._buffer)
