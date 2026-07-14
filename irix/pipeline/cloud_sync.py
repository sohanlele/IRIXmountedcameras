"""Cloud sync layer (Section 6.3 / 8.2).

Stores only derived data -- rep counts, form scores, personalization
profiles -- never raw video. ``CloudSync`` is an interface so the edge
side of the pipeline can be tested without a real backend; ``HTTPCloudSync``
sketches the real integration point.
"""
from __future__ import annotations

import json
from typing import List, Protocol

from .schema import DerivedMetricsEvent


class CloudSync(Protocol):
    def send(self, events: List[DerivedMetricsEvent]) -> None: ...


class InMemoryCloudSync:
    """Test/demo cloud sync -- just accumulates events in memory."""

    def __init__(self):
        self.received: List[DerivedMetricsEvent] = []

    def send(self, events: List[DerivedMetricsEvent]) -> None:
        self.received.extend(events)


class HTTPCloudSync:
    """Sketch of a real cloud sync: POSTs derived-metrics JSON to an API.

    Not wired to a real backend in this scaffold -- ``requests`` is not a
    hard dependency of this repo. Swap in your actual HTTP client and
    auth/retry policy.
    """

    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def send(self, events: List[DerivedMetricsEvent]) -> None:
        import urllib.request

        payload = json.dumps([e.to_dict() for e in events]).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
