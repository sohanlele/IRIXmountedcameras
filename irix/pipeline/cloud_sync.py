"""Sync layer to irix-mvp-app's backend (Section 6.3 / 8.2).

Forwards only structured derived events -- rep counts, form scores,
weight reads, band-placement prompts -- never raw video. ``CloudSync`` is
an interface so the edge side of the pipeline can be tested without a
real backend; ``HTTPCloudSync`` sketches the real integration point
against jeffreyjy/irix-mvp-app's FastAPI backend, which doesn't yet
expose a live-camera-data ingestion endpoint (its `api/v1` currently only
covers workout plans/sessions and auth) -- the endpoint path here is a
placeholder to be swapped once that route exists.
"""
from __future__ import annotations

import json
from typing import List, Protocol

from .schema import CameraEvent


class CloudSync(Protocol):
    def send(self, events: List[CameraEvent]) -> None: ...


class InMemoryCloudSync:
    """Test/demo cloud sync -- just accumulates events in memory."""

    def __init__(self):
        self.received: List[CameraEvent] = []

    def send(self, events: List[CameraEvent]) -> None:
        self.received.extend(events)


class HTTPCloudSync:
    """Sketch of a real sync: POSTs structured event JSON to
    irix-mvp-app's backend.

    Not wired to a real endpoint in this scaffold -- ``requests`` is not a
    hard dependency of this repo, and the app doesn't have a camera-event
    ingestion route yet. Swap in the real endpoint path, auth scheme
    (the app's `models/auth` suggests JWT), and retry policy once that
    exists.
    """

    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.api_key = api_key

    def send(self, events: List[CameraEvent]) -> None:
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
