"""Building-level aggregator (Section 6.2 / 6.3).

Each zone's edge box sits on the same local network segment as its
cameras to keep inference latency low; only derived data crosses up to a
building-level aggregator and out to the cloud. This class stands in for
that aggregator: it pulls from each zone's ``LocalBuffer`` and forwards to
a ``CloudSync`` implementation.
"""
from __future__ import annotations

from typing import Dict, List

from .cloud_sync import CloudSync
from .edge_buffer import LocalBuffer
from .schema import DerivedMetricsEvent


class Aggregator:
    def __init__(self, cloud_sync: CloudSync):
        self.zones: Dict[str, LocalBuffer] = {}
        self.cloud_sync = cloud_sync

    def register_zone(self, zone_id: str, buffer: LocalBuffer) -> None:
        self.zones[zone_id] = buffer

    def sync(self) -> int:
        """Drain every registered zone buffer and forward events to the cloud.

        Returns the number of events synced.
        """
        all_events: List[DerivedMetricsEvent] = []
        for buffer in self.zones.values():
            all_events.extend(buffer.drain())
        if all_events:
            self.cloud_sync.send(all_events)
        return len(all_events)
