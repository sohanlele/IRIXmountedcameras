"""BLE identity linking (Section 5.1).

The wristband broadcasts a BLE identifier that the nearest station's edge
node picks up to associate a lifter with that camera feed -- no facial
recognition. BLE RSSI-based proximity is inherently noisy (~5-10 m typical
accuracy in cluttered indoor environments), so for a first version, BLE
proximity combined with simple heuristics (closest station, most recent
motion) should be sufficient to resolve identity. If false pairings become
a problem in practice, the upgrade path is BLE Angle-of-Arrival or UWB
anchors per station (Section 12.3).

This module implements the pairing *resolution logic* that would run on
the edge node given RSSI readings -- not the BLE radio stack itself, which
is hardware/firmware, not software-scaffold scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BLEReading:
    station_id: str
    rssi: float          # dBm, less negative = stronger/closer
    timestamp: float
    recent_motion: bool = False  # True if this station's camera saw motion recently
    # Which physical band this reading is from. Optional (defaults to
    # None) because StationPairing.resolve() below never needed it -- its
    # whole input is already "these are all readings for one already-known
    # band, which station wins" -- but irix.live.station_runner needs it:
    # a station's radio can see several different members' bands at once,
    # so it has to know *which* reading belongs to *which* band before it
    # can resolve that band to an account via irix.identity.checkout.
    wristband_id: Optional[str] = None


class StationPairing:
    """Resolves which station a wristband should be paired to.

    v1 heuristic: strongest RSSI wins; ties (within ``rssi_tie_margin`` dBm)
    are broken in favor of the station with recent correlated motion.
    """

    def __init__(self, rssi_tie_margin: float = 3.0):
        self.rssi_tie_margin = rssi_tie_margin

    def resolve(self, readings: List[BLEReading]) -> Optional[str]:
        if not readings:
            return None
        best = max(readings, key=lambda r: r.rssi)
        contenders = [r for r in readings if best.rssi - r.rssi <= self.rssi_tie_margin]
        if len(contenders) > 1:
            motion_contenders = [r for r in contenders if r.recent_motion]
            if motion_contenders:
                best = max(motion_contenders, key=lambda r: r.rssi)
        return best.station_id
