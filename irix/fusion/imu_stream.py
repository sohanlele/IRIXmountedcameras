"""Pluggable live/recorded wristband IMU sample source.

``imu_io.load_imu_samples`` (added alongside ``irix.demo.run_upload``)
loads an already-recorded wristband export all at once -- the right
shape for offline analysis of a finished workout, the wrong shape for "a
member checks a band out at the front desk, and while they're doing a
set we're pulling samples off that live band in real time." This module
is the interface for that: a ``poll()``-based protocol so
``irix.pipeline.rep_session.RepSession`` (and, live, ``irix.live.
station_runner.StationSessionRunner``) can consume IMU samples the same
way whether they're already-recorded or arriving in real time from a
live BLE connection, without caring which.

Same pattern as ``irix.weight_recognition.vlm_backend.VLMBackend``
(``LocalVLMBackend``/``GeminiVLMBackend``) and ``irix.pipeline.
cloud_sync.CloudSync`` (``InMemoryCloudSync``/``HTTPCloudSync``)
elsewhere in this repo: a real, recorded/offline implementation plus a
documented not-yet-implemented sketch for the live/hardware-backed one,
so callers already code against the interface that will keep working
once the live one exists.
"""
from __future__ import annotations

from typing import List, Protocol

from .imu import IMUSample


class IMUStream(Protocol):
    def poll(self) -> List[IMUSample]:
        """Return any new samples available since the last ``poll()``
        call (empty list if none) -- non-blocking, safe to call once per
        outer processing-loop tick (e.g. once per video frame)."""
        ...


class RecordedIMUStream:
    """Replays an already-loaded (e.g. via ``irix.fusion.imu_io.
    load_imu_samples``) list of samples through the ``IMUStream``
    interface.

    Exists so an offline caller (``irix.demo.run_upload``) and a live
    caller can share exactly the same consumption code in ``RepSession``
    -- ``run_upload`` wraps its loaded file in one of these and calls
    ``poll()`` once at the start (all samples "arrive" immediately, since
    the whole file is already on disk); a live caller instead uses
    ``LiveBLEIMUStream`` (or a real subclass of it once one exists),
    whose ``poll()`` genuinely returns different things call to call as
    real samples arrive over the BLE connection.
    """

    def __init__(self, samples: List[IMUSample]):
        self._samples = list(samples)
        self._polled = False

    def poll(self) -> List[IMUSample]:
        if self._polled:
            return []
        self._polled = True
        return list(self._samples)


class LiveBLEIMUStream:
    """Sketch of the real live path: samples arriving in real time off a
    wristband's BLE connection, starting from whenever a station's
    ``StationSessionRunner`` decides this member's session is active
    (see that module) and ending when it isn't.

    Not implemented here -- same reasoning as ``irix.weight_recognition.
    vlm_backend.LocalVLMBackend`` staying a documented stub: which BLE
    GATT client library and wristband firmware characteristic/notify
    protocol a real device actually exposes is hardware/firmware detail
    this software scaffold has no way to guess at correctly. What *is*
    settled is the shape a real implementation needs to satisfy
    (``IMUStream.poll()``) so the rest of the pipeline (``RepSession``,
    ``RepCountFusion``) doesn't need to change once it exists -- a real
    subclass would maintain an internal buffer that a BLE notify
    callback appends to, and ``poll()`` would drain and return it.
    """

    def __init__(self, wristband_id: str):
        self.wristband_id = wristband_id

    def poll(self) -> List[IMUSample]:
        raise NotImplementedError(
            "LiveBLEIMUStream is a sketch: point it at a real BLE GATT client "
            "(e.g. bleak) subscribed to the wristband's IMU notify characteristic, "
            "buffer incoming samples, and have poll() drain that buffer. Kept "
            "unimplemented rather than guessed at, since the right implementation "
            "depends on the actual wristband firmware's BLE protocol."
        )
