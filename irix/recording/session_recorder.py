"""Session recording: everything Priority 8 asks for (camera video, IMU
packets, camera timestamps, clock synchronization, station metadata,
session metadata, events, metrics) written to one directory, and the
matching loader, so a recorded session can be replayed deterministically,
compared across algorithm versions, or (once enough are collected and
annotated) become a real training/validation dataset.

## Why this didn't already exist

Every "load a recording" path in this repo predates this module:
``irix.fusion.imu_io.load_imu_samples`` reads a wristband export someone
already has; ``irix.demo.run_upload`` replays a pre-recorded video file
plus that IMU export. Nothing in this repo ever *produced* either --
there was no write side at all, which meant no live/simulated run could
ever be captured for later comparison, and no path toward a real
annotated dataset existed even in principle. This module is that write
side, deliberately reusing ``irix.fusion.imu_io``'s existing IMU file
format (``save_imu_samples``, new alongside this module) rather than
inventing a second one, so a recorded session's IMU file loads back in
through the exact same path a real uploaded recording already does.

## Privacy: raw video is opt-in and off by default

``docs/API_SPEC.md`` states this repo's production event pipeline
contract plainly: "never raw video, never a statutorily-defined
biometric identifier." That principle is about what the *event* pipeline
sends onward (``irix.pipeline.cloud_sync``) -- it says nothing about
whether an explicitly-invoked, local recording tool used for algorithm
development/validation may capture frames at all, and Priority 8
explicitly asks for camera video recording. This module reconciles both:
``save_raw_frames`` defaults to ``False`` (only per-frame *timestamps*
and metadata are recorded -- enough to validate timing/fusion/event
correctness against, matching the deterministic-replay use case) and
must be explicitly set ``True`` to persist actual pixel data, which a
real deployment should gate behind its own explicit member-consent/
data-retention policy before ever enabling -- a policy decision this
module cannot make on a caller's behalf, so it defaults to the safer
choice rather than assuming consent.

## Directory layout

    <output_dir>/
        metadata.json       -- station/session/gym config, start/end ts,
                                clock_sync estimate snapshot if available
        events.jsonl         -- one CameraEvent.to_dict() per line, in
                                emission order
        imu.json             -- irix.fusion.imu_io format (empty file
                                omitted if no IMU was ever recorded)
        frames.jsonl          -- one {"timestamp":, "camera_id":,
                                "frame_index":} per line, always written
        frames/               -- only if save_raw_frames=True:
                                frame_<frame_index>.npy per frame
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..fusion.clock_sync import ClockSyncEstimate
from ..fusion.imu import IMUSample
from ..fusion.imu_io import load_imu_samples, save_imu_samples
from ..pipeline.schema import CameraEvent


@dataclass
class RecordedFrameInfo:
    timestamp: float
    frame_index: int
    camera_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "frame_index": self.frame_index, "camera_id": self.camera_id}


class SessionRecorder:
    """Records one session's worth of everything Priority 8 names. Not
    wired into any live entry point by default (opt-in -- a caller
    constructs one alongside a ``StationSessionRunner``/``RepSession``
    and calls its ``record_*`` methods from that runner's own
    ``on_events``/frame-loop hooks; see ``docs/TODO.md`` for the
    concrete next wiring step)."""

    def __init__(
        self,
        output_dir: str,
        station_id: str,
        exercise_name: str,
        member_id: Optional[str] = None,
        save_raw_frames: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = output_dir
        self.station_id = station_id
        self.exercise_name = exercise_name
        self.member_id = member_id
        self.save_raw_frames = save_raw_frames
        self.extra_metadata = extra_metadata or {}

        os.makedirs(output_dir, exist_ok=True)
        if save_raw_frames:
            os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)

        self._events: List[CameraEvent] = []
        self._imu_samples: List[IMUSample] = []
        self._frames: List[RecordedFrameInfo] = []
        self._start_ts: Optional[float] = None
        self._end_ts: Optional[float] = None
        self._clock_sync_snapshot: Optional[Dict[str, Any]] = None

    def record_frame(self, frame: np.ndarray, ts: float, camera_id: Optional[str] = None) -> None:
        """Log one frame's timing/identity (always) and, only if
        ``save_raw_frames``, its actual pixel data. ``frame_index`` is
        this recorder's own running count, not tied to any upstream
        frame-source numbering -- stable and gap-free even if the
        upstream source's own indexing isn't (e.g. after a camera
        reconnect)."""
        if self._start_ts is None:
            self._start_ts = ts
        self._end_ts = ts
        frame_index = len(self._frames)
        self._frames.append(RecordedFrameInfo(timestamp=ts, frame_index=frame_index, camera_id=camera_id))
        if self.save_raw_frames:
            np.save(os.path.join(self.output_dir, "frames", f"frame_{frame_index}.npy"), frame)

    def record_imu_samples(self, samples: List[IMUSample]) -> None:
        self._imu_samples.extend(samples)

    def record_events(self, events: List[CameraEvent]) -> None:
        self._events.extend(events)

    def record_clock_sync_snapshot(self, estimate: ClockSyncEstimate) -> None:
        """A point-in-time snapshot of the session's clock-sync state
        (Priority 8's "clock synchronization" recording target) --
        called whenever a caller wants one logged (e.g. at session close,
        or after each set); not automatically sampled on a timer by this
        class, since only the caller knows a meaningful cadence for its
        own live loop."""
        self._clock_sync_snapshot = {
            "offset_s": estimate.offset_s, "drift_ppm": estimate.drift_ppm,
            "confidence": estimate.confidence, "n_observations": estimate.n_observations,
        }

    def close(self) -> Dict[str, str]:
        """Flush everything to disk. Returns the written file paths
        (only the ones actually written -- ``imu.json`` is omitted
        entirely if no IMU was ever recorded, rather than writing an
        empty, misleading file)."""
        written: Dict[str, str] = {}

        metadata = {
            "station_id": self.station_id,
            "exercise_name": self.exercise_name,
            "member_id": self.member_id,
            "start_ts": self._start_ts,
            "end_ts": self._end_ts,
            "n_frames": len(self._frames),
            "n_imu_samples": len(self._imu_samples),
            "n_events": len(self._events),
            "save_raw_frames": self.save_raw_frames,
            "clock_sync": self._clock_sync_snapshot,
            **self.extra_metadata,
        }
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        written["metadata"] = metadata_path

        frames_path = os.path.join(self.output_dir, "frames.jsonl")
        with open(frames_path, "w") as f:
            for frame_info in self._frames:
                f.write(json.dumps(frame_info.to_dict()) + "\n")
        written["frames"] = frames_path

        events_path = os.path.join(self.output_dir, "events.jsonl")
        with open(events_path, "w") as f:
            for event in self._events:
                f.write(json.dumps(event.to_dict()) + "\n")
        written["events"] = events_path

        if self._imu_samples:
            imu_path = os.path.join(self.output_dir, "imu.json")
            save_imu_samples(sorted(self._imu_samples, key=lambda s: s.timestamp), imu_path)
            written["imu"] = imu_path

        return written


@dataclass
class RecordedSession:
    """Everything ``load_recorded_session`` reads back -- events are
    returned as plain dicts (``CameraEvent.to_dict()``'s own shape), not
    reconstructed dataclass instances: a recording made by an older
    schema version should still load (and be inspectable/replayable at
    the dict level) even after a field gets added or renamed, rather than
    failing to deserialize entirely."""

    metadata: Dict[str, Any]
    frames: List[RecordedFrameInfo]
    imu_samples: List[IMUSample]
    events: List[Dict[str, Any]]


def load_recorded_session(input_dir: str) -> RecordedSession:
    """Load a directory ``SessionRecorder.close()`` wrote. Raises
    ``FileNotFoundError`` if ``metadata.json`` is missing (the one file
    every recording always has) -- every other file is optional and
    simply produces an empty list if absent (e.g. no IMU was ever
    recorded for a camera-only session)."""
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{input_dir}: no metadata.json -- not a recorded session directory")
    with open(metadata_path) as f:
        metadata = json.load(f)

    frames: List[RecordedFrameInfo] = []
    frames_path = os.path.join(input_dir, "frames.jsonl")
    if os.path.exists(frames_path):
        with open(frames_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frames.append(RecordedFrameInfo(**d))

    events: List[Dict[str, Any]] = []
    events_path = os.path.join(input_dir, "events.jsonl")
    if os.path.exists(events_path):
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))

    imu_samples: List[IMUSample] = []
    imu_path = os.path.join(input_dir, "imu.json")
    if os.path.exists(imu_path):
        imu_samples = load_imu_samples(imu_path)

    return RecordedSession(metadata=metadata, frames=frames, imu_samples=imu_samples, events=events)
