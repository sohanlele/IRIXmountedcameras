"""A camera source that stays up for a 24/7 station, instead of exiting
the first time a read fails.

Every existing frame-reading loop in this repo (``run_demo.py --source``/
``run_live``, and ``run_upload.py``) opens one ``cv2.VideoCapture``, reads
until it returns ``False`` once, and stops -- correct for "play this video
file to the end" or "watch this webcam for one demo run," wrong for a
station that's supposed to keep watching its camera all day. A live RTSP
feed genuinely drops sometimes (network blip, camera reboot, DHCP
lease renewal) -- a station's software shouldn't exit and stay down until
someone notices and restarts it.

``ReconnectingFrameSource`` is that fix: it wraps whatever
``cv2.VideoCapture`` already understands (webcam index, video file path,
or a live stream URL -- the same ``source`` argument ``run_live`` already
accepts, since OpenCV treats an RTSP/HTTP stream URL exactly like a file
path) and, on any read failure, releases and reopens with exponential
backoff instead of raising or stopping. This is genuine, testable logic
(the retry/backoff state machine) that doesn't need a real camera to
verify -- see ``tests/test_camera_source.py``, which drives it against a
fake capture object that fails on cue.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, Optional

import numpy as np


def _default_capture_factory(source):
    import cv2

    return cv2.VideoCapture(source)


class ReconnectingFrameSource:
    def __init__(
        self,
        source,
        backoff_s: float = 2.0,
        max_backoff_s: float = 30.0,
        capture_factory: Optional[Callable] = None,
    ):
        """``source``: anything ``cv2.VideoCapture`` accepts (webcam
        index, file path, or a live stream URL). ``capture_factory``
        defaults to real ``cv2.VideoCapture`` -- overridden in tests with
        a fake that can be told to fail on cue, so this class's retry
        logic can be verified without a real camera."""
        self.source = source
        self.backoff_s = backoff_s
        self.max_backoff_s = max_backoff_s
        self._capture_factory = capture_factory or _default_capture_factory
        self._cap = None
        self._current_backoff = backoff_s

    def _ensure_open(self) -> bool:
        if self._cap is None:
            self._cap = self._capture_factory(self.source)
        return bool(self._cap.isOpened())

    def _reconnect(self, sleep_fn: Callable[[float], None]) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        sleep_fn(self._current_backoff)
        self._current_backoff = min(self._current_backoff * 2.0, self.max_backoff_s)

    def frames(
        self, max_frames: Optional[int] = None, sleep: Optional[Callable[[float], None]] = None,
    ) -> Iterator[np.ndarray]:
        """Yields frames indefinitely (or until ``max_frames`` have been
        yielded, mainly for tests) -- reconnects with exponential backoff
        on any open/read failure instead of stopping. ``sleep`` is
        injectable so tests don't have to wait through a real backoff."""
        sleep_fn = sleep or time.sleep
        count = 0
        while max_frames is None or count < max_frames:
            if not self._ensure_open():
                self._reconnect(sleep_fn)
                continue
            ok, frame = self._cap.read()
            if not ok:
                self._reconnect(sleep_fn)
                continue
            self._current_backoff = self.backoff_s  # reset once a read actually succeeds
            count += 1
            yield frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
