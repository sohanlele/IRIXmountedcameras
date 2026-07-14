import numpy as np

from irix.live.camera_source import ReconnectingFrameSource


class _FakeCapture:
    """A cv2.VideoCapture stand-in: `items` is a list where each entry is
    either a frame (np.ndarray, read() succeeds) or None (read() fails,
    same shape cv2 uses -- (False, None))."""

    def __init__(self, items):
        self._items = list(items)
        self._opened = True
        self.released = False

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._items:
            return False, None
        item = self._items.pop(0)
        if item is None:
            return False, None
        return True, item

    def release(self):
        self._opened = False
        self.released = True


def test_yields_frames_from_a_healthy_source():
    frames = [np.zeros((2, 2)), np.ones((2, 2))]
    cap = _FakeCapture(frames)
    source = ReconnectingFrameSource("fake", capture_factory=lambda s: cap)
    yielded = list(source.frames(max_frames=2))
    assert len(yielded) == 2


def test_reconnects_after_a_read_failure_instead_of_stopping():
    cap1 = _FakeCapture([None])  # fails immediately
    cap2 = _FakeCapture([np.zeros((2, 2))])
    captures = [cap1, cap2]

    def factory(source):
        return captures.pop(0)

    sleeps = []
    source = ReconnectingFrameSource("fake", backoff_s=1.0, capture_factory=factory)
    yielded = list(source.frames(max_frames=1, sleep=sleeps.append))

    assert len(yielded) == 1
    assert cap1.released is True, "the failed capture should be released before reconnecting"
    assert sleeps == [1.0]


def test_reconnects_after_a_failed_open_not_just_a_failed_read():
    class _NeverOpens:
        def isOpened(self):
            return False

        def read(self):
            return False, None

        def release(self):
            pass

    opens = [_NeverOpens(), _FakeCapture([np.zeros((2, 2))])]

    def factory(source):
        return opens.pop(0)

    sleeps = []
    source = ReconnectingFrameSource("fake", backoff_s=0.5, capture_factory=factory)
    yielded = list(source.frames(max_frames=1, sleep=sleeps.append))

    assert len(yielded) == 1
    assert sleeps == [0.5]


def test_backoff_doubles_on_repeated_failures():
    class _FlakyThenGood:
        def __init__(self, fail_times):
            self._fails_left = fail_times
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            if self._fails_left > 0:
                self._fails_left -= 1
                return False, None
            return True, np.zeros((2, 2))

        def release(self):
            self._opened = False

    flaky = _FlakyThenGood(fail_times=3)

    def factory(source):
        flaky._opened = True  # simulates a fresh connection attempt each reconnect
        return flaky

    sleeps = []
    source = ReconnectingFrameSource("fake", backoff_s=1.0, max_backoff_s=100.0, capture_factory=factory)
    yielded = list(source.frames(max_frames=1, sleep=sleeps.append))

    assert len(yielded) == 1
    assert sleeps == [1.0, 2.0, 4.0]


def test_backoff_caps_at_max_backoff_s():
    class _AlwaysFailsNTimes:
        def __init__(self, fail_times):
            self._fails_left = fail_times
            self._opened = True

        def isOpened(self):
            return self._opened

        def read(self):
            if self._fails_left > 0:
                self._fails_left -= 1
                return False, None
            return True, np.zeros((2, 2))

        def release(self):
            self._opened = False

    flaky = _AlwaysFailsNTimes(fail_times=5)

    def factory(source):
        flaky._opened = True
        return flaky

    sleeps = []
    source = ReconnectingFrameSource("fake", backoff_s=1.0, max_backoff_s=3.0, capture_factory=factory)
    list(source.frames(max_frames=1, sleep=sleeps.append))

    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_backoff_resets_after_a_successful_read():
    """A failure, successful reconnect, then a *later* failure should
    start backing off from backoff_s again, not continue escalating from
    wherever it left off -- a transient blip shouldn't leave the source
    thinking the camera is chronically unhealthy."""
    cap1 = _FakeCapture([None])           # fails once
    cap2 = _FakeCapture([np.zeros((2, 2)), None])  # one good frame, then fails
    cap3 = _FakeCapture([np.zeros((2, 2))])  # recovers again
    captures = [cap1, cap2, cap3]

    def factory(source):
        return captures.pop(0)

    sleeps = []
    source = ReconnectingFrameSource("fake", backoff_s=1.0, capture_factory=factory)
    yielded = list(source.frames(max_frames=2, sleep=sleeps.append))

    assert len(yielded) == 2
    # first failure backs off at 1.0; the frame from cap2 resets backoff;
    # cap2's own subsequent failure then backs off at 1.0 again, not 2.0.
    assert sleeps == [1.0, 1.0]
