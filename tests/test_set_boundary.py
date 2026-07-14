from irix.pipeline.events import RestGapSetBoundaryDetector


def test_first_rep_is_never_a_boundary():
    detector = RestGapSetBoundaryDetector(rest_gap_s=20.0)
    assert detector.observe(0.0) is False


def test_reps_within_gap_are_not_boundaries():
    detector = RestGapSetBoundaryDetector(rest_gap_s=20.0)
    detector.observe(0.0)
    assert detector.observe(2.0) is False
    assert detector.observe(5.0) is False
    assert detector.observe(9.5) is False


def test_gap_at_or_past_threshold_is_a_boundary():
    detector = RestGapSetBoundaryDetector(rest_gap_s=20.0)
    detector.observe(0.0)
    assert detector.observe(25.0) is True


def test_boundary_only_fires_once_per_gap():
    detector = RestGapSetBoundaryDetector(rest_gap_s=20.0)
    detector.observe(0.0)
    assert detector.observe(25.0) is True
    # next rep is close to the one that just fired the boundary -- no
    # second boundary immediately after
    assert detector.observe(26.0) is False


def test_reset_clears_history():
    detector = RestGapSetBoundaryDetector(rest_gap_s=20.0)
    detector.observe(0.0)
    detector.reset()
    # after reset, the next rep is treated as the first rep of a fresh
    # session -- no boundary even though real time has "passed"
    assert detector.observe(1000.0) is False
