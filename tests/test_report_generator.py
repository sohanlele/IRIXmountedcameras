"""Tests for irix.validation.report_generator (Priority 12) -- the
script that turns "run the suite and the benchmarks" into one dated,
machine-generated report instead of docs/VALIDATION.md's numbers going
stale by hand (which they had: 237 claimed vs. 389 actual before this
phase's work).
"""
from __future__ import annotations

from irix.validation.report_generator import (
    _find_summary_line,
    _parse_summary_counts,
    format_report_markdown,
    generate_report,
    run_tests,
)


def test_parse_summary_counts_all_passed():
    counts = _parse_summary_counts("237 passed, 3 skipped in 1.62s")
    assert counts == {"passed": 237, "skipped": 3}


def test_parse_summary_counts_with_failures():
    counts = _parse_summary_counts("5 failed, 232 passed, 3 skipped in 2.10s")
    assert counts == {"failed": 5, "passed": 232, "skipped": 3}


def test_find_summary_line_picks_the_real_summary_not_a_decoy():
    output = (
        "tests/test_foo.py::test_bar PASSED\n"
        "some other line mentioning passed inline but no ' in ' timing suffix\n"
        "===== 12 passed, 1 skipped in 0.53s ====="
    )
    line = _find_summary_line(output)
    assert line is not None
    assert "12 passed" in line


def test_find_summary_line_returns_none_when_absent():
    assert _find_summary_line("no summary here at all\njust noise") is None


def test_run_tests_against_a_small_real_subset_reports_real_counts():
    """Real subprocess pytest run (not mocked) against a small, known-
    passing subset of this repo's own suite -- proves the subprocess
    invocation, cwd, and summary-line parsing all actually work end to
    end, not just against synthetic strings."""
    result = run_tests(pytest_args=["tests/test_studio_api.py"])
    assert result.all_passed
    assert result.passed == 12
    assert result.failed == 0
    assert result.duration_s >= 0.0


def test_run_tests_reports_real_failures_without_fabricating():
    """A subset containing a genuinely failing test -- report_generator
    must surface the real failure count and failing test names, not
    silently report success."""
    import os
    import tempfile

    failing_test = (
        "def test_this_always_fails():\n"
        "    assert 1 == 2\n"
    )
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp_path = os.path.join(repo_root, "tests", "_tmp_deliberately_failing_test.py")
    with open(tmp_path, "w") as f:
        f.write(failing_test)
    try:
        result = run_tests(pytest_args=["tests/_tmp_deliberately_failing_test.py"])
        assert not result.all_passed
        assert result.failed == 1
        assert len(result.failure_names) == 1
    finally:
        os.remove(tmp_path)


def test_generate_report_without_benchmarks_has_no_fabricated_benchmark_section():
    report = generate_report(run_benchmarks=False, pytest_args=["tests/test_studio_api.py"])
    assert report["benchmarks"] is None
    assert report["tests"]["passed"] == 12
    assert report["git_commit"] is not None  # this repo IS a git checkout


def test_format_report_markdown_reflects_failure_status():
    report = {
        "generated_at": "2026-07-14T00:00:00+00:00",
        "git_commit": "abc1234",
        "environment": {"platform": "test-platform", "python_version": "3.10.0"},
        "tests": {
            "passed": 5, "failed": 1, "skipped": 0, "errors": 0, "duration_s": 1.0,
            "all_passed": False, "failure_names": ["tests/test_x.py::test_y"],
            "raw_summary": "1 failed, 5 passed in 1.00s",
        },
        "benchmarks": None,
    }
    md = format_report_markdown(report)
    assert "FAIL" in md
    assert "tests/test_x.py::test_y" in md
    assert "abc1234" in md


def test_format_report_markdown_reflects_pass_status():
    report = {
        "generated_at": "2026-07-14T00:00:00+00:00",
        "git_commit": "abc1234",
        "environment": {"platform": "test-platform", "python_version": "3.10.0"},
        "tests": {
            "passed": 393, "failed": 0, "skipped": 3, "errors": 0, "duration_s": 7.0,
            "all_passed": True, "failure_names": [],
            "raw_summary": "393 passed, 3 skipped in 7.00s",
        },
        "benchmarks": None,
    }
    md = format_report_markdown(report)
    assert "PASS" in md
    assert "393 passed" in md
