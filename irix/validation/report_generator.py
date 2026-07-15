"""Priority 12: automated validation report generation.

``docs/VALIDATION.md`` states validation status by hand, and it drifts
-- it claimed "237 passed" while the suite had actually grown to 377,
then 389, then 393 across this phase's work, because nothing forced it
to stay current. This module generates the numbers that go in a report
like that programmatically, from the actual test suite and benchmark
run at the moment it's invoked, so "the docs are stale" stops being
possible for the parts a machine can verify (pass/fail/skip counts,
timing/throughput numbers, git commit). It deliberately does **not**
try to auto-generate the hand-written qualitative sections of
``docs/VALIDATION.md`` ("what's genuinely validated end to end," "what's
not validated") -- those require human judgment about what a given test
actually proves, which is exactly the kind of thing this repo's other
principles (never fabricate, honestly report "unknown") say a script
shouldn't pretend to know.

Usage::

    python -m irix.validation.report_generator [--output PATH] [--skip-benchmarks]

Writes a dated Markdown report and prints it to stdout. Import
``generate_report()``/``run_tests()`` directly for programmatic use
(e.g. from a CI job)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class TestRunResult:
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_s: float
    raw_summary_line: str
    failure_names: List[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.errors == 0


def run_tests(pytest_args: Optional[List[str]] = None) -> TestRunResult:
    """Runs the real test suite as a subprocess (not `pytest.main()` in
    -process -- a subprocess is the only way to get a clean, unaffected
    exit code and stdout regardless of what this module itself imports)
    and parses pytest's own summary line. Never estimates or fabricates
    a count if pytest's output doesn't parse cleanly -- raises instead,
    since a validation report with a guessed pass count would be worse
    than no report at all."""
    args = [sys.executable, "-m", "pytest", "-q"] + (pytest_args or [])
    start = datetime.now(timezone.utc)
    proc = subprocess.run(args, capture_output=True, text=True, cwd=_repo_root())
    duration_s = (datetime.now(timezone.utc) - start).total_seconds()

    output = proc.stdout + "\n" + proc.stderr
    summary_line = _find_summary_line(output)
    if summary_line is None:
        raise RuntimeError(
            f"could not find a pytest summary line in test output -- refusing to "
            f"report fabricated numbers. Raw output tail:\n{output[-2000:]}"
        )

    counts = _parse_summary_counts(summary_line)
    failure_names = [
        line.split(" ", 1)[0].removeprefix("FAILED ") if line.startswith("FAILED") else line
        for line in output.splitlines()
        if line.startswith("FAILED")
    ]
    return TestRunResult(
        passed=counts.get("passed", 0), failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0), errors=counts.get("error", 0) + counts.get("errors", 0),
        duration_s=round(duration_s, 2), raw_summary_line=summary_line.strip(),
        failure_names=failure_names,
    )


def _repo_root() -> str:
    import os
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_summary_line(output: str) -> Optional[str]:
    # pytest's final summary line looks like "237 passed, 3 skipped in 1.6s"
    # or "5 failed, 232 passed in 2.1s" -- always the last non-empty line
    # of the "====...====" bracketed footer.
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if stripped and (" passed" in stripped or " failed" in stripped or " error" in stripped) and " in " in stripped:
            return stripped
    return None


def _parse_summary_counts(summary_line: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    # strip pytest's "===" padding and timing suffix, split on commas
    core = summary_line.strip("= ").rsplit(" in ", 1)[0]
    for part in core.split(","):
        part = part.strip()
        tokens = part.split()
        if len(tokens) != 2:
            continue
        n, label = tokens
        try:
            counts[label] = int(n)
        except ValueError:
            continue
    return counts


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=_repo_root(),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        return None


def generate_report(run_benchmarks: bool = True, pytest_args: Optional[List[str]] = None) -> Dict:
    """The full structured report: environment, git commit, test
    results, and (optionally -- ``run_benchmarks=False`` skips the
    slower/hardware-dependent part) the performance benchmark suite
    from ``irix.benchmark.run_benchmarks``. ``pytest_args``: forwarded
    to ``run_tests`` -- exposed mainly so tests of this module itself
    can point it at a small subset instead of recursively running the
    entire suite (see ``tests/test_report_generator.py``); real callers
    should leave this ``None`` to validate everything."""
    test_result = run_tests(pytest_args)

    report: Dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "tests": {
            "passed": test_result.passed, "failed": test_result.failed,
            "skipped": test_result.skipped, "errors": test_result.errors,
            "duration_s": test_result.duration_s, "all_passed": test_result.all_passed,
            "failure_names": test_result.failure_names,
            "raw_summary": test_result.raw_summary_line,
        },
        "benchmarks": None,
    }

    if run_benchmarks:
        from irix.benchmark.run_benchmarks import run_all
        report["benchmarks"] = run_all()

    return report


def format_report_markdown(report: Dict) -> str:
    lines = ["# IRIX validation report", ""]
    lines.append(f"Generated: {report['generated_at']}")
    lines.append(f"Git commit: `{report['git_commit'] or 'unknown'}`")
    env = report["environment"]
    lines.append(f"Environment: {env['platform']}, Python {env['python_version']}")
    lines.append("")

    t = report["tests"]
    status = "PASS" if t["all_passed"] else "FAIL"
    lines.append(f"## Test suite: {status}")
    lines.append("")
    lines.append(
        f"{t['passed']} passed, {t['failed']} failed, {t['skipped']} skipped, "
        f"{t['errors']} errors ({t['duration_s']}s)"
    )
    if t["failure_names"]:
        lines.append("")
        lines.append("Failing tests:")
        for name in t["failure_names"]:
            lines.append(f"- `{name}`")
    lines.append("")

    if report["benchmarks"] is not None:
        from irix.benchmark.run_benchmarks import format_report as format_benchmark_report
        lines.append("## Performance benchmarks")
        lines.append("")
        lines.append("```")
        lines.append(format_benchmark_report(report["benchmarks"]))
        lines.append("```")
        lines.append("")

    lines.append("## Scope note")
    lines.append("")
    lines.append(
        "This report covers what a script can verify mechanically: test "
        "pass/fail counts and measured performance numbers. It does not "
        "assess *what* the passing tests actually validate (mocked vs. "
        "real end-to-end, synthetic vs. real-world data) -- see "
        "`docs/VALIDATION.md` for that qualitative assessment, which "
        "requires human judgment this generator does not attempt to "
        "automate."
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="write the Markdown report to this path")
    parser.add_argument("--skip-benchmarks", action="store_true", help="skip the (slower) performance benchmark suite")
    parser.add_argument("--json", action="store_true", help="print the raw JSON report instead of Markdown")
    args = parser.parse_args(argv)

    report = generate_report(run_benchmarks=not args.skip_benchmarks)

    if args.json:
        output_text = json.dumps(report, indent=2, default=str)
    else:
        output_text = format_report_markdown(report)

    print(output_text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"\n(written to {args.output})", file=sys.stderr)

    return 0 if report["tests"]["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
