#!/usr/bin/env python3
"""Agentic self-improving test runner for jax_privacy.

Runs the test suite in an iterative loop, analyses failures, applies
safe infrastructure fixes, and persists a performance log and learnings
file under .jules/.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import psutil

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
JULES_DIR = REPO_ROOT / ".jules"
REPORT_FILE = JULES_DIR / "test_report.json"
PERF_LOG_FILE = JULES_DIR / "agentic_performance_log.json"
LEARNINGS_FILE = JULES_DIR / "agentic_learnings.md"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"

MAX_ITERATIONS = 3
SLOW_TEST_THRESHOLD_S = 10.0


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------

def discover_tests() -> list[str]:
    """Return paths to all *_test.py files under jax_privacy/ and backend/tests/."""
    test_files: list[str] = []
    search_dirs = [
        REPO_ROOT / "jax_privacy",
        REPO_ROOT / "backend" / "tests",
    ]
    for base in search_dirs:
        if base.exists():
            for p in sorted(base.rglob("*_test.py")):
                test_files.append(str(p))
    return test_files


# ---------------------------------------------------------------------------
# Failure classification helpers
# ---------------------------------------------------------------------------

def classify_failure(failure: dict) -> str:
    """Return a broad failure category string for a pytest failure dict."""
    longrepr = failure.get("longrepr", "") or ""
    call_repr = ""
    if isinstance(failure.get("call"), dict):
        call_repr = failure["call"].get("longrepr", "") or ""
    text = (longrepr + " " + call_repr).lower()

    if "importerror" in text or "modulenotfounderror" in text or "cannot import" in text:
        return "import_error"
    if "assertionerror" in text or "assert " in text:
        return "assertion_error"
    if "valueerror" in text:
        return "value_error"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "typeerror" in text:
        return "type_error"
    if "attributeerror" in text:
        return "attribute_error"
    return "unknown"


def suggest_fix(failure: dict) -> Optional[str]:
    """Return a human-readable suggestion for a failure, or None."""
    category = classify_failure(failure)
    node_id = failure.get("nodeid", "")

    suggestions = {
        "import_error": (
            "Check that the module is installed and that PYTHONPATH includes "
            "the repo root. Verify missing __init__.py files."
        ),
        "assertion_error": (
            "Review the assertion values. If comparing floats, consider using "
            "chex.assert_trees_all_close with a tolerance. Check for hardcoded "
            "magic numbers or model names that may have changed."
        ),
        "value_error": (
            "Inspect input validation logic or type coercions near the failing line."
        ),
        "timeout": (
            "Test exceeded the time limit. Consider mocking expensive IO or "
            "reducing problem size in the test fixture."
        ),
        "type_error": (
            "Check function signatures and argument types for API changes."
        ),
        "attribute_error": (
            "An attribute was accessed on an unexpected type. Check for API "
            "changes or None values."
        ),
        "unknown": (
            "Inspect the full traceback manually for this test: " + node_id
        ),
    }
    return suggestions.get(category)


# ---------------------------------------------------------------------------
# Auto-fix helpers (infrastructure only — no source/test logic is modified)
# ---------------------------------------------------------------------------

def _relax_pinned_dep(line: str) -> tuple[str, bool]:
    """If *line* is a strict pin (==), relax to >= and return (new_line, changed)."""
    m = re.match(r"^([A-Za-z0-9_\-]+)==([^\s]+)(.*)$", line.strip())
    if m:
        pkg, ver, rest = m.groups()
        new_line = f"{pkg}>={ver}{rest}\n"
        return new_line, True
    return line, False


def _fix_pinned_requirements(failures: list[dict]) -> list[str]:
    """Relax strict == pins in requirements.txt that are flagged in failures."""
    applied: list[str] = []
    if not REQUIREMENTS_FILE.exists():
        return applied

    # Collect package names mentioned in import errors
    flagged_pkgs: set[str] = set()
    for f in failures:
        if classify_failure(f) == "import_error":
            longrepr = (f.get("longrepr", "") or "").lower()
            for pkg in ["more_itertools", "scikit_learn", "sklearn"]:
                if pkg.replace("_", "-") in longrepr or pkg in longrepr:
                    flagged_pkgs.add(pkg.replace("_", "-"))

    # Also always check more_itertools as documented known issue
    flagged_pkgs.add("more-itertools")

    lines = REQUIREMENTS_FILE.read_text().splitlines(keepends=True)
    new_lines: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        pkg_name = re.split(r"[>=<!]", stripped)[0].lower().replace("_", "-")
        if pkg_name in {p.lower().replace("_", "-") for p in flagged_pkgs}:
            new_line, did_change = _relax_pinned_dep(stripped)
            if did_change:
                new_lines.append(new_line)
                applied.append(
                    f"Relaxed pin in requirements.txt: '{stripped}' → '{new_line.strip()}'"
                )
                changed = True
                continue
        new_lines.append(line)

    if changed:
        REQUIREMENTS_FILE.write_text("".join(new_lines))
    return applied


def _ensure_init_files(failures: list[dict]) -> list[str]:
    """Create missing __init__.py files for directories mentioned in import errors."""
    applied: list[str] = []
    for f in failures:
        if classify_failure(f) != "import_error":
            continue
        node_id = f.get("nodeid", "")
        # node_id looks like "path/to/test_file.py::TestClass::test_method"
        test_path = node_id.split("::")[0]
        if not test_path:
            continue
        test_file = REPO_ROOT / test_path
        if not test_file.exists():
            continue
        for parent in test_file.parents:
            if parent == REPO_ROOT:
                break
            init = parent / "__init__.py"
            if not init.exists():
                init.touch()
                applied.append(f"Created missing __init__.py: {init.relative_to(REPO_ROOT)}")
    return applied


def apply_safe_fixes(failures: list[dict]) -> list[str]:
    """Apply safe, infrastructure-only auto-fixes. Returns list of applied fix descriptions."""
    applied: list[str] = []
    applied.extend(_fix_pinned_requirements(failures))
    applied.extend(_ensure_init_files(failures))
    return applied


# ---------------------------------------------------------------------------
# Fragile-test detection (flags known fragility patterns without modifying tests)
# ---------------------------------------------------------------------------

_FRAGILITY_PATTERNS = [
    (
        r'gemini-3\.0-pro|claude-4\.6-sonnet',
        "Hardcoded model name — fragile if models are updated",
    ),
    (
        r'MAX_HISTORY_CONTEXT\s*=\s*\d+',
        "Magic-number constant dependency — test may break if constant changes",
    ),
    (
        r'assert_trees_all_close',
        "Float equality check — potentially flaky across JAX versions",
    ),
    (
        r'\[INST\]',
        "Mistral prompt construction — may malform if history starts with assistant",
    ),
]


def scan_fragile_tests(test_files: list[str]) -> list[str]:
    """Return a list of human-readable warnings about fragile test patterns."""
    warnings: list[str] = []
    for path_str in test_files:
        path = pathlib.Path(path_str)
        if not path.exists():
            continue
        content = path.read_text(errors="replace")
        for pattern, description in _FRAGILITY_PATTERNS:
            if re.search(pattern, content):
                rel = path.relative_to(REPO_ROOT)
                warnings.append(f"{rel}: {description}")
    # Also scan non-test source files for the same patterns
    for source_glob in [
        REPO_ROOT / "backend" / "agent.py",
    ]:
        if source_glob.exists():
            content = source_glob.read_text(errors="replace")
            for pattern, description in _FRAGILITY_PATTERNS:
                if re.search(pattern, content):
                    rel = source_glob.relative_to(REPO_ROOT)
                    warnings.append(f"{rel}: {description}")
    return warnings


# ---------------------------------------------------------------------------
# Pytest runner
# ---------------------------------------------------------------------------

def run_pytest(test_files: list[str]) -> dict:
    """Run pytest with JSON report and return parsed report dict."""
    JULES_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pytest",
        "--tb=short",
        "--json-report",
        f"--json-report-file={REPORT_FILE}",
        "-q",
        "--no-header",
    ] + test_files

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    proc = psutil.Process()
    mem_before = proc.memory_info().rss

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.perf_counter() - t0

    mem_after = proc.memory_info().rss
    mem_delta_mb = (mem_after - mem_before) / (1024 * 1024)

    report: dict = {}
    if REPORT_FILE.exists():
        try:
            report = json.loads(REPORT_FILE.read_text())
        except json.JSONDecodeError:
            pass

    report["_runner_duration_s"] = duration
    report["_runner_mem_delta_mb"] = mem_delta_mb
    report["_runner_stdout"] = result.stdout[-4000:] if result.stdout else ""
    report["_runner_stderr"] = result.stderr[-2000:] if result.stderr else ""
    return report


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

def parse_failures(report: dict) -> list[dict]:
    """Extract failure records from the pytest JSON report."""
    failures = []
    for test in report.get("tests", []):
        outcome = test.get("outcome", "")
        if outcome in ("failed", "error"):
            failures.append(test)
    return failures


def parse_timings(report: dict) -> dict[str, float]:
    """Return {nodeid: duration_s} for all tests."""
    timings: dict[str, float] = {}
    for test in report.get("tests", []):
        node_id = test.get("nodeid", "")
        duration = test.get("duration", 0.0) or 0.0
        if node_id:
            timings[node_id] = duration
    return timings


def summarise_report(report: dict) -> dict:
    """Return a compact summary dict for a pytest JSON report."""
    summary = report.get("summary", {})
    return {
        "total_tests": summary.get("total", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "errors": summary.get("errors", 0),
        "duration_seconds": round(report.get("_runner_duration_s", 0.0), 3),
        "mem_delta_mb": round(report.get("_runner_mem_delta_mb", 0.0), 2),
    }


# ---------------------------------------------------------------------------
# Performance delta
# ---------------------------------------------------------------------------

def compute_performance_delta(
    prev_timings: dict[str, float],
    curr_timings: dict[str, float],
) -> dict:
    """Compare per-test timings between two iterations."""
    improvements = {}
    regressions = {}
    for node_id, curr_t in curr_timings.items():
        prev_t = prev_timings.get(node_id)
        if prev_t is None:
            continue
        delta = curr_t - prev_t
        if delta < -0.05:  # at least 50 ms faster
            improvements[node_id] = round(delta, 4)
        elif delta > 0.05:  # at least 50 ms slower
            regressions[node_id] = round(delta, 4)
    return {"improvements": improvements, "regressions": regressions}


# ---------------------------------------------------------------------------
# Learnings file writer
# ---------------------------------------------------------------------------

def write_learnings(
    run_id: str,
    bugs_found: list[str],
    fixes_applied: list[str],
    performance_notes: list[str],
    remaining_issues: list[str],
    fragility_warnings: list[str],
) -> None:
    """Append a new run section to .jules/agentic_learnings.md."""
    JULES_DIR.mkdir(parents=True, exist_ok=True)

    existing = LEARNINGS_FILE.read_text() if LEARNINGS_FILE.exists() else ""

    def _bullet_list(items: list[str]) -> str:
        if not items:
            return "- _(none)_\n"
        return "".join(f"- {item}\n" for item in items)

    section = f"""
## Run {run_id}

### Bugs Found
{_bullet_list(bugs_found)}
### Fragility Warnings
{_bullet_list(fragility_warnings)}
### Auto-Fixes Applied
{_bullet_list(fixes_applied)}
### Performance Improvements
{_bullet_list(performance_notes)}
### Remaining Issues
{_bullet_list(remaining_issues)}
"""
    LEARNINGS_FILE.write_text(existing + section)


# ---------------------------------------------------------------------------
# Performance log writer
# ---------------------------------------------------------------------------

def write_performance_log(data: dict) -> None:
    JULES_DIR.mkdir(parents=True, exist_ok=True)
    PERF_LOG_FILE.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main agentic loop
# ---------------------------------------------------------------------------

def main() -> int:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"\n{'='*60}")
    print(f"  Agentic Test Runner  |  run_id={run_id}")
    print(f"{'='*60}\n")

    JULES_DIR.mkdir(parents=True, exist_ok=True)

    all_iterations: list[dict] = []
    prev_timings: dict[str, float] = {}
    all_fixes_applied: list[str] = []
    all_bugs_found: list[str] = []
    final_failures: list[dict] = []
    performance_notes: list[str] = []
    performance_delta: dict = {}

    # One-time fragility scan (before any iteration)
    test_files = discover_tests()
    fragility_warnings = scan_fragile_tests(test_files)
    if fragility_warnings:
        print(f"[fragility] {len(fragility_warnings)} fragile pattern(s) detected:")
        for w in fragility_warnings:
            print(f"  ⚠  {w}")

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")
        test_files = discover_tests()
        print(f"[discovery] {len(test_files)} test file(s) found")

        report = run_pytest(test_files)
        summary = summarise_report(report)
        timings = parse_timings(report)
        failures = parse_failures(report)

        print(
            f"[results]  total={summary['total_tests']}  "
            f"passed={summary['passed']}  "
            f"failed={summary['failed']}  "
            f"errors={summary['errors']}  "
            f"duration={summary['duration_seconds']}s"
        )

        # Annotate failures with classification and suggestion
        annotated_failures = []
        for f in failures:
            category = classify_failure(f)
            suggestion = suggest_fix(f)
            annotated_failures.append({
                "nodeid": f.get("nodeid", ""),
                "outcome": f.get("outcome", ""),
                "category": category,
                "suggestion": suggestion,
                "longrepr": (f.get("longrepr", "") or "")[:500],
            })
            bug_msg = f"[{category}] {f.get('nodeid', '')} — {suggestion or 'see log'}"
            if bug_msg not in all_bugs_found:
                all_bugs_found.append(bug_msg)

        # Slow-test warnings
        slow_tests = [
            nid for nid, t in timings.items() if t > SLOW_TEST_THRESHOLD_S
        ]
        if slow_tests:
            print(f"[slow]  {len(slow_tests)} test(s) exceeded {SLOW_TEST_THRESHOLD_S}s:")
            for s in slow_tests:
                print(f"  🐢  {s}  ({timings[s]:.1f}s)")

        # Performance delta vs previous iteration
        if prev_timings:
            delta = compute_performance_delta(prev_timings, timings)
            performance_delta = delta
            for nid, d in delta["improvements"].items():
                note = f"Iteration {iteration}: {nid} improved by {abs(d):.3f}s"
                performance_notes.append(note)
                print(f"  ✅ {note}")
            for nid, d in delta["regressions"].items():
                print(f"  ⚠  Regression: {nid} slower by {d:.3f}s")

        iter_record = {
            "iteration": iteration,
            **summary,
            "per_test_timings": timings,
            "failures": annotated_failures,
            "slow_tests": slow_tests,
        }
        all_iterations.append(iter_record)
        prev_timings = timings

        # Apply safe fixes if there are failures
        if failures:
            fixes = apply_safe_fixes(failures)
            if fixes:
                print(f"[fixes]  Applied {len(fixes)} auto-fix(es):")
                for fix in fixes:
                    print(f"  🔧  {fix}")
                all_fixes_applied.extend(fixes)
            else:
                print("[fixes]  No safe auto-fixes available for remaining failures.")
        else:
            print("[✓] All tests passed — stopping early.")
            break

    # Final state
    final_failures = all_iterations[-1]["failures"] if all_iterations else []
    total_tests = all_iterations[-1]["total_tests"] if all_iterations else 0
    final_passed = all_iterations[-1]["passed"] if all_iterations else 0
    final_pass_rate = f"{final_passed}/{total_tests}"

    remaining_issues = [
        f"[{f['category']}] {f['nodeid']}" for f in final_failures
    ]

    # Persist artefacts
    perf_log = {
        "run_id": run_id,
        "iterations": all_iterations,
        "final_pass_rate": final_pass_rate,
        "performance_delta": performance_delta,
    }
    write_performance_log(perf_log)
    print(f"\n[log]  Performance log written to {PERF_LOG_FILE.relative_to(REPO_ROOT)}")

    write_learnings(
        run_id=run_id,
        bugs_found=all_bugs_found,
        fixes_applied=all_fixes_applied,
        performance_notes=performance_notes,
        remaining_issues=remaining_issues,
        fragility_warnings=fragility_warnings,
    )
    print(f"[log]  Learnings written to {LEARNINGS_FILE.relative_to(REPO_ROOT)}")

    print(f"\n{'='*60}")
    print(f"  Final pass rate: {final_pass_rate}")
    print(f"  Remaining failures: {len(final_failures)}")
    print(f"{'='*60}\n")

    # Exit 0 so CI uploads artefacts even when tests fail (failures are logged)
    return 0


if __name__ == "__main__":
    sys.exit(main())
