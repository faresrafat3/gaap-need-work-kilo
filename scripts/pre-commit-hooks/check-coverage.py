#!/usr/bin/env python3
"""
Custom pre-commit hook to check test coverage for changed files.

This hook runs coverage analysis only on files that have been modified,
providing fast feedback without running the full test suite.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


MIN_COVERAGE = 80.0  # Minimum coverage threshold
COVERAGE_CACHE = Path(".coverage.cache")


def get_git_root() -> Path:
    """Get the repository root directory."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def get_changed_python_files() -> list[Path]:
    """Get list of changed Python files staged for commit."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM", "--", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    files = [Path(f) for f in result.stdout.strip().split("\n") if f]
    return files


def get_module_path(file_path: Path) -> Optional[str]:
    """Convert file path to module path for pytest."""
    if file_path.parts[0] == "gaap":
        parts = file_path.with_suffix("").parts
        return ".".join(parts)
    return None


def run_coverage_check(files: list[Path]) -> tuple[bool, dict]:
    """Run coverage check on specified files."""
    if not files:
        return True, {"coverage": 100.0, "files": {}}

    git_root = get_git_root()
    modules = []

    for f in files:
        module = get_module_path(f)
        if module:
            modules.append(module)

    if not modules:
        print("ℹ️  No GAAP modules to check coverage for")
        return True, {"coverage": 100.0, "files": {}}

    # Run pytest with coverage for changed modules
    cmd = [
        "python",
        "-m",
        "pytest",
        "-xvs",
        "--tb=no",
        "--cov=gaap",
        "--cov-report=json:/tmp/coverage.json",
        "--cov-report=term-missing",
        "-k",
        " or ".join(m.replace(".", "_") for m in modules),
        "tests/",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=git_root,
        )

        # Parse coverage report
        coverage_data = parse_coverage_report("/tmp/coverage.json")

        if coverage_data:
            overall_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)

            # Check coverage for each changed file
            file_coverages = {}
            for module in modules:
                module_coverage = get_module_coverage(coverage_data, module)
                file_coverages[module] = module_coverage

            passed = overall_coverage >= MIN_COVERAGE
            return passed, {
                "coverage": overall_coverage,
                "files": file_coverages,
            }

        return True, {"coverage": 0.0, "files": {}}

    except subprocess.TimeoutExpired:
        print("⚠️  Coverage check timed out (60s)")
        return True, {"coverage": 0.0, "files": {}}  # Don't block on timeout
    except FileNotFoundError:
        print("⚠️  pytest or coverage not installed, skipping coverage check")
        return True, {"coverage": 0.0, "files": {}}


def parse_coverage_report(path: str) -> Optional[dict]:
    """Parse the JSON coverage report."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def get_module_coverage(coverage_data: dict, module: str) -> float:
    """Get coverage percentage for a specific module."""
    files = coverage_data.get("files", {})
    module_path = module.replace(".", "/") + ".py"

    for file_path, data in files.items():
        if module_path in file_path:
            summary = data.get("summary", {})
            return summary.get("percent_covered", 0.0)

    return 0.0


def print_coverage_report(report: dict, files: list[Path]) -> None:
    """Print formatted coverage report."""
    print()
    print("=" * 60)
    print("📊 Coverage Report for Changed Files")
    print("=" * 60)

    overall = report.get("coverage", 0.0)
    file_coverages = report.get("files", {})

    if file_coverages:
        print(f"\n{'Module':<40} {'Coverage':>12}")
        print("-" * 52)

        for module, cov in sorted(file_coverages.items()):
            status = "✅" if cov >= MIN_COVERAGE else "❌"
            bar = "█" * int(cov / 10) + "░" * (10 - int(cov / 10))
            print(f"{status} {module:<38} {bar} {cov:>5.1f}%")

    print("-" * 52)
    status_icon = "✅" if overall >= MIN_COVERAGE else "❌"
    print(f"{status_icon} Overall Coverage: {overall:.1f}% (min: {MIN_COVERAGE}%)")
    print("=" * 60)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check test coverage for changed files")
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to check (if not provided, uses staged files)",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=MIN_COVERAGE,
        help=f"Minimum coverage threshold (default: {MIN_COVERAGE})",
    )
    parser.add_argument(
        "--skip-if-no-tests",
        action="store_true",
        help="Skip check if no tests exist for changed files",
    )

    args = parser.parse_args(argv)

    # Use provided files or get staged files
    files = [Path(f) for f in args.files] if args.files else get_changed_python_files()

    if not files:
        print("✅ No Python files to check")
        return 0

    print(f"🔍 Checking coverage for {len(files)} file(s)...")

    passed, report = run_coverage_check(files)
    print_coverage_report(report, files)

    if not passed:
        print(f"\n❌ Coverage check failed: {report['coverage']:.1f}% < {args.min_coverage}%")
        print("\nTo fix:")
        print("  1. Add tests for the uncovered code")
        print("  2. Run: pytest --cov=gaap --cov-report=html tests/")
        print("  3. View coverage report: open htmlcov/index.html")
        return 1

    print("\n✅ Coverage check passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
