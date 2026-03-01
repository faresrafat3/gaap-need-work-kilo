#!/usr/bin/env python3
"""
GAAP Dependency Security Scanner
================================

Scans project dependencies for:
- Known vulnerabilities (via OSV/PyPA database)
- Outdated packages
- License compatibility
- Suspicious packages

Usage:
    python scripts/security/check-dependencies.py [--requirements FILE] [--output FILE] [--format {json,text}]

Exit codes:
    0 - No issues found
    1 - Issues found
    2 - Error during scan
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
from packaging.requirements import Requirement
from packaging.version import Version, parse as parse_version


@dataclass
class Vulnerability:
    """Represents a vulnerability in a package."""

    package: str
    installed_version: str
    vulnerability_id: str
    severity: str
    summary: str
    fixed_versions: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass
class OutdatedPackage:
    """Represents an outdated package."""

    package: str
    installed_version: str
    latest_version: str
    release_date: str | None = None
    days_outdated: int | None = None


@dataclass
class LicenseInfo:
    """Represents package license information."""

    package: str
    version: str
    license: str
    is_compatible: bool
    warning: str | None = None


@dataclass
class DependencyReport:
    """Complete dependency health report."""

    timestamp: str
    requirements_file: str
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    outdated: list[OutdatedPackage] = field(default_factory=list)
    license_issues: list[LicenseInfo] = field(default_factory=list)
    suspicious_packages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "requirements_file": self.requirements_file,
            "summary": {
                "vulnerabilities": len(self.vulnerabilities),
                "outdated_packages": len(self.outdated),
                "license_issues": len(self.license_issues),
                "suspicious_packages": len(self.suspicious_packages),
            },
            "vulnerabilities": [
                {
                    "package": v.package,
                    "installed_version": v.installed_version,
                    "vulnerability_id": v.vulnerability_id,
                    "severity": v.severity,
                    "summary": v.summary,
                    "fixed_versions": v.fixed_versions,
                    "aliases": v.aliases,
                }
                for v in self.vulnerabilities
            ],
            "outdated_packages": [
                {
                    "package": o.package,
                    "installed_version": o.installed_version,
                    "latest_version": o.latest_version,
                    "release_date": o.release_date,
                    "days_outdated": o.days_outdated,
                }
                for o in self.outdated
            ],
            "license_issues": [
                {
                    "package": l.package,
                    "version": l.version,
                    "license": l.license,
                    "is_compatible": l.is_compatible,
                    "warning": l.warning,
                }
                for l in self.license_issues
            ],
            "suspicious_packages": self.suspicious_packages,
        }


class DependencyScanner:
    """Scans dependencies for security and health issues."""

    # Known malicious/suspicious package patterns
    SUSPICIOUS_PATTERNS = [
        # Typosquatting patterns
        re.compile(r"^[a-z]+-?sdk$", re.I),  # Generic SDK names
        re.compile(r"^[a-z]+-?api$", re.I),  # Generic API names
        re.compile(r"^[a-z]+-?client$", re.I),  # Generic client names
        # Suspicious names
        re.compile(r"test.*test", re.I),  # Double test
        re.compile(r"^.{1,2}$"),  # Very short names
    ]

    # Known typosquatting packages (from public databases)
    KNOWN_TYPO_SQUATS = {
        "djanga",
        "djnago",
        "djago",  # django typos
        "reqeusts",
        "reuqests",  # requests typos
        "urllib3s",  # urllib3 typos
        "nmap-python",  # fake nmap
        "python-nmap",  # potentially fake
    }

    # Incompatible licenses for typical commercial use
    INCOMPATIBLE_LICENSES = {
        "GPL-2.0",
        "GPL-3.0",
        "AGPL-3.0",
        "SSPL-1.0",
    }

    # Warn about these licenses
    WARNING_LICENSES = {
        "LGPL-2.1",
        "LGPL-3.0",
    }

    def __init__(self, requirements_file: str = "requirements.txt") -> None:
        self.requirements_file = Path(requirements_file)
        self.report = DependencyReport(
            timestamp=datetime.utcnow().isoformat(),
            requirements_file=str(self.requirements_file),
        )

    def parse_requirements(self) -> list[tuple[str, str | None]]:
        """Parse requirements file into (package, version_spec) tuples."""
        packages = []

        if not self.requirements_file.exists():
            print(f"Error: Requirements file not found: {self.requirements_file}", file=sys.stderr)
            return packages

        with open(self.requirements_file, "r") as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Skip options
                if line.startswith("-"):
                    continue

                # Skip inline comments
                if "#" in line:
                    line = line.split("#")[0].strip()

                try:
                    req = Requirement(line)
                    # Get the package name
                    name = req.name

                    # Try to get version specifier
                    version = None
                    for spec in req.specifier:
                        if spec.operator in ["==", ">=", "~="]:
                            version = spec.version
                            break

                    packages.append((name, version))
                except Exception:
                    # Fallback for simple parsing
                    if "==" in line:
                        parts = line.split("==")
                        packages.append((parts[0].strip(), parts[1].strip()))
                    elif ">=" in line:
                        parts = line.split(">=")
                        packages.append((parts[0].strip(), parts[1].strip().split(",")[0]))
                    else:
                        packages.append((line, None))

        return packages

    def get_installed_version(self, package: str) -> str | None:
        """Get the installed version of a package."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        except subprocess.CalledProcessError:
            pass
        return None

    def check_vulnerabilities_osv(self, package: str, version: str) -> list[Vulnerability]:
        """Check for vulnerabilities using OSV API."""
        vulnerabilities = []

        try:
            response = requests.post(
                "https://api.osv.dev/v1/query",
                json={
                    "package": {
                        "name": package,
                        "ecosystem": "PyPI",
                    },
                    "version": version,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            for vuln in data.get("vulns", []):
                # Determine severity
                severity = "unknown"
                for item in vuln.get("severity", []):
                    if item.get("type") == "CVSS_V3":
                        score = item.get("score", "0")
                        if isinstance(score, str) and score.startswith("CVSS:"):
                            # Parse CVSS score
                            try:
                                numeric_score = float(score.split("/")[0].split(":")[-1])
                                if numeric_score >= 7.0:
                                    severity = "high"
                                elif numeric_score >= 4.0:
                                    severity = "medium"
                                else:
                                    severity = "low"
                            except (ValueError, IndexError):
                                pass

                # Get fixed versions
                fixed_versions = []
                for event in vuln.get("affected", [{}])[0].get("ranges", [{}])[0].get("events", []):
                    if "fixed" in event:
                        fixed_versions.append(event["fixed"])

                vulnerabilities.append(
                    Vulnerability(
                        package=package,
                        installed_version=version,
                        vulnerability_id=vuln.get("id", "UNKNOWN"),
                        severity=severity,
                        summary=vuln.get("summary", "No summary available"),
                        fixed_versions=fixed_versions,
                        aliases=vuln.get("aliases", []),
                    )
                )

        except requests.RequestException as e:
            print(f"Warning: Could not check vulnerabilities for {package}: {e}", file=sys.stderr)

        return vulnerabilities

    def check_outdated(self, package: str, current_version: str) -> OutdatedPackage | None:
        """Check if a package is outdated using PyPI."""
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package}/json",
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            latest_version = data["info"]["version"]

            # Compare versions
            try:
                current = parse_version(current_version)
                latest = parse_version(latest_version)

                if latest > current:
                    # Get release date
                    release_date = None
                    days_outdated = None

                    if latest_version in data.get("releases", {}):
                        releases = data["releases"][latest_version]
                        if releases:
                            upload_time = releases[0].get("upload_time")
                            if upload_time:
                                release_date = upload_time
                                try:
                                    release_dt = datetime.fromisoformat(
                                        upload_time.replace("Z", "+00:00")
                                    )
                                    days_outdated = (
                                        datetime.now().replace(tzinfo=release_dt.tzinfo)
                                        - release_dt
                                    ).days
                                except ValueError:
                                    pass

                    return OutdatedPackage(
                        package=package,
                        installed_version=current_version,
                        latest_version=latest_version,
                        release_date=release_date,
                        days_outdated=days_outdated,
                    )
            except Exception:
                pass

        except requests.RequestException:
            pass

        return None

    def check_license(self, package: str, version: str) -> LicenseInfo | None:
        """Check package license compatibility."""
        try:
            response = requests.get(
                f"https://pypi.org/pypi/{package}/json",
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            license_str = data["info"].get("license", "Unknown")

            # Check for incompatible licenses
            is_compatible = True
            warning = None

            license_upper = license_str.upper()
            for bad_license in self.INCOMPATIBLE_LICENSES:
                if bad_license.upper() in license_upper:
                    is_compatible = False
                    warning = f"License '{license_str}' may be incompatible with commercial use"
                    break

            for warn_license in self.WARNING_LICENSES:
                if warn_license.upper() in license_upper:
                    warning = f"License '{license_str}' has copyleft requirements"
                    break

            # Check for unknown/unset licenses
            if not license_str or license_str in ["Unknown", "UNKNOWN", "", "LICENSE"]:
                warning = "License information not properly specified"

            return LicenseInfo(
                package=package,
                version=version,
                license=license_str,
                is_compatible=is_compatible,
                warning=warning,
            )

        except requests.RequestException:
            return None

    def check_suspicious(self, package: str) -> dict | None:
        """Check if a package name is suspicious."""
        warnings = []

        # Check for known typosquats
        if package.lower() in self.KNOWN_TYPO_SQUATS:
            warnings.append("Known typosquatting package")

        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.match(package):
                warnings.append(f"Matches suspicious pattern: {pattern.pattern}")

        # Check character homographs (basic)
        suspicious_chars = set("оοο０Ｏ")  # Cyrillic/Greek lookalikes
        for char in package:
            if char in suspicious_chars:
                warnings.append("Contains potential homograph attack characters")
                break

        if warnings:
            return {
                "package": package,
                "warnings": warnings,
            }

        return None

    def scan(self) -> DependencyReport:
        """Run full dependency scan."""
        packages = self.parse_requirements()

        print(f"Scanning {len(packages)} packages from {self.requirements_file}...")

        for package, specified_version in packages:
            print(f"  Checking {package}...", end=" ", flush=True)

            # Get installed version
            installed_version = self.get_installed_version(package)
            if not installed_version:
                installed_version = specified_version or "unknown"

            # Check vulnerabilities
            vulns = self.check_vulnerabilities_osv(package, installed_version)
            self.report.vulnerabilities.extend(vulns)

            # Check outdated
            outdated = self.check_outdated(package, installed_version)
            if outdated:
                self.report.outdated.append(outdated)

            # Check license
            license_info = self.check_license(package, installed_version)
            if license_info and (not license_info.is_compatible or license_info.warning):
                self.report.license_issues.append(license_info)

            # Check suspicious
            suspicious = self.check_suspicious(package)
            if suspicious:
                self.report.suspicious_packages.append(suspicious)

            # Print summary for this package
            issues = (
                len(vulns)
                + (1 if outdated else 0)
                + (1 if license_info and license_info.warning else 0)
                + (1 if suspicious else 0)
            )
            if issues == 0:
                print("✓")
            else:
                print(f"⚠ ({issues} issues)")

        return self.report


def format_text_report(report: DependencyReport) -> str:
    """Format report as human-readable text."""
    lines = [
        "=" * 80,
        "GAAP DEPENDENCY SECURITY SCAN REPORT",
        "=" * 80,
        "",
        f"Timestamp: {report.timestamp}",
        f"Requirements File: {report.requirements_file}",
        "",
        "-" * 80,
        "SUMMARY",
        "-" * 80,
        f"  Vulnerabilities:    {len(report.vulnerabilities)}",
        f"  Outdated Packages:  {len(report.outdated)}",
        f"  License Issues:     {len(report.license_issues)}",
        f"  Suspicious Packages: {len(report.suspicious_packages)}",
        "",
    ]

    # Vulnerabilities
    if report.vulnerabilities:
        lines.extend(
            [
                "-" * 80,
                f"VULNERABILITIES ({len(report.vulnerabilities)})",
                "-" * 80,
                "",
            ]
        )

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
        sorted_vulns = sorted(
            report.vulnerabilities, key=lambda v: severity_order.get(v.severity, 5)
        )

        for i, vuln in enumerate(sorted_vulns, 1):
            lines.extend(
                [
                    f"  [{i}] {vuln.vulnerability_id} [{vuln.severity.upper()}]",
                    f"      Package: {vuln.package}@{vuln.installed_version}",
                    f"      Summary: {vuln.summary}",
                ]
            )
            if vuln.fixed_versions:
                lines.append(f"      Fixed in: {', '.join(vuln.fixed_versions)}")
            if vuln.aliases:
                lines.append(f"      Aliases: {', '.join(vuln.aliases[:3])}")
            lines.append("")

    # Outdated packages
    if report.outdated:
        lines.extend(
            [
                "-" * 80,
                f"OUTDATED PACKAGES ({len(report.outdated)})",
                "-" * 80,
                "",
            ]
        )

        # Sort by days outdated
        sorted_outdated = sorted(report.outdated, key=lambda o: o.days_outdated or 0, reverse=True)

        for pkg in sorted_outdated:
            days_str = f"({pkg.days_outdated} days old)" if pkg.days_outdated else ""
            lines.append(
                f"  {pkg.package}: {pkg.installed_version} → {pkg.latest_version} {days_str}"
            )
        lines.append("")

    # License issues
    if report.license_issues:
        lines.extend(
            [
                "-" * 80,
                f"LICENSE ISSUES ({len(report.license_issues)})",
                "-" * 80,
                "",
            ]
        )

        for lic in report.license_issues:
            status = "❌" if not lic.is_compatible else "⚠️"
            lines.append(f"  {status} {lic.package}@{lic.version}: {lic.license}")
            if lic.warning:
                lines.append(f"      Warning: {lic.warning}")
        lines.append("")

    # Suspicious packages
    if report.suspicious_packages:
        lines.extend(
            [
                "-" * 80,
                f"SUSPICIOUS PACKAGES ({len(report.suspicious_packages)})",
                "-" * 80,
                "",
            ]
        )

        for susp in report.suspicious_packages:
            lines.append(f"  ⚠️ {susp['package']}")
            for warning in susp["warnings"]:
                lines.append(f"      - {warning}")
        lines.append("")

    lines.extend(
        [
            "-" * 80,
            "END OF REPORT",
            "-" * 80,
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GAAP Dependency Security Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/security/check-dependencies.py
  python scripts/security/check-dependencies.py --requirements requirements.txt
  python scripts/security/check-dependencies.py --output report.json --format json
        """,
    )
    parser.add_argument(
        "--requirements",
        "-r",
        type=str,
        default="requirements.txt",
        help="Requirements file to scan (default: requirements.txt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--fail-on-vuln",
        action="store_true",
        help="Fail if vulnerabilities are found",
    )

    args = parser.parse_args()

    try:
        scanner = DependencyScanner(requirements_file=args.requirements)
        report = scanner.scan()

        # Generate output
        if args.format == "json":
            output = json.dumps(report.to_dict(), indent=2)
        else:
            output = format_text_report(report)

        # Write or print output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"\nReport written to: {args.output}")
        else:
            print(output)

        # Determine exit code
        total_issues = (
            len(report.vulnerabilities)
            + len(report.license_issues)
            + len(report.suspicious_packages)
        )

        if args.fail_on_vuln and report.vulnerabilities:
            print(f"\n❌ Found {len(report.vulnerabilities)} vulnerability(s)")
            return 1

        if total_issues > 0:
            print(f"\n⚠️ Found {total_issues} issue(s) - review recommended")
            # Don't fail on warnings unless explicitly requested
            return 0

        print("\n✅ No security issues found")
        return 0

    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error during scan: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
