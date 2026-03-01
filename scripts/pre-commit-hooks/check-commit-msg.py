#!/usr/bin/env python3
"""
Commit message format checker.

Validates commit messages follow conventional commits format:
<type>[optional scope]: <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style (formatting, missing semi colons, etc)
- refactor: Code refactoring
- perf: Performance improvements
- test: Test additions/updates
- chore: Build process or auxiliary tool changes
- ci: CI configuration changes
- security: Security-related changes
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional


# Conventional commit pattern
CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>feat|fix|docs|style|refactor|perf|test|chore|ci|security|build|revert)"
    r"(\((?P<scope>[a-z-]+)\))?"
    r"(?P<breaking>!)?:"
    r" (?P<message>.{10,})$",
    re.IGNORECASE,
)

# Pattern to detect issue references
ISSUE_REF_RE = re.compile(r"#\d+|\b(closes?|fixes?|resolves?|refs?)\s+#\d+", re.IGNORECASE)

# Maximum lengths
MAX_HEADER_LENGTH = 72
MAX_BODY_LINE_LENGTH = 100

VALID_TYPES = {
    "feat": "New feature",
    "fix": "Bug fix",
    "docs": "Documentation changes",
    "style": "Code style/formatting",
    "refactor": "Code refactoring",
    "perf": "Performance improvements",
    "test": "Test additions/updates",
    "chore": "Build/tooling changes",
    "ci": "CI configuration changes",
    "security": "Security-related changes",
    "build": "Build system changes",
    "revert": "Revert previous commit",
}


class CommitMessageError:
    """Represents a validation error."""

    def __init__(self, message: str, line: Optional[int] = None):
        self.message = message
        self.line = line

    def __str__(self) -> str:
        if self.line:
            return f"Line {self.line}: {self.message}"
        return self.message


def validate_commit_message(message: str) -> list[CommitMessageError]:
    """Validate a commit message."""
    errors: list[CommitMessageError] = []
    lines = message.split("\n")

    if not lines or not lines[0].strip():
        errors.append(CommitMessageError("Commit message cannot be empty"))
        return errors

    header = lines[0]

    # Check for merge commits
    if header.startswith("Merge ") or header.startswith("Revert "):
        return errors  # Merge/revert commits are valid

    # Check header length
    if len(header) > MAX_HEADER_LENGTH:
        errors.append(
            CommitMessageError(
                f"Header too long ({len(header)} > {MAX_HEADER_LENGTH} chars)",
                line=1,
            )
        )

    # Check conventional commit format
    match = CONVENTIONAL_COMMIT_RE.match(header)
    if not match:
        # Check if it starts with a type but missing colon
        if re.match(r"^(feat|fix|docs|style|refactor|perf|test|chore|ci|security)\b", header):
            errors.append(
                CommitMessageError(
                    "Missing colon after type/scope. Format: type: message",
                    line=1,
                )
            )
        else:
            errors.append(
                CommitMessageError(
                    f"Invalid format. Expected: <type>: <description>\n"
                    f"Valid types: {', '.join(VALID_TYPES.keys())}",
                    line=1,
                )
            )
        return errors

    # Validate type
    commit_type = match.group("type").lower()
    if commit_type not in VALID_TYPES:
        errors.append(
            CommitMessageError(
                f"Invalid type '{commit_type}'. Valid: {', '.join(VALID_TYPES.keys())}",
                line=1,
            )
        )

    # Validate message length (after type)
    msg_content = match.group("message")
    if len(msg_content) < 10:
        errors.append(
            CommitMessageError(
                f"Description too short ({len(msg_content)} < 10 chars). "
                "Please provide a meaningful description.",
                line=1,
            )
        )

    # Check if message starts with lowercase (unless it's a proper noun)
    if msg_content[0].isupper() and not msg_content[0:2].isupper():
        # Allow uppercase for proper nouns like "GAAP", "API", etc.
        pass  # This is a style preference, not enforced

    # Check for trailing period in header
    if header.rstrip().endswith("."):
        errors.append(
            CommitMessageError(
                "Header should not end with a period",
                line=1,
            )
        )

    # If there are body lines, validate them
    if len(lines) > 2:
        # Check for blank line between header and body
        if lines[1].strip():
            errors.append(
                CommitMessageError(
                    "Add a blank line between header and body",
                    line=2,
                )
            )

        # Validate body line lengths
        for i, line in enumerate(lines[2:], start=3):
            # Skip quoted lines and lines with URLs
            if line.strip().startswith(">") or "http" in line:
                continue

            if len(line) > MAX_BODY_LINE_LENGTH:
                errors.append(
                    CommitMessageError(
                        f"Body line too long ({len(line)} > {MAX_BODY_LINE_LENGTH} chars)",
                        line=i,
                    )
                )

    return errors


def format_error_report(errors: list[CommitMessageError], message: str) -> str:
    """Format error report."""
    lines = [
        "",
        "=" * 70,
        "❌ Commit Message Validation Failed",
        "=" * 70,
        "",
        "Your commit message:",
        "-" * 70,
    ]

    for i, line in enumerate(message.split("\n"), 1):
        lines.append(f"{i:3}: {line}")

    lines.extend(
        [
            "-" * 70,
            "",
            "Errors:",
        ]
    )

    for error in errors:
        lines.append(f"  • {error}")

    lines.extend(
        [
            "",
            "=" * 70,
            "Expected Format:",
            "  <type>[optional scope]: <description>",
            "",
            "Valid types:",
        ]
    )

    for t, desc in VALID_TYPES.items():
        lines.append(f"  {t:10} - {desc}")

    lines.extend(
        [
            "",
            "Examples:",
            "  feat: add user authentication system",
            "  fix(api): resolve timeout issue in provider calls",
            "  docs: update README with installation steps",
            "  test(router): add unit tests for fallback logic",
            "",
            "For more details: https://www.conventionalcommits.org/",
            "=" * 70,
        ]
    )

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate commit message format")
    parser.add_argument(
        "commit_msg_file",
        help="Path to commit message file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce strict validation (fail on warnings)",
    )

    args = parser.parse_args(argv)

    try:
        message = Path(args.commit_msg_file).read_text(encoding="utf-8")
    except (IOError, UnicodeDecodeError) as e:
        print(f"Error reading commit message file: {e}", file=sys.stderr)
        return 1

    errors = validate_commit_message(message)

    if errors:
        print(format_error_report(errors, message), file=sys.stderr)
        return 1

    # Optional: print success for verbose mode
    # print("✅ Commit message format valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
