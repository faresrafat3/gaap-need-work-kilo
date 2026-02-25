"""
Data Loss Prevention (DLP) Scanner
Implements: docs/evolution_plan_2026/39_SECURITY_AUDIT_SPEC.md

Features:
- Regex pattern matching for known secrets
- Shannon entropy analysis for unknown secrets
- PII detection (email, phone, SSN)
- File path detection
- Auto-redaction
"""

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class LeakType(Enum):
    API_KEY = auto()
    AWS_KEY = auto()
    GITHUB_TOKEN = auto()
    PRIVATE_KEY = auto()
    EMAIL = auto()
    PHONE = auto()
    SSN = auto()
    CREDIT_CARD = auto()
    IP_ADDRESS = auto()
    FILE_PATH = auto()
    HIGH_ENTROPY = auto()
    INTERNAL_URL = auto()


@dataclass
class DLPFinding:
    leak_type: LeakType
    value: str
    start_pos: int
    end_pos: int
    entropy: float = 0.0
    redacted_value: str = ""
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.leak_type.name,
            "severity": self.severity,
            "entropy": self.entropy,
            "redacted": self.redacted_value,
        }


@dataclass
class DLPScanResult:
    is_safe: bool
    findings: list[DLPFinding] = field(default_factory=list)
    redacted_text: str = ""
    scan_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "scan_time_ms": self.scan_time_ms,
        }


logger = logging.getLogger("gaap.security.dlp")


class DLPScanner:
    """
    Advanced scanner to prevent leakage of PII and Secrets.

    Features:
    - Multi-pattern detection (API keys, tokens, PII)
    - Shannon entropy analysis
    - File path detection
    - Auto-redaction

    Usage:
        scanner = DLPScanner()
        result = scanner.scan("My API key is sk-abc123...")
        print(result.redacted_text)  # My API key is [REDACTED_API_KEY]
    """

    PATTERNS: dict[LeakType, str] = {
        LeakType.API_KEY: r"(?:sk-[a-zA-Z0-9]{32,}|AIza[a-zA-Z0-9_-]{35}|xox[baprs]-[a-zA-Z0-9-]+)",
        LeakType.AWS_KEY: r"AKIA[0-9A-Z]{16}",
        LeakType.GITHUB_TOKEN: r"ghp_[a-zA-Z0-9]{36}|gho_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{22,}",
        LeakType.PRIVATE_KEY: r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+PRIVATE KEY-----",
        LeakType.EMAIL: r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        LeakType.PHONE: r"\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        LeakType.SSN: r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
        LeakType.CREDIT_CARD: r"\b(?:\d[ -]*?){13,16}\b",
        LeakType.IP_ADDRESS: r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        LeakType.FILE_PATH: r"(?:/home/|/Users/|/etc/|C:\\Users\\|~\\/)[a-zA-Z0-9_/.-]+",
        LeakType.INTERNAL_URL: r"(?:localhost|127\.0\.0\.1|192\.168\.|10\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)(?:[:/][a-zA-Z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?",
    }

    REDACTION_TEMPLATES: dict[LeakType, str] = {
        LeakType.API_KEY: "[REDACTED_API_KEY]",
        LeakType.AWS_KEY: "[REDACTED_AWS_KEY]",
        LeakType.GITHUB_TOKEN: "[REDACTED_GITHUB_TOKEN]",
        LeakType.PRIVATE_KEY: "[REDACTED_PRIVATE_KEY]",
        LeakType.EMAIL: "[REDACTED_EMAIL]",
        LeakType.PHONE: "[REDACTED_PHONE]",
        LeakType.SSN: "[REDACTED_SSN]",
        LeakType.CREDIT_CARD: "[REDACTED_CC]",
        LeakType.IP_ADDRESS: "[REDACTED_IP]",
        LeakType.FILE_PATH: "[REDACTED_PATH]",
        LeakType.HIGH_ENTROPY: "[REDACTED_SECRET]",
        LeakType.INTERNAL_URL: "[REDACTED_INTERNAL_URL]",
    }

    SEVERITY_MAP: dict[LeakType, str] = {
        LeakType.PRIVATE_KEY: "critical",
        LeakType.AWS_KEY: "critical",
        LeakType.GITHUB_TOKEN: "critical",
        LeakType.API_KEY: "high",
        LeakType.SSN: "high",
        LeakType.CREDIT_CARD: "high",
        LeakType.EMAIL: "medium",
        LeakType.PHONE: "medium",
        LeakType.IP_ADDRESS: "low",
        LeakType.FILE_PATH: "low",
        LeakType.HIGH_ENTROPY: "medium",
        LeakType.INTERNAL_URL: "medium",
    }

    def __init__(
        self,
        entropy_threshold: float = 4.0,
        min_secret_length: int = 16,
        redact: bool = True,
    ):
        self.entropy_threshold = entropy_threshold
        self.min_secret_length = min_secret_length
        self.redact = redact
        self._compiled_patterns: dict[LeakType, re.Pattern] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for leak_type, pattern in self.PATTERNS.items():
            try:
                self._compiled_patterns[leak_type] = re.compile(pattern, re.IGNORECASE)
            except re.error as e:
                logger.warning(f"Invalid pattern for {leak_type}: {e}")

    def _calculate_entropy(self, text: str) -> float:
        if not text or len(text) < 4:
            return 0.0

        freq: dict[str, int] = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        length = len(text)
        entropy = 0.0
        for count in freq.values():
            prob = count / length
            entropy -= prob * math.log2(prob)

        return entropy

    def scan(self, text: str) -> DLPScanResult:
        import time

        start_time = time.time()

        findings: list[DLPFinding] = []
        redacted_text = text

        for leak_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                value = match.group()
                entropy = self._calculate_entropy(value)

                if leak_type == LeakType.API_KEY and entropy < 3.5:
                    continue

                finding = DLPFinding(
                    leak_type=leak_type,
                    value=value,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    entropy=entropy,
                    redacted_value=self.REDACTION_TEMPLATES.get(leak_type, "[REDACTED]"),
                    severity=self.SEVERITY_MAP.get(leak_type, "medium"),
                )
                findings.append(finding)

                if self.redact:
                    redacted_text = redacted_text.replace(value, finding.redacted_value)

        for word in re.finditer(
            r"\b[A-Za-z0-9+/=_-]{" + str(self.min_secret_length) + r",}\b", text
        ):
            value = word.group()
            if any(f.value == value for f in findings):
                continue

            entropy = self._calculate_entropy(value)
            if entropy > self.entropy_threshold:
                finding = DLPFinding(
                    leak_type=LeakType.HIGH_ENTROPY,
                    value=value,
                    start_pos=word.start(),
                    end_pos=word.end(),
                    entropy=entropy,
                    redacted_value="[REDACTED_SECRET]",
                    severity="medium",
                )
                findings.append(finding)

                if self.redact:
                    redacted_text = redacted_text.replace(value, "[REDACTED_SECRET]")

        scan_time = (time.time() - start_time) * 1000

        is_safe = len(findings) == 0

        if findings:
            logger.warning(f"DLP found {len(findings)} potential leaks")

        return DLPScanResult(
            is_safe=is_safe,
            findings=findings,
            redacted_text=redacted_text,
            scan_time_ms=scan_time,
        )

    def scan_and_redact(self, text: str) -> str:
        return self.scan(text).redacted_text

    def audit_leaks(self, text: str) -> list[dict[str, Any]]:
        result = self.scan(text)
        return [f.to_dict() for f in result.findings]

    def get_stats(self) -> dict[str, Any]:
        return {
            "patterns_count": len(self.PATTERNS),
            "entropy_threshold": self.entropy_threshold,
            "min_secret_length": self.min_secret_length,
        }


def create_dlp_scanner(
    entropy_threshold: float = 4.0,
    strict: bool = False,
) -> DLPScanner:
    if strict:
        return DLPScanner(entropy_threshold=3.5, min_secret_length=12)
    return DLPScanner(entropy_threshold=entropy_threshold)
