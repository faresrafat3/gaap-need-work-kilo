"""
StreamingAuditor - Real-Time Thought Monitor
=============================================

The "Inner Monologue" that monitors agent's thoughts during execution.
Detects issues in real-time and injects system interrupts.

Detects:
- Circular Reasoning: Repeating patterns in thoughts
- Complexity Spike: Code >> planned complexity
- Safety Violations: Hardcoded secrets, dangerous patterns
- Off-topic Drift: Straying from task goals

Usage:
    auditor = StreamingAuditor()

    for thought in thought_stream:
        result = auditor.audit_thought(thought)

        if result.should_interrupt:
            inject_to_stream(result.interrupt_message)
"""

import hashlib
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("gaap.core.auditor")


class AuditIssueType(Enum):
    """أنواع مشاكل التدقيق"""

    CIRCULAR_REASONING = auto()
    COMPLEXITY_SPIKE = auto()
    SAFETY_VIOLATION = auto()
    OFF_TOPIC_DRIFT = auto()
    REPETITION = auto()
    LOGIC_ERROR = auto()
    SCOPE_CREEP = auto()
    INCOMPLETE_THOUGHT = auto()


class AuditSeverity(Enum):
    """مستويات الخطورة"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditIssue:
    """مشكلة مكتشفة في التفكير"""

    type: AuditIssueType
    severity: AuditSeverity
    message: str
    detected_pattern: str
    suggestion: str
    context: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name,
            "severity": self.severity.value,
            "message": self.message,
            "detected_pattern": self.detected_pattern,
            "suggestion": self.suggestion,
            "context": self.context[:200],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class AuditResult:
    """نتيجة تدقيق فكرة"""

    has_issues: bool
    issues: list[AuditIssue] = field(default_factory=list)
    should_interrupt: bool = False
    interrupt_message: str | None = None
    thought_hash: str = ""
    similarity_to_previous: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_issues": self.has_issues,
            "issues": [i.to_dict() for i in self.issues],
            "should_interrupt": self.should_interrupt,
            "interrupt_message": self.interrupt_message,
            "thought_hash": self.thought_hash,
            "similarity_to_previous": self.similarity_to_previous,
        }


@dataclass
class AuditEvent:
    """حدث تدقيق مسجل"""

    timestamp: datetime
    thought_snippet: str
    issue_type: str | None
    severity: str
    action_taken: str
    system_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "thought_snippet": self.thought_snippet[:200],
            "issue_type": self.issue_type,
            "severity": self.severity,
            "action_taken": self.action_taken,
            "system_message": self.system_message,
        }


class StreamingAuditor:
    """
    Real-time thought auditor for Layer3 execution.

    Monitors the agent's thought stream and injects system interrupts
    when problematic patterns are detected.

    Features:
    - Circular reasoning detection via pattern matching
    - Complexity monitoring via code analysis
    - Safety pattern scanning for secrets
    - Repetition tracking with similarity scores
    - Off-topic drift detection

    Attributes:
        enabled: Whether auditing is active
        max_repetition: Max allowed repetitions
        complexity_threshold: Multiplier for complexity spike
        _thought_history: Recent thoughts for pattern analysis
        _pattern_counts: Track pattern frequencies
        _audit_log: Record of all audit events
    """

    SAFETY_PATTERNS = [
        (r"password\s*=\s*['\"][^'\"]+['\"]", "hardcoded_password"),
        (r"api_key\s*=\s*['\"][^'\"]+['\"]", "hardcoded_api_key"),
        (r"secret\s*=\s*['\"][^'\"]+['\"]", "hardcoded_secret"),
        (r"token\s*=\s*['\"][^'\"]+['\"]", "hardcoded_token"),
        (r"credential\s*=\s*['\"][^'\"]+['\"]", "hardcoded_credential"),
        (r"private_key\s*=\s*['\"]", "hardcoded_private_key"),
        (r"aws_access_key\s*=\s*['\"]", "hardcoded_aws_key"),
        (r"database_url\s*=\s*['\"][^'\"]+['\"]", "hardcoded_db_url"),
    ]

    DANGEROUS_PATTERNS = [
        (r"eval\s*\(", "eval_usage"),
        (r"exec\s*\(", "exec_usage"),
        (r"__import__\s*\(", "dynamic_import"),
        (r"subprocess\.call\s*\([^)]*shell\s*=\s*True", "shell_injection"),
        (r"os\.system\s*\(", "os_system"),
        (r"pickle\.loads?\s*\(", "pickle_usage"),
        (r"yaml\.load\s*\([^)]*\)", "unsafe_yaml"),
    ]

    CIRCULAR_INDICATORS = [
        "let me try again",
        "going back to",
        "as I mentioned before",
        "like I said",
        "as stated earlier",
        "returning to the previous",
        "revisiting the same",
    ]

    def __init__(
        self,
        enabled: bool = True,
        max_repetition: int = 3,
        complexity_threshold: float = 2.0,
        history_size: int = 50,
    ) -> None:
        self.enabled = enabled
        self.max_repetition = max_repetition
        self.complexity_threshold = complexity_threshold

        self._thought_history: deque[str] = deque(maxlen=history_size)
        self._thought_hashes: deque[str] = deque(maxlen=history_size)
        self._pattern_counts: dict[str, int] = {}
        self._audit_log: list[AuditEvent] = []
        self._interrupts_injected = 0
        self._issues_found = 0

        self._logger = logger

        self._topic_keywords: set[str] = set()
        self._original_goal: str = ""

    def set_context(self, goal: str, keywords: list[str] | None = None) -> None:
        """
        Set the task context for drift detection.

        Args:
            goal: The original task goal
            keywords: Important keywords for the task
        """
        self._original_goal = goal
        self._topic_keywords = set(keywords or [])
        if goal:
            self._topic_keywords.update(self._extract_keywords(goal))

    def audit_thought(self, thought: str) -> AuditResult:
        """
        Audit a single thought from the stream.

        Args:
            thought: The thought text to audit

        Returns:
            AuditResult with any issues found
        """
        if not self.enabled:
            return AuditResult(has_issues=False, thought_hash=self._hash_thought(thought))

        issues: list[AuditIssue] = []
        thought_hash = self._hash_thought(thought)

        self._thought_history.append(thought)
        self._thought_hashes.append(thought_hash)

        similarity = self._check_similarity(thought)
        circular_issues = self._check_circular_reasoning(thought, similarity)
        issues.extend(circular_issues)

        safety_issues = self._check_safety_violations(thought)
        issues.extend(safety_issues)

        dangerous_issues = self._check_dangerous_patterns(thought)
        issues.extend(dangerous_issues)

        repetition_issues = self._check_repetition(thought, thought_hash)
        issues.extend(repetition_issues)

        drift_issues = self._check_topic_drift(thought)
        issues.extend(drift_issues)

        has_issues = len(issues) > 0
        should_interrupt = any(
            i.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL] for i in issues
        )

        interrupt_message = None
        if should_interrupt:
            interrupt_message = self._generate_interrupt(issues)

        event = AuditEvent(
            timestamp=datetime.now(),
            thought_snippet=thought[:200],
            issue_type=issues[0].type.name if issues else None,
            severity=issues[0].severity.value if issues else "none",
            action_taken="interrupt" if should_interrupt else "log",
            system_message=interrupt_message,
        )
        self._audit_log.append(event)

        if has_issues:
            self._issues_found += 1
            self._logger.info(f"Audit issue: {issues[0].type.name} - {issues[0].message[:50]}")

        if should_interrupt:
            self._interrupts_injected += 1
            self._logger.warning(
                f"Injecting interrupt: {interrupt_message[:100] if interrupt_message else ''}"
            )

        return AuditResult(
            has_issues=has_issues,
            issues=issues,
            should_interrupt=should_interrupt,
            interrupt_message=interrupt_message,
            thought_hash=thought_hash,
            similarity_to_previous=similarity,
        )

    def audit_code(self, code: str, planned_complexity: float = 1.0) -> AuditResult:
        """
        Audit generated code for complexity and issues.

        Args:
            code: Generated code to audit
            planned_complexity: Expected complexity level

        Returns:
            AuditResult with any issues found
        """
        issues: list[AuditIssue] = []

        actual_complexity = self._estimate_complexity(code)
        if actual_complexity > planned_complexity * self.complexity_threshold:
            issues.append(
                AuditIssue(
                    type=AuditIssueType.COMPLEXITY_SPIKE,
                    severity=AuditSeverity.MEDIUM,
                    message=f"Code complexity ({actual_complexity:.1f}) exceeds plan ({planned_complexity:.1f})",
                    detected_pattern="complexity_analysis",
                    suggestion="Consider breaking into smaller functions",
                    context=code[:500],
                )
            )

        safety_issues = self._check_safety_violations(code)
        issues.extend(safety_issues)

        dangerous_issues = self._check_dangerous_patterns(code)
        issues.extend(dangerous_issues)

        has_issues = len(issues) > 0
        should_interrupt = any(
            i.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL] for i in issues
        )

        return AuditResult(
            has_issues=has_issues,
            issues=issues,
            should_interrupt=should_interrupt,
            interrupt_message=self._generate_interrupt(issues) if should_interrupt else None,
        )

    def _check_circular_reasoning(self, thought: str, similarity: float) -> list[AuditIssue]:
        """Check for circular reasoning patterns."""
        issues = []

        thought_lower = thought.lower()
        for indicator in self.CIRCULAR_INDICATORS:
            if indicator in thought_lower:
                issues.append(
                    AuditIssue(
                        type=AuditIssueType.CIRCULAR_REASONING,
                        severity=AuditSeverity.MEDIUM,
                        message=f"Circular reasoning indicator: '{indicator}'",
                        detected_pattern=indicator,
                        suggestion="Take a different approach or step back to reassess",
                        context=thought[:200],
                    )
                )
                break

        if similarity > 0.8 and len(self._thought_history) > 2:
            issues.append(
                AuditIssue(
                    type=AuditIssueType.CIRCULAR_REASONING,
                    severity=AuditSeverity.HIGH,
                    message=f"High similarity ({similarity:.0%}) to previous thoughts - possible loop",
                    detected_pattern="high_similarity",
                    suggestion="Try a completely different approach",
                    context=thought[:200],
                )
            )

        return issues

    def _check_safety_violations(self, text: str) -> list[AuditIssue]:
        """Check for hardcoded secrets and safety violations."""
        issues = []

        for pattern, violation_type in self.SAFETY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(
                    AuditIssue(
                        type=AuditIssueType.SAFETY_VIOLATION,
                        severity=AuditSeverity.CRITICAL,
                        message=f"Safety violation: {violation_type}",
                        detected_pattern=matches[0] if matches else pattern,
                        suggestion="Use environment variables or secure configuration",
                        context=text[:200],
                    )
                )

        return issues

    def _check_dangerous_patterns(self, text: str) -> list[AuditIssue]:
        """Check for dangerous code patterns."""
        issues = []

        for pattern, danger_type in self.DANGEROUS_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                issues.append(
                    AuditIssue(
                        type=AuditIssueType.SAFETY_VIOLATION,
                        severity=AuditSeverity.HIGH,
                        message=f"Dangerous pattern: {danger_type}",
                        detected_pattern=matches[0] if matches else pattern,
                        suggestion=f"Avoid {danger_type} - use safer alternatives",
                        context=text[:200],
                    )
                )

        return issues

    def _check_repetition(self, thought: str, thought_hash: str) -> list[AuditIssue]:
        """Check for repeated thoughts."""
        issues = []

        hash_count = sum(1 for h in self._thought_hashes if h == thought_hash)
        if hash_count >= self.max_repetition:
            issues.append(
                AuditIssue(
                    type=AuditIssueType.REPETITION,
                    severity=AuditSeverity.MEDIUM,
                    message=f"Thought repeated {hash_count} times",
                    detected_pattern="exact_repetition",
                    suggestion="This has been considered before - move forward",
                    context=thought[:200],
                )
            )

        return issues

    def _check_topic_drift(self, thought: str) -> list[AuditIssue]:
        issues: list[AuditIssue] = []

        if not self._topic_keywords or not self._original_goal:
            return issues

        thought_keywords = set(self._extract_keywords(thought))

        overlap = len(thought_keywords & self._topic_keywords)
        relevance = overlap / len(self._topic_keywords) if self._topic_keywords else 1.0

        if relevance < 0.2 and len(thought_keywords) > 5:
            issues.append(
                AuditIssue(
                    type=AuditIssueType.OFF_TOPIC_DRIFT,
                    severity=AuditSeverity.LOW,
                    message=f"Thought may have drifted from topic (relevance: {relevance:.0%})",
                    detected_pattern="low_keyword_overlap",
                    suggestion=f"Refocus on: {', '.join(list(self._topic_keywords)[:5])}",
                    context=thought[:200],
                )
            )

        return issues

    def _estimate_complexity(self, code: str) -> float:
        """Estimate code complexity."""
        if not code:
            return 0.0

        complexity = 0.0

        complexity += code.count("if ") * 1.0
        complexity += code.count("elif ") * 0.8
        complexity += code.count("else:") * 0.5
        complexity += code.count("for ") * 1.2
        complexity += code.count("while ") * 1.5
        complexity += code.count("try:") * 1.0
        complexity += code.count("except ") * 0.8

        complexity += code.count("def ") * 0.5

        complexity += code.count("class ") * 2.0

        max_nesting = 0
        current_nesting = 0
        for char in code:
            if char == "{":
                current_nesting += 1
                max_nesting = max(max_nesting, current_nesting)
            elif char == "}":
                current_nesting -= 1

        complexity += max_nesting * 0.5

        complexity += len(code) / 500

        return complexity

    def _check_similarity(self, thought: str) -> float:
        """Check similarity to recent thoughts."""
        if not self._thought_history:
            return 0.0

        words = set(self._extract_keywords(thought))
        if not words:
            return 0.0

        max_similarity = 0.0
        for prev_thought in list(self._thought_history)[:-1]:
            prev_words = set(self._extract_keywords(prev_thought))
            if not prev_words:
                continue

            intersection = len(words & prev_words)
            union = len(words | prev_words)
            similarity = intersection / union if union > 0 else 0.0
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def _hash_thought(self, thought: str) -> str:
        """Create a hash of thought for repetition detection."""
        normalized = " ".join(thought.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from text."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "was",
            "were",
            "been",
            "being",
            "have",
            "has",
            "had",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "with",
            "from",
            "when",
            "what",
            "where",
            "which",
            "their",
            "there",
            "about",
            "into",
            "than",
            "then",
            "them",
            "these",
            "those",
            "they",
            "some",
            "such",
        }
        return [w for w in words if w not in stopwords]

    def _generate_interrupt(self, issues: list[AuditIssue]) -> str:
        """Generate a system interrupt message."""
        if not issues:
            return ""

        primary = issues[0]

        interrupt_templates = {
            AuditIssueType.CIRCULAR_REASONING: (
                "System Message: Your reasoning appears to be circling back. Consider: {suggestion}"
            ),
            AuditIssueType.SAFETY_VIOLATION: (
                "System Message: SECURITY ALERT - {message}. Action required: {suggestion}"
            ),
            AuditIssueType.REPETITION: (
                "System Message: This line of thinking has been repeated. "
                "Please proceed with a decision or new approach."
            ),
            AuditIssueType.OFF_TOPIC_DRIFT: (
                "System Message: Attention drift detected. Please refocus on: {suggestion}"
            ),
            AuditIssueType.COMPLEXITY_SPIKE: (
                "System Message: Generated code exceeds planned complexity. {suggestion}"
            ),
        }

        template = interrupt_templates.get(
            primary.type, "System Message: {message}. Suggestion: {suggestion}"
        )

        return template.format(
            message=primary.message,
            suggestion=primary.suggestion,
        )

    def get_stats(self) -> dict[str, Any]:
        """Get auditor statistics."""
        by_severity: dict[str, int] = {}
        by_type: dict[str, int] = {}

        for event in self._audit_log:
            if event.issue_type:
                by_type[event.issue_type] = by_type.get(event.issue_type, 0) + 1
            by_severity[event.severity] = by_severity.get(event.severity, 0) + 1

        return {
            "enabled": self.enabled,
            "thoughts_audited": len(self._thought_history),
            "issues_found": self._issues_found,
            "interrupts_injected": self._interrupts_injected,
            "by_severity": by_severity,
            "by_type": by_type,
            "audit_log_size": len(self._audit_log),
        }

    def get_audit_log(self, limit: int = 20) -> list[AuditEvent]:
        """Get recent audit events."""
        return self._audit_log[-limit:]

    def reset(self) -> None:
        """Reset auditor state for new task."""
        self._thought_history.clear()
        self._thought_hashes.clear()
        self._pattern_counts.clear()
        self._audit_log.clear()
        self._topic_keywords.clear()
        self._original_goal = ""
        self._interrupts_injected = 0
        self._issues_found = 0


def create_streaming_auditor(
    enabled: bool = True,
    max_repetition: int = 3,
) -> StreamingAuditor:
    """Create a StreamingAuditor instance."""
    return StreamingAuditor(
        enabled=enabled,
        max_repetition=max_repetition,
    )
