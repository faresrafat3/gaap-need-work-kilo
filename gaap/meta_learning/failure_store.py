"""
Failure Store - Contrastive Experience Memory
=============================================

Learns from mistakes by storing:
- What the agent thought (hypothesis)
- What went wrong (error)
- What fixed it (corrective action)

This creates a "pitfall database" that can be searched
before starting new tasks to avoid similar mistakes.

Usage:
    store = FailureStore()

    # Record a failure
    trace = FailedTrace(
        task_type="code_generation",
        hypothesis="Using simple regex would work",
        error="Regex failed on nested structures",
        context={"pattern": r"\\{.*\\}"}
    )
    store.record(trace, corrective_action="Use proper parser instead")

    # Search for similar pitfalls
    pitfalls = store.find_similar("parse nested JSON")
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

from gaap.memory.hierarchical import EpisodicMemoryStore
from gaap.storage.atomic import atomic_write

logger = logging.getLogger("gaap.meta_learning.failures")


class FailureType(Enum):
    """تصنيف أنواع الفشل"""

    SYNTAX = auto()
    LOGIC = auto()
    ASSUMPTION = auto()
    RESOURCE = auto()
    TIMEOUT = auto()
    PERMISSION = auto()
    DEPENDENCY = auto()
    INTEGRATION = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    UNKNOWN = auto()


@dataclass
class FailedTrace:
    """تتبع الفشل مع السياق"""

    task_type: str
    hypothesis: str
    error: str
    error_type: FailureType = FailureType.UNKNOWN
    context: dict[str, Any] = field(default_factory=dict)
    agent_thoughts: str | None = None
    task_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "hypothesis": self.hypothesis,
            "error": self.error,
            "error_type": self.error_type.name,
            "context": self.context,
            "agent_thoughts": self.agent_thoughts,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_count": self.resolution_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailedTrace":
        return cls(
            task_type=data.get("task_type", "unknown"),
            hypothesis=data.get("hypothesis", ""),
            error=data.get("error", ""),
            error_type=FailureType[data.get("error_type", "UNKNOWN")],
            context=data.get("context", {}),
            agent_thoughts=data.get("agent_thoughts"),
            task_id=data.get("task_id"),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(),
            resolved=data.get("resolved", False),
            resolution_count=data.get("resolution_count", 0),
        )

    def get_id(self) -> str:
        content = f"{self.task_type}:{self.hypothesis}:{self.error}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class CorrectiveAction:
    """الإجراء التصحيحي للفشل"""

    failure_id: str
    solution: str
    explanation: str
    success_rate: float = 1.0
    applied_count: int = 1
    source: str = "manual"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "solution": self.solution,
            "explanation": self.explanation,
            "success_rate": self.success_rate,
            "applied_count": self.applied_count,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectiveAction":
        return cls(
            failure_id=data.get("failure_id", ""),
            solution=data.get("solution", ""),
            explanation=data.get("explanation", ""),
            success_rate=data.get("success_rate", 1.0),
            applied_count=data.get("applied_count", 1),
            source=data.get("source", "manual"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            metadata=data.get("metadata", {}),
        )


class FailureStore:
    """
    Contrastive Experience Store for learning from failures.

    Features:
    - Record failures with full context
    - Store corrective actions that worked
    - Semantic search for similar pitfalls
    - Track resolution success rates
    - Auto-classify failure types
    """

    DEFAULT_STORAGE_PATH = ".gaap/memory/failures"

    def __init__(
        self,
        storage_path: str | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
    ) -> None:
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._episodic = episodic_store

        self._failures: dict[str, FailedTrace] = {}
        self._corrections: dict[str, list[CorrectiveAction]] = {}
        self._keyword_index: dict[str, set[str]] = {}

        self._logger = logger

        self._load()

    def record(
        self,
        trace: FailedTrace,
        corrective_action: str | CorrectiveAction | None = None,
    ) -> str:
        """
        Record a failure trace with optional corrective action.

        Args:
            trace: The failure trace to record
            corrective_action: Optional solution that fixed the issue

        Returns:
            Failure ID for reference
        """
        failure_id = trace.get_id()

        if failure_id in self._failures:
            existing = self._failures[failure_id]
            existing.resolution_count += 1
            self._logger.info(
                f"Recurring failure: {failure_id} (count: {existing.resolution_count})"
            )
        else:
            self._failures[failure_id] = trace
            self._index_keywords(failure_id, trace)

        if corrective_action:
            if isinstance(corrective_action, str):
                action = CorrectiveAction(
                    failure_id=failure_id,
                    solution=corrective_action,
                    explanation="Manual correction",
                )
            else:
                action = corrective_action

            if failure_id not in self._corrections:
                self._corrections[failure_id] = []
            self._corrections[failure_id].append(action)

            trace.resolved = True

        self._save()
        return failure_id

    def find_similar(
        self,
        context: str,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list[tuple[FailedTrace, list[CorrectiveAction]]]:
        """
        Find failures similar to the given context.

        Args:
            context: Description of current task/problem
            task_type: Optional filter by task type
            limit: Maximum results to return

        Returns:
            List of (failure, corrections) tuples sorted by relevance
        """
        keywords = self._extract_keywords(context)

        failure_scores: dict[str, float] = {}

        for kw in keywords:
            for fid in self._keyword_index.get(kw, set()):
                failure_scores[fid] = failure_scores.get(fid, 0) + 1

        if task_type:
            failure_scores = {
                fid: score
                for fid, score in failure_scores.items()
                if fid in self._failures and self._failures[fid].task_type == task_type
            }

        sorted_failures = sorted(
            failure_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:limit]

        results = []
        for fid, _ in sorted_failures:
            if fid in self._failures:
                trace = self._failures[fid]
                corrections = self._corrections.get(fid, [])
                results.append((trace, corrections))

        return results

    def get_pitfall_warnings(self, task_description: str) -> list[str]:
        """
        Get warnings about potential pitfalls for a task.

        Args:
            task_description: Description of the task

        Returns:
            List of warning messages
        """
        similar = self.find_similar(task_description, limit=3)

        warnings = []
        for trace, corrections in similar:
            if not trace.resolved:
                warnings.append(
                    f"PITFALL: In {trace.task_type}, similar approach failed: {trace.error}"
                )
            elif corrections:
                best = max(corrections, key=lambda c: c.success_rate)
                warnings.append(
                    f"WARNING: {trace.task_type} - {trace.error}. Solution: {best.solution}"
                )

        return warnings

    def update_correction_success(
        self,
        failure_id: str,
        solution: str,
        success: bool,
    ) -> None:
        """
        Update success rate of a corrective action.

        Args:
            failure_id: ID of the failure
            solution: The solution that was applied
            success: Whether it worked
        """
        if failure_id not in self._corrections:
            return

        for action in self._corrections[failure_id]:
            if action.solution == solution:
                total = action.applied_count + 1
                successes = action.success_rate * action.applied_count
                if success:
                    successes += 1
                action.success_rate = successes / total
                action.applied_count = total
                self._logger.debug(f"Updated correction success rate: {action.success_rate:.2f}")
                break

        self._save()

    def get_unresolved_failures(self, limit: int = 20) -> list[FailedTrace]:
        """Get failures without known corrections."""
        unresolved = [f for f in self._failures.values() if not f.resolved]
        return sorted(
            unresolved,
            key=lambda x: x.timestamp,
            reverse=True,
        )[:limit]

    def get_recurring_failures(self, min_count: int = 2) -> list[FailedTrace]:
        """Get failures that happened multiple times."""
        return [f for f in self._failures.values() if f.resolution_count >= min_count]

    def get_stats(self) -> dict[str, Any]:
        """Get failure store statistics."""
        total = len(self._failures)
        resolved = sum(1 for f in self._failures.values() if f.resolved)
        corrections = sum(len(c) for c in self._corrections.values())

        by_type: dict[str, int] = {}
        for f in self._failures.values():
            by_type[f.error_type.name] = by_type.get(f.error_type.name, 0) + 1

        return {
            "total_failures": total,
            "resolved": resolved,
            "resolution_rate": resolved / max(total, 1),
            "total_corrections": corrections,
            "by_type": by_type,
            "recurring": len(self.get_recurring_failures()),
        }

    def classify_error(self, error: str) -> FailureType:
        """
        Auto-classify error type from error message.

        Args:
            error: Error message string

        Returns:
            Classified FailureType
        """
        error_lower = error.lower()

        patterns = {
            FailureType.SYNTAX: [
                r"syntax\s*error",
                r"unexpected\s+token",
                r"invalid\s+syntax",
                r"parse\s*error",
            ],
            FailureType.LOGIC: [
                r"assertion\s+failed",
                r"logic\s+error",
                r"incorrect\s+result",
                r"wrong\s+output",
            ],
            FailureType.ASSUMPTION: [
                r"assumption\s+failed",
                r"expected\s+but\s+got",
                r"not\s+found",
                r"unexpected\s+type",
            ],
            FailureType.RESOURCE: [
                r"out\s+of\s+memory",
                r"disk\s+full",
                r"resource\s+exhausted",
                r"quota\s+exceeded",
            ],
            FailureType.TIMEOUT: [
                r"timeout",
                r"timed\s+out",
                r"deadline\s+exceeded",
            ],
            FailureType.PERMISSION: [
                r"permission\s+denied",
                r"access\s+denied",
                r"unauthorized",
                r"forbidden",
            ],
            FailureType.DEPENDENCY: [
                r"module\s+not\s+found",
                r"import\s+error",
                r"dependency\s+error",
                r"package\s+not\s+found",
            ],
            FailureType.INTEGRATION: [
                r"connection\s+refused",
                r"api\s+error",
                r"service\s+unavailable",
                r"integration\s+failed",
            ],
            FailureType.SECURITY: [
                r"security\s+violation",
                r"sandbox\s+violation",
                r"unsafe\s+operation",
                r"blocked\s+by\s+firewall",
            ],
            FailureType.PERFORMANCE: [
                r"too\s+slow",
                r"performance\s+degraded",
                r"memory\s+leak",
            ],
        }

        for ftype, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, error_lower):
                    return ftype

        return FailureType.UNKNOWN

    def _index_keywords(self, failure_id: str, trace: FailedTrace) -> None:
        """Index failure by keywords for fast search."""
        text = f"{trace.task_type} {trace.hypothesis} {trace.error}"
        keywords = self._extract_keywords(text)

        for kw in keywords:
            if kw not in self._keyword_index:
                self._keyword_index[kw] = set()
            self._keyword_index[kw].add(failure_id)

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from text."""
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
            "error",
            "failed",
        }

        return [w for w in words if w not in stopwords]

    def _save(self) -> bool:
        """Save failure store to disk."""
        try:
            failures_file = self.storage_path / "failures.json"
            corrections_file = self.storage_path / "corrections.json"
            index_file = self.storage_path / "index.json"

            atomic_write(
                failures_file,
                json.dumps(
                    {fid: t.to_dict() for fid, t in self._failures.items()},
                    indent=2,
                ),
            )

            atomic_write(
                corrections_file,
                json.dumps(
                    {
                        fid: [c.to_dict() for c in actions]
                        for fid, actions in self._corrections.items()
                    },
                    indent=2,
                ),
            )

            atomic_write(
                index_file,
                json.dumps(
                    {kw: list(ids) for kw, ids in self._keyword_index.items()},
                    indent=2,
                ),
            )

            self._logger.debug(f"Saved {len(self._failures)} failures to {self.storage_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save failure store: {e}")
            return False

    def _load(self) -> bool:
        """Load failure store from disk."""
        try:
            failures_file = self.storage_path / "failures.json"
            corrections_file = self.storage_path / "corrections.json"
            index_file = self.storage_path / "index.json"

            if failures_file.exists():
                with open(failures_file) as f:
                    data = json.load(f)
                self._failures = {fid: FailedTrace.from_dict(t) for fid, t in data.items()}

            if corrections_file.exists():
                with open(corrections_file) as f:
                    data = json.load(f)
                self._corrections = {
                    fid: [CorrectiveAction.from_dict(c) for c in actions]
                    for fid, actions in data.items()
                }

            if index_file.exists():
                with open(index_file) as f:
                    data = json.load(f)
                self._keyword_index = {kw: set(ids) for kw, ids in data.items()}

            self._logger.info(f"Loaded {len(self._failures)} failures from {self.storage_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load failure store: {e}")
            return False

    def clear_old_failures(self, days: int = 90) -> int:
        """
        Remove failures older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of failures removed
        """
        cutoff = datetime.now() - timedelta(days=days)
        to_remove = [
            fid
            for fid, trace in self._failures.items()
            if trace.timestamp < cutoff and trace.resolved
        ]

        for fid in to_remove:
            del self._failures[fid]
            if fid in self._corrections:
                del self._corrections[fid]

        self._rebuild_index()
        self._save()

        return len(to_remove)

    def _rebuild_index(self) -> None:
        """Rebuild keyword index from current failures."""
        self._keyword_index = {}
        for fid, trace in self._failures.items():
            self._index_keywords(fid, trace)


def create_failure_store(
    storage_path: str | None = None,
) -> FailureStore:
    """Create a FailureStore instance."""
    return FailureStore(storage_path=storage_path)
