"""
Hierarchical Memory System

GAAP's 4-tier hierarchical memory system inspired by human memory:

Memory Tiers:
    - L1 Working Memory: Fast, limited (100 items) - Current context
    - L2 Episodic Memory: Event history - Learning from experience
    - L3 Semantic Memory: Patterns & rules - Extracted knowledge
    - L4 Procedural Memory: Acquired skills - Templates & procedures

Features:
    - Memory decay over time
    - Access-based strengthening
    - Priority-based retention
    - Automatic consolidation

Usage:
    from gaap.memory import HierarchicalMemory, EpisodicMemory

    memory = HierarchicalMemory()

    # Record episode
    episode = EpisodicMemory(
        task_id="task-123",
        action="code_generation",
        result="Success",
        success=True,
        duration_ms=1500,
        tokens_used=2500,
        cost_usd=0.05,
        model="llama-3.3-70b",
        provider="groq",
        lessons=["Use iterative approach"]
    )
    memory.record_episode(episode)

    # Search for lessons
    lessons = memory.search_lessons("code generation best practices")
"""

# Hierarchical Memory
import hashlib
import logging
import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================


class MemoryTier(Enum):
    """
    Memory tier enumeration.

    Represents the 4-tier hierarchical memory system:
    - WORKING: Fast, limited capacity (L1)
    - EPISODIC: Event history (L2)
    - SEMANTIC: Patterns and rules (L3)
    - PROCEDURAL: Skills and procedures (L4)

    Usage:
        >>> tier = MemoryTier.WORKING
        >>> print(tier.value)
        1
    """

    WORKING = 1  # L1: ذاكرة العمل (سريعة، محدودة)
    EPISODIC = 2  # L2: ذاكرة الأحداث (متوسطة)
    SEMANTIC = 3  # L3: ذاكرة دلالية (أنماط)
    PROCEDURAL = 4  # L4: ذاكرة إجرائية (مهارات)


class MemoryPriority(Enum):
    """
    Memory priority levels.

    Determines retention and retrieval priority:
    - CRITICAL: Highest priority, never decays
    - HIGH: High priority, slow decay
    - NORMAL: Standard priority
    - LOW: Low priority, faster decay
    - BACKGROUND: Lowest priority, quick decay

    Usage:
        >>> priority = MemoryPriority.HIGH
        >>> print(priority.value)
        2
    """

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class MemoryEntry(Generic[T]):
    """
    Memory entry for storing information.

    Generic container for memory entries with decay and access tracking.

    Attributes:
        id: Unique identifier
        tier: Memory tier (WORKING, EPISODIC, SEMANTIC, PROCEDURAL)
        content: Stored content (generic type)
        priority: Retention priority
        created_at: Creation timestamp
        last_accessed: Last access timestamp
        access_count: Number of accesses
        importance: Base importance (0.0-1.0)
        decay_rate: Decay rate per hour
        tags: Associated tags
        metadata: Additional metadata

    Usage:
        >>> entry = MemoryEntry(
        ...     id="mem-123",
        ...     tier=MemoryTier.WORKING,
        ...     content={"key": "value"},
        ...     priority=MemoryPriority.NORMAL
        ... )
        >>> entry.access()
        >>> strength = entry.get_strength()
    """

    id: str
    tier: MemoryTier
    content: T
    priority: MemoryPriority = MemoryPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 1.0
    decay_rate: float = 0.1
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def access(self) -> None:
        """
        Record an access to this memory entry.

        Updates last_accessed timestamp and increments access_count.
        Access strengthens memory retention.

        Example:
            >>> entry.access()
            >>> print(entry.access_count)
            1
        """
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_strength(self) -> float:
        """
        Calculate current memory strength.

        Memory strength decays over time but is boosted by access.

        Returns:
            Memory strength (0.0 to importance + 0.5)

        Formula:
            strength = max(0, importance + access_boost - decay)
            where:
                access_boost = min(access_count * 0.1, 0.5)
                decay = decay_rate * age_in_hours

        Example:
            >>> entry = MemoryEntry(id="1", tier=MemoryTier.WORKING, content="test")
            >>> entry.access()
            >>> strength = entry.get_strength()
            >>> print(f"Strength: {strength:.2f}")
        """
        age = (datetime.now() - self.created_at).total_seconds() / 3600  # hours
        # Scale decay by priority: CRITICAL=1 (slowest), BACKGROUND=5 (fastest)
        # priority.value: CRITICAL=1, HIGH=2, NORMAL=3, LOW=4, BACKGROUND=5
        priority_multiplier = max(
            1, 6 - self.priority.value
        )  # CRITICAL=5x, HIGH=4x, NORMAL=3x, LOW=2x, BACKGROUND=1x
        adjusted_decay_rate = self.decay_rate / priority_multiplier
        decay = adjusted_decay_rate * age
        access_boost = min(self.access_count * 0.1, 0.5)
        return max(0, self.importance + access_boost - decay)


@dataclass
class EpisodicMemory:
    """
    الذاكرة العرضية لتخزين الأحداث مع السياق الكامل.
    """

    task_id: str
    action: str
    result: str
    success: bool
    category: str = "general"  # التصنيف (مثلاً: research, code, diagnostic)
    duration_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0
    model: str = "unknown"
    provider: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """تحويل لقاموس للتخزين"""
        return {
            "task_id": self.task_id,
            "action": self.action[:200],
            "result": self.result[:500] if hasattr(self, "result") else "",
            "success": self.success,
            "category": self.category,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens_used,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
            "lessons": self.lessons,
        }


@dataclass
class SemanticRule:
    """
    Semantic rule extracted from episodes.

    Generalized knowledge extracted from specific events.

    Attributes:
        id: Unique identifier
        condition: Rule condition/pattern
        action: Rule action/recommendation
        confidence: Confidence level (0.0-1.0)
        support_count: Number of supporting episodes
        source_episodes: Source episode IDs
        created_at: Creation timestamp

    Usage:
        >>> rule = SemanticRule(
        ...     id="rule-1",
        ...     condition="code_generation with security requirements",
        ...     action="add security critic to MAD panel",
        ...     confidence=0.85,
        ...     support_count=5
        ... )
    """

    id: str
    condition: str  # الشرط
    action: str  # الإجراء
    confidence: float  # الثقة
    support_count: int  # عدد الدعم
    source_episodes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with rule data

        Example:
            >>> rule_dict = rule.to_dict()
            >>> print(rule_dict["confidence"])
            0.85
        """
        return {
            "id": self.id,
            "condition": self.condition,
            "action": self.action,
            "confidence": self.confidence,
            "support": self.support_count,
        }


# =============================================================================
# Working Memory (L1)
# =============================================================================


class WorkingMemory:
    """
    Working Memory (L1) - Fast, limited capacity memory.

    Used for current task context with automatic eviction of old entries.

    Attributes:
        max_size: Maximum number of items (default: 100)
        _items: Deque of memory entries
        _index: Dictionary for O(1) key lookup
        _logger: Logger instance

    Features:
        - O(1) store and retrieve operations
        - Automatic eviction of oldest items
        - Access tracking for each entry
        - Fixed maximum size (deque with maxlen)

    Usage:
        >>> memory = WorkingMemory(max_size=100)
        >>> memory.store("key1", {"data": "value"})
        >>> value = memory.retrieve("key1")
        >>> print(memory.get_size())
        1
    """

    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize working memory.

        Args:
            max_size: Maximum number of items to retain (default: 100)
        """
        self.max_size = max_size
        self._items: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._logger = logging.getLogger("gaap.memory.working")

    def store(
        self,
        key: str,
        content: Any,
        priority: MemoryPriority = MemoryPriority.NORMAL,
    ) -> None:
        """
        Store content in working memory.

        Args:
            key: Unique key for retrieval
            content: Content to store (any type)
            priority: Retention priority (default: NORMAL)

        Note:
            - Automatically evicts oldest items if at max_size
            - Replaces existing entry if key already exists

        Example:
            >>> memory = WorkingMemory()
            >>> memory.store("config", {"timeout": 30})
        """
        entry = MemoryEntry(
            id=self._generate_id(key),
            tier=MemoryTier.WORKING,
            content=content,
            priority=priority,
        )

        # Remove old entry if exists (move_to_end is O(1))
        if key in self._items:
            del self._items[key]

        # Evict oldest if at capacity
        while len(self._items) >= self.max_size:
            self._items.popitem(last=False)  # FIFO eviction

        self._items[key] = entry

    def retrieve(self, key: str) -> Any | None:
        """
        Retrieve content from working memory.

        Args:
            key: Key to retrieve

        Returns:
            Stored content or None if not found

        Note:
            Updates last_accessed timestamp and access_count

        Example:
            >>> memory.store("key", "value")
            >>> value = memory.retrieve("key")
            >>> print(value)
            'value'
        """
        entry = self._items.get(key)
        if entry:
            entry.access()
            # Move to end (most recently used)
            self._items.move_to_end(key)
            return entry.content
        return None

    def clear(self) -> None:
        """
        Clear all items from working memory.

        Resets the ordered dict.

        Example:
            >>> memory.clear()
            >>> print(memory.get_size())
            0
        """
        self._items.clear()

    def get_size(self) -> int:
        """
        Get current number of items in memory.

        Returns:
            Number of stored items

        Example:
            >>> memory.store("key", "value")
            >>> print(memory.get_size())
            1
        """
        return len(self._items)

    def _generate_id(self, key: str) -> str:
        """
        Generate unique ID for a memory entry.

        Args:
            key: Base key string

        Returns:
            8-character MD5 hash with timestamp

        Example:
            >>> memory._generate_id("test")
            'a1b2c3d4'
        """
        return hashlib.md5(f"{key}:{time.time()}".encode()).hexdigest()[:8]


# =============================================================================
# Episodic Memory (L2)
# =============================================================================


class EpisodicMemoryStore:
    """
    Episodic Memory Store (L2) - Event history memory.

    Records specific events with full context for learning from experience.

    Attributes:
        storage_path: Optional path for disk persistence
        _episodes: List of recorded episodes
        _task_index: Index for fast task-based lookup
        _logger: Logger instance

    Features:
        - Chronological episode storage
        - Task-based indexing
        - Success rate calculation
        - Lesson extraction
        - Optional disk persistence (JSON)

    Usage:
        >>> store = EpisodicMemoryStore(storage_path=".gaap/memory")
        >>> episode = EpisodicMemory(
        ...     task_id="task-123",
        ...     action="code_generation",
        ...     result="Success",
        ...     success=True,
        ...     duration_ms=1500,
        ...     tokens_used=2500,
        ...     cost_usd=0.05,
        ...     model="llama-3.3-70b",
        ...     provider="groq",
        ...     lessons=["Use iterative approach"]
        ... )
        >>> store.record(episode)
        >>> rate = store.get_success_rate()
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """
        Initialize episodic memory store.

        Args:
            storage_path: Optional path for disk persistence
        """
        self.storage_path = storage_path
        self._episodes: list[EpisodicMemory] = []
        self._task_index: dict[str, list[int]] = {}
        self._logger = logging.getLogger("gaap.memory.episodic")

    def record(self, episode: EpisodicMemory) -> None:
        """
        Record an episode to memory.

        Args:
            episode: Episode to record

        Note:
            Automatically indexes by task_id for fast retrieval

        Example:
            >>> episode = EpisodicMemory(...)
            >>> store.record(episode)
        """
        idx = len(self._episodes)
        self._episodes.append(episode)

        # Index by task_id
        if episode.task_id not in self._task_index:
            self._task_index[episode.task_id] = []
        self._task_index[episode.task_id].append(idx)

    def get_episodes(
        self,
        task_id: str | None = None,
        success_only: bool = False,
        limit: int = 100,
    ) -> list[EpisodicMemory]:
        """
        Retrieve episodes with optional filtering.

        Args:
            task_id: Optional task ID to filter by
            success_only: Filter to successful episodes only
            limit: Maximum number of episodes to return

        Returns:
            List of matching episodes (most recent first)

        Example:
            >>> episodes = store.get_episodes(task_id="task-123", success_only=True)
            >>> print(len(episodes))
            5
        """
        if task_id:
            indices = self._task_index.get(task_id, [])
            episodes = [self._episodes[i] for i in indices if i < len(self._episodes)]
        else:
            episodes = self._episodes

        if success_only:
            episodes = [e for e in episodes if e.success]

        return episodes[-limit:]

    def get_success_rate(self, task_type: str | None = None) -> float:
        """
        Calculate overall or per-task-type success rate.

        Args:
            task_type: Optional task type to filter by

        Returns:
            Success rate (0.0 to 1.0)

        Example:
            >>> rate = store.get_success_rate()
            >>> print(f"Success rate: {rate:.2%}")
            Success rate: 85%
        """
        episodes = self._episodes

        if not episodes:
            return 0.0

        successes = sum(1 for e in episodes if e.success)
        return successes / len(episodes)

    def get_recent_lessons(self, limit: int = 20) -> list[str]:
        """
        Extract recent lessons from all episodes.

        Args:
            limit: Maximum number of lessons to return

        Returns:
            List of recent lessons (most recent first)

        Example:
            >>> lessons = store.get_recent_lessons(limit=10)
            >>> for lesson in lessons:
            ...     print(lesson)
        """
        lessons = []
        for episode in reversed(self._episodes):
            lessons.extend(episode.lessons)
            if len(lessons) >= limit:
                break
        return lessons[:limit]

    def save(self) -> bool:
        """
        Save episodes to disk as JSON.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.save()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping save")
            return False

        import json
        from pathlib import Path

        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            filepath = Path(self.storage_path) / "episodic_memory.json"

            data = {
                "episodes": [
                    {
                        "task_id": e.task_id,
                        "action": e.action,
                        "result": e.result,
                        "success": e.success,
                        "duration_ms": e.duration_ms,
                        "tokens_used": e.tokens_used,
                        "cost_usd": e.cost_usd,
                        "model": e.model,
                        "provider": e.provider,
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        "lessons": e.lessons,
                    }
                    for e in self._episodes
                ],
                "task_index": self._task_index,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._logger.info(f"Saved {len(self._episodes)} episodes to {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save episodic memory: {e}")
            return False

    def load(self) -> bool:
        """
        Load episodes from disk.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.load()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping load")
            return False

        import json
        from pathlib import Path

        try:
            filepath = Path(self.storage_path) / "episodic_memory.json"
            if not filepath.exists():
                self._logger.info(f"No existing memory file at {filepath}")
                return False

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            self._episodes = []
            self._task_index = data.get("task_index", {})

            for ep_data in data.get("episodes", []):
                from datetime import datetime

                timestamp = ep_data.get("timestamp")
                if timestamp:
                    try:
                        timestamp = datetime.fromisoformat(timestamp)
                    except Exception:
                        self._logger.debug(f"Invalid timestamp format, using now")
                        timestamp = datetime.now()

                episode = EpisodicMemory(
                    task_id=ep_data.get("task_id", ""),
                    action=ep_data.get("action", ""),
                    result=ep_data.get("result", ""),
                    success=ep_data.get("success", False),
                    duration_ms=ep_data.get("duration_ms", 0.0),
                    tokens_used=ep_data.get("tokens_used", 0),
                    cost_usd=ep_data.get("cost_usd", 0.0),
                    model=ep_data.get("model", ""),
                    provider=ep_data.get("provider", ""),
                    timestamp=timestamp,
                    lessons=ep_data.get("lessons", []),
                )
                self._episodes.append(episode)

            self._logger.info(f"Loaded {len(self._episodes)} episodes from {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load episodic memory: {e}")
            return False


# =============================================================================
# Semantic Memory (L3)
# =============================================================================


class SemanticMemoryStore:
    """
    Semantic Memory Store (L3) - Patterns and rules.

    Extracts generalized knowledge from accumulated experience.

    Attributes:
        storage_path: Optional path for disk persistence
        _rules: Dictionary of semantic rules
        _pattern_index: Keyword-based rule indexing
        _logger: Logger instance

    Features:
        - Rule extraction from episodes
        - Confidence-based ranking
        - Keyword indexing
        - Pattern matching
        - Optional disk persistence (JSON)

    Usage:
        >>> store = SemanticMemoryStore(storage_path=".gaap/memory")
        >>> rule = store.add_rule(
        ...     condition="code generation with security requirements",
        ...     action="add security critic to MAD panel",
        ...     confidence=0.85,
        ...     source_episode="task-123"
        ... )
        >>> rules = store.find_rules("secure API development")
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """
        Initialize semantic memory store.

        Args:
            storage_path: Optional path for disk persistence
        """
        self.storage_path = storage_path
        self._rules: dict[str, SemanticRule] = {}
        self._pattern_index: dict[str, list[str]] = {}
        self._logger = logging.getLogger("gaap.memory.semantic")

    def add_rule(
        self,
        condition: str,
        action: str,
        confidence: float = 0.5,
        source_episode: str | None = None,
    ) -> SemanticRule:
        """
        Add or strengthen a semantic rule.

        Args:
            condition: Rule condition/pattern
            action: Rule action/recommendation
            confidence: Initial confidence (default: 0.5)
            source_episode: Source episode ID

        Returns:
            Created or strengthened SemanticRule

        Note:
            - Existing rules are strengthened (confidence +0.1)
            - Support count is incremented
            - Keywords are indexed for fast retrieval

        Example:
            >>> rule = store.add_rule(
            ...     condition="security requirements",
            ...     action="add security critic",
            ...     confidence=0.8
            ... )
        """
        rule_id = self._generate_id(condition, action)

        if rule_id in self._rules:
            # Strengthen existing rule
            rule = self._rules[rule_id]
            rule.support_count += 1
            rule.confidence = min(rule.confidence + 0.1, 1.0)
            if source_episode:
                rule.source_episodes.append(source_episode)
        else:
            # New rule
            rule = SemanticRule(
                id=rule_id,
                condition=condition,
                action=action,
                confidence=confidence,
                support_count=1,
                source_episodes=[source_episode] if source_episode else [],
            )
            self._rules[rule_id] = rule

        # Index by keywords
        keywords = self._extract_keywords(condition)
        for kw in keywords:
            if kw not in self._pattern_index:
                self._pattern_index[kw] = []
            if rule_id not in self._pattern_index[kw]:
                self._pattern_index[kw].append(rule_id)

        return rule

    def find_rules(self, context: str, min_confidence: float = 0.3) -> list[SemanticRule]:
        """
        Find rules matching a context.

        Args:
            context: Context string to match
            min_confidence: Minimum confidence threshold (default: 0.3)

        Returns:
            List of matching rules (sorted by relevance, max 10)

        Example:
            >>> rules = store.find_rules("secure API development")
            >>> for rule in rules:
            ...     print(f"{rule.condition} → {rule.action}")
        """
        keywords = self._extract_keywords(context)

        rule_scores: dict[str, float] = {}
        for kw in keywords:
            for rule_id in self._pattern_index.get(kw, []):
                rule_scores[rule_id] = rule_scores.get(rule_id, 0) + 1

        # Sort by score (number of keyword matches)
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)

        rules = []
        for rule_id, _ in sorted_rules:
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                if rule.confidence >= min_confidence:
                    rules.append(rule)

        return rules[:10]

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: Input text

        Returns:
            List of keywords (stopwords removed, min length 3)

        Example:
            >>> keywords = store._extract_keywords("Secure API Development")
            >>> print(keywords)
            ['secure', 'api', 'development']
        """
        words = text.lower().split()
        # Remove stopwords
        stopwords = {"the", "a", "an", "is", "are", "to", "for", "and", "or", "in", "on"}
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _generate_id(self, condition: str, action: str) -> str:
        """
        Generate unique rule ID.

        Args:
            condition: Rule condition
            action: Rule action

        Returns:
            12-character MD5 hash
        """
        return hashlib.md5(f"{condition}:{action}".encode()).hexdigest()[:12]

    def save(self) -> bool:
        """
        Save semantic rules to disk as JSON.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.save()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping save")
            return False

        import json
        from pathlib import Path

        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            filepath = Path(self.storage_path) / "semantic_memory.json"

            data = {
                "rules": {
                    rid: {
                        "condition": r.condition,
                        "action": r.action,
                        "confidence": r.confidence,
                        "support_count": r.support_count,
                        "source_episodes": r.source_episodes,
                        "created_at": r.created_at.isoformat() if r.created_at else None,
                    }
                    for rid, r in self._rules.items()
                },
                "pattern_index": self._pattern_index,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._logger.info(f"Saved {len(self._rules)} rules to {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save semantic memory: {e}")
            return False

    def load(self) -> bool:
        """
        Load semantic rules from disk.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.load()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping load")
            return False

        import json
        from pathlib import Path

        try:
            filepath = Path(self.storage_path) / "semantic_memory.json"
            if not filepath.exists():
                self._logger.info(f"No existing semantic memory file at {filepath}")
                return False

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            from datetime import datetime

            self._rules = {}
            for rid, rdata in data.get("rules", {}).items():
                created_at = rdata.get("created_at")
                if created_at:
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except Exception:
                        self._logger.debug(f"Invalid created_at format, using now")
                        created_at = datetime.now()

                rule = SemanticRule(
                    id=rid,
                    condition=rdata.get("condition", ""),
                    action=rdata.get("action", ""),
                    confidence=rdata.get("confidence", 0.5),
                    support_count=rdata.get("support_count", 1),
                    source_episodes=rdata.get("source_episodes", []),
                    created_at=created_at,
                )
                self._rules[rid] = rule

            self._pattern_index = data.get("pattern_index", {})

            self._logger.info(f"Loaded {len(self._rules)} rules from {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load semantic memory: {e}")
            return False


# =============================================================================
# Procedural Memory (L4)
# =============================================================================


class ProceduralMemoryStore:
    """
    Procedural Memory Store (L4) - Acquired skills.

    Represents fine-tuned prompts, templates, and procedures.

    Attributes:
        storage_path: Optional path for disk persistence
        _procedures: Dictionary of stored procedures
        _logger: Logger instance

    Features:
        - Procedure storage with success rate tracking
        - Template-based procedures
        - Example storage
        - Best procedure selection by task type
        - Optional disk persistence (JSON)

    Usage:
        >>> store = ProceduralMemoryStore(storage_path=".gaap/memory")
        >>> store.store_procedure(
        ...     name="code_review",
        ...     prompt_template="Review this code for {criteria}...",
        ...     success_rate=0.85,
        ...     examples=[{"input": "...", "output": "..."}]
        ... )
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """
        Initialize procedural memory store.

        Args:
            storage_path: Optional path for disk persistence
        """
        self.storage_path = storage_path
        self._procedures: dict[str, dict[str, Any]] = {}
        self._logger = logging.getLogger("gaap.memory.procedural")

    def store_procedure(
        self,
        name: str,
        prompt_template: str,
        success_rate: float,
        examples: list[dict[str, Any]],
    ) -> None:
        """
        Store a procedure with template and examples.

        Args:
            name: Procedure name
            prompt_template: Template string for the procedure
            success_rate: Success rate (0.0-1.0)
            examples: List of example input/output pairs

        Example:
            >>> store.store_procedure(
            ...     name="security_review",
            ...     prompt_template="Check for vulnerabilities...",
            ...     success_rate=0.9,
            ...     examples=[{"input": "code", "output": "review"}]
            ... )
        """
        self._procedures[name] = {
            "template": prompt_template,
            "success_rate": success_rate,
            "examples": examples,
            "created_at": datetime.now().isoformat(),
        }

    def get_procedure(self, name: str) -> dict[str, Any] | None:
        """
        Retrieve a procedure by name.

        Args:
            name: Procedure name

        Returns:
            Procedure dictionary or None if not found

        Example:
            >>> proc = store.get_procedure("code_review")
            >>> if proc:
            ...     print(proc['template'])
        """
        return self._procedures.get(name)

    def find_best_procedure(self, task_type: str) -> dict[str, Any] | None:
        """
        Find best procedure for a task type.

        Args:
            task_type: Type of task to match

        Returns:
            Best matching procedure or None

        Note:
            Selects procedure with highest success rate that matches task type

        Example:
            >>> best = store.find_best_procedure("code_review")
            >>> if best:
            ...     print(f"Success rate: {best['success_rate']:.2%}")
        """
        best = None
        best_rate = 0

        for name, proc in self._procedures.items():
            if task_type.lower() in name.lower() and proc["success_rate"] > best_rate:
                best = proc
                best_rate = proc["success_rate"]

        return best

    def save(self) -> bool:
        """
        Save procedures to disk as JSON.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.save()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping save")
            return False

        import json
        from pathlib import Path

        try:
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            filepath = Path(self.storage_path) / "procedural_memory.json"

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self._procedures, f, ensure_ascii=False, indent=2, default=str)

            self._logger.info(f"Saved {len(self._procedures)} procedures to {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save procedural memory: {e}")
            return False

    def load(self) -> bool:
        """
        Load procedures from disk.

        Returns:
            True if successful, False otherwise

        Note:
            Requires storage_path to be set

        Example:
            >>> store.storage_path = ".gaap/memory"
            >>> success = store.load()
        """
        if not self.storage_path:
            self._logger.warning("No storage_path configured, skipping load")
            return False

        import json
        from pathlib import Path

        try:
            filepath = Path(self.storage_path) / "procedural_memory.json"
            if not filepath.exists():
                self._logger.info(f"No existing procedural memory file at {filepath}")
                return False

            with open(filepath, encoding="utf-8") as f:
                self._procedures = json.load(f)

            self._logger.info(f"Loaded {len(self._procedures)} procedures from {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load procedural memory: {e}")
            return False


# =============================================================================
# Hierarchical Memory Manager
# =============================================================================


class HierarchicalMemory:
    """
    Hierarchical Memory Manager - 4-tier memory system.

    Integrates all memory tiers:
    - L1: Working Memory (fast, limited)
    - L2: Episodic Memory (event history)
    - L3: Semantic Memory (patterns & rules)
    - L4: Procedural Memory (skills & templates)

    Attributes:
        working: Working memory instance
        episodic: Episodic memory store
        semantic: Semantic memory store
        procedural: Procedural memory store
        _logger: Logger instance
        _promotion_threshold: Access count for memory promotion

    Features:
        - Multi-tier storage and retrieval
        - Automatic lesson extraction from episodes
        - Rule generation from lessons
        - Memory promotion (L1 → L2)
        - Relevant context retrieval
        - Comprehensive statistics

    Usage:
        >>> memory = HierarchicalMemory(working_size=100, storage_path=".gaap/memory")
        >>> memory.store("config", {"timeout": 30}, MemoryTier.WORKING)
        >>> episode = EpisodicMemory(...)
        >>> memory.record_episode(episode)
        >>> context = memory.get_relevant_context("code security")
    """

    def __init__(
        self,
        working_size: int = 100,
        storage_path: str | None = None,
    ) -> None:
        """
        Initialize hierarchical memory manager.

        Args:
            working_size: Working memory max size (default: 100)
            storage_path: Optional path for disk persistence
        """
        self.working = WorkingMemory(max_size=working_size)
        self.episodic = EpisodicMemoryStore(storage_path)
        self.semantic = SemanticMemoryStore(storage_path)
        self.procedural = ProceduralMemoryStore(storage_path)

        self._logger = logging.getLogger("gaap.memory")

        # Memory promotion threshold
        self._promotion_threshold = 3  # accesses required for promotion

    def __repr__(self) -> str:
        return f"HierarchicalMemory(promotion_threshold={self._promotion_threshold})"

    def store(self, key: str, content: Any, tier: MemoryTier = MemoryTier.WORKING) -> None:
        """
        Store content in specified memory tier.

        Args:
            key: Unique key for retrieval
            content: Content to store
            tier: Memory tier (default: WORKING)

        Example:
            >>> memory.store("config", {"timeout": 30}, MemoryTier.WORKING)
        """
        if tier == MemoryTier.WORKING:
            self.working.store(key, content)

    def retrieve(self, key: str) -> Any | None:
        """
        Retrieve content by key (searches all tiers).

        Args:
            key: Key to retrieve

        Returns:
            Stored content or None if not found

        Note:
            Searches L1 (Working) first, then other tiers

        Example:
            >>> result = memory.retrieve("config")
            >>> if result:
            ...     print(result)
        """
        # L1 - Working Memory
        result = self.working.retrieve(key)
        if result is not None:
            return result

        # Can extend to search other tiers

        return None

    def record_episode(self, episode: EpisodicMemory) -> None:
        """
        Record an episode and extract lessons.

        Args:
            episode: Episode to record

        Note:
            Automatically extracts lessons and creates semantic rules

        Example:
            >>> episode = EpisodicMemory(...)
            >>> memory.record_episode(episode)
        """
        self.episodic.record(episode)

        # Extract lessons from episode
        if episode.lessons:
            for lesson in episode.lessons:
                # Convert lesson to rule (format: "condition: action")
                parts = lesson.split(":", 1)
                if len(parts) == 2:
                    self.semantic.add_rule(
                        condition=parts[0].strip(),
                        action=parts[1].strip(),
                        confidence=0.6 if episode.success else 0.3,
                        source_episode=episode.task_id,
                    )

    def get_relevant_context(self, query: str) -> dict[str, Any]:
        """
        Get relevant context from all memory tiers.

        Args:
            query: Context query string

        Returns:
            Dictionary with rules, lessons, and procedures

        Example:
            >>> context = memory.get_relevant_context("code security")
            >>> print(f"Found {len(context['rules'])} rules")
        """
        context = {
            "rules": self.semantic.find_rules(query),
            "lessons": self.episodic.get_recent_lessons(5),
            "procedures": [],
        }

        # Search for relevant procedures
        # Can be extended based on query

        return context

    def promote_memory(self, entry: MemoryEntry) -> None:
        """
        Promote memory entry to higher tier.

        Args:
            entry: Memory entry to promote

        Note:
            Promotes L1 (Working) → L2 (Episodic) based on access count

        Example:
            >>> entry = MemoryEntry(...)
            >>> if entry.access_count >= 3:
            ...     memory.promote_memory(entry)
        """
        # L1 → L2 promotion
        if entry.tier == MemoryTier.WORKING and entry.access_count >= self._promotion_threshold:
            self._logger.info(f"Promoting memory {entry.id} to episodic")
            # Can be extended to actually move to episodic

    def decay(self) -> None:
        """
        Apply memory decay (cleanup weak memories).

        Note:
            Removes or weakens memories with low strength

        Example:
            >>> memory.decay()
        """
        # Remove weak memories
        pass

    def get_stats(self) -> dict[str, Any]:
        """
        Get comprehensive memory statistics.

        Returns:
            Dictionary with stats for all memory tiers

        Example:
            >>> stats = memory.get_stats()
            >>> print(f"Working: {stats['working']['size']}/{stats['working']['max_size']}")
            >>> print(f"Episodic: {stats['episodic']['total_episodes']} episodes")
        """
        return {
            "working": {
                "size": self.working.get_size(),
                "max_size": self.working.max_size,
            },
            "episodic": {
                "total_episodes": len(self.episodic._episodes),
                "success_rate": self.episodic.get_success_rate(),
            },
            "semantic": {
                "total_rules": len(self.semantic._rules),
            },
            "procedural": {
                "total_procedures": len(self.procedural._procedures),
            },
        }

    def save(self) -> dict[str, bool]:
        """
        Save all memory tiers to disk.

        Returns:
            Dictionary with save results for each tier

        Example:
            >>> results = memory.save()
            >>> print(f"Episodic: {results['episodic']}")
        """
        results = {
            "episodic": self.episodic.save(),
            "semantic": self.semantic.save(),
            "procedural": self.procedural.save(),
        }
        self._logger.info(f"Memory save results: {results}")
        return results

    def load(self) -> dict[str, bool]:
        """
        Load all memory tiers from disk.

        Returns:
            Dictionary with load results for each tier

        Example:
            >>> results = memory.load()
            >>> print(f"Episodic: {results['episodic']}")
        """
        results = {
            "episodic": self.episodic.load(),
            "semantic": self.semantic.load(),
            "procedural": self.procedural.load(),
        }
        self._logger.info(f"Memory load results: {results}")
        return results
