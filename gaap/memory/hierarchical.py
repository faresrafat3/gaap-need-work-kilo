# Hierarchical Memory
import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# =============================================================================
# Enums
# =============================================================================


class MemoryTier(Enum):
    """طبقات الذاكرة"""

    WORKING = 1  # L1: ذاكرة العمل (سريعة، محدودة)
    EPISODIC = 2  # L2: ذاكرة الأحداث (متوسطة)
    SEMANTIC = 3  # L3: ذاكرة دلالية (أنماط)
    PROCEDURAL = 4  # L4: ذاكرة إجرائية (مهارات)


class MemoryPriority(Enum):
    """أولوية الذاكرة"""

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
    """مدخل ذاكرة"""

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
        """تسجيل وصول"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def get_strength(self) -> float:
        """قوة الذاكرة (تتلاشى مع الوقت)"""
        age = (datetime.now() - self.created_at).total_seconds() / 3600  # ساعات
        decay = self.decay_rate * age
        access_boost = min(self.access_count * 0.1, 0.5)
        return max(0, self.importance + access_boost - decay)


@dataclass
class EpisodicMemory:
    """ذاكرة حدث"""

    task_id: str
    action: str
    result: str
    success: bool
    duration_ms: float
    tokens_used: int
    cost_usd: float
    model: str
    provider: str
    timestamp: datetime = field(default_factory=datetime.now)
    lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action[:100],
            "success": self.success,
            "duration_ms": self.duration_ms,
            "tokens": self.tokens_used,
            "model": self.model,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class SemanticRule:
    """قاعدة دلالية"""

    id: str
    condition: str  # الشرط
    action: str  # الإجراء
    confidence: float  # الثقة
    support_count: int  # عدد الدعم
    source_episodes: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
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
    ذاكرة العمل - سريعة ومحدودة

    تستخدم للسياق الحالي للمهمة
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._items: deque = deque(maxlen=max_size)
        self._index: dict[str, MemoryEntry] = {}
        self._logger = logging.getLogger("gaap.memory.working")

    def store(
        self, key: str, content: Any, priority: MemoryPriority = MemoryPriority.NORMAL
    ) -> None:
        """تخزين"""
        entry = MemoryEntry(
            id=self._generate_id(key), tier=MemoryTier.WORKING, content=content, priority=priority
        )

        # إزالة القديم إذا موجود
        if key in self._index:
            old_entry = self._index[key]
            if old_entry in self._items:
                self._items.remove(old_entry)

        self._items.append(entry)
        self._index[key] = entry

    def retrieve(self, key: str) -> Any | None:
        """استرجاع"""
        entry = self._index.get(key)
        if entry:
            entry.access()
            return entry.content
        return None

    def clear(self) -> None:
        """مسح"""
        self._items.clear()
        self._index.clear()

    def get_size(self) -> int:
        """الحجم"""
        return len(self._items)

    def _generate_id(self, key: str) -> str:
        return hashlib.md5(f"{key}:{time.time()}".encode()).hexdigest()[:8]


# =============================================================================
# Episodic Memory (L2)
# =============================================================================


class EpisodicMemoryStore:
    """
    ذاكرة الأحداث - تسجل تجارب المشروع

    تستخدم للتعلم من النجاح والفشل
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path
        self._episodes: list[EpisodicMemory] = []
        self._task_index: dict[str, list[int]] = {}
        self._logger = logging.getLogger("gaap.memory.episodic")

    def record(self, episode: EpisodicMemory) -> None:
        """تسجيل حدث"""
        idx = len(self._episodes)
        self._episodes.append(episode)

        # فهرسة
        if episode.task_id not in self._task_index:
            self._task_index[episode.task_id] = []
        self._task_index[episode.task_id].append(idx)

    def get_episodes(
        self, task_id: str | None = None, success_only: bool = False, limit: int = 100
    ) -> list[EpisodicMemory]:
        """الحصول على أحداث"""
        if task_id:
            indices = self._task_index.get(task_id, [])
            episodes = [self._episodes[i] for i in indices if i < len(self._episodes)]
        else:
            episodes = self._episodes

        if success_only:
            episodes = [e for e in episodes if e.success]

        return episodes[-limit:]

    def get_success_rate(self, task_type: str | None = None) -> float:
        """معدل النجاح"""
        episodes = self._episodes

        if not episodes:
            return 0.0

        successes = sum(1 for e in episodes if e.success)
        return successes / len(episodes)

    def get_recent_lessons(self, limit: int = 20) -> list[str]:
        """الدروس المستفادة"""
        lessons = []
        for episode in reversed(self._episodes):
            lessons.extend(episode.lessons)
            if len(lessons) >= limit:
                break
        return lessons[:limit]

    def save(self) -> bool:
        """حفظ الذاكرة للقرص"""
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
        """تحميل الذاكرة من القرص"""
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
    الذاكرة الدلالية - أنماط وقواعد

    تستخلص من الخبرة المتراكمة
    """

    def __init__(self, storage_path: str | None = None):
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
        """إضافة قاعدة"""
        rule_id = self._generate_id(condition, action)

        if rule_id in self._rules:
            # تعزيز القاعدة الموجودة
            rule = self._rules[rule_id]
            rule.support_count += 1
            rule.confidence = min(rule.confidence + 0.1, 1.0)
            if source_episode:
                rule.source_episodes.append(source_episode)
        else:
            # قاعدة جديدة
            rule = SemanticRule(
                id=rule_id,
                condition=condition,
                action=action,
                confidence=confidence,
                support_count=1,
                source_episodes=[source_episode] if source_episode else [],
            )
            self._rules[rule_id] = rule

        # فهرسة
        keywords = self._extract_keywords(condition)
        for kw in keywords:
            if kw not in self._pattern_index:
                self._pattern_index[kw] = []
            if rule_id not in self._pattern_index[kw]:
                self._pattern_index[kw].append(rule_id)

        return rule

    def find_rules(self, context: str, min_confidence: float = 0.3) -> list[SemanticRule]:
        """البحث عن قواعد مناسبة"""
        keywords = self._extract_keywords(context)

        rule_scores: dict[str, float] = {}
        for kw in keywords:
            for rule_id in self._pattern_index.get(kw, []):
                rule_scores[rule_id] = rule_scores.get(rule_id, 0) + 1

        # ترتيب
        sorted_rules = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)

        rules = []
        for rule_id, _ in sorted_rules:
            if rule_id in self._rules:
                rule = self._rules[rule_id]
                if rule.confidence >= min_confidence:
                    rules.append(rule)

        return rules[:10]

    def _extract_keywords(self, text: str) -> list[str]:
        """استخراج الكلمات المفتاحية"""
        words = text.lower().split()
        # إزالة الكلمات الشائعة
        stopwords = {"the", "a", "an", "is", "are", "to", "for", "and", "or", "in", "on"}
        return [w for w in words if w not in stopwords and len(w) > 2]

    def _generate_id(self, condition: str, action: str) -> str:
        return hashlib.md5(f"{condition}:{action}".encode()).hexdigest()[:12]

    def save(self) -> bool:
        """حفظ الذاكرة للقرص"""
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
        """تحميل الذاكرة من القرص"""
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
    الذاكرة الإجرائية - مهارات مكتسبة

    تمثل كـ fine-tuned prompts أو templates
    """

    def __init__(self, storage_path: str | None = None):
        self.storage_path = storage_path
        self._procedures: dict[str, dict[str, Any]] = {}
        self._logger = logging.getLogger("gaap.memory.procedural")

    def store_procedure(
        self, name: str, prompt_template: str, success_rate: float, examples: list[dict[str, Any]]
    ) -> None:
        """تخزين إجراء"""
        self._procedures[name] = {
            "template": prompt_template,
            "success_rate": success_rate,
            "examples": examples,
            "created_at": datetime.now().isoformat(),
        }

    def get_procedure(self, name: str) -> dict[str, Any] | None:
        """الحصول على إجراء"""
        return self._procedures.get(name)

    def find_best_procedure(self, task_type: str) -> dict[str, Any] | None:
        """أفضل إجراء لمهمة"""
        best = None
        best_rate = 0

        for name, proc in self._procedures.items():
            if task_type.lower() in name.lower() and proc["success_rate"] > best_rate:
                best = proc
                best_rate = proc["success_rate"]

        return best

    def save(self) -> bool:
        """حفظ الذاكرة للقرص"""
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
        """تحميل الذاكرة من القرص"""
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
    مدير الذاكرة الهرمية

    يجمع الطبقات الأربع:
    - L1: Working Memory
    - L2: Episodic Memory
    - L3: Semantic Memory
    - L4: Procedural Memory
    """

    def __init__(self, working_size: int = 100, storage_path: str | None = None):
        self.working = WorkingMemory(max_size=working_size)
        self.episodic = EpisodicMemoryStore(storage_path)
        self.semantic = SemanticMemoryStore(storage_path)
        self.procedural = ProceduralMemoryStore(storage_path)

        self._logger = logging.getLogger("gaap.memory")

        # ترقية الذاكرة
        self._promotion_threshold = 3  # عدد الوصولات للترقية

    def store(self, key: str, content: Any, tier: MemoryTier = MemoryTier.WORKING) -> None:
        """تخزين في طبقة معينة"""
        if tier == MemoryTier.WORKING:
            self.working.store(key, content)

    def retrieve(self, key: str) -> Any | None:
        """استرجاع (يبحث في كل الطبقات)"""
        # L1
        result = self.working.retrieve(key)
        if result is not None:
            return result

        # يمكن إضافة بحث في الطبقات الأخرى

        return None

    def record_episode(self, episode: EpisodicMemory) -> None:
        """تسجيل حدث"""
        self.episodic.record(episode)

        # استخلاص دروس
        if episode.lessons:
            for lesson in episode.lessons:
                # تحويل الدرس لقاعدة
                parts = lesson.split(":", 1)
                if len(parts) == 2:
                    self.semantic.add_rule(
                        condition=parts[0].strip(),
                        action=parts[1].strip(),
                        confidence=0.6 if episode.success else 0.3,
                        source_episode=episode.task_id,
                    )

    def get_relevant_context(self, query: str) -> dict[str, Any]:
        """الحصول على سياق ذي صلة"""
        context = {
            "rules": self.semantic.find_rules(query),
            "lessons": self.episodic.get_recent_lessons(5),
            "procedures": [],
        }

        # البحث عن إجراءات مناسبة
        # ...

        return context

    def promote_memory(self, entry: MemoryEntry) -> None:
        """ترقية الذاكرة (نقل لطبقة أعلى)"""
        # L1 -> L2
        if entry.tier == MemoryTier.WORKING and entry.access_count >= self._promotion_threshold:
            # نقل للحلقات
            self._logger.info(f"Promoting memory {entry.id} to episodic")

    def decay(self) -> None:
        """تلاشي الذاكرة (تنظيف)"""
        # إزالة الذكريات الضعيفة
        pass

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
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
        """حفظ جميع طبقات الذاكرة"""
        results = {
            "episodic": self.episodic.save(),
            "semantic": self.semantic.save(),
            "procedural": self.procedural.save(),
        }
        self._logger.info(f"Memory save results: {results}")
        return results

    def load(self) -> dict[str, bool]:
        """تحميل جميع طبقات الذاكرة"""
        results = {
            "episodic": self.episodic.load(),
            "semantic": self.semantic.load(),
            "procedural": self.procedural.load(),
        }
        self._logger.info(f"Memory load results: {results}")
        return results
