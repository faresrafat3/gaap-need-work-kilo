"""
Wisdom Distiller - Extract Principles from Success
===================================================

Analyzes successful episodes and extracts generalized principles
that can guide future behavior.

Key insight: Instead of just remembering "this worked", we ask:
"What universal principle made this work?"

Usage:
    distiller = WisdomDistiller(llm_provider)

    # Distill from episodes
    result = await distiller.distill(episodes)
    print(result.heuristic.principle)  # "Always validate input before parsing"
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

from gaap.storage.atomic import atomic_write

from gaap.memory.hierarchical import EpisodicMemory, EpisodicMemoryStore

logger = logging.getLogger("gaap.meta_learning.wisdom")


class HeuristicCategory(Enum):
    """تصنيفات الحكم المستخرجة"""

    CODING = auto()
    TESTING = auto()
    DEBUGGING = auto()
    ARCHITECTURE = auto()
    SECURITY = auto()
    PERFORMANCE = auto()
    RESEARCH = auto()
    COMMUNICATION = auto()
    PROCESS = auto()
    GENERAL = auto()


class HeuristicStatus(Enum):
    """حالة الحكمة"""

    DRAFT = auto()
    VALIDATED = auto()
    NEEDS_MORE_EVIDENCE = auto()
    DEPRECATED = auto()
    PROMOTED_TO_AXIOM = auto()


@dataclass
class ProjectHeuristic:
    """
    A distilled principle from successful experiences.

    Represents generalized knowledge that can guide future behavior.
    """

    principle: str
    category: HeuristicCategory = HeuristicCategory.GENERAL
    status: HeuristicStatus = HeuristicStatus.DRAFT
    confidence: float = 0.5
    evidence_count: int = 1
    source_episodes: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    counter_examples: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime | None = None
    success_rate: float = 0.0
    applicability_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_id(self) -> str:
        content = f"{self.category.name}:{self.principle}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.get_id(),
            "principle": self.principle,
            "category": self.category.name,
            "status": self.status.name,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_episodes": self.source_episodes,
            "examples": self.examples,
            "counter_examples": self.counter_examples,
            "created_at": self.created_at.isoformat(),
            "last_validated": self.last_validated.isoformat() if self.last_validated else None,
            "success_rate": self.success_rate,
            "applicability_score": self.applicability_score,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectHeuristic":
        return cls(
            principle=data.get("principle", ""),
            category=HeuristicCategory[data.get("category", "GENERAL")],
            status=HeuristicStatus[data.get("status", "DRAFT")],
            confidence=data.get("confidence", 0.5),
            evidence_count=data.get("evidence_count", 1),
            source_episodes=data.get("source_episodes", []),
            examples=data.get("examples", []),
            counter_examples=data.get("counter_examples", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            last_validated=datetime.fromisoformat(data["last_validated"])
            if data.get("last_validated")
            else None,
            success_rate=data.get("success_rate", 0.0),
            applicability_score=data.get("applicability_score", 0.5),
            metadata=data.get("metadata", {}),
        )

    def is_ready_for_axiom(self) -> bool:
        """Check if heuristic is strong enough to become an axiom."""
        return (
            self.confidence >= 0.9
            and self.evidence_count >= 10
            and self.success_rate >= 0.85
            and self.status == HeuristicStatus.VALIDATED
            and len(self.counter_examples) == 0
        )


@dataclass
class DistillationResult:
    """نتيجة عملية التقطير"""

    heuristic: ProjectHeuristic | None
    episodes_analyzed: int
    similar_patterns_found: int
    confidence: float
    reasoning: str
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "heuristic": self.heuristic.to_dict() if self.heuristic else None,
            "episodes_analyzed": self.episodes_analyzed,
            "similar_patterns_found": self.similar_patterns_found,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


class WisdomDistiller:
    """
    Extracts generalized principles from successful experiences.

    Features:
    - Pattern detection across similar successful episodes
    - LLM-powered principle extraction
    - Confidence scoring based on evidence
    - Epistemic humility (knows when it doesn't know)
    - Cross-domain pattern transfer
    """

    DEFAULT_STORAGE_PATH = ".gaap/memory/wisdom"

    MIN_EPISODES_FOR_DISTILLATION = 3
    MIN_EPISODES_FOR_VALIDATION = 5

    def __init__(
        self,
        storage_path: str | None = None,
        episodic_store: EpisodicMemoryStore | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._episodic = episodic_store
        self._llm = llm_client

        self._heuristics: dict[str, ProjectHeuristic] = {}
        self._category_index: dict[HeuristicCategory, set[str]] = {}
        self._keyword_index: dict[str, set[str]] = {}

        self._logger = logger

        self._load()

    async def distill(
        self,
        episodes: list[EpisodicMemory],
        min_similarity: float = 0.5,
    ) -> DistillationResult:
        """
        Distill a principle from similar successful episodes.

        Args:
            episodes: List of successful episodes to analyze
            min_similarity: Minimum similarity threshold

        Returns:
            DistillationResult with extracted heuristic
        """
        successful = [e for e in episodes if e.success]

        if len(successful) < self.MIN_EPISODES_FOR_DISTILLATION:
            return DistillationResult(
                heuristic=None,
                episodes_analyzed=len(successful),
                similar_patterns_found=0,
                confidence=0.0,
                reasoning="Not enough successful episodes",
                skipped=True,
                skip_reason=f"Need at least {self.MIN_EPISODES_FOR_DISTILLATION} episodes",
            )

        grouped = self._group_by_similarity(successful, min_similarity)

        if not grouped:
            return DistillationResult(
                heuristic=None,
                episodes_analyzed=len(successful),
                similar_patterns_found=0,
                confidence=0.0,
                reasoning="No similar patterns found",
                skipped=True,
                skip_reason="Episodes too dissimilar",
            )

        best_group = max(grouped, key=len)
        pattern_count = len(best_group)

        principle = await self._extract_principle(best_group)

        if not principle:
            return DistillationResult(
                heuristic=None,
                episodes_analyzed=len(successful),
                similar_patterns_found=pattern_count,
                confidence=0.0,
                reasoning="Could not extract clear principle",
                skipped=True,
                skip_reason="LLM failed to extract principle",
            )

        category = self._classify_category(best_group[0])

        heuristic = ProjectHeuristic(
            principle=principle,
            category=category,
            status=HeuristicStatus.DRAFT,
            confidence=self._calculate_confidence(best_group),
            evidence_count=pattern_count,
            source_episodes=[e.task_id for e in best_group if e.task_id],
            examples=[e.action[:200] for e in best_group[:3]],
            success_rate=1.0,
            applicability_score=self._calculate_applicability(best_group),
        )

        existing = self._find_similar_heuristic(heuristic)
        if existing:
            existing.evidence_count += pattern_count
            existing.confidence = min(existing.confidence + 0.1, 1.0)
            existing.source_episodes.extend([e.task_id for e in best_group if e.task_id])
            existing.last_validated = datetime.now()
            self._save()

            return DistillationResult(
                heuristic=existing,
                episodes_analyzed=len(successful),
                similar_patterns_found=pattern_count,
                confidence=existing.confidence,
                reasoning="Strengthened existing heuristic",
            )

        self._add_heuristic(heuristic)
        self._save()

        return DistillationResult(
            heuristic=heuristic,
            episodes_analyzed=len(successful),
            similar_patterns_found=pattern_count,
            confidence=heuristic.confidence,
            reasoning="New heuristic extracted successfully",
        )

    async def distill_from_category(
        self,
        category: str,
        days: int = 7,
        limit: int = 50,
    ) -> list[DistillationResult]:
        """
        Distill heuristics from recent episodes in a category.

        Args:
            category: Episode category to analyze
            days: Look back period
            limit: Maximum episodes to analyze

        Returns:
            List of distillation results
        """
        if not self._episodic:
            return []

        cutoff = datetime.now() - timedelta(days=days)
        episodes = [
            e
            for e in self._episodic._episodes
            if e.category == category and e.timestamp >= cutoff and e.success
        ][:limit]

        if len(episodes) < self.MIN_EPISODES_FOR_DISTILLATION:
            return []

        results = []

        for i in range(0, len(episodes), 5):
            batch = episodes[i : i + 5]
            if len(batch) >= self.MIN_EPISODES_FOR_DISTILLATION:
                result = await self.distill(batch)
                if result.heuristic:
                    results.append(result)

        return results

    def get_heuristics_for_context(
        self,
        context: str,
        min_confidence: float = 0.5,
        limit: int = 5,
    ) -> list[ProjectHeuristic]:
        """
        Get relevant heuristics for a given context.

        Args:
            context: Task or problem description
            min_confidence: Minimum confidence threshold
            limit: Maximum heuristics to return

        Returns:
            List of relevant heuristics sorted by applicability
        """
        keywords = self._extract_keywords(context)

        scores: dict[str, float] = {}
        for kw in keywords:
            for hid in self._keyword_index.get(kw, set()):
                if hid in self._heuristics:
                    h = self._heuristics[hid]
                    if h.confidence >= min_confidence:
                        scores[hid] = scores.get(hid, 0) + h.confidence

        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]

        return [self._heuristics[hid] for hid, _ in sorted_ids]

    def validate_heuristic(
        self,
        heuristic_id: str,
        recent_episodes: list[EpisodicMemory],
    ) -> bool:
        """
        Validate a heuristic against recent episodes.

        Args:
            heuristic_id: ID of heuristic to validate
            recent_episodes: Recent episodes to test against

        Returns:
            True if heuristic still holds, False otherwise
        """
        heuristic = self._heuristics.get(heuristic_id)
        if not heuristic:
            return False

        relevant = [e for e in recent_episodes if self._is_episode_relevant(e, heuristic)]

        if len(relevant) < self.MIN_EPISODES_FOR_VALIDATION:
            return True

        successes = sum(1 for e in relevant if e.success)
        new_rate = successes / len(relevant)

        heuristic.success_rate = heuristic.success_rate * 0.7 + new_rate * 0.3
        heuristic.last_validated = datetime.now()

        if new_rate < 0.5:
            heuristic.status = HeuristicStatus.DEPRECATED
            self._logger.warning(
                f"Heuristic deprecated due to low success rate: {heuristic.principle[:50]}"
            )
            return False

        if heuristic.success_rate >= 0.8 and heuristic.confidence >= 0.7:
            heuristic.status = HeuristicStatus.VALIDATED

        self._save()
        return True

    def get_heuristics_ready_for_axiom(self) -> list[ProjectHeuristic]:
        """Get heuristics that are strong enough to become axioms."""
        return [h for h in self._heuristics.values() if h.is_ready_for_axiom()]

    def get_stats(self) -> dict[str, Any]:
        """Get distiller statistics."""
        by_category: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for h in self._heuristics.values():
            by_category[h.category.name] = by_category.get(h.category.name, 0) + 1
            by_status[h.status.name] = by_status.get(h.status.name, 0) + 1

        ready_for_axiom = len(self.get_heuristics_ready_for_axiom())

        return {
            "total_heuristics": len(self._heuristics),
            "by_category": by_category,
            "by_status": by_status,
            "ready_for_axiom": ready_for_axiom,
            "avg_confidence": sum(h.confidence for h in self._heuristics.values())
            / max(len(self._heuristics), 1),
        }

    async def _extract_principle(
        self,
        episodes: list[EpisodicMemory],
    ) -> str | None:
        """
        Use LLM to extract a universal principle from episodes.

        If no LLM is available, uses rule-based extraction.
        """
        if self._llm:
            return await self._llm_extract_principle(episodes)
        return self._rule_based_extract_principle(episodes)

    async def _llm_extract_principle(
        self,
        episodes: list[EpisodicMemory],
    ) -> str | None:
        """Extract principle using LLM."""
        actions = [e.action for e in episodes[:5]]
        results = [e.result for e in episodes[:5]]

        prompt = f"""
Analyze these successful problem-solving episodes and extract a universal engineering principle.

Episodes:
{json.dumps([{"action": a, "result": r[:200]} for a, r in zip(actions, results)], indent=2)}

What is the underlying principle that made these approaches successful?
Provide a concise, actionable principle (one sentence).
"""

        llm = self._llm
        if not llm:
            return None

        try:
            if hasattr(llm, "generate"):
                response = await llm.generate(prompt)
            else:
                response = str(llm)

            return self._clean_principle(response)

        except Exception as e:
            self._logger.error(f"LLM principle extraction failed: {e}")
            return None

    def _rule_based_extract_principle(
        self,
        episodes: list[EpisodicMemory],
    ) -> str | None:
        """Extract principle using rule-based analysis."""
        if not episodes:
            return None

        actions = [e.action.lower() for e in episodes]
        common_words: dict[str, int] = {}

        for action in actions:
            words = set(self._extract_keywords(action))
            for word in words:
                common_words[word] = common_words.get(word, 0) + 1

        recurring = [
            w
            for w, c in sorted(common_words.items(), key=lambda x: x[1], reverse=True)
            if c >= len(episodes) // 2
        ][:5]

        if recurring:
            category = episodes[0].category
            return f"For {category}: Consider focusing on {', '.join(recurring[:3])}"

        return None

    def _clean_principle(self, text: str) -> str:
        """Clean and format extracted principle."""
        text = text.strip()

        if len(text) > 300:
            sentences = re.split(r"[.!?]", text)
            text = sentences[0] if sentences else text[:300]

        return text.strip()

    def _group_by_similarity(
        self,
        episodes: list[EpisodicMemory],
        min_similarity: float,
    ) -> list[list[EpisodicMemory]]:
        """Group episodes by similarity."""
        if not episodes:
            return []

        groups: list[list[EpisodicMemory]] = []

        for episode in episodes:
            added = False
            for group in groups:
                if self._calculate_similarity(episode, group[0]) >= min_similarity:
                    group.append(episode)
                    added = True
                    break

            if not added:
                groups.append([episode])

        return [g for g in groups if len(g) >= self.MIN_EPISODES_FOR_DISTILLATION]

    def _calculate_similarity(
        self,
        e1: EpisodicMemory,
        e2: EpisodicMemory,
    ) -> float:
        """Calculate similarity between two episodes."""
        if e1.category != e2.category:
            return 0.0

        kw1 = set(self._extract_keywords(e1.action))
        kw2 = set(self._extract_keywords(e2.action))

        if not kw1 or not kw2:
            return 0.0

        intersection = kw1 & kw2
        union = kw1 | kw2

        return len(intersection) / len(union)

    def _calculate_confidence(self, episodes: list[EpisodicMemory]) -> float:
        """Calculate confidence score for extracted heuristic."""
        base = 0.3

        count_factor = min(len(episodes) / 10, 0.3)

        consistency = 1.0 if all(e.success for e in episodes) else 0.5

        recency_factor = 0.2 if episodes else 0.0

        return min(base + count_factor + consistency * 0.2 + recency_factor, 1.0)

    def _calculate_applicability(
        self,
        episodes: list[EpisodicMemory],
    ) -> float:
        """Calculate how widely applicable the heuristic is."""
        unique_tasks = len(set(e.task_id for e in episodes))
        unique_categories = len(set(e.category for e in episodes))

        task_factor = min(unique_tasks / 5, 0.5)
        category_factor = min(unique_categories / 3, 0.5)

        return task_factor + category_factor

    def _classify_category(self, episode: EpisodicMemory) -> HeuristicCategory:
        """Classify heuristic category from episode."""
        cat_map = {
            "code": HeuristicCategory.CODING,
            "test": HeuristicCategory.TESTING,
            "debug": HeuristicCategory.DEBUGGING,
            "architect": HeuristicCategory.ARCHITECTURE,
            "security": HeuristicCategory.SECURITY,
            "performance": HeuristicCategory.PERFORMANCE,
            "research": HeuristicCategory.RESEARCH,
            "communicate": HeuristicCategory.COMMUNICATION,
        }

        cat_lower = episode.category.lower()
        for key, hcat in cat_map.items():
            if key in cat_lower:
                return hcat

        return HeuristicCategory.GENERAL

    def _is_episode_relevant(
        self,
        episode: EpisodicMemory,
        heuristic: ProjectHeuristic,
    ) -> bool:
        """Check if episode is relevant to a heuristic."""
        kw_episode = set(self._extract_keywords(episode.action))
        kw_heuristic = set(self._extract_keywords(heuristic.principle))

        if not kw_episode or not kw_heuristic:
            return False

        overlap = kw_episode & kw_heuristic
        return len(overlap) >= 1

    def _find_similar_heuristic(
        self,
        heuristic: ProjectHeuristic,
    ) -> ProjectHeuristic | None:
        """Find existing heuristic with similar principle."""
        for existing in self._heuristics.values():
            if existing.category != heuristic.category:
                continue

            kw1 = set(self._extract_keywords(existing.principle))
            kw2 = set(self._extract_keywords(heuristic.principle))

            if not kw1 or not kw2:
                continue

            overlap = len(kw1 & kw2) / min(len(kw1), len(kw2))
            if overlap >= 0.7:
                return existing

        return None

    def _add_heuristic(self, heuristic: ProjectHeuristic) -> None:
        """Add heuristic to store with indexing."""
        hid = heuristic.get_id()
        self._heuristics[hid] = heuristic

        if heuristic.category not in self._category_index:
            self._category_index[heuristic.category] = set()
        self._category_index[heuristic.category].add(hid)

        for kw in self._extract_keywords(heuristic.principle):
            if kw not in self._keyword_index:
                self._keyword_index[kw] = set()
            self._keyword_index[kw].add(hid)

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
            "when",
            "what",
            "where",
            "which",
            "their",
            "there",
            "about",
            "into",
            "than",
        }

        return [w for w in words if w not in stopwords]

    def _save(self) -> bool:
        """Save heuristics to disk."""
        try:
            filepath = self.storage_path / "heuristics.json"

            atomic_write(
                filepath,
                json.dumps(
                    {hid: h.to_dict() for hid, h in self._heuristics.items()},
                    indent=2,
                ),
            )

            self._logger.debug(f"Saved {len(self._heuristics)} heuristics to {self.storage_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save heuristics: {e}")
            return False

    def _load(self) -> bool:
        """Load heuristics from disk."""
        try:
            filepath = self.storage_path / "heuristics.json"

            if not filepath.exists():
                return True

            with open(filepath) as f:
                data = json.load(f)

            self._heuristics = {hid: ProjectHeuristic.from_dict(h) for hid, h in data.items()}

            for hid, h in self._heuristics.items():
                if h.category not in self._category_index:
                    self._category_index[h.category] = set()
                self._category_index[h.category].add(hid)

                for kw in self._extract_keywords(h.principle):
                    if kw not in self._keyword_index:
                        self._keyword_index[kw] = set()
                    self._keyword_index[kw].add(hid)

            self._logger.info(f"Loaded {len(self._heuristics)} heuristics from {self.storage_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load heuristics: {e}")
            return False


def create_wisdom_distiller(
    storage_path: str | None = None,
    llm_client: Any | None = None,
) -> WisdomDistiller:
    """Create a WisdomDistiller instance."""
    return WisdomDistiller(storage_path=storage_path, llm_client=llm_client)
