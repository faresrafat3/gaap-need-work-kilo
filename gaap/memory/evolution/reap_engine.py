"""
REAP Engine Module
==================

Review, Extract, Abstract, Prune - Memory consolidation algorithm.

The REAP algorithm processes memories during "sleep" cycles:
1. Review: Cluster similar episodes
2. Extract: Identify invariants/patterns
3. Abstract: Create strategic heuristics
4. Prune: Remove low-confidence/noisy memories

Reference: docs/evolution_plan_2026/01_MEMORY_AND_DREAMING.md
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.evolution.reap")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class EpisodeCluster:
    """مجموعة حلقات متشابهة"""

    cluster_id: str
    episodes: list[dict[str, Any]]
    centroid_topic: str
    similarity_score: float
    common_patterns: list[str] = field(default_factory=list)
    success_rate: float = 0.0


@dataclass
class ExtractedPattern:
    """نمط مستخرج"""

    pattern: str
    episodes_involved: list[str]
    confidence: float
    domain: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class AbstractedRule:
    """قاعدة مجردة"""

    rule: str
    source_pattern: str
    confidence: float
    domain: str
    applicability: str
    exceptions: list[str] = field(default_factory=list)


@dataclass
class REAPResult:
    """نتيجة خوارزمية REAP"""

    clusters_formed: int
    patterns_extracted: int
    rules_created: int
    memories_pruned: int
    processing_time_ms: float
    details: dict[str, Any] = field(default_factory=dict)


class REAPEngine:
    """
    محرك REAP للتوحيد

    Process:
    1. Review: تجميع الحلقات المتشابهة
    2. Extract: استخراج الأنماط والثوابت
    3. Abstract: تحويل الأنماط لقواعد
    4. Prune: حذف الضوضاء
    """

    CLUSTERING_PROMPT = """Analyze these episodes and group them into clusters based on similarity.

Episodes:
{episodes}

Respond with JSON:
{{
  "clusters": [
    {{
      "episode_ids": ["id1", "id2"],
      "common_theme": "description",
      "similarity": 0.0-1.0
    }}
  ]
}}"""

    PATTERN_EXTRACTION_PROMPT = """Extract patterns from these related episodes.

Episodes:
{episodes}

Identify:
1. What is common across successful episodes?
2. What led to failures?
3. What decisions were made?

Respond with JSON:
{{
  "patterns": [
    {{
      "pattern": "description",
      "confidence": 0.0-1.0,
      "evidence": ["quote from episode"]
    }}
  ]
}}"""

    ABSTRACTION_PROMPT = """Convert this pattern into a general rule.

Pattern: {pattern}
Domain: {domain}
Success Rate: {success_rate}

Create a strategic heuristic that can guide future decisions.

Respond with JSON:
{{
  "rule": "the abstracted rule",
  "applicability": "when to apply this rule",
  "exceptions": ["cases where rule doesn't apply"],
  "confidence": 0.0-1.0
}}"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        min_cluster_size: int = 3,
        min_confidence: float = 0.5,
        pruning_threshold: float = 0.2,
    ):
        self._llm_provider = llm_provider
        self._model = model
        self._min_cluster_size = min_cluster_size
        self._min_confidence = min_confidence
        self._pruning_threshold = pruning_threshold

        self._clusters: list[EpisodeCluster] = []
        self._patterns: list[ExtractedPattern] = []
        self._rules: list[AbstractedRule] = []

        self._logger = logger

    async def run_cycle(
        self,
        episodes: list[dict[str, Any]],
        memory_store: Any = None,
    ) -> REAPResult:
        """
        تشغيل دورة REAP كاملة

        Args:
            episodes: قائمة الحلقات
            memory_store: مخزن الذاكرة (للحذف)

        Returns:
            نتيجة REAP
        """
        import time

        start_time = time.time()

        self._logger.info(f"Starting REAP cycle with {len(episodes)} episodes")

        clusters = await self._review(episodes)

        patterns = await self._extract(clusters)

        rules = await self._abstract(patterns)

        pruned = await self._prune(episodes, memory_store)

        result = REAPResult(
            clusters_formed=len(clusters),
            patterns_extracted=len(patterns),
            rules_created=len(rules),
            memories_pruned=pruned,
            processing_time_ms=(time.time() - start_time) * 1000,
            details={
                "clusters": [c.centroid_topic for c in clusters],
                "patterns": [p.pattern[:100] for p in patterns],
                "rules": [r.rule[:100] for r in rules],
            },
        )

        self._logger.info(
            f"REAP cycle completed: {result.clusters_formed} clusters, "
            f"{result.patterns_extracted} patterns, "
            f"{result.rules_created} rules, "
            f"{result.memories_pruned} pruned"
        )

        return result

    async def _review(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[EpisodeCluster]:
        """
        Review: تجميع الحلقات المتشابهة
        """
        clusters = []

        if self._llm_provider and len(episodes) >= self._min_cluster_size:
            try:
                llm_clusters = await self._llm_cluster(episodes)
                clusters.extend(llm_clusters)
            except Exception as e:
                self._logger.warning(f"LLM clustering failed: {e}")

        heuristic_clusters = self._heuristic_cluster(episodes)

        for hc in heuristic_clusters:
            existing = next(
                (c for c in clusters if c.cluster_id == hc.cluster_id),
                None,
            )
            if not existing:
                clusters.append(hc)

        self._clusters = clusters
        return clusters

    async def _llm_cluster(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[EpisodeCluster]:
        """تجميع باستخدام LLM"""

        episodes_str = "\n".join(
            [
                f"[{e.get('id', i)}] {e.get('action', '')}: {e.get('result', '')[:100]}"
                for i, e in enumerate(episodes[:20])
            ]
        )

        prompt = self.CLUSTERING_PROMPT.format(episodes=episodes_str)

        if self._llm_provider is None:
            return []

        response = await self._llm_provider.complete(
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            temperature=0.1,
            max_tokens=500,
        )

        return self._parse_clusters(response, episodes)

    def _heuristic_cluster(
        self,
        episodes: list[dict[str, Any]],
    ) -> list[EpisodeCluster]:
        """تجميع استدلالي"""

        import hashlib
        from collections import defaultdict

        domain_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for episode in episodes:
            domain = episode.get("domain", "general")
            domain_clusters[domain].append(episode)

        action_clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for episode in episodes:
            action = episode.get("action", "unknown")
            action_key = action[:50].lower()
            action_clusters[action_key].append(episode)

        clusters = []

        for domain, eps in domain_clusters.items():
            if len(eps) >= self._min_cluster_size:
                success_count = sum(1 for e in eps if e.get("success", False))

                clusters.append(
                    EpisodeCluster(
                        cluster_id=hashlib.md5(domain.encode()).hexdigest()[:8],
                        episodes=eps,
                        centroid_topic=f"Domain: {domain}",
                        similarity_score=0.7,
                        success_rate=success_count / len(eps),
                    )
                )

        for action_key, eps in action_clusters.items():
            if len(eps) >= self._min_cluster_size and len(eps) <= 50:
                success_count = sum(1 for e in eps if e.get("success", False))

                clusters.append(
                    EpisodeCluster(
                        cluster_id=hashlib.md5(action_key.encode()).hexdigest()[:8],
                        episodes=eps,
                        centroid_topic=f"Action: {action_key}",
                        similarity_score=0.6,
                        success_rate=success_count / len(eps),
                    )
                )

        return clusters

    def _parse_clusters(
        self,
        response: str,
        episodes: list[dict[str, Any]],
    ) -> list[EpisodeCluster]:
        """تحليل استجابة التجميع"""

        import json

        clusters = []

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                episode_map = {e.get("id", str(i)): e for i, e in enumerate(episodes)}

                for i, cluster_data in enumerate(data.get("clusters", [])):
                    episode_ids = cluster_data.get("episode_ids", [])
                    cluster_episodes = [
                        episode_map[eid] for eid in episode_ids if eid in episode_map
                    ]

                    if len(cluster_episodes) >= self._min_cluster_size:
                        clusters.append(
                            EpisodeCluster(
                                cluster_id=f"llm_{i}",
                                episodes=cluster_episodes,
                                centroid_topic=cluster_data.get("common_theme", "Unknown"),
                                similarity_score=cluster_data.get("similarity", 0.5),
                            )
                        )

        except Exception as e:
            self._logger.debug(f"Failed to parse cluster response: {e}")

        return clusters

    async def _extract(
        self,
        clusters: list[EpisodeCluster],
    ) -> list[ExtractedPattern]:
        """
        Extract: استخراج الأنماط
        """

        patterns = []

        for cluster in clusters:
            if cluster.success_rate >= 0.7:
                pattern = await self._extract_success_pattern(cluster)
                if pattern:
                    patterns.append(pattern)

            elif cluster.success_rate <= 0.3:
                pattern = await self._extract_failure_pattern(cluster)
                if pattern:
                    patterns.append(pattern)

        self._patterns = patterns
        return patterns

    async def _extract_success_pattern(
        self,
        cluster: EpisodeCluster,
    ) -> ExtractedPattern | None:
        """استخراج نمط النجاح"""

        successful = [e for e in cluster.episodes if e.get("success", False)]

        if len(successful) < self._min_cluster_size:
            return None

        common_actions = self._find_common_elements([e.get("action", "") for e in successful])

        common_lessons = self._find_common_elements(
            [l for e in successful for l in e.get("lessons", [])]
        )

        pattern_text = f"In {cluster.centroid_topic}: "

        if common_actions:
            pattern_text += f"Actions that work: {', '.join(common_actions[:3])}. "

        if common_lessons:
            pattern_text += f"Key lessons: {', '.join(common_lessons[:3])}."

        return ExtractedPattern(
            pattern=pattern_text,
            episodes_involved=[e.get("id", "") for e in successful],
            confidence=cluster.similarity_score * cluster.success_rate,
            domain=self._infer_domain(cluster),
            evidence=common_lessons[:3],
        )

    async def _extract_failure_pattern(
        self,
        cluster: EpisodeCluster,
    ) -> ExtractedPattern | None:
        """استخراج نمط الفشل"""

        failed = [e for e in cluster.episodes if not e.get("success", False)]

        if len(failed) < self._min_cluster_size:
            return None

        common_errors = self._find_common_elements(
            [e.get("error", e.get("result", "")) for e in failed]
        )

        pattern_text = f"In {cluster.centroid_topic}: "
        pattern_text += f"Common failure causes: {', '.join(common_errors[:3])}. "
        pattern_text += "Avoid these patterns."

        return ExtractedPattern(
            pattern=pattern_text,
            episodes_involved=[e.get("id", "") for e in failed],
            confidence=cluster.similarity_score * (1 - cluster.success_rate),
            domain=self._infer_domain(cluster),
            evidence=common_errors[:3],
        )

    def _find_common_elements(self, items: list[str]) -> list[str]:
        """إيجاد العناصر المشتركة"""

        from collections import Counter

        words = []
        for item in items:
            words.extend(item.lower().split())

        word_counts = Counter(words)

        common = [
            word for word, count in word_counts.most_common(10) if count >= 2 and len(word) > 3
        ]

        return common

    def _infer_domain(self, cluster: EpisodeCluster) -> str:
        """استنتاج التخصص"""

        for episode in cluster.episodes:
            domain = episode.get("domain")
            if domain:
                return str(domain)

        return "general"

    async def _abstract(
        self,
        patterns: list[ExtractedPattern],
    ) -> list[AbstractedRule]:
        """
        Abstract: تحويل الأنماط لقواعد
        """

        rules = []

        for pattern in patterns:
            if pattern.confidence >= self._min_confidence:
                rule = await self._create_rule(pattern)
                if rule:
                    rules.append(rule)

        self._rules = rules
        return rules

    async def _create_rule(
        self,
        pattern: ExtractedPattern,
    ) -> AbstractedRule | None:
        """إنشاء قاعدة من نمط"""

        if self._llm_provider:
            try:
                prompt = self.ABSTRACTION_PROMPT.format(
                    pattern=pattern.pattern,
                    domain=pattern.domain,
                    success_rate=pattern.confidence,
                )

                response = await self._llm_provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._model,
                    temperature=0.2,
                    max_tokens=200,
                )

                return self._parse_rule(response, pattern)

            except Exception as e:
                self._logger.debug(f"LLM rule creation failed: {e}")

        return AbstractedRule(
            rule=pattern.pattern,
            source_pattern=pattern.pattern,
            confidence=pattern.confidence * 0.8,
            domain=pattern.domain,
            applicability=f"When working on {pattern.domain} tasks",
        )

    def _parse_rule(
        self,
        response: str,
        pattern: ExtractedPattern,
    ) -> AbstractedRule | None:
        """تحليل استجابة القاعدة"""

        import json

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                return AbstractedRule(
                    rule=data.get("rule", pattern.pattern),
                    source_pattern=pattern.pattern,
                    confidence=float(data.get("confidence", pattern.confidence)),
                    domain=pattern.domain,
                    applicability=data.get("applicability", ""),
                    exceptions=data.get("exceptions", []),
                )

        except Exception:
            pass

        return None

    async def _prune(
        self,
        episodes: list[dict[str, Any]],
        memory_store: Any,
    ) -> int:
        """
        Prune: حذف الذكريات الضعيفة
        """

        pruned = 0

        to_prune = []

        for episode in episodes:
            confidence = episode.get("confidence", 1.0)
            importance = episode.get("importance", 1.0)
            access_count = episode.get("access_count", 0)

            score = confidence * 0.4 + importance * 0.4 + min(access_count / 10, 0.2)

            if score < self._pruning_threshold:
                age_days = (
                    (
                        datetime.now()
                        - datetime.fromisoformat(
                            episode.get("timestamp", datetime.now().isoformat())
                        )
                    ).days
                    if "timestamp" in episode
                    else 30
                )

                if age_days > 7:
                    to_prune.append(episode)

        if memory_store and hasattr(memory_store, "delete"):
            for episode in to_prune:
                try:
                    if memory_store.delete(episode.get("id")):
                        pruned += 1
                except Exception as e:
                    self._logger.debug(f"Failed to prune episode: {e}")

        self._logger.info(f"Pruned {pruned} low-quality memories")

        return pruned

    def get_clusters(self) -> list[EpisodeCluster]:
        return self._clusters

    def get_patterns(self) -> list[ExtractedPattern]:
        return self._patterns

    def get_rules(self) -> list[AbstractedRule]:
        return self._rules
