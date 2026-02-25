"""
Relation Extractor Module
=========================

Extracts relations between concepts from content.

Features:
- Pattern-based extraction
- LLM-powered extraction
- Co-occurrence analysis
- Temporal relation detection
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .graph_builder import RelationType

logger = logging.getLogger("gaap.memory.knowledge.relation")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@dataclass
class ExtractedRelation:
    """
    علاقة مستخرجة

    Attributes:
        source: النص المصدر
        target: النص الهدف
        relation: نوع العلاقة
        confidence: ثقة الاستخراج
        evidence: الدليل من النص
        context: السياق المحيط
    """

    source: str
    target: str
    relation: RelationType
    confidence: float
    evidence: str = ""
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "context": self.context[:200],
        }


class RelationExtractor:
    """
    مستخرج العلاقات

    Methods:
    - Pattern-based: قواعد وأنماط محددة
    - LLM-based: تحليل ذكي
    - Co-occurrence: تواجد مشترك
    - Temporal: علاقات زمنية
    """

    RELATION_PATTERNS = {
        RelationType.CAUSED: [
            r"(.+?)\s+(?:caused|سبب|أدى إلى|نتج عن)\s+(.+)",
            r"(.+?)\s+(?:error|خطأ|exception|مشكلة)\s+(?:in|في)\s+(.+)",
            r"(.+?)\s+(?:failed|فشل)\s+(?:because|لأن|بسبب)\s+(.+)",
        ],
        RelationType.FIXED: [
            r"(.+?)\s+(?:fixed|أصلح|حل|resolved)\s+(.+)",
            r"(.+?)\s+(?:solution|حل)\s+(?:for|لـ)\s+(.+)",
            r"to fix (.+?), (.+)",
            r"لحل (.+?), (.+)",
        ],
        RelationType.DEPENDS_ON: [
            r"(.+?)\s+(?:depends|يعتمد)\s+(?:on|على)\s+(.+)",
            r"(.+?)\s+(?:requires|يتطلب|needs)\s+(.+)",
            r"(.+?)\s+(?:import|from)\s+(.+)",
        ],
        RelationType.IS_A: [
            r"(.+?)\s+(?:is a|is an|نوع من|هو نوع)\s+(.+)",
            r"(.+?)\s*:\s*(.+)",  # Type annotation style
        ],
        RelationType.USES: [
            r"(.+?)\s+(?:uses|يستخدم|utilizes)\s+(.+)",
            r"(.+?)\s+(?:with|باستخدام)\s+(.+)",
        ],
        RelationType.SIMILAR_TO: [
            r"(.+?)\s+(?:similar to|مشابه لـ|like)\s+(.+)",
            r"(.+?)\s+(?:vs|versus|مقارنة بـ)\s+(.+)",
        ],
    }

    LLM_EXTRACTION_PROMPT = """Analyze the following text and extract relationships between concepts.

Text:
"{content}"

Extract relationships in this format:
{{
  "relations": [
    {{
      "source": "concept1",
      "target": "concept2",
      "relation": "caused|fixed|depends_on|is_a|uses|similar_to|related_to",
      "confidence": 0.0-1.0,
      "evidence": "the text that indicates this relation"
    }}
  ]
}}

Only extract clear, meaningful relationships. Respond with JSON only."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        min_confidence: float = 0.3,
    ):
        self._llm_provider = llm_provider
        self._model = model
        self._min_confidence = min_confidence

        self._compiled_patterns = self._compile_patterns()

        self._logger = logger

    def _compile_patterns(self) -> dict[RelationType, list[re.Pattern]]:
        """تجميع الأنماط"""
        compiled = {}

        for relation, patterns in self.RELATION_PATTERNS.items():
            compiled[relation] = [re.compile(p, re.IGNORECASE) for p in patterns]

        return compiled

    async def extract(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> list[ExtractedRelation]:
        """
        استخراج العلاقات من المحتوى

        Args:
            content: المحتوى النصي
            context: سياق إضافي

        Returns:
            قائمة العلاقات المستخرجة
        """
        relations = []

        pattern_relations = self._extract_with_patterns(content)
        relations.extend(pattern_relations)

        if self._llm_provider:
            llm_relations = await self._extract_with_llm(content)
            relations.extend(llm_relations)

        cooccur_relations = self._extract_co_occurrences(content)
        relations.extend(cooccur_relations)

        unique_relations = self._deduplicate_relations(relations)

        filtered = [r for r in unique_relations if r.confidence >= self._min_confidence]

        self._logger.info(f"Extracted {len(filtered)} relations from content")

        return filtered

    def _extract_with_patterns(self, content: str) -> list[ExtractedRelation]:
        """استخراج بالأنماط"""
        relations = []

        for relation, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(content)

                for match in matches:
                    groups = match.groups()

                    if len(groups) >= 2:
                        source = groups[0].strip()
                        target = groups[1].strip()

                        if len(source) > 3 and len(target) > 3:
                            relations.append(
                                ExtractedRelation(
                                    source=source[:100],
                                    target=target[:100],
                                    relation=relation,
                                    confidence=0.7,
                                    evidence=match.group(0),
                                    context=content[max(0, match.start() - 50) : match.end() + 50],
                                )
                            )

        return relations

    async def _extract_with_llm(self, content: str) -> list[ExtractedRelation]:
        """استخراج باستخدام LLM"""
        if not self._llm_provider:
            return []

        try:
            truncated = content[:2000] if len(content) > 2000 else content

            prompt = self.LLM_EXTRACTION_PROMPT.format(content=truncated)

            response = await self._llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=0.1,
                max_tokens=500,
            )

            return self._parse_llm_response(response)

        except Exception as e:
            self._logger.warning(f"LLM extraction failed: {e}")
            return []

    def _parse_llm_response(self, response: str) -> list[ExtractedRelation]:
        """تحليل استجابة LLM"""
        import json

        relations = []

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                for item in data.get("relations", []):
                    relation_str = item.get("relation", "related_to")

                    try:
                        relation = RelationType(relation_str)
                    except ValueError:
                        relation = RelationType.RELATED_TO

                    relations.append(
                        ExtractedRelation(
                            source=item.get("source", "")[:100],
                            target=item.get("target", "")[:100],
                            relation=relation,
                            confidence=float(item.get("confidence", 0.5)),
                            evidence=item.get("evidence", ""),
                        )
                    )

        except Exception as e:
            self._logger.debug(f"Failed to parse LLM response: {e}")

        return relations

    def _extract_co_occurrences(self, content: str) -> list[ExtractedRelation]:
        """استخراج من التواجد المشترك"""
        relations = []

        words = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", content)

        word_positions: dict[str, list[int]] = {}
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower not in word_positions:
                word_positions[word_lower] = []
            word_positions[word_lower].append(i)

        significant_words = [
            w for w, positions in word_positions.items() if len(positions) >= 1 and len(w) > 4
        ]

        window_size = 10

        for i, word1 in enumerate(significant_words):
            for word2 in significant_words[i + 1 :]:
                pos1 = word_positions[word1]
                pos2 = word_positions[word2]

                min_distance = min(abs(p1 - p2) for p1 in pos1 for p2 in pos2)

                if min_distance <= window_size:
                    confidence = 1.0 - (min_distance / window_size) * 0.5

                    relations.append(
                        ExtractedRelation(
                            source=word1,
                            target=word2,
                            relation=RelationType.RELATED_TO,
                            confidence=confidence * 0.5,
                            evidence=f"Co-occurrence within {min_distance} words",
                        )
                    )

        return relations

    def _deduplicate_relations(
        self,
        relations: list[ExtractedRelation],
    ) -> list[ExtractedRelation]:
        """إزالة التكرارات"""
        seen = set()
        unique = []

        for r in relations:
            key = (
                r.source.lower(),
                r.target.lower(),
                r.relation.value,
            )

            reverse_key = (
                r.target.lower(),
                r.source.lower(),
                r.relation.value,
            )

            if key not in seen and reverse_key not in seen:
                seen.add(key)
                unique.append(r)

        return unique

    def extract_from_episode(
        self,
        episode: dict[str, Any],
    ) -> list[ExtractedRelation]:
        """
        استخراج العلاقات من حلقة

        Args:
            episode: بيانات الحلقة

        Returns:
            قائمة العلاقات
        """
        relations = []

        if "error" in episode and "solution" in episode:
            relations.append(
                ExtractedRelation(
                    source=episode.get("error", "")[:100],
                    target=episode.get("solution", "")[:100],
                    relation=RelationType.FIXED,
                    confidence=0.8,
                    evidence="Episode error-solution pair",
                )
            )

        if "action" in episode and "result" in episode:
            if not episode.get("success", True):
                relations.append(
                    ExtractedRelation(
                        source=episode.get("action", "")[:100],
                        target=episode.get("result", "")[:100],
                        relation=RelationType.CAUSED,
                        confidence=0.7,
                        evidence="Failed action-result pair",
                    )
                )

        if "lessons" in episode:
            lessons = episode.get("lessons", [])
            for lesson in lessons:
                relations.append(
                    ExtractedRelation(
                        source=episode.get("action", "task")[:100],
                        target=lesson[:100] if isinstance(lesson, str) else str(lesson)[:100],
                        relation=RelationType.RELATED_TO,
                        confidence=0.6,
                        evidence="Lesson from episode",
                    )
                )

        return relations
