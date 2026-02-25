"""
LLM Reranker Module
===================

Intelligent reranking using LLM for complex queries.
Provides reasoning for each ranking decision.
"""

import logging
import time
from typing import Any, Protocol, runtime_checkable

from .base import BaseReranker, RerankRequest, RerankResult

logger = logging.getLogger("gaap.memory.rerankers.llm")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class LLMReranker(BaseReranker):
    """
    LLM-powered reranker for intelligent relevance assessment.

    Uses LLM to:
    - Understand complex queries
    - Provide reasoning for rankings
    - Handle ambiguous cases
    - Consider context deeply

    Best used when:
    - Cross-encoder confidence is low
    - Query is complex or ambiguous
    - Need explanation for ranking
    """

    RERANK_PROMPT = """You are a relevance ranking expert. Given a query and a list of documents, rank them by relevance.

## Query
{query}

## Context
{context}

## Documents
{documents}

## Task
For each document, provide:
1. A relevance score from 0.0 to 1.0
2. A brief reason for the score

Respond in this exact JSON format:
{{
  "rankings": [
    {{"id": 0, "score": 0.95, "reason": "Directly answers the query"}},
    {{"id": 1, "score": 0.3, "reason": "Only tangentially related"}},
    ...
  ]
}}

Only respond with the JSON, no other text."""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        max_candidates: int = 10,
        temperature: float = 0.1,
    ):
        super().__init__(name="llm")
        self._llm_provider = llm_provider
        self._model = model
        self._max_candidates = max_candidates
        self._temperature = temperature

    @property
    def available(self) -> bool:
        return self._llm_provider is not None

    async def rerank(self, request: RerankRequest) -> list[RerankResult]:
        """
        Rerank candidates using LLM.

        Process:
        1. Prepare prompt with query and candidates
        2. Get LLM response with scores and reasoning
        3. Parse and return ranked results
        """
        start_time = time.time()

        if not request.candidates:
            return []

        candidates = request.candidates[: self._max_candidates]

        candidates_with_meta = self._prepare_candidates(
            candidates,
            request.metadata_list if request.metadata_list else None,
        )

        if not self._llm_provider:
            results = await self._fallback_rerank(request.query, candidates_with_meta)
        else:
            results = await self._llm_rerank(
                request.query,
                candidates_with_meta,
                request.context,
            )

        final_results = self._sort_and_rank(results, request.top_k)

        self._last_rerank_time_ms = (time.time() - start_time) * 1000
        self._total_reranks += 1

        logger.debug(
            f"LLM reranked {len(candidates_with_meta)} candidates in "
            f"{self._last_rerank_time_ms:.2f}ms"
        )

        return final_results

    async def _llm_rerank(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any]]],
        context: dict[str, Any],
    ) -> list[RerankResult]:
        """Rerank using LLM."""
        results = []

        context_str = self._format_context(context)
        documents_str = self._format_documents(candidates)

        prompt = self.RERANK_PROMPT.format(
            query=query,
            context=context_str,
            documents=documents_str,
        )

        try:
            if self._llm_provider is None:
                raise ValueError("LLM provider is not available")
            response = await self._llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=self._temperature,
                max_tokens=500,
            )

            rankings = self._parse_rankings(response)

            for i, (content, meta) in enumerate(candidates):
                ranking = rankings.get(i, {"score": 0.5, "reason": "No ranking provided"})

                results.append(
                    RerankResult(
                        content=content,
                        score=ranking["score"],
                        original_score=ranking["score"],
                        source="llm",
                        metadata=meta,
                        reasoning=ranking.get("reason"),
                    )
                )

        except Exception as e:
            logger.error(f"LLM reranking failed: {e}")
            return await self._fallback_rerank(query, candidates)

        return results

    async def _fallback_rerank(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any]]],
    ) -> list[RerankResult]:
        """Fallback when LLM is not available."""
        results = []

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for content, meta in candidates:
            content_lower = content.lower()
            content_words = set(content_lower.split())

            exact_match = 1.0 if query_lower in content_lower else 0.0

            word_overlap = len(query_words & content_words) / len(query_words) if query_words else 0

            meta_relevance = self._check_meta_relevance(meta, query_lower)

            score = exact_match * 0.5 + word_overlap * 0.3 + meta_relevance * 0.2

            results.append(
                RerankResult(
                    content=content,
                    score=score,
                    original_score=score,
                    source="fallback",
                    metadata=meta,
                    reasoning="Fallback scoring (no LLM available)",
                )
            )

        return results

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context for prompt."""
        if not context:
            return "No additional context available"

        parts = []
        for key, value in context.items():
            parts.append(f"- {key}: {str(value)[:200]}")

        return "\n".join(parts)

    def _format_documents(self, candidates: list[tuple[str, dict[str, Any]]]) -> str:
        """Format documents for prompt."""
        parts = []
        for i, (content, meta) in enumerate(candidates):
            truncated = content[:300] + "..." if len(content) > 300 else content
            parts.append(f"[{i}] {truncated}")

        return "\n\n".join(parts)

    def _parse_rankings(self, response: str) -> dict[int, dict[str, Any]]:
        """Parse LLM response to get rankings."""
        import json

        rankings = {}

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                for item in data.get("rankings", []):
                    idx = item.get("id", 0)
                    score = float(item.get("score", 0.5))
                    reason = item.get("reason", "")

                    score = max(0.0, min(1.0, score))

                    rankings[idx] = {
                        "score": score,
                        "reason": reason,
                    }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")

        except Exception as e:
            logger.warning(f"Error parsing rankings: {e}")

        return rankings

    def _check_meta_relevance(self, meta: dict[str, Any], query: str) -> float:
        """Check if metadata is relevant to query."""
        if not meta:
            return 0.0

        query_words = set(query.split())

        relevant_fields = ["type", "category", "domain", "tags", "topic"]

        matches = 0
        total = 0

        for field in relevant_fields:
            if field in meta:
                total += 1
                value = str(meta[field]).lower()
                if any(word in value for word in query_words):
                    matches += 1

        return matches / total if total > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "model": self._model,
                "available": self.available,
                "max_candidates": self._max_candidates,
            }
        )
        return stats
