"""
Base Reranker Module
====================

Abstract base class for all rerankers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RerankResult:
    """
    Result from reranking process.

    Attributes:
        content: The content being ranked
        score: Relevance score (0-1)
        original_score: Original score before reranking
        rank: Final rank position
        source: Where this result came from
        metadata: Additional metadata
        reasoning: Why this score was given (optional, for LLM reranker)
    """

    content: str
    score: float
    original_score: float = 0.0
    rank: int = 0
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content[:500],
            "score": self.score,
            "original_score": self.original_score,
            "rank": self.rank,
            "source": self.source,
            "metadata": self.metadata,
            "reasoning": self.reasoning,
        }


@dataclass
class RerankRequest:
    """
    Request for reranking.

    Attributes:
        query: The original query
        candidates: List of candidate contents to rerank
        top_k: Number of results to return
        context: Additional context for better reranking
        metadata_list: Metadata for each candidate
    """

    query: str
    candidates: list[str]
    top_k: int = 5
    context: dict[str, Any] = field(default_factory=dict)
    metadata_list: list[dict[str, Any]] = field(default_factory=list)


class BaseReranker(ABC):
    """
    Abstract base class for rerankers.

    Rerankers take a list of candidates and re-score them
    based on actual relevance to the query.
    """

    def __init__(self, name: str = "base"):
        self.name = name
        self._last_rerank_time_ms: float = 0.0
        self._total_reranks: int = 0

    @abstractmethod
    async def rerank(self, request: RerankRequest) -> list[RerankResult]:
        """
        Rerank candidates based on relevance to query.

        Args:
            request: The rerank request with query and candidates

        Returns:
            List of RerankResult sorted by score descending
        """
        pass

    def _prepare_candidates(
        self,
        candidates: list[str],
        metadata_list: list[dict[str, Any]] | None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Prepare candidates with their metadata."""
        if metadata_list is None:
            metadata_list = []

        result = []
        for i, content in enumerate(candidates):
            meta = metadata_list[i] if i < len(metadata_list) else {}
            result.append((content, meta))

        return result

    def _sort_and_rank(self, results: list[RerankResult], top_k: int) -> list[RerankResult]:
        """Sort results by score and assign ranks."""
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)

        for i, result in enumerate(sorted_results[:top_k]):
            result.rank = i + 1

        return sorted_results[:top_k]

    def get_stats(self) -> dict[str, Any]:
        """Get reranker statistics."""
        return {
            "name": self.name,
            "total_reranks": self._total_reranks,
            "last_rerank_time_ms": self._last_rerank_time_ms,
        }
