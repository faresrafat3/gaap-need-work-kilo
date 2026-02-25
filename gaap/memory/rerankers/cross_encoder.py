"""
Cross-Encoder Reranker Module
=============================

Fast reranking using cross-encoder models.
Uses sentence-transformers for accurate relevance scoring.
"""

import logging
import time
from typing import Any

from .base import BaseReranker, RerankRequest, RerankResult

logger = logging.getLogger("gaap.memory.rerankers.cross_encoder")


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder based reranker.

    Uses a cross-encoder model to score query-document pairs.
    More accurate than bi-encoder but slower.

    Features:
    - Uses sentence-transformers cross-encoder
    - Fallback to similarity-based scoring
    - Batch processing for efficiency
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 512,
        batch_size: int = 8,
    ):
        super().__init__(name="cross_encoder")
        self.model_name = model_name or self.DEFAULT_MODEL
        self.max_length = max_length
        self.batch_size = batch_size

        self._model: Any = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
            )
            self._available = True
            logger.info(f"CrossEncoderReranker initialized with {self.model_name}")

        except ImportError:
            logger.warning(
                "sentence_transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._available = False

        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder model: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def rerank(self, request: RerankRequest) -> list[RerankResult]:
        """
        Rerank candidates using cross-encoder.

        Process:
        1. Create query-document pairs
        2. Score each pair with cross-encoder
        3. Sort by score and return top_k
        """
        start_time = time.time()

        if not request.candidates:
            return []

        candidates_with_meta = self._prepare_candidates(
            request.candidates,
            request.metadata_list if request.metadata_list else None,
        )

        if self._available and self._model:
            results = await self._rerank_with_model(request.query, candidates_with_meta)
        else:
            results = await self._rerank_with_fallback(request.query, candidates_with_meta)

        final_results = self._sort_and_rank(results, request.top_k)

        self._last_rerank_time_ms = (time.time() - start_time) * 1000
        self._total_reranks += 1

        logger.debug(
            f"Reranked {len(candidates_with_meta)} candidates in {self._last_rerank_time_ms:.2f}ms"
        )

        return final_results

    async def _rerank_with_model(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any]]],
    ) -> list[RerankResult]:
        """Rerank using the cross-encoder model."""
        results = []

        pairs = [[query, content] for content, _ in candidates]

        try:
            scores = self._model.predict(pairs, batch_size=self.batch_size)

            if hasattr(scores, "tolist"):
                scores = scores.tolist()

            for i, (content, meta) in enumerate(candidates):
                score = float(scores[i]) if i < len(scores) else 0.0
                normalized_score = self._normalize_score(score)

                results.append(
                    RerankResult(
                        content=content,
                        score=normalized_score,
                        original_score=normalized_score,
                        source="cross_encoder",
                        metadata=meta,
                    )
                )

        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            return await self._rerank_with_fallback(query, candidates)

        return results

    async def _rerank_with_fallback(
        self,
        query: str,
        candidates: list[tuple[str, dict[str, Any]]],
    ) -> list[RerankResult]:
        """Fallback reranking using simple similarity."""
        results = []

        query_words = set(query.lower().split())

        for content, meta in candidates:
            content_words = set(content.lower().split())

            intersection = len(query_words & content_words)
            union = len(query_words | content_words)
            jaccard = intersection / union if union > 0 else 0.0

            query_coverage = intersection / len(query_words) if query_words else 0.0
            content_coverage = intersection / len(content_words) if content_words else 0.0

            score = jaccard * 0.4 + query_coverage * 0.4 + content_coverage * 0.2

            results.append(
                RerankResult(
                    content=content,
                    score=score,
                    original_score=score,
                    source="fallback_similarity",
                    metadata=meta,
                )
            )

        return results

    def _normalize_score(self, score: float) -> float:
        """Normalize cross-encoder score to 0-1 range."""
        if score <= -10:
            return 0.0
        elif score >= 10:
            return 1.0
        else:
            return max(0.0, min(1.0, (score + 10) / 20))

    def get_stats(self) -> dict[str, Any]:
        stats = super().get_stats()
        stats.update(
            {
                "model_name": self.model_name,
                "available": self._available,
                "max_length": self.max_length,
            }
        )
        return stats
