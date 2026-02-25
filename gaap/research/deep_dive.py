"""
Deep Dive - Citation Mapping and Cross-Validation
================================================

Deep exploration protocol for research.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .types import (
    Source,
    SourceStatus,
    Hypothesis,
    HypothesisStatus,
    AssociativeTriple,
    ResearchFinding,
    Contradiction,
)
from .config import DeepDiveConfig

from .web_fetcher import WebFetcher
from .content_extractor import ContentExtractor
from .source_auditor import SourceAuditor
from .synthesizer import Synthesizer

logger = logging.getLogger("gaap.research.deep_dive")


@dataclass
class DeepDiveResult:
    """Result of deep dive exploration."""

    sources: list[Source]
    primary_sources: list[Source]
    citations_followed: int
    cross_validation_score: float
    exploration_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_count": len(self.sources),
            "primary_sources_count": len(self.primary_sources),
            "citations_followed": self.citations_followed,
            "cross_validation_score": f"{self.cross_validation_score:.2f}",
            "exploration_time_ms": f"{self.exploration_time_ms:.1f}",
        }


class DeepDive:
    """
    Deep exploration protocol for research.

    Protocol:
    1. Exploration: Get top N results from web search
    2. Citation Mapping: Follow citations to primary sources
    3. Cross-Validation: Compare primary vs secondary sources
    4. Knowledge Extraction: Build hypotheses and triples

    Depth Levels:
    - 1: Basic search + content extraction
    - 2: + Citation following
    - 3: + Cross-validation + hypothesis building
    - 4+: + Recursive exploration of related topics
    - 5: Full deep dive with all features

    Usage:
        deep_dive = DeepDive(config)
        result = await deep_dive.explore("FastAPI async")
    """

    def __init__(
        self,
        config: DeepDiveConfig | None = None,
        web_fetcher: WebFetcher | None = None,
        content_extractor: ContentExtractor | None = None,
        source_auditor: SourceAuditor | None = None,
        synthesizer: Synthesizer | None = None,
    ) -> None:
        self.config = config or DeepDiveConfig()
        self._web_fetcher = web_fetcher
        self._content_extractor = content_extractor
        self._source_auditor = source_auditor
        self._synthesizer = synthesizer

        self._explorations_count = 0
        self._citations_followed = 0
        self._cross_validations = 0
        self._total_time_ms = 0.0

        self._logger = logger

    async def explore(
        self,
        query: str,
        depth: int | None = None,
        initial_sources: list[Source] | None = None,
    ) -> DeepDiveResult:
        """
        Execute deep dive exploration.

        Args:
            query: Research query
            depth: Exploration depth (1-5)
            initial_sources: Pre-existing sources to include

        Returns:
            DeepDiveResult with all discovered sources
        """
        start_time = time.time()
        depth = min(depth or self.config.default_depth, self.config.max_depth)

        self._explorations_count += 1
        self._logger.info(f"Starting deep dive: '{query[:50]}...' at depth {depth}")

        try:
            if initial_sources:
                sources = initial_sources
            else:
                sources = await self._search_sources(query)

            if depth >= 2:
                sources = await self._follow_citations(sources)

            if depth >= 3 and self._synthesizer:
                sources = await self._cross_validate_sources(sources)

            if depth >= 4:
                related_sources = await self._explore_related_topics(query, sources)
                sources.extend(related_sources)

            primary_sources = self._identify_primary_sources(sources)

            exploration_time_ms = (time.time() - start_time) * 1000
            self._total_time_ms += exploration_time_ms

            return DeepDiveResult(
                sources=sources,
                primary_sources=primary_sources,
                citations_followed=self._citations_followed,
                cross_validation_score=self._calculate_cross_validation_score(sources),
                exploration_time_ms=exploration_time_ms,
            )

        except Exception as e:
            self._logger.error(f"Deep dive failed: {e}")
            return DeepDiveResult(
                sources=[],
                primary_sources=[],
                citations_followed=0,
                cross_validation_score=0.0,
                exploration_time_ms=(time.time() - start_time) * 1000,
            )

    async def _search_sources(self, query: str) -> list[Source]:
        """Search for initial sources"""
        if not self._web_fetcher:
            return []

        results = await self._web_fetcher.search(query)
        sources = [r.to_source() for r in results[: self.config.max_sources_per_depth]]
        return sources

    async def _follow_citations(
        self,
        sources: list[Source],
    ) -> list[Source]:
        """Follow citations to primary sources"""
        if not self._content_extractor:
            return sources

        all_sources = list(sources)
        citation_urls: set[str] = set()

        for source in sources:
            if not source.content:
                continue

            content = await self._content_extractor.extract(source.url)
            if content.links:
                for link in content.links[: self.config.max_citations_to_follow]:
                    if self._is_valid_citation_link(link, source.url):
                        citation_urls.add(link)

        for url in list(citation_urls)[: self.config.max_citations_to_follow]:
            try:
                extracted = await self._content_extractor.extract(url)
                if extracted.extraction_success:
                    new_source = extracted.to_source(self._extract_domain(url))
                    all_sources.append(new_source)
                    self._citations_followed += 1
            except Exception as e:
                self._logger.warning(f"Failed to follow citation {url}: {e}")

        return all_sources

    def _is_valid_citation_link(self, link: str, source_url: str) -> bool:
        """Check if a link is a valid citation to follow"""
        parsed = re.search(r"https?://([^/]+)", link)
        if not parsed:
            return False

        domain = parsed.group(1)
        invalid_domains = {"google.com", "bing.com", "yahoo.com", "duckduckgo.com"}
        if domain in invalid_domains:
            return False

        if link == source_url:
            return False

        return True

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        return parsed.netloc

    async def _cross_validate_sources(
        self,
        sources: list[Source],
    ) -> list[Source]:
        """Cross-validate sources for consistency"""
        if not self._synthesizer or len(sources) < 2:
            return sources

        self._cross_validations += 1

        validated_sources = []

        for source in sources:
            if not source.content:
                validated_sources.append(source)
                continue

            try:
                claims = await self._synthesizer.extract_claims(
                    source.content,
                    source,
                    max_claims=2,
                )
                if claims:
                    source.metadata["claims"] = [c.text for c in claims]
                validated_sources.append(source)
            except Exception as e:
                self._logger.warning(f"Failed to validate source {source.url}: {e}")
                validated_sources.append(source)

        return validated_sources

    async def _explore_related_topics(
        self,
        original_query: str,
        sources: list[Source],
    ) -> list[Source]:
        """Explore topics related to the original query"""
        if not self._web_fetcher:
            return []

        related_queries = self._generate_related_queries(original_query, sources)

        related_sources = []

        for query in related_queries[:3]:
            try:
                results = await self._web_fetcher.search(query, max_results=5)
                for r in results:
                    related_sources.append(r.to_source())
            except Exception as e:
                self._logger.warning(f"Failed related search '{query}': {e}")

        return related_sources

    def _generate_related_queries(
        self,
        original_query: str,
        sources: list[Source],
    ) -> list[str]:
        """Generate related search queries"""
        queries = []

        words = original_query.lower().split()
        if len(words) > 1:
            queries.append(f"{words[0]} vs {words[-1]}")
            queries.append(f"{original_query} tutorial")
            queries.append(f"{original_query} examples")
            queries.append(f"best practices {words[0]}")

        return queries[:5]

    def _identify_primary_sources(
        self,
        sources: list[Source],
    ) -> list[Source]:
        """Identify primary (original) sources"""
        primary = []

        for source in sources:
            if source.citation_count > 2:
                primary.append(source)
            elif source.ets_score >= 0.8:
                primary.append(source)

        return primary

    def _calculate_cross_validation_score(
        self,
        sources: list[Source],
    ) -> float:
        """Calculate cross-validation consistency score"""
        if not sources:
            return 0.0

        with_claims = sum(1 for s in sources if s.metadata.get("claims"))
        return with_claims / len(sources)

    def get_stats(self) -> dict[str, Any]:
        """Get deep dive statistics"""
        return {
            "explorations_count": self._explorations_count,
            "citations_followed": self._citations_followed,
            "cross_validations": self._cross_validations,
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "config": {
                "default_depth": self.config.default_depth,
                "max_depth": self.config.max_depth,
                "citation_follow_depth": self.config.citation_follow_depth,
                "max_sources_per_depth": self.config.max_sources_per_depth,
            },
        }


def create_deep_dive(
    default_depth: int = 3,
    max_sources: int = 20,
) -> DeepDive:
    """Create a DeepDive instance"""
    config = DeepDiveConfig(
        default_depth=default_depth,
        max_sources_per_depth=max_sources,
    )
    return DeepDive(config)
