"""
Deep Discovery Engine - Main Orchestrator
========================================

Main engine orchestrating all research components.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .types import (
    Source,
    Hypothesis,
    AssociativeTriple,
    ResearchFinding,
    ResearchResult,
    ResearchMetrics,
    ExecutionStep,
    HypothesisStatus,
)
from .config import DDEConfig
from .web_fetcher import WebFetcher
from .content_extractor import ContentExtractor
from .source_auditor import SourceAuditor
from .synthesizer import Synthesizer
from .deep_dive import DeepDive
from .knowledge_integrator import KnowledgeIntegrator

if TYPE_CHECKING:
    from gaap.core.base import BaseProvider
    from gaap.memory.knowledge.graph_builder import KnowledgeGraphBuilder
    from gaap.storage.sqlite_store import SQLiteStore

logger = logging.getLogger("gaap.research.engine")


class DeepDiscoveryEngine:
    """
    Main orchestrator for Deep Discovery Engine.

    Coordinates all research components:
    - WebFetcher: Search and fetch content
    - SourceAuditor: ETS scoring
    - ContentExtractor: Clean text extraction
    - Synthesizer: Hypothesis building
    - DeepDive: Citation mapping
    - KnowledgeIntegrator: Permanent storage

    Usage:
        config = DDEConfig(research_depth=3)
        engine = DeepDiscoveryEngine(config=config, llm_provider=provider)

        result = await engine.research("FastAPI async best practices")

        if result.success:
            print(f"Found {len(result.finding.sources)} sources")
            print(f"Built {len(result.finding.hypotheses)} hypotheses")
    """

    def __init__(
        self,
        config: DDEConfig | None = None,
        llm_provider: BaseProvider | None = None,
        knowledge_graph: KnowledgeGraphBuilder | None = None,
        sqlite_store: SQLiteStore | None = None,
    ) -> None:
        self.config = config or DDEConfig()
        self._llm_provider = llm_provider

        self._web_fetcher = WebFetcher(self.config.web_fetcher)
        self._content_extractor = ContentExtractor(self.config.content_extractor)
        self._source_auditor = SourceAuditor(self.config.source_audit)
        self._synthesizer = Synthesizer(llm_provider, self.config.synthesizer)
        self._deep_dive = DeepDive(
            self.config.deep_dive,
            self._web_fetcher,
            self._content_extractor,
            self._source_auditor,
            self._synthesizer,
        )
        self._knowledge_integrator = KnowledgeIntegrator(
            knowledge_graph,
            sqlite_store,
            self.config.storage,
        )

        self._research_count = 0
        self._cache_hits = 0
        self._total_time_ms = 0.0

        self._logger = logger

    async def research(
        self,
        query: str,
        depth: int | None = None,
        config_override: dict[str, Any] | None = None,
        force_fresh: bool = False,
    ) -> ResearchResult:
        """
        Execute full research protocol.

        Steps:
        1. Check cache/KG for existing research
        2. Web search for sources
        3. Audit sources (ETS scoring)
        4. Deep dive (citation mapping, cross-validation)
        5. Build and verify hypotheses
        6. Extract associative triples
        7. Store in KnowledgeGraph
        8. Return findings

        Args:
            query: Research query
            depth: Override research depth
            config_override: Override specific config values
            force_fresh: Skip cache and do fresh research

        Returns:
            ResearchResult with findings and metrics
        """
        start_time = time.time()
        execution_trace: list[ExecutionStep] = []
        metrics = ResearchMetrics()

        effective_config = self.config
        if config_override:
            effective_config = DDEConfig.from_dict(
                {
                    **self.config.to_dict(),
                    **config_override,
                }
            )

        depth = depth or effective_config.research_depth

        self._logger.info(f"Starting research: '{query[:50]}...' at depth {depth}")

        try:
            if self.config.check_existing_research and not force_fresh:
                step = ExecutionStep(step_name="cache_check", input_summary=query)
                existing = await self._knowledge_integrator.find_similar(query)
                step.duration_ms = (time.time() - start_time) * 1000
                execution_trace.append(step)

                if existing:
                    self._cache_hits += 1
                    self._logger.info(f"Cache hit for query: {query[:50]}")

                    return ResearchResult(
                        success=True,
                        query=query,
                        config_used=effective_config.to_dict(),
                        finding=existing,
                        metrics=metrics,
                        execution_trace=execution_trace,
                        total_time_ms=(time.time() - start_time) * 1000,
                    )

            step = ExecutionStep(step_name="web_search", input_summary=query)
            search_results = await self._web_fetcher.search(query)
            metrics.web_requests += 1
            metrics.sources_found = len(search_results)
            step.output_summary = f"Found {len(search_results)} results"
            step.duration_ms = (time.time() - start_time) * 1000
            execution_trace.append(step)

            sources = [r.to_source() for r in search_results]

            step = ExecutionStep(
                step_name="content_extraction", input_summary=f"{len(sources)} sources"
            )

            async def extract_content(source: Source) -> Source:
                content = await self._content_extractor.extract(source.url)
                if content.extraction_success:
                    source.content = content.content
                    source.title = content.title or source.title
                    source.author = content.author
                    if content.publish_date:
                        source.publish_date = content.publish_date
                return source

            gather_results = await asyncio.gather(
                *[extract_content(s) for s in sources[: effective_config.max_total_sources]],
                return_exceptions=True,
            )
            sources = [s for s in gather_results if isinstance(s, Source)]
            metrics.sources_fetched = len([s for s in sources if s.content])
            step.output_summary = f"Extracted {metrics.sources_fetched} sources"
            step.duration_ms = (time.time() - start_time) * 1000
            execution_trace.append(step)

            step = ExecutionStep(
                step_name="source_auditing", input_summary=f"{len(sources)} sources"
            )
            passed, filtered = self._source_auditor.audit_batch(sources)
            sources = passed
            metrics.sources_filtered = len(filtered)
            metrics.sources_passed_ets = len(passed)
            if sources:
                metrics.avg_ets_score = sum(s.ets_score for s in sources) / len(sources)
            step.output_summary = f"Passed: {len(passed)}, Filtered: {len(filtered)}"
            step.duration_ms = (time.time() - start_time) * 1000
            execution_trace.append(step)

            if depth >= 2:
                step = ExecutionStep(step_name="deep_dive", input_summary=f"{len(sources)} sources")
                dive_result = await self._deep_dive.explore(query, depth, sources)
                sources = dive_result.sources
                metrics.citations_followed = dive_result.citations_followed
                step.output_summary = f"Expanded to {len(sources)} sources"
                step.duration_ms = (time.time() - start_time) * 1000
                execution_trace.append(step)

            hypotheses: list[Hypothesis] = []
            if self._synthesizer and sources:
                step = ExecutionStep(
                    step_name="hypothesis_building", input_summary=f"{len(sources)} sources"
                )

                all_claims = []
                for source in sources[:10]:
                    if source.content:
                        claims = await self._synthesizer.extract_claims(source.content, source)
                        all_claims.extend(claims)
                        metrics.llm_calls += 1

                for claim in all_claims[: effective_config.max_total_hypotheses]:
                    hypothesis = await self._synthesizer.build_hypothesis(claim, sources)
                    hypotheses.append(hypothesis)
                    metrics.llm_calls += 1

                metrics.hypotheses_generated = len(hypotheses)
                step.output_summary = f"Generated {len(hypotheses)} hypotheses"
                step.duration_ms = (time.time() - start_time) * 1000
                execution_trace.append(step)

                if effective_config.synthesizer.cross_validate_enabled:
                    step = ExecutionStep(
                        step_name="hypothesis_verification",
                        input_summary=f"{len(hypotheses)} hypotheses",
                    )

                    for hypothesis in hypotheses:
                        verified = await self._synthesizer.verify_hypothesis(hypothesis, sources)
                        metrics.llm_calls += 1

                        if verified.status == HypothesisStatus.VERIFIED:
                            metrics.hypotheses_verified += 1
                        elif verified.status == HypothesisStatus.FALSIFIED:
                            metrics.hypotheses_falsified += 1
                        elif verified.status == HypothesisStatus.CONFLICTED:
                            metrics.hypotheses_conflicted += 1

                    if hypotheses:
                        metrics.avg_hypothesis_confidence = sum(
                            h.confidence for h in hypotheses
                        ) / len(hypotheses)

                    step.output_summary = f"Verified: {metrics.hypotheses_verified}, Falsified: {metrics.hypotheses_falsified}"
                    step.duration_ms = (time.time() - start_time) * 1000
                    execution_trace.append(step)

            triples: list[AssociativeTriple] = []
            if self._synthesizer and sources:
                step = ExecutionStep(
                    step_name="triple_extraction", input_summary=f"{len(sources)} sources"
                )

                for source in sources[:10]:
                    if source.content:
                        source_triples = await self._synthesizer.extract_triples(
                            source.content, source
                        )
                        triples.extend(source_triples)
                        metrics.llm_calls += 1

                seen = set()
                unique_triples = []
                for t in triples:
                    key = t.to_tuple()
                    if key not in seen:
                        seen.add(key)
                        unique_triples.append(t)
                triples = unique_triples

                metrics.triples_extracted = len(triples)
                step.output_summary = f"Extracted {len(triples)} triples"
                step.duration_ms = (time.time() - start_time) * 1000
                execution_trace.append(step)

            finding = ResearchFinding(
                query=query,
                sources=sources,
                hypotheses=hypotheses,
                triples=triples,
                summary=self._generate_summary(sources, hypotheses),
                confidence=metrics.avg_hypothesis_confidence,
                research_depth=depth,
            )

            step = ExecutionStep(step_name="storage", input_summary=f"Finding: {finding.id}")
            await self._knowledge_integrator.store_research(finding)
            step.output_summary = f"Stored finding {finding.id}"
            step.duration_ms = (time.time() - start_time) * 1000
            execution_trace.append(step)

            self._research_count += 1
            total_time = (time.time() - start_time) * 1000
            self._total_time_ms += total_time

            self._logger.info(
                f"Research complete: {len(sources)} sources, "
                f"{len(hypotheses)} hypotheses, {len(triples)} triples "
                f"in {total_time:.0f}ms"
            )

            return ResearchResult(
                success=True,
                query=query,
                config_used=effective_config.to_dict(),
                finding=finding,
                metrics=metrics,
                execution_trace=execution_trace,
                total_time_ms=total_time,
            )

        except Exception as e:
            self._logger.error(f"Research failed: {e}")
            return ResearchResult(
                success=False,
                query=query,
                config_used=effective_config.to_dict(),
                metrics=metrics,
                execution_trace=execution_trace,
                total_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

    async def quick_search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[Source]:
        """
        Quick search without deep analysis.

        Args:
            query: Search query
            max_results: Maximum results

        Returns:
            List of Sources
        """
        results = await self._web_fetcher.search(query, max_results=max_results)
        return [r.to_source() for r in results]

    def _generate_summary(
        self,
        sources: list[Source],
        hypotheses: list[Hypothesis],
    ) -> str:
        """Generate research summary."""
        verified = [h for h in hypotheses if h.status == HypothesisStatus.VERIFIED]

        if not sources:
            return "No sources found for this query."

        summary_parts = [
            f"Research found {len(sources)} sources with average ETS score of "
            f"{sum(s.ets_score for s in sources) / len(sources):.2f}.",
        ]

        if verified:
            summary_parts.append(
                f"Verified {len(verified)} hypotheses with confidence "
                f"{sum(h.confidence for h in verified) / len(verified):.0%}."
            )

        return " ".join(summary_parts)

    def update_config(self, new_config: DDEConfig) -> None:
        """Update engine configuration."""
        self.config = new_config
        self._web_fetcher.config = new_config.web_fetcher
        self._content_extractor.config = new_config.content_extractor
        self._source_auditor.config = new_config.source_audit
        self._synthesizer.config = new_config.synthesizer
        self._deep_dive.config = new_config.deep_dive
        self._knowledge_integrator.config = new_config.storage
        self._logger.info("Configuration updated")

    def get_config(self) -> DDEConfig:
        """Get current configuration."""
        return self.config

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics."""
        return {
            "research_count": self._research_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": f"{self._cache_hits / max(1, self._research_count + self._cache_hits):.1%}",
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "avg_time_per_research_ms": f"{self._total_time_ms / max(1, self._research_count):.1f}",
            "components": {
                "web_fetcher": self._web_fetcher.get_stats(),
                "source_auditor": self._source_auditor.get_stats(),
                "content_extractor": self._content_extractor.get_stats(),
                "synthesizer": self._synthesizer.get_stats(),
                "deep_dive": self._deep_dive.get_stats(),
                "knowledge_integrator": self._knowledge_integrator.get_stats(),
            },
        }

    async def close(self) -> None:
        """Close all resources."""
        if hasattr(self._web_fetcher, "close"):
            await self._web_fetcher.close()


async def create_engine(
    config: DDEConfig | None = None,
    llm_provider: BaseProvider | None = None,
) -> DeepDiscoveryEngine:
    """Create a DeepDiscoveryEngine instance."""
    return DeepDiscoveryEngine(config=config, llm_provider=llm_provider)
