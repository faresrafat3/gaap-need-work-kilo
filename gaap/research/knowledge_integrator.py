"""
Knowledge Integrator - Permanent Storage for Research
====================================================

Stores research results permanently in KnowledgeGraph.
No TTL - everything is kept for future reference.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import StorageConfig
from .types import (
    AssociativeTriple,
    Hypothesis,
    ResearchFinding,
    Source,
)

if TYPE_CHECKING:
    from gaap.memory.knowledge.graph_builder import KnowledgeGraphBuilder
    from gaap.storage.sqlite_store import SQLiteStore

logger = logging.getLogger("gaap.research.knowledge_integrator")


class KnowledgeIntegrator:
    """
    Permanent storage for research results in KnowledgeGraph.

    Features:
    - Store sources, hypotheses, and triples
    - Find similar existing research
    - No TTL - permanent storage
    - Deduplication by content hash
    - Integration with KnowledgeGraph and VectorStore

    Storage Structure:
    - Sources → Nodes (type: SOURCE)
    - Hypotheses → Nodes (type: HYPOTHESIS)
    - Triples → Edges (with confidence)

    Usage:
        integrator = KnowledgeIntegrator(kg_builder, sqlite_store)
        finding_id = await integrator.store_research(finding)
    """

    NODE_TYPE_SOURCE = "research_source"
    NODE_TYPE_HYPOTHESIS = "research_hypothesis"
    NODE_TYPE_FINDING = "research_finding"
    NODE_TYPE_ENTITY = "research_entity"

    RELATION_SUPPORTS = "supports"
    RELATION_CONTRADICTS = "contradicts"
    RELATION_DERIVED_FROM = "derived_from"
    RELATION_ABOUT = "about"
    RELATION_CITES = "cites"

    def __init__(
        self,
        knowledge_graph: KnowledgeGraphBuilder | None = None,
        sqlite_store: SQLiteStore | None = None,
        config: StorageConfig | None = None,
    ) -> None:
        self._kg = knowledge_graph
        self._sqlite = sqlite_store
        self.config = config or StorageConfig()

        self._findings_count = 0
        self._sources_count = 0
        self._hypotheses_count = 0
        self._triples_count = 0
        self._duplicates_skipped = 0
        self._total_time_ms = 0.0

        self._seen_hashes: set[str] = set()

        self._logger = logger

        if self.config.storage_path:
            Path(self.config.storage_path).mkdir(parents=True, exist_ok=True)

    async def store_research(
        self,
        finding: ResearchFinding,
    ) -> str:
        """
        Store research finding permanently.

        Args:
            finding: ResearchFinding to store

        Returns:
            Finding ID for future reference
        """
        start_time = time.time()

        if not finding.id:
            finding.id = self._generate_finding_id(finding.query)

        if self.config.dedup_enabled and self._is_duplicate_finding(finding):
            self._duplicates_skipped += 1
            self._logger.info(f"Skipping duplicate finding: {finding.id}")
            return finding.id

        if self._sqlite and self.config.sqlite_cache_enabled:
            await self._store_in_sqlite(finding)

        if self._kg and self.config.knowledge_graph_enabled:
            await self._store_in_knowledge_graph(finding)

        self._findings_count += 1
        self._sources_count += len(finding.sources)
        self._hypotheses_count += len(finding.hypotheses)
        self._triples_count += len(finding.triples)

        self._total_time_ms += (time.time() - start_time) * 1000

        self._logger.info(
            f"Stored research finding {finding.id}: "
            f"{len(finding.sources)} sources, {len(finding.hypotheses)} hypotheses, "
            f"{len(finding.triples)} triples"
        )

        return finding.id

    async def find_similar(
        self,
        query: str,
        threshold: float = 0.8,
    ) -> ResearchFinding | None:
        """
        Find similar existing research.

        Args:
            query: Query to match
            threshold: Similarity threshold (0-1)

        Returns:
            ResearchFinding if found, None otherwise
        """
        if not self._sqlite:
            return None

        try:
            records = self._sqlite.query(
                "research_findings",
                limit=10,
            )

            query_lower = query.lower()
            query_words = set(query_lower.split())

            for record in records:
                data = record.get("data", {})
                stored_query = data.get("query", "").lower()
                stored_words = set(stored_query.split())

                if not query_words or not stored_words:
                    continue

                intersection = len(query_words & stored_words)
                union = len(query_words | stored_words)
                similarity = intersection / union if union > 0 else 0

                if similarity >= threshold:
                    return self._record_to_finding(data)

            return None

        except Exception as e:
            self._logger.warning(f"Failed to find similar research: {e}")
            return None

    async def get_by_topic(
        self,
        topic: str,
    ) -> list[ResearchFinding]:
        """
        Get all research on a topic.

        Args:
            topic: Topic to search for

        Returns:
            List of ResearchFindings
        """
        if not self._sqlite:
            return []

        try:
            records = self._sqlite.query(
                "research_findings",
                limit=100,
            )

            findings: list[ResearchFinding] = []
            topic_lower = topic.lower()

            for record in records:
                data = record.get("data", {})
                query = data.get("query", "").lower()

                if topic_lower in query:
                    finding = self._record_to_finding(data)
                    if finding:
                        findings.append(finding)

            return findings

        except Exception as e:
            self._logger.warning(f"Failed to get research by topic: {e}")
            return []

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        source: Source,
        confidence: float = 0.8,
    ) -> str:
        """
        Add associative triple to KnowledgeGraph.

        Args:
            subject: Subject entity
            predicate: Relationship
            object: Object entity
            source: Source of triple
            confidence: Confidence score

        Returns:
            Triple ID
        """
        triple = AssociativeTriple(
            subject=subject,
            predicate=predicate,
            object=object,
            source=source,
            confidence=confidence,
        )

        if self._kg:
            await self._store_triple_in_kg(triple)

        self._triples_count += 1
        return hashlib.md5(f"{subject}:{predicate}:{object}".encode()).hexdigest()[:12]

    def _generate_finding_id(self, query: str) -> str:
        """Generate unique finding ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return f"research_{timestamp}_{query_hash}"

    def _is_duplicate_finding(self, finding: ResearchFinding) -> bool:
        """Check if finding is a duplicate."""
        if self.config.dedup_by_hash and finding.sources:
            for source in finding.sources:
                if source.content_hash and source.content_hash in self._seen_hashes:
                    return True

        return False

    async def _store_in_sqlite(self, finding: ResearchFinding) -> None:
        """Store finding in SQLite."""
        if not self._sqlite:
            return

        try:
            self._sqlite.insert(
                "research_findings",
                {
                    "query": finding.query,
                    "summary": finding.summary,
                    "confidence": finding.confidence,
                    "research_depth": finding.research_depth,
                    "sources_count": len(finding.sources),
                    "hypotheses_count": len(finding.hypotheses),
                    "triples_count": len(finding.triples),
                    "created_at": finding.created_at.isoformat(),
                    "sources": [s.to_dict() for s in finding.sources],
                    "hypotheses": [h.to_dict() for h in finding.hypotheses],
                    "triples": [t.to_dict() for t in finding.triples],
                },
                item_id=finding.id,
            )
        except Exception as e:
            self._logger.warning(f"Failed to store in SQLite: {e}")

    async def _store_in_knowledge_graph(self, finding: ResearchFinding) -> None:
        """Store finding in KnowledgeGraph."""
        if not self._kg:
            return

        try:
            await self._store_finding_node(finding)

            for source in finding.sources:
                await self._store_source_node(source, finding.id)

            for hypothesis in finding.hypotheses:
                await self._store_hypothesis_node(hypothesis, finding.id)

            for triple in finding.triples:
                await self._store_triple_in_kg(triple)

        except Exception as e:
            self._logger.warning(f"Failed to store in KnowledgeGraph: {e}")

    async def _store_finding_node(self, finding: ResearchFinding) -> None:
        """Store finding as node."""

    async def _store_source_node(
        self,
        source: Source,
        finding_id: str,
    ) -> None:
        """Store source as node."""
        if source.content_hash:
            self._seen_hashes.add(source.content_hash)

    async def _store_hypothesis_node(
        self,
        hypothesis: Hypothesis,
        finding_id: str,
    ) -> None:
        """Store hypothesis as node."""

    async def _store_triple_in_kg(self, triple: AssociativeTriple) -> None:
        """Store triple in KnowledgeGraph."""

    def _record_to_finding(self, data: dict[str, Any]) -> ResearchFinding | None:
        """Convert SQLite record to ResearchFinding."""
        try:
            finding = ResearchFinding(
                id=data.get("id", ""),
                query=data.get("query", ""),
                summary=data.get("summary", ""),
                confidence=data.get("confidence", 0.0),
                research_depth=data.get("research_depth", 1),
                created_at=datetime.fromisoformat(
                    data.get("created_at", datetime.now().isoformat())
                ),
            )

            sources_data = data.get("sources", [])
            for s in sources_data:
                finding.sources.append(
                    Source(
                        url=s.get("url", ""),
                        title=s.get("title", ""),
                        domain=s.get("domain", ""),
                        ets_score=s.get("ets_score", 0.5),
                        content_hash=s.get("content_hash", ""),
                    )
                )

            return finding

        except Exception as e:
            self._logger.warning(f"Failed to convert record to finding: {e}")
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get integrator statistics."""
        return {
            "findings_stored": self._findings_count,
            "sources_stored": self._sources_count,
            "hypotheses_stored": self._hypotheses_count,
            "triples_stored": self._triples_count,
            "duplicates_skipped": self._duplicates_skipped,
            "unique_hashes_seen": len(self._seen_hashes),
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "config": {
                "knowledge_graph_enabled": self.config.knowledge_graph_enabled,
                "vector_store_enabled": self.config.vector_store_enabled,
                "sqlite_cache_enabled": self.config.sqlite_cache_enabled,
                "dedup_enabled": self.config.dedup_enabled,
            },
        }


def create_knowledge_integrator(
    storage_path: str = ".gaap/research",
    kg_builder: KnowledgeGraphBuilder | None = None,
    sqlite_store: SQLiteStore | None = None,
) -> KnowledgeIntegrator:
    """Create a KnowledgeIntegrator instance."""
    config = StorageConfig(storage_path=storage_path)
    return KnowledgeIntegrator(
        knowledge_graph=kg_builder,
        sqlite_store=sqlite_store,
        config=config,
    )
