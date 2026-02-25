"""
Deep Discovery Engine - Type Definitions
========================================

Core data structures for the research system.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("gaap.research.types")


class ETSLevel(Enum):
    """
    Epistemic Trust Score Levels.

    Every source is assigned an ETS based on domain reputation,
    author credibility, and content quality.
    """

    VERIFIED = 1.0  # Official docs, verified GitHub repos
    RELIABLE = 0.7  # Peer-reviewed papers, high-reputation SO
    QUESTIONABLE = 0.5  # Medium articles, tutorials
    UNRELIABLE = 0.3  # Random blogs, AI summaries
    BLACKLISTED = 0.0  # Contradictory or banned domains


class SourceStatus(Enum):
    """Status of a source in the research pipeline."""

    DISCOVERED = auto()
    FETCHED = auto()
    AUDITED = auto()
    VALIDATED = auto()
    STORED = auto()
    FAILED = auto()


class HypothesisStatus(Enum):
    """Status of a hypothesis in verification."""

    UNVERIFIED = auto()
    VERIFIED = auto()
    FALSIFIED = auto()
    CONFLICTED = auto()  # Conflicting evidence


@dataclass
class Source:
    """
    A source of information with full metadata.

    Attributes:
        url: Source URL
        title: Page title
        domain: Domain name for ETS scoring
        ets_score: Epistemic Trust Score (0.0 - 1.0)
        content: Extracted text content
        content_hash: SHA256 hash of content for deduplication
        retrieval_timestamp: When this source was fetched
        citation_count: How many other sources cite this
        author: Content author if available
        publish_date: Publication date if available
        status: Current status in pipeline
        metadata: Additional metadata
    """

    url: str
    title: str = ""
    domain: str = ""
    ets_score: float = 0.5
    content: str | None = None
    content_hash: str = ""
    retrieval_timestamp: datetime = field(default_factory=datetime.now)
    citation_count: int = 0
    author: str | None = None
    publish_date: date | None = None
    status: SourceStatus = SourceStatus.DISCOVERED
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            from urllib.parse import urlparse

            parsed = urlparse(self.url)
            self.domain = parsed.netloc

    def compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        if self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return self.content_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "domain": self.domain,
            "ets_score": self.ets_score,
            "content_hash": self.content_hash,
            "retrieval_timestamp": self.retrieval_timestamp.isoformat(),
            "citation_count": self.citation_count,
            "author": self.author,
            "publish_date": self.publish_date.isoformat() if self.publish_date else None,
            "status": self.status.name,
        }


@dataclass
class Claim:
    """
    A factual claim extracted from a source.

    Attributes:
        text: The claim text
        source: Source where claim was found
        confidence: Extraction confidence
        topic: Related topic/category
    """

    text: str
    source: Source
    confidence: float = 0.8
    topic: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_url": self.source.url,
            "confidence": self.confidence,
            "topic": self.topic,
        }


@dataclass
class Hypothesis:
    """
    A formal hypothesis built from claims.

    The Synthesizer builds hypotheses from claims and
    verifies them through cross-validation.

    Attributes:
        id: Unique hypothesis ID
        statement: The hypothesis statement
        status: Verification status
        supporting_sources: Sources that support this hypothesis
        contradicting_sources: Sources that contradict
        confidence: Overall confidence (0.0 - 1.0)
        verification_timestamp: When verified/falsified
        reasoning: LLM reasoning for the status
    """

    id: str
    statement: str
    status: HypothesisStatus = HypothesisStatus.UNVERIFIED
    supporting_sources: list[Source] = field(default_factory=list)
    contradicting_sources: list[Source] = field(default_factory=list)
    confidence: float = 0.0
    verification_timestamp: datetime | None = None
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_verified(self) -> bool:
        return self.status == HypothesisStatus.VERIFIED

    @property
    def is_falsified(self) -> bool:
        return self.status == HypothesisStatus.FALSIFIED

    @property
    def is_conflicted(self) -> bool:
        return self.status == HypothesisStatus.CONFLICTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "status": self.status.name,
            "supporting_count": len(self.supporting_sources),
            "contradicting_count": len(self.contradicting_sources),
            "confidence": self.confidence,
            "reasoning": self.reasoning[:500] if self.reasoning else None,
        }


@dataclass
class Contradiction:
    """
    A detected contradiction between sources or hypotheses.

    Attributes:
        claim1: First claim/hypothesis
        claim2: Contradicting claim/hypothesis
        source1: Source for first claim
        source2: Source for second claim
        severity: How significant the contradiction is
        resolution: How to resolve (if determined)
    """

    claim1: str
    claim2: str
    source1: Source
    source2: Source
    severity: str = "medium"  # low, medium, high
    resolution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim1": self.claim1[:100],
            "claim2": self.claim2[:100],
            "source1_url": self.source1.url,
            "source2_url": self.source2.url,
            "severity": self.severity,
            "resolution": self.resolution,
        }


@dataclass
class AssociativeTriple:
    """
    Subject-Predicate-Object triple for Knowledge Graph.

    These triples represent extracted knowledge relationships
    that are stored permanently for future queries.

    Attributes:
        subject: The subject entity (e.g., "FastAPI")
        predicate: The relationship (e.g., "supports")
        object: The object entity (e.g., "async/await")
        source: Source where this was extracted
        confidence: Extraction confidence
        created_at: When this was created
    """

    subject: str
    predicate: str
    object: str
    source: Source
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source_url": self.source.url,
            "confidence": self.confidence,
        }

    def to_tuple(self) -> tuple[str, str, str]:
        """Return as simple tuple."""
        return (self.subject, self.predicate, self.object)


@dataclass
class SearchResult:
    """
    Raw search result from web search.

    Attributes:
        url: Result URL
        title: Page title
        snippet: Search result snippet
        rank: Position in search results
        provider: Search provider used
    """

    url: str
    title: str = ""
    snippet: str = ""
    rank: int = 0
    provider: str = "duckduckgo"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_source(self) -> Source:
        """Convert to Source object."""
        return Source(
            url=self.url,
            title=self.title,
            status=SourceStatus.DISCOVERED,
            metadata={"snippet": self.snippet, "rank": self.rank, "provider": self.provider},
        )


@dataclass
class ExtractedContent:
    """
    Content extracted from a URL.

    Attributes:
        url: Source URL
        title: Page title
        content: Clean text content
        author: Author if found
        publish_date: Publication date if found
        links: Links found in content
        code_blocks: Code blocks found
        extraction_success: Whether extraction succeeded
        error: Error message if failed
    """

    url: str
    title: str = ""
    content: str = ""
    author: str | None = None
    publish_date: date | None = None
    links: list[str] = field(default_factory=list)
    code_blocks: list[str] = field(default_factory=list)
    extraction_success: bool = True
    error: str | None = None
    extraction_time_ms: float = 0.0

    def to_source(self, domain: str = "") -> Source:
        """Convert to Source object."""
        source = Source(
            url=self.url,
            title=self.title,
            domain=domain,
            content=self.content,
            author=self.author,
            publish_date=self.publish_date,
            status=SourceStatus.FETCHED,
        )
        source.compute_hash()
        return source


@dataclass
class ResearchMetrics:
    """
    Statistics about a research execution.

    For display in API responses and Web GUI.
    """

    sources_found: int = 0
    sources_fetched: int = 0
    sources_filtered: int = 0
    sources_passed_ets: int = 0
    hypotheses_generated: int = 0
    hypotheses_verified: int = 0
    hypotheses_falsified: int = 0
    hypotheses_conflicted: int = 0
    triples_extracted: int = 0
    citations_followed: int = 0
    contradictions_found: int = 0
    avg_ets_score: float = 0.0
    avg_hypothesis_confidence: float = 0.0
    llm_calls: int = 0
    web_requests: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_found": self.sources_found,
            "sources_fetched": self.sources_fetched,
            "sources_filtered": self.sources_filtered,
            "sources_passed_ets": self.sources_passed_ets,
            "hypotheses_generated": self.hypotheses_generated,
            "hypotheses_verified": self.hypotheses_verified,
            "hypotheses_falsified": self.hypotheses_falsified,
            "hypotheses_conflicted": self.hypotheses_conflicted,
            "triples_extracted": self.triples_extracted,
            "citations_followed": self.citations_followed,
            "contradictions_found": self.contradictions_found,
            "avg_ets_score": f"{self.avg_ets_score:.2f}",
            "avg_hypothesis_confidence": f"{self.avg_hypothesis_confidence:.2f}",
            "llm_calls": self.llm_calls,
            "web_requests": self.web_requests,
        }


@dataclass
class ExecutionStep:
    """
    Step in the research execution trace.

    For debugging and transparency.
    """

    step_name: str
    started_at: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_name": self.step_name,
            "started_at": self.started_at.isoformat(),
            "duration_ms": f"{self.duration_ms:.1f}",
            "input_summary": self.input_summary[:100],
            "output_summary": self.output_summary[:100],
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ResearchFinding:
    """
    Complete research finding for storage.

    This is what gets stored permanently in KnowledgeGraph.
    """

    id: str = ""
    query: str = ""
    sources: list[Source] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    triples: list[AssociativeTriple] = field(default_factory=list)
    contradictions: list[Contradiction] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0
    research_depth: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "sources_count": len(self.sources),
            "hypotheses_count": len(self.hypotheses),
            "triples_count": len(self.triples),
            "contradictions_count": len(self.contradictions),
            "summary": self.summary[:500],
            "confidence": self.confidence,
            "research_depth": self.research_depth,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ResearchResult:
    """
    Complete result for API response.

    Includes everything needed for display and debugging.
    """

    success: bool
    query: str
    config_used: dict[str, Any] = field(default_factory=dict)
    finding: ResearchFinding | None = None
    metrics: ResearchMetrics = field(default_factory=ResearchMetrics)
    execution_trace: list[ExecutionStep] = field(default_factory=list)
    total_time_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "query": self.query,
            "config_used": self.config_used,
            "finding": self.finding.to_dict() if self.finding else None,
            "metrics": self.metrics.to_dict(),
            "execution_trace": [s.to_dict() for s in self.execution_trace],
            "total_time_ms": f"{self.total_time_ms:.1f}",
            "error": self.error,
        }
