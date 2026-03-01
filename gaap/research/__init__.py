"""
Deep Discovery Engine (DDE) - Spec 17
=====================================

Inference-First research system that builds a provable knowledge base.

Components:
- WebFetcher: Web search with multiple providers
- SourceAuditor: Epistemic Trust Score (ETS) evaluation
- ContentExtractor: Clean text extraction from URLs
- Synthesizer: Hypothesis building and LLM verification
- DeepDive: Citation mapping and cross-validation
- KnowledgeIntegrator: Permanent storage in KnowledgeGraph
- DeepDiscoveryEngine: Main orchestrator

Usage:
    from gaap.research import DeepDiscoveryEngine, DDEConfig

    config = DDEConfig(research_depth=3)
    engine = DeepDiscoveryEngine(config=config, llm_provider=provider)

    result = await engine.research("FastAPI async best practices")
"""

from .config import (
    ContentExtractorConfig,
    DDEConfig,
    DeepDiveConfig,
    LLMValidatorConfig,
    SourceAuditConfig,
    StorageConfig,
    SynthesizerConfig,
    WebFetcherConfig,
)
from .engine import DeepDiscoveryEngine, create_engine
from .source_auditor import create_source_auditor
from .synthesizer import create_synthesizer
from .types import (
    AssociativeTriple,
    Claim,
    Contradiction,
    ETSLevel,
    ExecutionStep,
    ExtractedContent,
    Hypothesis,
    HypothesisStatus,
    ResearchFinding,
    ResearchMetrics,
    ResearchResult,
    SearchResult,
    Source,
    SourceStatus,
)
from .web_fetcher import create_web_fetcher

__all__ = [
    # Types
    "ETSLevel",
    "Source",
    "SourceStatus",
    "Hypothesis",
    "HypothesisStatus",
    "AssociativeTriple",
    "ResearchFinding",
    "ResearchResult",
    "ResearchMetrics",
    "ExecutionStep",
    "Claim",
    "Contradiction",
    "SearchResult",
    "ExtractedContent",
    # Config
    "WebFetcherConfig",
    "LLMValidatorConfig",
    "SourceAuditConfig",
    "ContentExtractorConfig",
    "SynthesizerConfig",
    "DeepDiveConfig",
    "StorageConfig",
    "DDEConfig",
    # Engine
    "DeepDiscoveryEngine",
    "create_engine",
    # Factories
    "create_web_fetcher",
    "create_source_auditor",
    "create_synthesizer",
]
