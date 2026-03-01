"""
Deep Discovery Engine - Configuration
====================================

All configurable parameters for the research system.
Every parameter can be controlled from:
- Config file
- API endpoint (for Web GUI)
- Environment variables
- Code

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class WebFetcherConfig:
    """
    Configuration for web search and fetching.

    Default: DuckDuckGo (free, no API key required)
    Options: Google (Serper), Perplexity (requires API key)
    """

    provider: Literal["duckduckgo", "serper", "perplexity", "brave"] = "duckduckgo"
    api_key: str | None = None
    max_results: int = 10
    timeout_seconds: int = 30
    user_agent: str = "GAAP-Research-Bot/1.0"
    retry_count: int = 3
    retry_delay_ms: int = 500
    respect_robots_txt: bool = True
    rate_limit_per_second: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "api_key"} | {
            "api_key": "***" if self.api_key else None
        }


@dataclass
class LLMValidatorConfig:
    """
    Configuration for LLM-based validation and synthesis.

    Supports multiple providers for flexibility.
    """

    provider: Literal["auto", "kimi", "kilo", "openai", "anthropic", "gemini"] = "auto"
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout_seconds: int = 120
    retry_count: int = 2
    system_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceAuditConfig:
    """
    Configuration for source auditing (ETS scoring).
    """

    min_ets_threshold: float = 0.3
    max_ets_threshold: float = 1.0
    domain_overrides: dict[str, float] = field(default_factory=dict)
    blacklist_domains: list[str] = field(default_factory=list)
    whitelist_domains: list[str] = field(default_factory=list)
    check_author: bool = True
    check_date: bool = True
    freshness_weight: float = 0.2
    citation_weight: float = 0.3
    domain_weight: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContentExtractorConfig:
    """
    Configuration for content extraction from URLs.
    """

    max_content_length: int = 50000
    min_content_length: int = 100
    timeout_seconds: int = 30
    extract_code_blocks: bool = True
    extract_links: bool = True
    extract_metadata: bool = True
    clean_html: bool = True
    remove_scripts: bool = True
    remove_styles: bool = True
    max_concurrent_fetches: int = 5
    request_delay_ms: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SynthesizerConfig:
    """
    Configuration for hypothesis synthesis and verification.
    """

    max_hypotheses: int = 10
    min_confidence_threshold: float = 0.5
    verification_confidence_threshold: float = 0.7
    cross_validate_enabled: bool = True
    extract_claims_per_source: int = 5
    max_triples_per_hypothesis: int = 20
    detect_contradictions: bool = True
    llm_temperature: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeepDiveConfig:
    """
    Configuration for deep exploration protocol.
    """

    default_depth: int = 3
    max_depth: int = 5
    min_depth: int = 1
    citation_follow_depth: int = 2
    max_sources_per_depth: int = 20
    follow_external_links: bool = True
    cross_validate_sources: bool = True
    max_citations_to_follow: int = 10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StorageConfig:
    """
    Configuration for permanent storage (no TTL - everything is kept).
    """

    knowledge_graph_enabled: bool = True
    vector_store_enabled: bool = True
    sqlite_cache_enabled: bool = True
    dedup_enabled: bool = True
    dedup_by_hash: bool = True
    dedup_by_url: bool = True
    store_raw_content: bool = True
    store_extracted_content: bool = True
    compress_content: bool = False
    storage_path: str = ".gaap/research"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DDEConfig:
    """
    Master configuration for Deep Discovery Engine.

    All parameters are configurable for full control from Web GUI.

    Usage:
        config = DDEConfig()
        config.research_depth = 4
        config.web_fetcher.max_results = 20

        # Or from dict (API request)
        config = DDEConfig.from_dict({"research_depth": 4})
    """

    # Web Fetcher Settings
    web_fetcher: WebFetcherConfig = field(default_factory=WebFetcherConfig)

    # LLM Validator Settings
    llm_validator: LLMValidatorConfig = field(default_factory=LLMValidatorConfig)

    # Source Audit Settings
    source_audit: SourceAuditConfig = field(default_factory=SourceAuditConfig)

    # Content Extractor Settings
    content_extractor: ContentExtractorConfig = field(default_factory=ContentExtractorConfig)

    # Synthesizer Settings
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)

    # Deep Dive Settings
    deep_dive: DeepDiveConfig = field(default_factory=DeepDiveConfig)

    # Storage Settings
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Global Settings
    research_depth: int = 3
    max_total_sources: int = 50
    max_total_hypotheses: int = 20
    max_execution_time_seconds: int = 300
    parallel_processing: bool = True
    max_parallel_tasks: int = 5

    # Caching
    check_existing_research: bool = True
    force_fresh: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API responses."""
        return {
            "web_fetcher": self.web_fetcher.to_dict(),
            "llm_validator": self.llm_validator.to_dict(),
            "source_audit": self.source_audit.to_dict(),
            "content_extractor": self.content_extractor.to_dict(),
            "synthesizer": self.synthesizer.to_dict(),
            "deep_dive": self.deep_dive.to_dict(),
            "storage": self.storage.to_dict(),
            "research_depth": self.research_depth,
            "max_total_sources": self.max_total_sources,
            "max_total_hypotheses": self.max_total_hypotheses,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "parallel_processing": self.parallel_processing,
            "max_parallel_tasks": self.max_parallel_tasks,
            "check_existing_research": self.check_existing_research,
            "force_fresh": self.force_fresh,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DDEConfig":
        """Deserialize from API request."""
        config = cls()

        if "web_fetcher" in data:
            config.web_fetcher = WebFetcherConfig(**data["web_fetcher"])
        if "llm_validator" in data:
            config.llm_validator = LLMValidatorConfig(**data["llm_validator"])
        if "source_audit" in data:
            config.source_audit = SourceAuditConfig(**data["source_audit"])
        if "content_extractor" in data:
            config.content_extractor = ContentExtractorConfig(**data["content_extractor"])
        if "synthesizer" in data:
            config.synthesizer = SynthesizerConfig(**data["synthesizer"])
        if "deep_dive" in data:
            config.deep_dive = DeepDiveConfig(**data["deep_dive"])
        if "storage" in data:
            config.storage = StorageConfig(**data["storage"])

        for key in [
            "research_depth",
            "max_total_sources",
            "max_total_hypotheses",
            "max_execution_time_seconds",
            "parallel_processing",
            "max_parallel_tasks",
            "check_existing_research",
            "force_fresh",
        ]:
            if key in data:
                setattr(config, key, data[key])

        return config

    @classmethod
    def quick(cls) -> "DDEConfig":
        """Quick research preset."""
        return cls(
            research_depth=1,
            max_total_sources=10,
            web_fetcher=WebFetcherConfig(max_results=5),
        )

    @classmethod
    def standard(cls) -> "DDEConfig":
        """Standard research preset."""
        return cls(
            research_depth=3,
            max_total_sources=30,
        )

    @classmethod
    def deep(cls) -> "DDEConfig":
        """Deep research preset."""
        return cls(
            research_depth=5,
            max_total_sources=100,
            deep_dive=DeepDiveConfig(citation_follow_depth=3),
            synthesizer=SynthesizerConfig(max_hypotheses=20),
        )

    @classmethod
    def academic(cls) -> "DDEConfig":
        """Academic research preset with high ETS standards."""
        return cls(
            research_depth=4,
            source_audit=SourceAuditConfig(
                min_ets_threshold=0.5,
                check_author=True,
                check_date=True,
            ),
            synthesizer=SynthesizerConfig(
                cross_validate_enabled=True,
                verification_confidence_threshold=0.8,
            ),
        )
