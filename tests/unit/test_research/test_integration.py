"""
Integration Tests for Deep Discovery Engine
"""

import pytest
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.research import (
    DeepDiscoveryEngine,
    DDEConfig,
    Source,
    SourceStatus,
    ETSLevel,
    WebFetcherConfig,
    SourceAuditConfig,
    SynthesizerConfig,
)
from gaap.research.web_fetcher import WebFetcher, create_web_fetcher
from gaap.research.source_auditor import SourceAuditor, create_source_auditor
from gaap.research.types import SearchResult, Hypothesis, HypothesisStatus


class TestWebFetcher:
    """Tests for WebFetcher component."""

    def test_web_fetcher_creation(self):
        config = WebFetcherConfig(
            provider="duckduckgo",
            max_results=10,
            timeout_seconds=30,
        )
        fetcher = WebFetcher(config)

        assert fetcher.config.provider == "duckduckgo"
        assert fetcher.config.max_results == 10

    def test_create_web_fetcher_factory(self):
        fetcher = create_web_fetcher(
            provider="duckduckgo",
            max_results=5,
        )

        assert fetcher.config.provider == "duckduckgo"
        assert fetcher.config.max_results == 5

    def test_web_fetcher_stats(self):
        fetcher = WebFetcher()
        stats = fetcher.get_stats()

        assert "provider" in stats
        assert "search_count" in stats
        assert "fetch_count" in stats

    def test_clean_ddg_url(self):
        fetcher = WebFetcher()

        url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com"
        cleaned = fetcher._clean_ddg_url(url)

        assert cleaned == "https://example.com"

    def test_clean_text(self):
        fetcher = WebFetcher()

        text = "<p>Hello  &amp;  World</p>"
        cleaned = fetcher._clean_text(text)

        assert "<p>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_set_provider(self):
        fetcher = WebFetcher()
        fetcher.set_provider("serper", api_key="test-key")

        assert fetcher.config.provider == "serper"


class TestSourceAuditor:
    """Tests for SourceAuditor component."""

    def test_auditor_creation(self):
        config = SourceAuditConfig(
            min_ets_threshold=0.3,
            check_author=True,
            check_date=True,
        )
        auditor = SourceAuditor(config)

        assert auditor.config.min_ets_threshold == 0.3
        assert auditor.config.check_author is True

    def test_create_source_auditor_factory(self):
        auditor = create_source_auditor(
            min_ets_threshold=0.5,
            domain_overrides={"example.com": 0.9},
        )

        assert auditor.config.min_ets_threshold == 0.5
        assert auditor.get_domain_score("example.com") == 0.9

    def test_audit_official_domain(self):
        auditor = SourceAuditor()
        source = Source(
            url="https://docs.python.org/3/library/asyncio.html",
            title="asyncio Documentation",
            domain="docs.python.org",
            content="This is comprehensive documentation about asyncio. " * 50,
            author="Python Documentation Team",
        )

        result = auditor.audit(source)

        assert result.domain_score >= 0.9
        assert "Official/verified domain" in result.reasons

    def test_audit_github_domain(self):
        auditor = SourceAuditor()
        source = Source(
            url="https://github.com/fastapi/fastapi",
            title="FastAPI Repository",
            domain="github.com",
            content="FastAPI framework repository with source code...",
        )

        result = auditor.audit(source)

        assert result.ets_score >= 0.7
        assert result.ets_level in [ETSLevel.VERIFIED, ETSLevel.RELIABLE]

    def test_audit_medium_domain(self):
        auditor = SourceAuditor()
        source = Source(
            url="https://medium.com/some-article",
            title="Some Blog Post",
            domain="medium.com",
            content="Blog post content...",
        )

        result = auditor.audit(source)

        assert result.ets_score < 0.7
        assert result.ets_level in [ETSLevel.QUESTIONABLE, ETSLevel.UNRELIABLE]

    def test_audit_blacklisted_domain(self):
        auditor = SourceAuditor()
        auditor.add_blacklist_domain("example.com")
        source = Source(
            url="https://example.com/article",
            title="Test",
            domain="example.com",
        )

        result = auditor.audit(source)

        assert result.ets_score == 0.0
        assert result.ets_level == ETSLevel.BLACKLISTED

    def test_audit_batch(self):
        auditor = SourceAuditor()
        auditor.add_blacklist_domain("spam.com")
        sources = [
            Source(url="https://docs.python.org/1", domain="docs.python.org", content="x" * 500),
            Source(url="https://spam.com/2", domain="spam.com"),
            Source(url="https://github.com/3", domain="github.com", content="x" * 500),
        ]

        passed, filtered = auditor.audit_batch(sources)

        assert len(passed) == 2
        assert len(filtered) == 1

    def test_freshness_scoring(self):
        auditor = SourceAuditor()

        recent_source = Source(
            url="https://docs.python.org/recent",
            domain="docs.python.org",
            publish_date=date.today(),
        )
        recent_result = auditor.audit(recent_source)

        old_source = Source(
            url="https://docs.python.org/old",
            domain="docs.python.org",
            publish_date=date(2020, 1, 1),
        )
        old_result = auditor.audit(old_source)

        assert recent_result.freshness_score > old_result.freshness_score

    def test_content_quality_scoring(self):
        auditor = SourceAuditor()

        good_content = Source(
            url="https://docs.python.org/good",
            domain="docs.python.org",
            content="This is a comprehensive guide.\n\n```python\ndef example():\n    pass\n```\n\nMore details here..."
            * 10,
        )
        good_result = auditor.audit(good_content)

        poor_content = Source(
            url="https://docs.python.org/poor",
            domain="docs.python.org",
            content="Short",
        )
        poor_result = auditor.audit(poor_content)

        assert good_result.content_score > poor_result.content_score

    def test_custom_domain_override(self):
        auditor = SourceAuditor()
        auditor.set_domain_score("mycompany.com", 0.95)

        source = Source(
            url="https://mycompany.com/docs",
            domain="mycompany.com",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.95

    def test_blacklist_domain(self):
        auditor = SourceAuditor()
        auditor.add_blacklist_domain("spam.com")

        source = Source(
            url="https://spam.com/article",
            domain="spam.com",
        )

        result = auditor.audit(source)

        assert result.ets_score == 0.0

    def test_whitelist_domain(self):
        auditor = SourceAuditor()
        auditor.add_whitelist_domain("trusted.com")

        source = Source(
            url="https://trusted.com/article",
            domain="trusted.com",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.8

    def test_get_stats(self):
        auditor = SourceAuditor()

        sources = [
            Source(url="https://docs.python.org/1", domain="docs.python.org", content="x" * 500),
            Source(url="https://github.com/2", domain="github.com", content="x" * 500),
        ]

        auditor.audit_batch(sources)
        stats = auditor.get_stats()

        assert stats["audited_count"] == 2


class TestDDEConfig:
    """Tests for DDEConfig."""

    def test_default_config(self):
        config = DDEConfig()

        assert config.research_depth == 3
        assert config.max_total_sources == 50
        assert config.web_fetcher.provider == "duckduckgo"

    def test_quick_preset(self):
        config = DDEConfig.quick()

        assert config.research_depth == 1
        assert config.max_total_sources == 10

    def test_deep_preset(self):
        config = DDEConfig.deep()

        assert config.research_depth == 5
        assert config.max_total_sources == 100

    def test_academic_preset(self):
        config = DDEConfig.academic()

        assert config.research_depth == 4
        assert config.source_audit.min_ets_threshold == 0.5

    def test_config_serialization(self):
        config = DDEConfig(
            research_depth=4,
            web_fetcher=WebFetcherConfig(max_results=15),
        )

        data = config.to_dict()

        assert data["research_depth"] == 4
        assert data["web_fetcher"]["max_results"] == 15

    def test_config_from_dict(self):
        data = {
            "research_depth": 5,
            "web_fetcher": {
                "provider": "duckduckgo",
                "max_results": 20,
            },
        }

        config = DDEConfig.from_dict(data)

        assert config.research_depth == 5
        assert config.web_fetcher.max_results == 20


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_to_source(self):
        result = SearchResult(
            url="https://docs.python.org/asyncio",
            title="Asyncio Documentation",
            snippet="Asynchronous I/O support",
            rank=1,
            provider="duckduckgo",
        )

        source = result.to_source()

        assert source.url == "https://docs.python.org/asyncio"
        assert source.title == "Asyncio Documentation"
        assert source.domain == "docs.python.org"
        assert source.status == SourceStatus.DISCOVERED
        assert source.metadata["rank"] == 1
        assert source.metadata["provider"] == "duckduckgo"


class TestHypothesis:
    """Tests for Hypothesis."""

    def test_hypothesis_creation(self):
        hyp = Hypothesis(
            id="hyp_001",
            statement="FastAPI supports async endpoints",
            confidence=0.85,
        )

        assert hyp.id == "hyp_001"
        assert hyp.status == HypothesisStatus.UNVERIFIED
        assert hyp.confidence == 0.85

    def test_hypothesis_status_checks(self):
        verified = Hypothesis(
            id="v1",
            statement="Test",
            status=HypothesisStatus.VERIFIED,
        )

        falsified = Hypothesis(
            id="f1",
            statement="Test",
            status=HypothesisStatus.FALSIFIED,
        )

        conflicted = Hypothesis(
            id="c1",
            statement="Test",
            status=HypothesisStatus.CONFLICTED,
        )

        assert verified.is_verified
        assert not verified.is_falsified
        assert falsified.is_falsified
        assert conflicted.is_conflicted


class TestDeepDiscoveryEngine:
    """Tests for DeepDiscoveryEngine."""

    def test_engine_creation(self):
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config)

        assert engine.config.research_depth == 1

    def test_engine_components(self):
        engine = DeepDiscoveryEngine()

        assert engine._web_fetcher is not None
        assert engine._content_extractor is not None
        assert engine._source_auditor is not None
        assert engine._synthesizer is not None

    def test_get_stats(self):
        engine = DeepDiscoveryEngine()
        stats = engine.get_stats()

        assert "research_count" in stats
        assert "components" in stats
        assert "web_fetcher" in stats["components"]

    def test_update_config(self):
        engine = DeepDiscoveryEngine()
        new_config = DDEConfig.deep()

        engine.update_config(new_config)

        assert engine.config.research_depth == 5

    def test_generate_summary(self):
        engine = DeepDiscoveryEngine()

        sources = [
            Source(url="https://example.com/1", domain="example.com", ets_score=0.8),
            Source(url="https://example.com/2", domain="example.com", ets_score=0.6),
        ]

        hypotheses = [
            Hypothesis(
                id="h1",
                statement="Test",
                status=HypothesisStatus.VERIFIED,
                confidence=0.9,
            ),
        ]

        summary = engine._generate_summary(sources, hypotheses)

        assert "sources" in summary.lower()
        assert "0.70" in summary or "0.7" in summary
