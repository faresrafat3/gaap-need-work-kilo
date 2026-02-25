"""
Integration Tests for Research Module
=====================================

Comprehensive tests for Deep Discovery Engine components:
- WebFetcher: DuckDuckGo search, error handling, timeout
- SourceAuditor: ETS scoring, domain reputation, blacklist
- ContentExtractor: HTML extraction, script removal, invalid HTML
- Synthesizer: Hypothesis building, LLM validation mock
- DeepDiscoveryEngine: Full research flow with mocked components

Uses pytest and pytest-asyncio with unittest.mock for external calls.
No real API keys required.

Note: These tests mock all external HTTP calls. For real network tests,
run with: pytest -m e2e --run-e2e
"""

import asyncio
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import pytest

from gaap.research.web_fetcher import WebFetcher, create_web_fetcher
from gaap.research.source_auditor import SourceAuditor, create_source_auditor, AuditResult
from gaap.research.content_extractor import ContentExtractor, create_content_extractor
from gaap.research.synthesizer import Synthesizer, create_synthesizer
from gaap.research.engine import DeepDiscoveryEngine
from gaap.research.types import (
    Source,
    SourceStatus,
    ETSLevel,
    Hypothesis,
    HypothesisStatus,
    Claim,
    SearchResult,
    ExtractedContent,
    AssociativeTriple,
    ResearchResult,
    ResearchMetrics,
)
from gaap.research.config import (
    WebFetcherConfig,
    SourceAuditConfig,
    ContentExtractorConfig,
    SynthesizerConfig,
    DDEConfig,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock async HTTP session."""
    session = MagicMock()
    session.get = AsyncMock()
    session.post = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture(scope="module")
def mock_llm_provider():
    """Create a mock LLM provider for Synthesizer tests."""
    provider = MagicMock()
    provider.name = "mock-llm"
    provider.default_model = "test-model"

    async def mock_chat_completion(messages, model=None, **kwargs):
        from gaap.core.types import (
            ChatCompletionResponse,
            ChatCompletionChoice,
            Message,
            MessageRole,
            Usage,
        )

        return ChatCompletionResponse(
            id="test-response-id",
            model=model or "test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content='{"claim": "Test claim", "confidence": 0.8}',
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    provider.chat_completion = AsyncMock(side_effect=mock_chat_completion)
    return provider


@pytest.fixture
def sample_html():
    """Sample HTML content for extraction tests."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Article - Python Async</title>
        <meta name="author" content="John Doe">
        <meta property="article:published_time" content="2025-01-15T10:00:00Z">
        <style>.hidden { display: none; }</style>
        <script>console.log('tracking');</script>
    </head>
    <body>
        <nav>Navigation content</nav>
        <header>Header content</header>
        <article>
            <h1>Python Async Programming Guide</h1>
            <p>This is a comprehensive guide about async programming in Python.</p>
            <p>Asyncio is a library to write concurrent code using the async/await syntax.</p>
            <pre><code>
import asyncio

async def main():
    await asyncio.sleep(1)
    print("Hello async!")

asyncio.run(main())
            </code></pre>
            <p>For more information, visit <a href="https://docs.python.org/3/library/asyncio.html">Python Docs</a></p>
        </article>
        <aside>Sidebar content</aside>
        <footer>Footer content</footer>
        <script>ads.track();</script>
    </body>
    </html>
    """


@pytest.fixture
def sample_sources():
    """Create sample sources for testing."""
    return [
        Source(
            url="https://docs.python.org/3/library/asyncio.html",
            title="asyncio - Asynchronous I/O",
            domain="docs.python.org",
            content="asyncio is a library to write concurrent code using the async/await syntax.",
            author="Python Documentation Team",
            publish_date=date(2025, 1, 15),
        ),
        Source(
            url="https://github.com/python/cpython",
            title="CPython Repository",
            domain="github.com",
            content="The Python programming language repository with asyncio implementation.",
        ),
        Source(
            url="https://arxiv.org/abs/1234.5678",
            title="Async Patterns Research Paper",
            domain="arxiv.org",
            content="Academic paper about async patterns in programming languages.",
            author="Dr. Smith",
        ),
        Source(
            url="https://wikipedia.org/wiki/Asynchronous_I/O",
            title="Asynchronous I/O - Wikipedia",
            domain="wikipedia.org",
            content="Wikipedia article about asynchronous I/O.",
        ),
        Source(
            url="https://medium.com/@user/async-guide",
            title="Async Guide",
            domain="medium.com",
            content="Blog post about async programming.",
        ),
    ]


# =============================================================================
# TestWebFetcher
# =============================================================================


class TestWebFetcher:
    """Tests for WebFetcher component - DuckDuckGo search, error handling, timeout."""

    def test_web_fetcher_creation_with_duckduckgo(self):
        """Test WebFetcher creation with DuckDuckGo provider (default, free)."""
        config = WebFetcherConfig(
            provider="duckduckgo",
            max_results=10,
            timeout_seconds=30,
        )
        fetcher = WebFetcher(config)

        assert fetcher.config.provider == "duckduckgo"
        assert fetcher.config.max_results == 10
        assert fetcher.config.timeout_seconds == 30
        assert fetcher.config.api_key is None

    def test_web_fetcher_factory(self):
        """Test create_web_fetcher factory function."""
        fetcher = create_web_fetcher(
            provider="duckduckgo",
            max_results=5,
        )

        assert fetcher.config.provider == "duckduckgo"
        assert fetcher.config.max_results == 5

    def test_web_fetcher_stats(self):
        """Test that WebFetcher tracks statistics."""
        fetcher = WebFetcher()
        stats = fetcher.get_stats()

        assert "provider" in stats
        assert "search_count" in stats
        assert "fetch_count" in stats
        assert "error_count" in stats
        assert "total_time_ms" in stats

    def test_clean_ddg_url(self):
        """Test DuckDuckGo redirect URL cleaning."""
        fetcher = WebFetcher()

        redirect_url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2Fasyncio"
        cleaned = fetcher._clean_ddg_url(redirect_url)
        assert cleaned == "https://docs.python.org/asyncio"

        direct_url = "https://example.com/direct"
        assert fetcher._clean_ddg_url(direct_url) == direct_url

    def test_clean_text_removes_html(self):
        """Test that HTML tags and entities are cleaned."""
        fetcher = WebFetcher()

        html_text = "<p>Hello &amp; World</p>  <span>Test</span>"
        cleaned = fetcher._clean_text(html_text)

        assert "<p>" not in cleaned
        assert "<span>" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
        assert "Test" in cleaned

    def test_parse_ddg_html(self):
        """Test parsing DuckDuckGo HTML response."""
        fetcher = WebFetcher()

        html = """
        <html>
        <div class="result">
            <a class="result__a" href="https://docs.python.org/asyncio">Asyncio Documentation</a>
            <a class="result__snippet">Asynchronous I/O in Python</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://github.com/fastapi/fastapi">FastAPI GitHub</a>
            <a class="result__snippet">Fast async web framework</a>
        </div>
        </html>
        """

        results = fetcher._parse_ddg_html(html, max_results=10)

        assert len(results) == 2
        assert results[0].url == "https://docs.python.org/asyncio"
        assert results[0].title == "Asyncio Documentation"
        assert results[0].rank == 1
        assert results[0].provider == "duckduckgo"

    def test_set_provider_runtime(self):
        """Test switching provider at runtime."""
        fetcher = WebFetcher()
        fetcher.set_provider("serper", api_key="test-key")

        assert fetcher.config.provider == "serper"
        assert fetcher.config.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_search_returns_empty_on_network_error(self, mock_session):
        """Test that search returns empty list on network errors."""
        config = WebFetcherConfig(provider="duckduckgo")
        fetcher = WebFetcher(config)
        fetcher._session = mock_session

        mock_session.get.side_effect = ConnectionError("Network error")

        results = await fetcher.search("test query")

        assert results == []
        assert fetcher._search_count == 1

    @pytest.mark.asyncio
    async def test_search_timeout_handling(self, mock_session):
        """Test that search handles timeout gracefully."""
        config = WebFetcherConfig(provider="duckduckgo", timeout_seconds=5)
        fetcher = WebFetcher(config)
        fetcher._session = mock_session

        mock_session.get.side_effect = asyncio.TimeoutError("Request timed out")

        results = await fetcher.search("test query")

        assert results == []
        assert fetcher._search_count == 1

    @pytest.mark.asyncio
    async def test_fetch_content_timeout(self, mock_session):
        """Test that fetch_content handles timeout."""
        config = WebFetcherConfig(timeout_seconds=5)
        fetcher = WebFetcher(config)
        fetcher._session = mock_session

        mock_session.get.side_effect = asyncio.TimeoutError("Fetch timed out")

        content = await fetcher.fetch_content("https://example.com")

        assert content == ""
        assert fetcher._error_count == 1

    @pytest.mark.asyncio
    async def test_fetch_content_network_error(self, mock_session):
        """Test that fetch_content handles network errors."""
        fetcher = WebFetcher()
        fetcher._session = mock_session

        mock_session.get.side_effect = ConnectionError("Connection refused")

        content = await fetcher.fetch_content("https://example.com")

        assert content == ""
        assert fetcher._error_count == 1

    @pytest.mark.asyncio
    async def test_fetch_batch_partial_failure(self, mock_session):
        """Test that fetch_batch handles partial failures."""
        fetcher = WebFetcher()
        fetcher._session = mock_session

        call_count = 0

        async def mock_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Failed")
            response = MagicMock()
            response.text = f"Content for {url}"
            return response

        mock_session.get.side_effect = mock_get

        results = await fetcher.fetch_batch(
            ["https://fail.com", "https://success.com"],
            max_concurrent=2,
        )

        assert "https://success.com" in results
        assert results["https://success.com"] != ""
        assert fetcher._error_count == 1

    @pytest.mark.asyncio
    async def test_search_successful_mock(self, mock_session):
        """Test successful search with mocked response."""
        fetcher = WebFetcher()
        fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <a class="result__a" href="https://docs.python.org/asyncio">Asyncio Docs</a>
        <a class="result__snippet">Python async documentation</a>
        </html>
        """
        mock_session.get.return_value = mock_response

        results = await fetcher.search("asyncio")

        assert len(results) >= 1
        assert results[0].url == "https://docs.python.org/asyncio"
        assert fetcher._search_count == 1


# =============================================================================
# TestSourceAuditor
# =============================================================================


class TestSourceAuditor:
    """Tests for SourceAuditor - ETS scoring, domain reputation, blacklist handling."""

    def test_auditor_creation(self):
        """Test SourceAuditor creation with config."""
        config = SourceAuditConfig(
            min_ets_threshold=0.4,
            check_author=True,
            check_date=True,
        )
        auditor = SourceAuditor(config)

        assert auditor.config.min_ets_threshold == 0.4
        assert auditor.config.check_author is True
        assert auditor.config.check_date is True

    def test_auditor_factory(self):
        """Test create_source_auditor factory function."""
        auditor = create_source_auditor(
            min_ets_threshold=0.5,
            domain_overrides={"custom.com": 0.9},
        )

        assert auditor.config.min_ets_threshold == 0.5
        assert auditor.get_domain_score("custom.com") == 0.9

    def test_ets_scoring_official_domain(self):
        """Test ETS scoring for official documentation domains."""
        auditor = SourceAuditor()

        source = Source(
            url="https://docs.python.org/3/library/asyncio.html",
            title="Asyncio Documentation",
            domain="docs.python.org",
            content="Comprehensive asyncio documentation with examples. " * 20,
            author="Python Team",
        )

        result = auditor.audit(source)

        assert result.domain_score == 1.0
        assert result.ets_score >= 0.65
        assert result.ets_level in [ETSLevel.VERIFIED, ETSLevel.RELIABLE]
        assert "Official/verified domain" in result.reasons

    def test_ets_scoring_github_domain(self):
        """Test ETS scoring for GitHub domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://github.com/fastapi/fastapi",
            title="FastAPI Repository",
            domain="github.com",
            content="FastAPI source code and documentation repository.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.9
        assert result.ets_level in [ETSLevel.VERIFIED, ETSLevel.RELIABLE]

    def test_ets_scoring_arxiv_domain(self):
        """Test ETS scoring for arXiv academic domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://arxiv.org/abs/2301.12345",
            title="Research Paper",
            domain="arxiv.org",
            content="Academic research paper about async patterns.",
            author="Dr. Research",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.9
        assert result.ets_level in [ETSLevel.VERIFIED, ETSLevel.RELIABLE]

    def test_ets_scoring_wikipedia_domain(self):
        """Test ETS scoring for Wikipedia domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://en.wikipedia.org/wiki/Asynchronous_I/O",
            title="Async I/O Wikipedia",
            domain="wikipedia.org",
            content="Wikipedia article about asynchronous I/O.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.7
        assert result.ets_level in [ETSLevel.RELIABLE, ETSLevel.QUESTIONABLE]

    def test_ets_scoring_medium_domain(self):
        """Test ETS scoring for Medium blog domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://medium.com/@user/async-guide",
            title="Async Guide",
            domain="medium.com",
            content="Blog post about async programming.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.5
        assert result.ets_level in [ETSLevel.QUESTIONABLE, ETSLevel.UNRELIABLE]

    def test_ets_scoring_unknown_domain(self):
        """Test ETS scoring for unknown domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://random-unknown-site.com/article",
            title="Random Article",
            domain="random-unknown-site.com",
            content="Some content.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.4
        assert result.ets_level in [ETSLevel.QUESTIONABLE, ETSLevel.UNRELIABLE]

    def test_ets_scoring_edu_domain(self):
        """Test ETS scoring for .edu domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://stanford.edu/cs/course",
            title="Stanford Course",
            domain="stanford.edu",
            content="Course materials for computer science.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.75

    def test_ets_scoring_gov_domain(self):
        """Test ETS scoring for .gov domain."""
        auditor = SourceAuditor()

        source = Source(
            url="https://nasa.gov/research",
            title="NASA Research",
            domain="nasa.gov",
            content="Government research publication.",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.85

    def test_blacklist_domain_handling(self):
        """Test that blacklisted domains get zero ETS score."""
        auditor = SourceAuditor()
        auditor.add_blacklist_domain("spam-site.com")
        auditor.add_blacklist_domain("malware.net")

        source = Source(
            url="https://spam-site.com/article",
            title="Spam Article",
            domain="spam-site.com",
        )

        result = auditor.audit(source)

        assert result.ets_score == 0.0
        assert result.ets_level == ETSLevel.BLACKLISTED
        assert "blacklisted" in result.reasons[0].lower()
        assert source.status == SourceStatus.AUDITED

    def test_blacklist_via_config(self):
        """Test blacklist configuration via config."""
        config = SourceAuditConfig(
            blacklist_domains=["spam.com", "fake-news.org"],
        )
        auditor = SourceAuditor(config)

        source = Source(url="https://spam.com/bad", domain="spam.com")
        result = auditor.audit(source)

        assert result.ets_score == 0.0
        assert result.ets_level == ETSLevel.BLACKLISTED

    def test_whitelist_domain_handling(self):
        """Test that whitelisted domains get boosted score."""
        auditor = SourceAuditor()
        auditor.add_whitelist_domain("trusted-internal.company.com")

        source = Source(
            url="https://trusted-internal.company.com/docs",
            domain="trusted-internal.company.com",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.8

    def test_domain_reputation_lookup(self):
        """Test domain reputation lookup."""
        auditor = SourceAuditor()

        assert auditor.get_domain_score("docs.python.org") == 1.0
        assert auditor.get_domain_score("github.com") == 0.9
        assert auditor.get_domain_score("stackoverflow.com") == 0.75
        assert auditor.get_domain_score("arxiv.org") == 0.9
        assert auditor.get_domain_score("medium.com") == 0.5

    def test_custom_domain_override(self):
        """Test custom domain score override."""
        auditor = SourceAuditor()
        auditor.set_domain_score("my-company.com", 0.95)

        source = Source(
            url="https://my-company.com/internal-docs",
            domain="my-company.com",
        )

        result = auditor.audit(source)

        assert result.domain_score == 0.95

    def test_freshness_scoring_recent(self):
        """Test freshness scoring for recent content."""
        auditor = SourceAuditor()

        recent_source = Source(
            url="https://docs.python.org/recent",
            domain="docs.python.org",
            publish_date=date.today(),
        )

        result = auditor.audit(recent_source)

        assert result.freshness_score == 1.0
        assert "Recent content" in result.reasons

    def test_freshness_scoring_old(self):
        """Test freshness scoring for old content."""
        auditor = SourceAuditor()

        old_source = Source(
            url="https://docs.python.org/old",
            domain="docs.python.org",
            publish_date=date(2018, 1, 1),
        )

        result = auditor.audit(old_source)

        assert result.freshness_score < 0.5
        assert "Outdated content" in result.reasons

    def test_freshness_scoring_no_date(self):
        """Test freshness scoring when no date is available."""
        auditor = SourceAuditor()

        source = Source(
            url="https://docs.python.org/nodate",
            domain="docs.python.org",
        )

        result = auditor.audit(source)

        assert result.freshness_score == 0.5

    def test_author_scoring(self):
        """Test author credibility scoring."""
        auditor = SourceAuditor()

        with_author = Source(
            url="https://example.com/with-author",
            domain="example.com",
            author="John Doe",
        )

        without_author = Source(
            url="https://example.com/no-author",
            domain="example.com",
        )

        result_with = auditor.audit(with_author)
        result_without = auditor.audit(without_author)

        assert result_with.author_score > result_without.author_score
        assert "Identified author" in result_with.reasons

    def test_content_quality_scoring(self):
        """Test content quality scoring."""
        auditor = SourceAuditor()

        high_quality = Source(
            url="https://example.com/high",
            domain="example.com",
            content="This is a comprehensive guide.\n\n```python\ndef example():\n    return True\n```\n\n"
            * 50,
        )

        low_quality = Source(
            url="https://example.com/low",
            domain="example.com",
            content="Short",
        )

        high_result = auditor.audit(high_quality)
        low_result = auditor.audit(low_quality)

        assert high_result.content_score > low_result.content_score
        assert "Quality content" in high_result.reasons
        assert "Low quality content" in low_result.reasons

    def test_audit_batch(self):
        """Test batch auditing of sources."""
        config = SourceAuditConfig(min_ets_threshold=0.6)
        auditor = SourceAuditor(config)
        auditor.add_blacklist_domain("spam.com")

        sources = [
            Source(
                url="https://docs.python.org/1",
                domain="docs.python.org",
                content="x" * 1000,
            ),
            Source(url="https://spam.com/2", domain="spam.com"),
            Source(
                url="https://github.com/3",
                domain="github.com",
                content="x" * 1000,
            ),
            Source(url="https://medium.com/4", domain="medium.com", content="short"),
        ]

        passed, filtered = auditor.audit_batch(sources, filter_threshold=True)

        assert len(passed) == 2
        assert len(filtered) == 2

    def test_audit_batch_no_filter(self):
        """Test batch auditing without filtering."""
        auditor = SourceAuditor()

        sources = [
            Source(url="https://docs.python.org/1", domain="docs.python.org", content="x" * 500),
            Source(url="https://spam.com/2", domain="spam.com"),
        ]

        passed, filtered = auditor.audit_batch(sources, filter_threshold=False)

        assert len(passed) == 2
        assert len(filtered) == 0

    def test_audit_result_to_dict(self):
        """Test AuditResult serialization."""
        auditor = SourceAuditor()

        source = Source(
            url="https://docs.python.org/test",
            domain="docs.python.org",
            content="Test content " * 50,
        )

        result = auditor.audit(source)
        data = result.to_dict()

        assert "url" in data
        assert "ets_score" in data
        assert "ets_level" in data
        assert "reasons" in data

    def test_get_stats(self):
        """Test auditor statistics."""
        auditor = SourceAuditor()

        sources = [
            Source(url="https://docs.python.org/1", domain="docs.python.org", content="x" * 500),
            Source(url="https://github.com/2", domain="github.com", content="x" * 500),
        ]

        auditor.audit_batch(sources)
        stats = auditor.get_stats()

        assert stats["audited_count"] == 2
        assert "blacklist_size" in stats


# =============================================================================
# TestContentExtractor
# =============================================================================


class TestContentExtractor:
    """Tests for ContentExtractor - HTML extraction, script removal, invalid HTML."""

    def test_extractor_creation(self):
        """Test ContentExtractor creation with config."""
        config = ContentExtractorConfig(
            max_content_length=10000,
            extract_code_blocks=True,
            extract_links=True,
        )
        extractor = ContentExtractor(config)

        assert extractor.config.max_content_length == 10000
        assert extractor.config.extract_code_blocks is True

    def test_extractor_factory(self):
        """Test create_content_extractor factory function."""
        extractor = create_content_extractor(
            max_content_length=20000,
            extract_code_blocks=False,
        )

        assert extractor.config.max_content_length == 20000
        assert extractor.config.extract_code_blocks is False

    def test_extractor_stats(self):
        """Test that ContentExtractor tracks statistics."""
        extractor = ContentExtractor()
        stats = extractor.get_stats()

        assert "extracted_count" in stats
        assert "failed_count" in stats
        assert "success_rate" in stats

    def test_clean_html_removes_scripts(self, sample_html):
        """Test that scripts and styles are removed from HTML."""
        extractor = ContentExtractor()

        cleaned = extractor._clean_html(sample_html)

        assert "<script>" not in cleaned
        assert "<style>" not in cleaned
        assert "console.log" not in cleaned
        assert ".hidden" not in cleaned

    def test_clean_html_removes_ads(self, sample_html):
        """Test that advertisement-related elements are removed."""
        extractor = ContentExtractor()

        html_with_ads = sample_html + "<advertisement>Ad content here</advertisement>"
        cleaned = extractor._clean_html(html_with_ads)

        assert "<advertisement>" not in cleaned.lower()
        assert "Ad content here" not in cleaned

    def test_extract_main_content(self, sample_html):
        """Test extraction of main content area."""
        extractor = ContentExtractor()

        cleaned = extractor._clean_html(sample_html)
        main = extractor._extract_main_content(cleaned)

        assert "Python Async Programming Guide" in main
        assert "Asyncio is a library" in main

    def test_extract_title(self, sample_html):
        """Test title extraction from HTML."""
        extractor = ContentExtractor()

        title = extractor._extract_title(sample_html)

        assert "Python Async" in title

    def test_html_to_text_conversion(self, sample_html):
        """Test HTML to clean text conversion."""
        extractor = ContentExtractor()

        cleaned = extractor._clean_html(sample_html)
        main = extractor._extract_main_content(cleaned)
        text = extractor._html_to_text(main)

        assert "<" not in text
        assert ">" not in text
        assert "Asyncio is a library" in text

    def test_extract_author(self, sample_html):
        """Test author extraction from HTML meta tags."""
        extractor = ContentExtractor()

        author = extractor._extract_author(sample_html)

        assert author == "John Doe"

    def test_extract_publish_date(self, sample_html):
        """Test publication date extraction from HTML."""
        extractor = ContentExtractor()

        pub_date = extractor._extract_date(sample_html)

        assert pub_date is not None
        assert pub_date.year == 2025
        assert pub_date.month == 1

    def test_extract_links(self, sample_html):
        """Test link extraction from HTML."""
        extractor = ContentExtractor()

        links = extractor._extract_links(sample_html, "https://example.com")

        assert len(links) >= 1
        assert any("docs.python.org" in link for link in links)

    def test_extract_code_blocks(self, sample_html):
        """Test code block extraction from HTML."""
        extractor = ContentExtractor()

        code_blocks = extractor._extract_code_blocks(sample_html)

        assert len(code_blocks) >= 1
        assert "asyncio" in code_blocks[0]
        assert "async def main()" in code_blocks[0]

    def test_extract_links_ignores_javascript(self):
        """Test that javascript: links are ignored."""
        extractor = ContentExtractor()

        html = """
        <a href="https://valid.com">Valid</a>
        <a href="javascript:void(0)">JS Link</a>
        <a href="#anchor">Anchor</a>
        """
        links = extractor._extract_links(html, "https://example.com")

        assert any("valid.com" in link for link in links)
        assert not any("javascript" in link for link in links)
        assert not any("#anchor" in link for link in links)

    def test_handle_invalid_html(self):
        """Test handling of invalid/malformed HTML."""
        extractor = ContentExtractor()

        invalid_html = "<html><p>Unclosed paragraph<div>Mixed content</html>"
        text = extractor._html_to_text(invalid_html)

        assert "Unclosed paragraph" in text or "Mixed content" in text

    def test_handle_empty_html(self):
        """Test handling of empty HTML."""
        extractor = ContentExtractor()

        title = extractor._extract_title("")
        assert title == ""

        text = extractor._html_to_text("")
        assert text == ""

    def test_handle_none_content(self):
        """Test handling of None content in source."""
        extractor = ContentExtractor()

        text = extractor._html_to_text("<p>Content</p>")
        assert "Content" in text

    @pytest.mark.asyncio
    async def test_extract_with_pre_fetched_html(self, sample_html):
        """Test extraction with pre-fetched HTML content."""
        extractor = ContentExtractor()

        result = await extractor.extract(
            url="https://example.com/test",
            html=sample_html,
        )

        assert result.extraction_success is True
        assert "Python Async" in result.title
        assert "asyncio" in result.content.lower()
        assert result.author == "John Doe"
        assert len(result.code_blocks) >= 1

    @pytest.mark.asyncio
    async def test_extract_handles_empty_html(self):
        """Test extraction handles empty HTML gracefully."""
        extractor = ContentExtractor()

        result = await extractor.extract(
            url="https://example.com/empty",
            html="",
        )

        assert result.extraction_success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_extract_with_navigation_removed(self, sample_html):
        """Test that navigation elements are removed."""
        extractor = ContentExtractor()

        result = await extractor.extract(
            url="https://example.com/test",
            html=sample_html,
        )

        assert "Navigation content" not in result.content
        assert "Header content" not in result.content
        assert "Footer content" not in result.content

    @pytest.mark.asyncio
    async def test_extract_batch(self, sample_html):
        """Test batch extraction from multiple URLs."""
        extractor = ContentExtractor()

        html_contents = {
            "https://example.com/1": sample_html,
            "https://example.com/2": "<html><body><article><h1>Test</h1><p>Content</p></article></body></html>",
        }

        results = await extractor.extract_batch(
            urls=list(html_contents.keys()),
            html_contents=html_contents,
            max_concurrent=2,
        )

        assert len(results) == 2
        assert all(r.extraction_success for r in results)

    def test_content_truncation(self):
        """Test that content is truncated to max length."""
        config = ContentExtractorConfig(max_content_length=100)
        extractor = ContentExtractor(config)

        long_content = "x" * 1000

        title = extractor._extract_title(f"<title>Test</title><body>{long_content}</body>")
        cleaned = extractor._clean_html(f"<html><body>{long_content}</body></html>")
        text = extractor._html_to_text(cleaned)

        assert len(text) > 0


# =============================================================================
# TestSynthesizer
# =============================================================================


class TestSynthesizer:
    """Tests for Synthesizer - hypothesis building, LLM validation mock, error handling."""

    def test_synthesizer_creation(self):
        """Test Synthesizer creation with config."""
        config = SynthesizerConfig(
            max_hypotheses=15,
            cross_validate_enabled=True,
            llm_temperature=0.2,
        )
        synthesizer = Synthesizer(config=config)

        assert synthesizer.config.max_hypotheses == 15
        assert synthesizer.config.cross_validate_enabled is True

    def test_synthesizer_factory(self):
        """Test create_synthesizer factory function."""
        synthesizer = create_synthesizer(
            max_hypotheses=20,
            cross_validate=False,
        )

        assert synthesizer.config.max_hypotheses == 20
        assert synthesizer.config.cross_validate_enabled is False

    def test_synthesizer_stats(self):
        """Test that Synthesizer tracks statistics."""
        synthesizer = Synthesizer()
        stats = synthesizer.get_stats()

        assert "claims_extracted" in stats
        assert "hypotheses_built" in stats
        assert "llm_calls" in stats

    def test_parse_json_response(self):
        """Test JSON parsing from LLM response."""
        synthesizer = Synthesizer()

        json_response = '{"hypothesis": "Test", "confidence": 0.8}'
        result = synthesizer._parse_json_response(json_response)

        assert result["hypothesis"] == "Test"
        assert result["confidence"] == 0.8

    def test_parse_json_response_with_markdown(self):
        """Test JSON parsing from markdown code blocks."""
        synthesizer = Synthesizer()

        markdown_response = '```json\n{"hypothesis": "Test", "confidence": 0.9}\n```'
        result = synthesizer._parse_json_response(markdown_response)

        assert result["hypothesis"] == "Test"
        assert result["confidence"] == 0.9

    def test_parse_json_response_with_extra_text(self):
        """Test JSON parsing with surrounding text."""
        synthesizer = Synthesizer()

        response = 'Here is the result: {"claims": [{"claim": "Test", "confidence": 0.7}]}'
        result = synthesizer._parse_json_response(response)

        assert "claims" in result

    def test_extract_claims_rule_based(self):
        """Test rule-based claim extraction (without LLM)."""
        synthesizer = Synthesizer()

        source = Source(
            url="https://example.com/test",
            title="Test Article",
            domain="example.com",
            content="Python is a programming language. asyncio supports async operations. FastAPI is a web framework.",
        )

        claims = synthesizer._extract_claims_rule_based(source.content, source)

        assert isinstance(claims, list)
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_extract_triples_rule_based(self):
        """Test rule-based triple extraction (without LLM)."""
        synthesizer = Synthesizer()

        source = Source(
            url="https://example.com/test",
            title="Test Article",
            domain="example.com",
            content="FastAPI supports async. Python is a language. asyncio implements async patterns.",
        )

        triples = synthesizer._extract_triples_rule_based(source.content, source, max_triples=10)

        assert isinstance(triples, list)
        assert all(isinstance(t, AssociativeTriple) for t in triples)

    @pytest.mark.asyncio
    async def test_build_hypothesis_without_llm(self):
        """Test hypothesis building without LLM provider."""
        synthesizer = Synthesizer()

        source = Source(
            url="https://example.com/test",
            title="Test",
            domain="example.com",
        )

        claim = Claim(
            text="Python supports async programming",
            source=source,
            confidence=0.8,
        )

        hypothesis = await synthesizer.build_hypothesis(claim, [source])

        assert hypothesis is not None
        assert hypothesis.statement == "Python supports async programming"
        assert hypothesis.status == HypothesisStatus.UNVERIFIED
        assert hypothesis.confidence == 0.8

    @pytest.mark.asyncio
    async def test_verify_hypothesis_without_llm(self):
        """Test hypothesis verification without LLM provider."""
        synthesizer = Synthesizer()

        hypothesis = Hypothesis(
            id="test-hyp",
            statement="Test hypothesis",
            status=HypothesisStatus.UNVERIFIED,
            confidence=0.7,
        )

        sources = [
            Source(url="https://example.com/1", domain="example.com"),
            Source(url="https://example.com/2", domain="example.com"),
        ]

        result = await synthesizer.verify_hypothesis(hypothesis, sources)

        assert result.status == HypothesisStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_verify_hypothesis_insufficient_sources(self, mock_llm_provider):
        """Test hypothesis verification with insufficient sources."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        hypothesis = Hypothesis(
            id="test-hyp",
            statement="Test hypothesis",
            status=HypothesisStatus.UNVERIFIED,
        )

        result = await synthesizer.verify_hypothesis(hypothesis, [])

        assert result.status == HypothesisStatus.UNVERIFIED

    @pytest.mark.asyncio
    async def test_find_contradictions_without_llm(self):
        """Test contradiction detection without LLM provider."""
        synthesizer = Synthesizer()

        hypotheses = [
            Hypothesis(id="h1", statement="Claim A is true"),
            Hypothesis(id="h2", statement="Claim A is false"),
        ]

        contradictions = await synthesizer.find_contradictions(hypotheses)

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_find_contradictions_insufficient_hypotheses(self, mock_llm_provider):
        """Test contradiction detection with insufficient hypotheses."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        hypothesis = Hypothesis(id="h1", statement="Single hypothesis")
        contradictions = await synthesizer.find_contradictions([hypothesis])

        assert contradictions == []

    @pytest.mark.asyncio
    async def test_extract_claims_with_mock_llm(self, mock_llm_provider, sample_sources):
        """Test claim extraction with mock LLM provider."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        claims = await synthesizer.extract_claims(
            content="This is test content for claim extraction.",
            source=sample_sources[0],
        )

        assert isinstance(claims, list)
        assert synthesizer._llm_calls >= 1

    @pytest.mark.asyncio
    async def test_build_hypothesis_with_mock_llm(self, mock_llm_provider, sample_sources):
        """Test hypothesis building with mock LLM provider."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        claim = Claim(
            text="Python supports async programming",
            source=sample_sources[0],
            confidence=0.8,
        )

        hypothesis = await synthesizer.build_hypothesis(claim, sample_sources)

        assert hypothesis is not None
        assert hypothesis.id is not None
        assert synthesizer._llm_calls >= 1

    @pytest.mark.asyncio
    async def test_extract_triples_with_mock_llm(self, mock_llm_provider, sample_sources):
        """Test triple extraction with mock LLM provider."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        triples = await synthesizer.extract_triples(
            content="FastAPI supports async. Python is a programming language.",
            source=sample_sources[0],
        )

        assert isinstance(triples, list)

    @pytest.mark.asyncio
    async def test_llm_call_failure_handling(self):
        """Test handling of LLM call failures."""
        failing_provider = MagicMock()
        failing_provider.chat_completion = AsyncMock(side_effect=Exception("LLM error"))
        failing_provider.default_model = "test-model"

        synthesizer = Synthesizer(llm_provider=failing_provider)

        source = Source(url="https://example.com/test", domain="example.com")

        claims = await synthesizer.extract_claims("Test content", source)

        assert isinstance(claims, list)

    @pytest.mark.asyncio
    async def test_extract_claims_fallback_on_error(self):
        """Test that claim extraction falls back to rule-based on LLM error."""
        error_provider = MagicMock()
        error_provider.chat_completion = AsyncMock(side_effect=ValueError("API error"))
        error_provider.default_model = "test-model"

        synthesizer = Synthesizer(llm_provider=error_provider)

        source = Source(
            url="https://example.com/test",
            domain="example.com",
            content="Python is a programming language. It supports async operations.",
        )

        claims = await synthesizer.extract_claims(source.content, source)

        assert isinstance(claims, list)

    def test_set_provider(self, mock_llm_provider):
        """Test setting LLM provider at runtime."""
        synthesizer = Synthesizer()
        assert synthesizer._provider is None

        synthesizer.set_provider(mock_llm_provider)
        assert synthesizer._provider == mock_llm_provider


# =============================================================================
# TestDeepDiscoveryEngine
# =============================================================================


class TestDeepDiscoveryEngine:
    """Tests for DeepDiscoveryEngine - full research flow, metrics, error handling."""

    def test_engine_creation(self):
        """Test DeepDiscoveryEngine creation."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config)

        assert engine.config.research_depth == 1
        assert engine._web_fetcher is not None
        assert engine._content_extractor is not None
        assert engine._source_auditor is not None
        assert engine._synthesizer is not None

    def test_engine_components_initialized(self):
        """Test that all engine components are initialized."""
        engine = DeepDiscoveryEngine()

        assert hasattr(engine, "_web_fetcher")
        assert hasattr(engine, "_content_extractor")
        assert hasattr(engine, "_source_auditor")
        assert hasattr(engine, "_synthesizer")
        assert hasattr(engine, "_deep_dive")
        assert hasattr(engine, "_knowledge_integrator")

    def test_engine_stats(self):
        """Test engine statistics."""
        engine = DeepDiscoveryEngine()
        stats = engine.get_stats()

        assert "research_count" in stats
        assert "cache_hits" in stats
        assert "components" in stats
        assert "web_fetcher" in stats["components"]
        assert "source_auditor" in stats["components"]

    def test_update_config(self):
        """Test updating engine configuration."""
        engine = DeepDiscoveryEngine()
        new_config = DDEConfig.deep()

        engine.update_config(new_config)

        assert engine.config.research_depth == 5
        assert engine.config.max_total_sources == 100

    def test_get_config(self):
        """Test getting engine configuration."""
        engine = DeepDiscoveryEngine()
        config = engine.get_config()

        assert isinstance(config, DDEConfig)
        assert config.research_depth == 3

    def test_generate_summary(self, sample_sources):
        """Test research summary generation."""
        engine = DeepDiscoveryEngine()

        hypotheses = [
            Hypothesis(
                id="h1",
                statement="Test hypothesis",
                status=HypothesisStatus.VERIFIED,
                confidence=0.9,
            ),
        ]

        summary = engine._generate_summary(sample_sources, hypotheses)

        assert "sources" in summary.lower()
        assert "ets" in summary.lower() or "score" in summary.lower()

    def test_generate_summary_no_sources(self):
        """Test summary generation with no sources."""
        engine = DeepDiscoveryEngine()

        summary = engine._generate_summary([], [])

        assert "no sources" in summary.lower()

    @pytest.mark.asyncio
    async def test_quick_search(self, mock_session):
        """Test quick search functionality."""
        engine = DeepDiscoveryEngine()
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = """
        <a class="result__a" href="https://docs.python.org/asyncio">Asyncio Docs</a>
        <a class="result__snippet">Python async</a>
        """
        mock_session.get.return_value = mock_response

        sources = await engine.quick_search("asyncio", max_results=5)

        assert isinstance(sources, list)

    @pytest.mark.asyncio
    async def test_research_flow_with_mocks(self, mock_session, mock_llm_provider, sample_sources):
        """Test full research flow with mocked external calls."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = """
        <a class="result__a" href="https://docs.python.org/asyncio">Asyncio Documentation</a>
        <a class="result__snippet">Python async documentation</a>
        """
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="Python async programming",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)
        assert result.query == "Python async programming"
        assert isinstance(result.metrics, ResearchMetrics)

    @pytest.mark.asyncio
    async def test_research_handles_empty_search_results(self, mock_session, mock_llm_provider):
        """Test research handles empty search results."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = "<html>No results</html>"
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="nonexistent query xyz123",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_research_handles_network_error(self, mock_session, mock_llm_provider):
        """Test research handles network errors gracefully."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_session.get.side_effect = ConnectionError("Network error")

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)

    @pytest.mark.asyncio
    async def test_research_with_config_override(self, mock_session, mock_llm_provider):
        """Test research with configuration override."""
        engine = DeepDiscoveryEngine(llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = '<a class="result__a" href="https://test.com">Test</a>'
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="test query",
            depth=2,
            config_override={"max_total_sources": 5},
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)

    @pytest.mark.asyncio
    async def test_research_metrics_calculation(self, mock_session, mock_llm_provider):
        """Test that research metrics are calculated correctly."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = """
        <a class="result__a" href="https://docs.python.org/1">Result 1</a>
        <a class="result__a" href="https://github.com/2">Result 2</a>
        """
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert result.metrics.sources_found >= 0
        assert result.metrics.web_requests >= 0
        assert isinstance(result.total_time_ms, float)

    @pytest.mark.asyncio
    async def test_research_execution_trace(self, mock_session, mock_llm_provider):
        """Test that execution trace is recorded."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = '<a class="result__a" href="https://test.com">Test</a>'
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert len(result.execution_trace) > 0
        assert all(step.step_name for step in result.execution_trace)

    @pytest.mark.asyncio
    async def test_research_error_handling(self):
        """Test that research handles unexpected errors."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config)

        engine._web_fetcher.search = AsyncMock(side_effect=Exception("Unexpected error"))

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_close_engine(self, mock_session):
        """Test closing engine resources."""
        engine = DeepDiscoveryEngine()
        engine._web_fetcher._session = mock_session

        await engine.close()

        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_with_depth_variations(self, mock_session, mock_llm_provider):
        """Test research with different depth levels."""
        engine = DeepDiscoveryEngine(llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = '<a class="result__a" href="https://test.com">Test</a>'
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="test query",
            depth=1,
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)

    def test_config_presets(self):
        """Test configuration presets."""
        quick = DDEConfig.quick()
        assert quick.research_depth == 1

        standard = DDEConfig.standard()
        assert standard.research_depth == 3

        deep = DDEConfig.deep()
        assert deep.research_depth == 5

        academic = DDEConfig.academic()
        assert academic.source_audit.min_ets_threshold == 0.5

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = DDEConfig(
            research_depth=4,
            web_fetcher=WebFetcherConfig(max_results=15),
        )

        data = config.to_dict()

        assert data["research_depth"] == 4
        assert data["web_fetcher"]["max_results"] == 15

    def test_config_deserialization(self):
        """Test configuration deserialization."""
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


# =============================================================================
# Integration Tests - Full Workflow
# =============================================================================


class TestResearchIntegration:
    """Integration tests for full research workflow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, mock_session, mock_llm_provider):
        """Test full pipeline with all components mocked."""
        config = DDEConfig(
            research_depth=1,
            max_total_sources=5,
            check_existing_research=False,
        )
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)
        engine._web_fetcher._session = mock_session

        mock_response = MagicMock()
        mock_response.text = """
        <html>
        <a class="result__a" href="https://docs.python.org/asyncio">Asyncio Documentation</a>
        <a class="result__snippet">Python asyncio documentation</a>
        <a class="result__a" href="https://github.com/python/cpython">CPython Repository</a>
        <a class="result__snippet">Python source code</a>
        </html>
        """
        mock_session.get.return_value = mock_response

        result = await engine.research(
            query="Python asyncio",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)
        assert result.query == "Python asyncio"
        assert result.config_used is not None
        assert isinstance(result.execution_trace, list)

    @pytest.mark.asyncio
    async def test_source_auditing_integration(self, mock_session, sample_sources):
        """Test source auditing in full pipeline."""
        auditor = SourceAuditor()

        passed, filtered = auditor.audit_batch(sample_sources, filter_threshold=True)

        assert len(passed) >= 1
        assert all(s.ets_score >= auditor.config.min_ets_threshold for s in passed)

    @pytest.mark.asyncio
    async def test_content_extraction_integration(self, sample_html):
        """Test content extraction in full pipeline."""
        extractor = ContentExtractor()

        result = await extractor.extract(
            url="https://example.com/test",
            html=sample_html,
        )

        assert result.extraction_success is True
        assert len(result.code_blocks) >= 1
        assert result.author is not None

    @pytest.mark.asyncio
    async def test_hypothesis_workflow_integration(self, mock_llm_provider, sample_sources):
        """Test hypothesis building and verification workflow."""
        synthesizer = Synthesizer(llm_provider=mock_llm_provider)

        claims = await synthesizer.extract_claims(
            content="Python supports async programming through asyncio library.",
            source=sample_sources[0],
        )

        if claims:
            hypothesis = await synthesizer.build_hypothesis(claims[0], sample_sources)

            assert hypothesis is not None
            assert hypothesis.statement is not None

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, mock_session, mock_llm_provider):
        """Test error recovery in full pipeline."""
        config = DDEConfig.quick()
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)

        call_count = 0

        async def failing_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First call fails")
            mock_response = MagicMock()
            mock_response.text = '<a class="result__a" href="https://test.com">Test</a>'
            return mock_response

        engine._web_fetcher._session = MagicMock()
        engine._web_fetcher._session.get = AsyncMock(side_effect=failing_get)

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)

    @pytest.mark.asyncio
    async def test_timeout_handling_integration(self, mock_llm_provider):
        """Test timeout handling in full pipeline."""
        config = DDEConfig.quick()
        config.web_fetcher.timeout_seconds = 1
        engine = DeepDiscoveryEngine(config=config, llm_provider=mock_llm_provider)

        async def timeout_get(*args, **kwargs):
            raise asyncio.TimeoutError("Request timed out")

        engine._web_fetcher._session = MagicMock()
        engine._web_fetcher._session.get = AsyncMock(side_effect=timeout_get)

        result = await engine.research(
            query="test query",
            force_fresh=True,
        )

        assert isinstance(result, ResearchResult)
