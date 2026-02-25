"""
Tests for Deep Discovery Engine Types
"""

import pytest
from datetime import datetime, date
from gaap.research.types import (
    ETSLevel,
    Source,
    SourceStatus,
    Hypothesis,
    HypothesisStatus,
    AssociativeTriple,
    ResearchFinding,
    ResearchResult,
    ResearchMetrics,
    ExecutionStep,
    Claim,
    SearchResult,
    ExtractedContent,
)


class TestETSLevel:
    def test_ets_levels(self):
        assert ETSLevel.VERIFIED.value == 1.0
        assert ETSLevel.RELIABLE.value == 0.7
        assert ETSLevel.QUESTIONABLE.value == 0.5
        assert ETSLevel.UNRELIABLE.value == 0.3
        assert ETSLevel.BLACKLISTED.value == 0.0


class TestSource:
    def test_source_creation(self):
        source = Source(
            url="https://docs.python.org/3/library/asyncio.html",
            title="asyncio - Asynchronous I/O",
            domain="docs.python.org",
            ets_score=1.0,
        )

        assert source.url == "https://docs.python.org/3/library/asyncio.html"
        assert source.domain == "docs.python.org"
        assert source.ets_score == 1.0
        assert source.status == SourceStatus.DISCOVERED

    def test_source_domain_extraction(self):
        source = Source(url="https://fastapi.tiangolo.com/tutorial/")

        assert source.domain == "fastapi.tiangolo.com"

    def test_source_hash(self):
        source = Source(
            url="https://example.com",
            content="This is some content for hashing",
        )

        hash_result = source.compute_hash()

        assert len(hash_result) == 16
        assert source.content_hash == hash_result

    def test_source_to_dict(self):
        source = Source(
            url="https://example.com",
            title="Example",
            ets_score=0.8,
        )

        result = source.to_dict()

        assert result["url"] == "https://example.com"
        assert result["title"] == "Example"
        assert result["ets_score"] == 0.8


class TestHypothesis:
    def test_hypothesis_creation(self):
        hypothesis = Hypothesis(
            id="hyp_123",
            statement="FastAPI supports async/await natively",
        )

        assert hypothesis.id == "hyp_123"
        assert hypothesis.status == HypothesisStatus.UNVERIFIED
        assert hypothesis.confidence == 0.0

    def test_hypothesis_status_checks(self):
        verified = Hypothesis(
            id="hyp_1",
            statement="Test",
            status=HypothesisStatus.VERIFIED,
        )

        falsified = Hypothesis(
            id="hyp_2",
            statement="Test",
            status=HypothesisStatus.FALSIFIED,
        )

        conflicted = Hypothesis(
            id="hyp_3",
            statement="Test",
            status=HypothesisStatus.CONFLICTED,
        )

        assert verified.is_verified
        assert not verified.is_falsified
        assert falsified.is_falsified
        assert conflicted.is_conflicted


class TestAssociativeTriple:
    def test_triple_creation(self):
        source = Source(url="https://example.com")
        triple = AssociativeTriple(
            subject="FastAPI",
            predicate="supports",
            object="async/await",
            source=source,
            confidence=0.9,
        )

        assert triple.subject == "FastAPI"
        assert triple.predicate == "supports"
        assert triple.object == "async/await"
        assert triple.confidence == 0.9

    def test_triple_to_tuple(self):
        source = Source(url="https://example.com")
        triple = AssociativeTriple(
            subject="A",
            predicate="B",
            object="C",
            source=source,
        )

        result = triple.to_tuple()

        assert result == ("A", "B", "C")


class TestSearchResult:
    def test_search_result_to_source(self):
        result = SearchResult(
            url="https://example.com",
            title="Example Title",
            snippet="Example snippet",
            rank=1,
            provider="duckduckgo",
        )

        source = result.to_source()

        assert source.url == "https://example.com"
        assert source.title == "Example Title"
        assert source.status == SourceStatus.DISCOVERED
        assert source.metadata["rank"] == 1


class TestExtractedContent:
    def test_extracted_content_to_source(self):
        content = ExtractedContent(
            url="https://example.com",
            title="Example",
            content="Some content here",
            author="John Doe",
        )

        source = content.to_source("example.com")

        assert source.url == "https://example.com"
        assert source.title == "Example"
        assert source.content == "Some content here"
        assert source.author == "John Doe"
        assert source.status == SourceStatus.FETCHED
        assert len(source.content_hash) == 16


class TestResearchMetrics:
    def test_metrics_to_dict(self):
        metrics = ResearchMetrics(
            sources_found=10,
            sources_fetched=8,
            hypotheses_verified=3,
        )

        result = metrics.to_dict()

        assert result["sources_found"] == 10
        assert result["sources_fetched"] == 8
        assert result["hypotheses_verified"] == 3


class TestResearchFinding:
    def test_finding_creation(self):
        source = Source(url="https://example.com")
        finding = ResearchFinding(
            id="finding_123",
            query="FastAPI async",
            sources=[source],
        )

        assert finding.id == "finding_123"
        assert finding.query == "FastAPI async"
        assert len(finding.sources) == 1

    def test_finding_to_dict(self):
        source = Source(url="https://example.com")
        finding = ResearchFinding(
            id="finding_1",
            query="Test query",
            sources=[source],
            hypotheses=[],
            triples=[],
        )

        result = finding.to_dict()

        assert result["id"] == "finding_1"
        assert result["query"] == "Test query"
        assert result["sources_count"] == 1


class TestResearchResult:
    def test_result_success(self):
        finding = ResearchFinding(query="test")
        result = ResearchResult(
            success=True,
            query="test query",
            finding=finding,
        )

        assert result.success
        assert result.query == "test query"
        assert result.finding is not None

    def test_result_failure(self):
        result = ResearchResult(
            success=False,
            query="test",
            error="Something went wrong",
        )

        assert not result.success
        assert result.error == "Something went wrong"

    def test_result_to_dict(self):
        finding = ResearchFinding(id="f1", query="test")
        metrics = ResearchMetrics(sources_found=5)
        result = ResearchResult(
            success=True,
            query="test",
            finding=finding,
            metrics=metrics,
            total_time_ms=100.5,
        )

        data = result.to_dict()

        assert data["success"]
        assert data["query"] == "test"
        assert data["total_time_ms"] == "100.5"
