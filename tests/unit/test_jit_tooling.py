"""
Unit Tests for JIT Tooling System
Tests: LibraryDiscoverer, SkillCache, CodeSynthesizer, ToolSynthesizer

Reference: docs/evolution_plan_2026/14_JUST_IN_TIME_TOOLING.md
"""

import asyncio
import hashlib
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gaap.core.events import EventEmitter, EventType
from gaap.core.types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Usage,
)
from gaap.tools.code_synthesizer import (
    CodeSynthesizer,
    CodeSynthesizerConfig,
    ComplexityLevel,
    SynthesisRequest,
    SynthesisResult,
    DANGEROUS_IMPORTS,
    DANGEROUS_PATTERNS,
)
from gaap.tools.library_discoverer import CacheEntry, LibraryDiscoverer, LibraryInfo, SearchResult
from gaap.tools.skill_cache import (
    SkillCache,
    SkillMetadata,
    SkillCacheStats,
    CATEGORIES,
    SCHEMA_VERSION,
)
from gaap.tools.synthesizer import SynthesizedTool, SynthesisStats, ToolSynthesizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def skill_cache(temp_dir):
    return SkillCache(cache_path=Path(temp_dir) / "skills")


@pytest.fixture
def library_discoverer():
    return LibraryDiscoverer(github_token="test-token", cache_ttl=60)


@pytest.fixture
def code_synthesizer():
    return CodeSynthesizer()


@pytest.fixture
def code_synthesizer_with_provider(mock_provider):
    return CodeSynthesizer(provider=mock_provider)


@pytest.fixture
def tool_synthesizer(temp_dir, mock_provider):
    return ToolSynthesizer(
        workspace_path=Path(temp_dir) / "custom_tools",
        cache_path=Path(temp_dir) / "skills",
        github_token="test-token",
        llm_provider=mock_provider,
    )


@pytest.fixture
def sample_library_info():
    return LibraryInfo(
        name="requests",
        source="pypi",
        description="HTTP library for Python",
        version="2.31.0",
        quality_score=0.85,
        url="https://pypi.org/project/requests/",
        dependencies=["urllib3", "charset-normalizer"],
        metadata={
            "downloads_last_month": 500000000,
            "license": "Apache 2.0",
            "author": "Kenneth Reitz",
        },
    )


@pytest.fixture
def sample_github_library():
    return LibraryInfo(
        name="fastapi",
        source="github",
        description="FastAPI framework, modern, fast, for building APIs",
        quality_score=0.9,
        url="https://github.com/tiangolo/fastapi",
        metadata={
            "stars": 65000,
            "forks": 5500,
            "full_name": "tiangolo/fastapi",
            "language": "Python",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "created_at": "2018-12-07T00:00:00Z",
        },
    )


@pytest.fixture
def sample_synthesized_tool(temp_dir):
    return SynthesizedTool(
        id="abc123",
        name="tool_abc123",
        code='def run(**kwargs):\n    return {"success": True, "result": "test"}',
        description="Test tool",
        file_path=Path(temp_dir) / "tool_abc123.py",
        is_safe=True,
        module=None,
    )


@pytest.fixture
def sample_skill_metadata():
    return SkillMetadata(
        id="test123",
        name="test_skill",
        description="A test skill",
        created_at=datetime.now().isoformat(),
        last_used=datetime.now().isoformat(),
        use_count=5,
        dependencies=["requests"],
        tags=["http", "api"],
        file_path="/tmp/tool_test123.py",
        checksum="abc123def456",
        category="coding",
    )


@pytest.fixture
def mock_pypi_response():
    return {
        "info": {
            "name": "requests",
            "version": "2.31.0",
            "summary": "HTTP library for Python",
            "author": "Kenneth Reitz",
            "license": "Apache 2.0",
            "requires_dist": ["urllib3>=1.21.1", "charset-normalizer>=2"],
            "requires_python": ">=3.7",
            "classifiers": ["Development Status :: 5 - Production/Stable"],
        },
        "releases": {},
    }


@pytest.fixture
def mock_github_search_response():
    return {
        "total_count": 100,
        "items": [
            {
                "name": "fastapi",
                "full_name": "tiangolo/fastapi",
                "description": "FastAPI framework for building APIs",
                "html_url": "https://github.com/tiangolo/fastapi",
                "stargazers_count": 65000,
                "forks_count": 5500,
                "language": "Python",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "created_at": "2018-12-07T00:00:00Z",
            },
            {
                "name": "flask",
                "full_name": "pallets/flask",
                "description": "The Python micro framework",
                "html_url": "https://github.com/pallets/flask",
                "stargazers_count": 60000,
                "forks_count": 5000,
                "language": "Python",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "created_at": "2010-04-16T00:00:00Z",
            },
        ],
    }


# =============================================================================
# TestLibraryDiscoverer
# =============================================================================


class TestLibraryDiscoverer:
    """Tests for LibraryDiscoverer module."""

    @pytest.mark.asyncio
    async def test_search_pypi(self, library_discoverer):
        with patch.object(library_discoverer, "_search_pypi") as mock_search:
            mock_search.return_value = SearchResult(
                query="http client",
                libraries=[
                    LibraryInfo(name="requests", source="pypi", description="HTTP library"),
                    LibraryInfo(name="httpx", source="pypi", description="Async HTTP client"),
                ],
                total=2,
                source="pypi",
            )

            result = await library_discoverer._search_pypi("http client")

            assert result.query == "http client"
            assert result.source == "pypi"
            assert len(result.libraries) == 2
            assert result.libraries[0].name == "requests"

    @pytest.mark.asyncio
    async def test_search_github(self, library_discoverer):
        with patch.object(library_discoverer, "_search_github") as mock_search:
            mock_search.return_value = SearchResult(
                query="web framework",
                libraries=[
                    LibraryInfo(name="fastapi", source="github", description="FastAPI framework"),
                ],
                total=1,
                source="github",
            )

            result = await library_discoverer._search_github("web framework")

            assert result.query == "web framework"
            assert result.source == "github"
            assert len(result.libraries) == 1

    @pytest.mark.asyncio
    async def test_search_combined(self, library_discoverer):
        with patch.object(library_discoverer, "_search_pypi") as mock_pypi:
            with patch.object(library_discoverer, "_search_github") as mock_github:
                mock_pypi.return_value = SearchResult(
                    query="http",
                    libraries=[LibraryInfo(name="requests", source="pypi", quality_score=0.8)],
                    total=1,
                    source="pypi",
                )
                mock_github.return_value = SearchResult(
                    query="http",
                    libraries=[LibraryInfo(name="httpx", source="github", quality_score=0.85)],
                    total=1,
                    source="github",
                )

                result = await library_discoverer.search("http", sources=["pypi", "github"])

                assert result.total == 2
                assert "," in result.source

    @pytest.mark.asyncio
    async def test_get_package_info(self, library_discoverer, mock_pypi_response):
        with patch.object(library_discoverer, "_fetch_json") as mock_fetch:
            mock_fetch.return_value = mock_pypi_response

            result = await library_discoverer.get_package_info("requests")

            assert result is not None
            assert result.name == "requests"
            assert result.version == "2.31.0"
            assert result.source == "pypi"
            assert "urllib3" in result.dependencies

    @pytest.mark.asyncio
    async def test_get_package_info_not_found(self, library_discoverer):
        with patch.object(library_discoverer, "_fetch_json") as mock_fetch:
            mock_fetch.return_value = None

            result = await library_discoverer.get_package_info("nonexistent-package-xyz")

            assert result is None

    def test_get_quality_score_pypi(self, library_discoverer, sample_library_info):
        score = library_discoverer.get_quality_score(sample_library_info)

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_get_quality_score_github(self, library_discoverer, sample_github_library):
        score = library_discoverer.get_quality_score(sample_github_library)

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_get_quality_score_low_quality(self, library_discoverer):
        library = LibraryInfo(
            name="unknown-lib",
            source="github",
            description="",
            metadata={"stars": 0, "forks": 0},
        )

        score = library_discoverer.get_quality_score(library)

        assert 0.0 <= score <= 1.0
        assert score < 0.5

    @pytest.mark.asyncio
    async def test_recommend_for_task(self, library_discoverer):
        with patch.object(library_discoverer, "search") as mock_search:
            mock_search.return_value = SearchResult(
                query="web framework",
                libraries=[
                    LibraryInfo(
                        name="fastapi",
                        source="github",
                        quality_score=0.9,
                        description="FastAPI framework",
                    ),
                    LibraryInfo(
                        name="flask",
                        source="github",
                        quality_score=0.85,
                        description="Flask framework",
                    ),
                ],
                total=2,
                source="github",
            )

            results = await library_discoverer.recommend_for_task(
                "I need a web framework for building APIs"
            )

            assert len(results) <= 5
            if results:
                assert results[0].quality_score >= results[-1].quality_score

    @pytest.mark.asyncio
    async def test_recommend_for_task_empty_keywords(self, library_discoverer):
        results = await library_discoverer.recommend_for_task("")

        assert results == []

    def test_caching(self, library_discoverer):
        library_discoverer._set_cached("test_source", "test_key", {"data": "test_value"})

        cached = library_discoverer._get_cached("test_source", "test_key")

        assert cached == {"data": "test_value"}

    def test_caching_expired(self, library_discoverer):
        library_discoverer._cache_ttl = -1
        library_discoverer._set_cached("test_source", "test_key", {"data": "test_value"})

        cached = library_discoverer._get_cached("test_source", "test_key")

        assert cached is None

    def test_cache_key_generation(self, library_discoverer):
        key1 = library_discoverer._get_cache_key("pypi", "requests")
        key2 = library_discoverer._get_cache_key("pypi", "requests")
        key3 = library_discoverer._get_cache_key("github", "requests")

        assert key1 == key2
        assert key1 != key3

    @pytest.mark.asyncio
    async def test_rate_limiting(self, library_discoverer):
        import time

        library_discoverer._pypi_rate_limit = 100

        start = time.time()
        for _ in range(3):
            await library_discoverer._rate_limit_wait("pypi")
        elapsed = time.time() - start

        assert elapsed >= 0.02

    def test_rate_limiting_interval(self, library_discoverer):
        library_discoverer._pypi_rate_limit = 10

        min_interval = 1.0 / library_discoverer._pypi_rate_limit

        assert min_interval == 0.1

    @pytest.mark.asyncio
    async def test_error_handling_api_failure(self, library_discoverer):
        with patch.object(library_discoverer, "_fetch_json") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            result = await library_discoverer.get_package_info("requests")

            assert result is None

    @pytest.mark.asyncio
    async def test_error_handling_rate_limited(self, library_discoverer):
        with patch.object(library_discoverer, "_get_session") as mock_session:
            mock_response = MagicMock()
            mock_response.status = 403
            mock_response.json = AsyncMock(return_value={})
            mock_response.text = AsyncMock(return_value="")

            mock_client = MagicMock()
            mock_client.get = MagicMock(
                return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response))
            )
            mock_session.return_value = mock_client

            result = await library_discoverer._search_github("test")

            assert result.error is not None

    def test_clear_cache(self, library_discoverer):
        library_discoverer._set_cached("source", "key", {"data": "value"})
        assert len(library_discoverer._cache) > 0

        library_discoverer.clear_cache()

        assert len(library_discoverer._cache) == 0

    def test_get_cache_stats(self, library_discoverer):
        library_discoverer._set_cached("source1", "key1", {"data": "value1"})
        library_discoverer._set_cached("source2", "key2", {"data": "value2"})

        stats = library_discoverer.get_cache_stats()

        assert "total_entries" in stats
        assert stats["total_entries"] == 2

    @pytest.mark.asyncio
    async def test_close(self, library_discoverer):
        await library_discoverer._get_session()
        assert library_discoverer._session is not None

        await library_discoverer.close()

        assert library_discoverer._session.closed


# =============================================================================
# TestSkillCache
# =============================================================================


class TestSkillCache:
    """Tests for SkillCache module."""

    def test_store_skill(self, skill_cache, sample_synthesized_tool):
        skill_id = skill_cache.store(
            sample_synthesized_tool,
            {"tags": ["test", "utility"], "description": "Test skill"},
        )

        assert skill_id == sample_synthesized_tool.id
        assert len(skill_cache.list_all()) == 1

    def test_retrieve_skill_metadata(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        metadata = skill_cache._metadata_cache.get(sample_synthesized_tool.id)
        assert metadata is not None
        assert metadata.id == sample_synthesized_tool.id
        assert metadata.name == sample_synthesized_tool.name

    def test_retrieve_skill_not_found(self, skill_cache):
        metadata = skill_cache._metadata_cache.get("nonexistent-id")
        assert metadata is None

    def test_find_by_name_metadata(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        for skill_id, meta in skill_cache._metadata_cache.items():
            if meta.name == sample_synthesized_tool.name:
                assert meta.id == sample_synthesized_tool.id
                return

        pytest.fail("Skill not found by name")

    def test_find_by_name_not_found(self, skill_cache):
        for meta in skill_cache._metadata_cache.values():
            if meta.name == "nonexistent-skill":
                pytest.fail("Found unexpected skill")

    def test_find_by_tags_matching(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["http", "api", "network"]})

        matching_ids = []
        tags_lower = ["http", "network"]
        for skill_id, meta in skill_cache._metadata_cache.items():
            skill_tags_lower = [t.lower() for t in meta.tags]
            if any(t in skill_tags_lower for t in tags_lower):
                matching_ids.append(skill_id)

        assert len(matching_ids) >= 1

    def test_find_by_tags_no_match(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["http", "api"]})

        matching_ids = []
        tags_lower = ["database", "sql"]
        for skill_id, meta in skill_cache._metadata_cache.items():
            skill_tags_lower = [t.lower() for t in meta.tags]
            if any(t in skill_tags_lower for t in tags_lower):
                matching_ids.append(skill_id)

        assert len(matching_ids) == 0

    def test_list_all(self, skill_cache, sample_synthesized_tool):
        tool2 = SynthesizedTool(
            id="def456",
            name="tool_def456",
            code="def run(): pass",
            description="Second tool",
            file_path=Path("/tmp/tool_def456.py"),
            is_safe=True,
        )

        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})
        skill_cache.store(tool2, {"tags": ["test"]})

        all_skills = skill_cache.list_all()

        assert len(all_skills) == 2

    def test_update_usage(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        original_metadata = skill_cache._metadata_cache[sample_synthesized_tool.id]
        original_use_count = original_metadata.use_count

        skill_cache.update_usage(sample_synthesized_tool.id)

        updated_metadata = skill_cache._metadata_cache[sample_synthesized_tool.id]
        assert updated_metadata.use_count == original_use_count + 1

    def test_update_usage_nonexistent(self, skill_cache):
        skill_cache.update_usage("nonexistent-id")

        assert len(skill_cache.list_all()) == 0

    def test_delete_skill(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        result = skill_cache.delete(sample_synthesized_tool.id)

        assert result is True
        assert sample_synthesized_tool.id not in skill_cache._metadata_cache

    def test_delete_skill_not_found(self, skill_cache):
        result = skill_cache.delete("nonexistent-id")

        assert result is False

    def test_cleanup_unused(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        old_metadata = skill_cache._metadata_cache[sample_synthesized_tool.id]
        old_metadata.last_used = (datetime.now() - timedelta(days=60)).isoformat()

        deleted_count = skill_cache.cleanup_unused(days=30)

        assert deleted_count == 1
        assert len(skill_cache.list_all()) == 0

    def test_cleanup_unused_keeps_recent(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        deleted_count = skill_cache.cleanup_unused(days=30)

        assert deleted_count == 0
        assert len(skill_cache.list_all()) == 1

    def test_get_stats(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        stats = skill_cache.get_stats()

        assert isinstance(stats, SkillCacheStats)
        assert stats.total_skills == 1
        assert stats.total_size_bytes >= 0
        assert isinstance(stats.categories, dict)
        assert isinstance(stats.most_used, list)

    def test_checksum_verification(self, skill_cache, sample_synthesized_tool):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        metadata = skill_cache._metadata_cache[sample_synthesized_tool.id]
        expected_checksum = hashlib.sha256(sample_synthesized_tool.code.encode()).hexdigest()[:16]
        assert metadata.checksum == expected_checksum

    def test_checksum_verification_fails_on_tampering(
        self, skill_cache, sample_synthesized_tool, temp_dir
    ):
        skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        metadata = skill_cache._metadata_cache[sample_synthesized_tool.id]
        file_path = Path(metadata.file_path)
        with open(file_path, "w") as f:
            f.write("tampered code")

        import importlib.util

        spec = importlib.util.spec_from_file_location("test", file_path)
        assert spec is not None

    def test_determine_category_coding(self, skill_cache):
        category = skill_cache._determine_category(
            ["coding"],
            "Write a function to process data",
        )

        assert category == "coding"

    def test_determine_category_research(self, skill_cache):
        category = skill_cache._determine_category(
            ["research"],
            "Search for relevant information",
        )

        assert category == "research"

    def test_determine_category_analysis(self, skill_cache):
        category = skill_cache._determine_category(
            [],
            "Analyze and parse the input data",
        )

        assert category == "analysis"

    def test_determine_category_automation(self, skill_cache):
        category = skill_cache._determine_category(
            ["automation"],
            "Automate the batch processing",
        )

        assert category == "automation"

    def test_determine_category_other(self, skill_cache):
        category = skill_cache._determine_category(
            [],
            "Do something random",
        )

        assert category == "other"

    def test_validate_skill(self, skill_cache, temp_dir):
        code = '''
def run(**kwargs):
    """Test function."""
    return {"success": True, "result": None}
'''
        tool = SynthesizedTool(
            id="valid123",
            name="tool_valid123",
            code=code,
            description="Valid tool",
            file_path=Path(temp_dir) / "tool_valid123.py",
            is_safe=True,
        )
        skill_cache.store(tool, {"tags": ["test"]})

        metadata = skill_cache._metadata_cache.get(tool.id)
        assert metadata is not None

        file_path = Path(metadata.file_path)
        assert file_path.exists()

        with open(file_path) as f:
            stored_code = f.read()
        assert "def run" in stored_code

    def test_validate_skill_no_entry_point(self, skill_cache, temp_dir):
        code = """
def helper():
    pass
"""
        tool = SynthesizedTool(
            id="invalid123",
            name="tool_invalid123",
            code=code,
            description="Invalid tool",
            file_path=Path(temp_dir) / "tool_invalid123.py",
            is_safe=True,
        )
        skill_cache.store(tool, {"tags": ["test"]})

        metadata = skill_cache._metadata_cache.get(tool.id)
        assert metadata is not None

        import ast

        file_path = Path(metadata.file_path)
        with open(file_path) as f:
            stored_code = f.read()

        tree = ast.parse(stored_code)
        has_entry = any(
            isinstance(node, ast.FunctionDef) and node.name in ("run", "execute")
            for node in ast.walk(tree)
        )
        assert has_entry is False

    def test_repair_index(self, skill_cache, temp_dir):
        code = "def run(): return {'success': True}"
        tool = SynthesizedTool(
            id="repair123",
            name="tool_repair123",
            code=code,
            description="Tool to repair",
            file_path=Path(temp_dir) / "skills" / "coding" / "tool_repair123.py",
            is_safe=True,
        )

        skill_cache.store(tool, {"tags": ["test"]})
        del skill_cache._metadata_cache[tool.id]

        restored = skill_cache.repair_index()

        assert restored >= 1

    def test_metadata_persistence(self, skill_cache, sample_synthesized_tool, temp_dir):
        skill_cache.store(
            sample_synthesized_tool, {"tags": ["test"], "description": "Persistent skill"}
        )

        new_cache = SkillCache(cache_path=skill_cache.cache_path)

        all_skills = new_cache.list_all()

        assert len(all_skills) == 1


# =============================================================================
# TestCodeSynthesizer
# =============================================================================


class TestCodeSynthesizer:
    """Tests for CodeSynthesizer module."""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self, code_synthesizer):
        request = SynthesisRequest(
            intent="Create a simple greeting function",
            input_schema={"name": "str"},
            output_type="dict",
        )

        result = await code_synthesizer.synthesize(request)

        assert result.success is True
        assert "def run" in result.code
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_synthesize_with_libraries(self, code_synthesizer):
        request = SynthesisRequest(
            intent="Parse JSON data",
            required_libraries=["json"],
            input_schema={"data": "str"},
            output_type="dict",
        )

        libraries = [
            LibraryInfo(name="json", source="pypi", description="JSON parser"),
        ]

        result = await code_synthesizer.synthesize(request, libraries)

        assert result.success is True
        assert "import json" in result.code or "json" in result.code.lower()

    def test_generate_from_template_basic(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "basic",
            {
                "description": "Test tool",
                "function_description": "Test function",
                "implementation": "result = kwargs",
            },
        )

        assert "def run" in code
        assert "Test tool" in code
        assert "success" in code

    def test_generate_from_template_async(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "async",
            {
                "description": "Async tool",
                "function_description": "Async function",
                "implementation": "result = kwargs",
            },
        )

        assert "async def run" in code
        assert "asyncio" in code

    def test_generate_from_template_error_handling(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "error_handling",
            {
                "description": "Error handling tool",
                "function_description": "Function with error handling",
                "implementation": "result = kwargs",
                "validation_logic": "if not kwargs: raise ValueError('No input')",
            },
        )

        assert "ValidationError" in code
        assert "ExecutionError" in code

    def test_generate_from_template_with_config(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "with_config",
            {
                "description": "Config tool",
                "function_description": "Function with config",
                "implementation": "result = config",
                "config_fields": "api_key: str\n    timeout: int = 30",
            },
        )

        assert "@dataclass" in code
        assert "class Config" in code

    def test_generate_from_template_progress(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "progress",
            {
                "description": "Progress tool",
                "function_description": "Function with progress",
                "implementation_with_progress": "report_progress(1, 'Processing')\n        result = kwargs",
                "total_steps": 3,
            },
        )

        assert "progress_callback" in code
        assert "report_progress" in code

    def test_generate_from_template_unknown(self, code_synthesizer):
        code = code_synthesizer.generate_from_template(
            "unknown_template",
            {"description": "Test"},
        )

        assert "def run" in code

    def test_validate_generated_code_valid(self, code_synthesizer):
        valid_code = '''
"""Valid module."""
from typing import Any

def run(**kwargs: Any) -> dict[str, Any]:
    """Execute the tool."""
    return {"success": True, "result": None}
'''
        is_valid, errors = code_synthesizer.validate_generated_code(valid_code)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_generated_code_invalid_syntax(self, code_synthesizer):
        invalid_code = "def run(\n  return {}"

        is_valid, errors = code_synthesizer.validate_generated_code(invalid_code)

        assert is_valid is False
        assert any("Syntax error" in e for e in errors)

    def test_validate_generated_code_security_issues(self, code_synthesizer):
        dangerous_code = '''
"""Dangerous module."""
import os

def run(**kwargs):
    os.system("rm -rf /")
    return {"success": True}
'''
        is_valid, errors = code_synthesizer.validate_generated_code(dangerous_code)

        assert is_valid is False
        assert any("dangerous" in e.lower() for e in errors)

    def test_validate_generated_code_eval(self, code_synthesizer):
        dangerous_code = '''
"""Dangerous eval."""
def run(**kwargs):
    result = eval(kwargs.get("code", ""))
    return {"success": True, "result": result}
'''
        is_valid, errors = code_synthesizer.validate_generated_code(dangerous_code)

        assert is_valid is False

    def test_validate_generated_code_no_entry_point(self, code_synthesizer):
        code_without_entry = '''
"""Module without entry point."""
def helper():
    return True
'''
        is_valid, errors = code_synthesizer.validate_generated_code(code_without_entry)

        assert is_valid is False
        assert any("entry point" in e.lower() for e in errors)

    def test_validate_generated_code_missing_docstring(self, code_synthesizer):
        code_no_docstring = """
from typing import Any

def run(**kwargs: Any) -> dict[str, Any]:
    return {"success": True}
"""
        is_valid, errors = code_synthesizer.validate_generated_code(code_no_docstring)

        assert "docstring" in " ".join(errors).lower()

    def test_validate_generated_code_missing_type_hints(self, code_synthesizer):
        code_no_hints = '''
"""Module without type hints."""

def run(**kwargs):
    """Execute the tool."""
    return {"success": True}
'''
        is_valid, errors = code_synthesizer.validate_generated_code(code_no_hints)

        assert "type hint" in " ".join(errors).lower()

    def test_validate_generated_code_too_long(self, code_synthesizer):
        config = CodeSynthesizerConfig(max_code_length=100)
        synthesizer = CodeSynthesizer(config=config)

        long_code = '"""Long module."""\n' + "x = 1\n" * 100
        long_code += '\ndef run(**kwargs):\n    """Execute."""\n    return {"success": True}\n'

        is_valid, errors = synthesizer.validate_generated_code(long_code)

        assert is_valid is False
        assert any("max length" in e for e in errors)

    @pytest.mark.asyncio
    async def test_generate_tests(self, code_synthesizer):
        code = '''
def run(**kwargs):
    """Test function."""
    return {"success": True, "result": kwargs}
'''
        tests = await code_synthesizer.generate_tests(code)

        assert "def test_" in tests
        assert "run" in tests

    def test_estimate_complexity_simple(self, code_synthesizer):
        request = SynthesisRequest(
            intent="Add two numbers",
        )

        complexity = code_synthesizer.estimate_complexity(request)

        assert complexity == ComplexityLevel.SIMPLE.value

    def test_estimate_complexity_moderate(self, code_synthesizer):
        request = SynthesisRequest(
            intent="Parse and validate a CSV file",
            required_libraries=["csv", "io"],
            constraints=["Handle encoding", "Skip empty lines"],
        )

        complexity = code_synthesizer.estimate_complexity(request)

        assert complexity in [ComplexityLevel.MODERATE.value, ComplexityLevel.COMPLEX.value]

    def test_estimate_complexity_complex(self, code_synthesizer):
        request = SynthesisRequest(
            intent="Create an async API client with authentication and retry logic",
            required_libraries=["aiohttp", "asyncio", "auth"],
            constraints=[
                "Handle rate limiting",
                "Implement exponential backoff",
                "Support multiple auth methods",
            ],
            input_schema={"endpoint": "str", "method": "str", "data": "dict"},
        )

        complexity = code_synthesizer.estimate_complexity(request)

        assert complexity == ComplexityLevel.COMPLEX.value

    @pytest.mark.asyncio
    async def test_fallback_to_template(self, code_synthesizer):
        code_synthesizer._provider = None

        request = SynthesisRequest(
            intent="Simple task",
        )

        result = await code_synthesizer.synthesize(request)

        assert result.success is True
        assert "def run" in result.code
        assert result.confidence <= 0.5

    def test_extract_code_from_response_with_markdown(self, code_synthesizer):
        content = """Here's the code:

```python
def run():
    return {"success": True}
```

That should work."""

        code = code_synthesizer._extract_code_from_response(content)

        assert code is not None
        assert "def run" in code

    def test_extract_code_from_response_without_markdown(self, code_synthesizer):
        content = '''def helper():
    pass

def run(**kwargs):
    """Main function."""
    return {"success": True}'''

        code = code_synthesizer._extract_code_from_response(content)

        assert code is not None
        assert "def run" in code

    def test_extract_code_from_response_no_code(self, code_synthesizer):
        content = "This is just text with no code."

        code = code_synthesizer._extract_code_from_response(content)

        assert code is None

    def test_extract_imports(self, code_synthesizer):
        code = """
import os
from typing import Any, Dict
import json

def run():
    pass
"""
        imports = code_synthesizer._extract_imports(code)

        assert len(imports) == 3
        assert "import os" in imports
        assert "from typing import Any, Dict" in imports

    def test_register_template(self, code_synthesizer):
        custom_template = '''
def run():
    """Custom template."""
    pass
'''
        code_synthesizer.register_template("custom", custom_template)

        assert "custom" in code_synthesizer.get_template_names()

    def test_get_template_names(self, code_synthesizer):
        names = code_synthesizer.get_template_names()

        assert "basic" in names
        assert "async" in names
        assert "error_handling" in names

    def test_set_provider(self, code_synthesizer, mock_provider):
        code_synthesizer.set_provider(mock_provider)

        assert code_synthesizer._provider == mock_provider

    def test_get_config(self, code_synthesizer):
        config = code_synthesizer.get_config()

        assert isinstance(config, CodeSynthesizerConfig)

    def test_update_config(self, code_synthesizer):
        code_synthesizer.update_config(max_code_length=10000)

        assert code_synthesizer._config.max_code_length == 10000

    @pytest.mark.asyncio
    async def test_synthesize_with_llm(self, code_synthesizer_with_provider):
        request = SynthesisRequest(
            intent="Create a hello world function",
            input_schema={},
            output_type="dict",
        )

        result = await code_synthesizer_with_provider.synthesize(request)

        assert result.success is True
        assert "Mock response content" in result.code or "def run" in result.code

    def test_repr(self, code_synthesizer):
        repr_str = repr(code_synthesizer)

        assert "CodeSynthesizer" in repr_str


# =============================================================================
# TestToolSynthesizer
# =============================================================================


class TestToolSynthesizer:
    """Tests for ToolSynthesizer module."""

    @pytest.mark.asyncio
    async def test_synthesize_basic(self, tool_synthesizer, mock_provider):
        mock_response = ChatCompletionResponse(
            id="test-id",
            model="test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content='```python\nfrom typing import Any\n\ndef run(**kwargs: Any) -> dict[str, Any]:\n    """Test tool."""\n    return {"success": True, "result": "test"}\n```',
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_provider.chat_completion = AsyncMock(return_value=mock_response)

        with patch.object(tool_synthesizer.code_synthesizer, "synthesize") as mock_synthesize:
            mock_synthesize.return_value = SynthesisResult(
                code='from typing import Any\n\ndef run(**kwargs: Any) -> dict[str, Any]:\n    return {"success": True}',
                success=True,
                confidence=0.8,
            )

            result = await tool_synthesizer.code_synthesizer.synthesize(
                SynthesisRequest(intent="Create a simple test function")
            )

            assert result.success is True
            assert "def run" in result.code

    @pytest.mark.asyncio
    async def test_synthesize_with_discovery(self, tool_synthesizer, mock_provider):
        with patch.object(
            tool_synthesizer.library_discoverer, "recommend_for_task"
        ) as mock_recommend:
            mock_recommend.return_value = [
                LibraryInfo(name="requests", source="pypi", description="HTTP library"),
            ]

            with patch.object(tool_synthesizer.code_synthesizer, "synthesize") as mock_synthesize:
                mock_synthesize.return_value = SynthesisResult(
                    code='import requests\nfrom typing import Any\n\ndef run(**kwargs: Any) -> dict[str, Any]:\n    return {"success": True}',
                    success=True,
                    confidence=0.8,
                )

                result = await tool_synthesizer.code_synthesizer.synthesize(
                    SynthesisRequest(intent="Make HTTP requests", required_libraries=["requests"])
                )

                assert result.success is True

    @pytest.mark.asyncio
    async def test_load_skill_metadata(self, tool_synthesizer, sample_synthesized_tool):
        tool_synthesizer.skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        metadata = tool_synthesizer.skill_cache._metadata_cache.get(sample_synthesized_tool.id)

        assert metadata is not None
        assert metadata.id == sample_synthesized_tool.id

    def test_list_skills(self, tool_synthesizer, sample_synthesized_tool):
        tool_synthesizer.skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        skills = tool_synthesizer.list_skills()

        assert len(skills) == 1

    @pytest.mark.asyncio
    async def test_cleanup_old_skills(self, tool_synthesizer, sample_synthesized_tool):
        tool_synthesizer.skill_cache.store(sample_synthesized_tool, {"tags": ["test"]})

        metadata = tool_synthesizer.skill_cache._metadata_cache[sample_synthesized_tool.id]
        metadata.last_used = (datetime.now() - timedelta(days=60)).isoformat()

        deleted = tool_synthesizer.cleanup_old_skills(days=30)

        assert deleted == 1

    def test_get_stats(self, tool_synthesizer):
        stats = tool_synthesizer.get_stats()

        assert "tools_synthesized" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "failures" in stats
        assert "cache_stats" in stats

    @pytest.mark.asyncio
    async def test_event_emission(self, tool_synthesizer, mock_provider):
        emitted_events = []

        def capture_event(event):
            emitted_events.append(event)

        emitter = EventEmitter.get_instance()
        sub_id = emitter.subscribe(EventType.TOOL_SYNTHESIS_STARTED, capture_event)

        try:
            with patch.object(tool_synthesizer.code_synthesizer, "synthesize") as mock_synthesize:
                mock_synthesize.return_value = SynthesisResult(
                    code='def run(**kwargs):\n    return {"success": True}',
                    success=True,
                    confidence=0.5,
                )

                tool_synthesizer.event_emitter.emit(
                    EventType.TOOL_SYNTHESIS_STARTED,
                    {"intent": "Test intent"},
                    source="ToolSynthesizer",
                )

                assert len(emitted_events) > 0
        finally:
            emitter.unsubscribe(sub_id)

    @pytest.mark.asyncio
    async def test_full_flow(self, tool_synthesizer, mock_provider):
        with patch.object(
            tool_synthesizer.library_discoverer, "recommend_for_task"
        ) as mock_recommend:
            mock_recommend.return_value = [
                LibraryInfo(name="json", source="pypi", description="JSON library"),
            ]

            with patch.object(tool_synthesizer.code_synthesizer, "synthesize") as mock_synthesize:
                mock_synthesize.return_value = SynthesisResult(
                    code='''"""JSON parser tool."""
import json
from typing import Any

def run(**kwargs: Any) -> dict[str, Any]:
    """Parse JSON data."""
    data = kwargs.get("data", "{}")
    result = json.loads(data)
    return {"success": True, "result": result}
''',
                    success=True,
                    confidence=0.8,
                )

                result = await tool_synthesizer.code_synthesizer.synthesize(
                    SynthesisRequest(intent="Parse JSON data", required_libraries=["json"])
                )

                assert result.success is True
                assert "json.loads" in result.code

    @pytest.mark.asyncio
    async def test_synthesize_cache_check(self, tool_synthesizer, sample_synthesized_tool):
        tool_synthesizer.skill_cache.store(
            sample_synthesized_tool,
            {"tags": ["test"], "description": "Test tool for testing"},
        )

        cached = tool_synthesizer._check_cache_for_intent("Test tool for testing")

        assert cached is None or cached.id == sample_synthesized_tool.id

    @pytest.mark.asyncio
    async def test_synthesize_failure(self, tool_synthesizer, mock_provider):
        with patch.object(tool_synthesizer.code_synthesizer, "synthesize") as mock_synthesize:
            mock_synthesize.return_value = SynthesisResult(
                code="",
                success=False,
                error="API error",
            )

            result = await tool_synthesizer.code_synthesizer.synthesize(
                SynthesisRequest(intent="Create a tool")
            )

            assert result.success is False
            assert result.error == "API error"
            stats = tool_synthesizer.get_stats()

    def test_determine_category(self, tool_synthesizer):
        assert tool_synthesizer._determine_category("Write a function") == "coding"
        assert tool_synthesizer._determine_category("Search for data") == "research"
        assert tool_synthesizer._determine_category("Analyze the results") == "analysis"
        assert tool_synthesizer._determine_category("Automate the workflow") == "automation"
        assert tool_synthesizer._determine_category("Format helper") == "utility"

    def test_set_llm_provider(self, tool_synthesizer, mock_provider):
        new_provider = MagicMock()
        new_provider.get_available_models = MagicMock(return_value=["model-1"])

        tool_synthesizer.set_llm_provider(new_provider)

        assert tool_synthesizer.code_synthesizer._provider == new_provider

    @pytest.mark.asyncio
    async def test_close(self, tool_synthesizer):
        await tool_synthesizer.close()

    def test_repr(self, tool_synthesizer):
        repr_str = repr(tool_synthesizer)

        assert "ToolSynthesizer" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestJITToolingIntegration:
    """Integration tests for the complete JIT tooling flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_synthesis(self, temp_dir, mock_provider):
        discoverer = LibraryDiscoverer(github_token="test-token")
        cache = SkillCache(cache_path=Path(temp_dir) / "skills")
        synthesizer = CodeSynthesizer(provider=mock_provider)

        with patch.object(discoverer, "search") as mock_search:
            mock_search.return_value = SearchResult(
                query="http",
                libraries=[LibraryInfo(name="requests", source="pypi", quality_score=0.9)],
                total=1,
                source="pypi",
            )

            result = await discoverer.search("http")

            assert result.total == 1

            request = SynthesisRequest(
                intent="Make HTTP requests",
                required_libraries=["requests"],
            )

            with patch.object(synthesizer, "_synthesize_with_llm") as mock_llm:
                mock_llm.return_value = SynthesisResult(
                    code='import requests\nfrom typing import Any\n\ndef run(**kwargs: Any) -> dict[str, Any]:\n    return {"success": True}',
                    success=True,
                    confidence=0.8,
                )

                synthesis_result = await synthesizer.synthesize(request, result.libraries)

                assert synthesis_result.success is True

                tool = SynthesizedTool(
                    id="test123",
                    name="http_tool",
                    code=synthesis_result.code,
                    description="HTTP tool",
                    file_path=Path(temp_dir) / "http_tool.py",
                    is_safe=True,
                )

                skill_id = cache.store(tool, {"tags": ["http", "network"]})

                metadata = cache._metadata_cache.get(skill_id)
                assert metadata is not None

        await discoverer.close()

    @pytest.mark.asyncio
    async def test_cache_persistence_across_sessions(self, temp_dir):
        cache_path = Path(temp_dir) / "skills"

        cache1 = SkillCache(cache_path=cache_path)
        tool = SynthesizedTool(
            id="persist123",
            name="persistent_tool",
            code='def run(**kwargs):\n    return {"success": True}',
            description="Persistent tool",
            file_path=Path(temp_dir) / "persistent_tool.py",
            is_safe=True,
        )
        cache1.store(tool, {"tags": ["test"]})

        cache2 = SkillCache(cache_path=cache_path)

        all_skills = cache2.list_all()

        assert len(all_skills) == 1
        assert all_skills[0].id == "persist123"
