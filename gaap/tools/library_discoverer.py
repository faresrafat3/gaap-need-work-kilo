"""
Library Discoverer Module

Discovers and evaluates Python libraries from PyPI and GitHub.

Features:
    - PyPI package search and metadata retrieval
    - GitHub repository search
    - Quality scoring based on multiple factors
    - Task-based recommendations
    - Rate limiting and caching

Usage:
    from gaap.tools.library_discoverer import LibraryDiscoverer, LibraryInfo

    discoverer = LibraryDiscoverer(github_token="optional-token")
    result = await discoverer.search("web framework")
    info = await discoverer.get_package_info("requests")
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp

from gaap.core.logging import get_standard_logger

logger = get_standard_logger("gaap.tools.library_discoverer")

PYPI_JSON_API = "https://pypi.org/pypi/{package}/json"
PYPI_SIMPLE_API = "https://pypi.org/simple/"
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
GITHUB_REPO_API = "https://api.github.com/repos/{owner}/{repo}"

CACHE_TTL_SECONDS = 3600
PYPI_RATE_LIMIT = 10
GITHUB_RATE_LIMIT_WITHOUT_TOKEN = 10
GITHUB_RATE_LIMIT_WITH_TOKEN = 5000

LICENSE_SCORES: dict[str, float] = {
    "mit": 1.0,
    "apache-2.0": 1.0,
    "apache 2.0": 1.0,
    "bsd-3-clause": 1.0,
    "bsd-2-clause": 0.95,
    "bsd": 0.95,
    "mpl-2.0": 0.9,
    "mozilla": 0.9,
    "lgpl": 0.85,
    "gpl-3.0": 0.7,
    "gpl": 0.7,
    "gpl-2.0": 0.65,
    "agpl": 0.6,
    "unlicense": 0.8,
    "cc0": 0.8,
    "proprietary": 0.3,
    "unknown": 0.5,
}


@dataclass
class LibraryInfo:
    """Information about a discovered library."""

    name: str
    source: str
    description: str = ""
    version: str = ""
    quality_score: float = 0.0
    url: str = ""
    dependencies: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "version": self.version,
            "quality_score": self.quality_score,
            "url": self.url,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }


@dataclass
class SearchResult:
    """Result of a library search."""

    query: str
    libraries: list[LibraryInfo] = field(default_factory=list)
    total: int = 0
    source: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "libraries": [lib.to_dict() for lib in self.libraries],
            "total": self.total,
            "source": self.source,
            "error": self.error,
        }


@dataclass
class CacheEntry:
    """Cached API response."""

    data: Any
    timestamp: float
    ttl: float

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class LibraryDiscoverer:
    """
    Discovers and evaluates Python libraries from multiple sources.

    Features:
        - PyPI package search and metadata
        - GitHub repository search
        - Quality scoring
        - Task-based recommendations
        - Rate limiting and caching
    """

    def __init__(
        self,
        github_token: str | None = None,
        cache_ttl: float = CACHE_TTL_SECONDS,
        timeout: float = 30.0,
    ):
        self._github_token = github_token or os.getenv("GITHUB_TOKEN")
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._cache: dict[str, CacheEntry] = {}
        self._session: aiohttp.ClientSession | None = None
        self._last_pypi_request: float = 0.0
        self._last_github_request: float = 0.0
        self._pypi_rate_limit = PYPI_RATE_LIMIT
        self._github_rate_limit = (
            GITHUB_RATE_LIMIT_WITH_TOKEN if self._github_token else GITHUB_RATE_LIMIT_WITHOUT_TOKEN
        )
        self._logger = logger

    def __repr__(self) -> str:
        return f"LibraryDiscoverer(cache_entries={len(self._cache)}, github_token={'set' if self._github_token else 'not set'})"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_cache_key(self, source: str, key: str) -> str:
        combined = f"{source}:{key}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cached(self, source: str, key: str) -> Any | None:
        cache_key = self._get_cache_key(source, key)
        entry = self._cache.get(cache_key)
        if entry and not entry.is_expired():
            return entry.data
        return None

    def _set_cached(self, source: str, key: str, data: Any) -> None:
        cache_key = self._get_cache_key(source, key)
        self._cache[cache_key] = CacheEntry(data=data, timestamp=time.time(), ttl=self._cache_ttl)

    async def _rate_limit_wait(self, source: str) -> None:
        if source == "pypi":
            min_interval = 1.0 / self._pypi_rate_limit
            elapsed = time.time() - self._last_pypi_request
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_pypi_request = time.time()
        elif source == "github":
            min_interval = 1.0 / self._github_rate_limit
            elapsed = time.time() - self._last_github_request
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_github_request = time.time()

    async def _fetch_json(
        self, url: str, headers: dict[str, str] | None = None, source: str = ""
    ) -> dict[str, Any] | None:
        try:
            await self._rate_limit_wait(source)
            session = await self._get_session()
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 403:
                    self._logger.warning(f"Rate limited by {source}: {url}")
                    return None
                elif response.status == 404:
                    return None
                else:
                    self._logger.error(f"HTTP {response.status} from {source}: {url}")
                    return None
        except asyncio.TimeoutError:
            self._logger.error(f"Timeout fetching {url}")
            return None
        except aiohttp.ClientError as e:
            self._logger.error(f"Client error fetching {url}: {e}")
            return None
        except Exception as e:
            self._logger.exception(f"Unexpected error fetching {url}: {e}")
            return None

    async def _search_pypi(self, query: str, limit: int = 20) -> SearchResult:
        cached = self._get_cached("pypi_search", query)
        if cached:
            return cached

        libraries: list[LibraryInfo] = []
        try:
            url = f"https://pypi.org/search/?q={query}&page=1"
            session = await self._get_session()
            await self._rate_limit_wait("pypi")

            async with session.get(url) as response:
                if response.status != 200:
                    return SearchResult(
                        query=query,
                        libraries=[],
                        source="pypi",
                        error=f"HTTP {response.status}",
                    )

                html = await response.text()

            import re

            package_pattern = r'<a[^>]*href="/project/([^/]+)/"[^>]*class="package-snippet"'
            desc_pattern = r'<p class="package-snippet__description">([^<]*)</p>'

            packages = re.findall(package_pattern, html)
            descriptions = re.findall(desc_pattern, html)

            for i, pkg in enumerate(packages[:limit]):
                desc = descriptions[i] if i < len(descriptions) else ""
                libraries.append(
                    LibraryInfo(
                        name=pkg,
                        source="pypi",
                        description=desc.strip(),
                        url=f"https://pypi.org/project/{pkg}/",
                    )
                )

            result = SearchResult(
                query=query,
                libraries=libraries,
                total=len(libraries),
                source="pypi",
            )
            self._set_cached("pypi_search", query, result)
            return result

        except Exception as e:
            self._logger.exception(f"Error searching PyPI: {e}")
            return SearchResult(query=query, source="pypi", error=str(e))

    async def _search_github(self, query: str, limit: int = 20) -> SearchResult:
        cached = self._get_cached("github_search", query)
        if cached:
            return cached

        libraries: list[LibraryInfo] = []
        try:
            headers = {}
            if self._github_token:
                headers["Authorization"] = f"token {self._github_token}"

            params = {
                "q": f"{query} language:Python",
                "sort": "stars",
                "order": "desc",
                "per_page": limit,
            }

            await self._rate_limit_wait("github")
            session = await self._get_session()
            url = f"{GITHUB_SEARCH_API}?q={params['q']}&sort={params['sort']}&order={params['order']}&per_page={params['per_page']}"

            async with session.get(url, headers=headers) as response:
                if response.status == 403:
                    return SearchResult(
                        query=query,
                        source="github",
                        error="Rate limited",
                    )
                if response.status != 200:
                    return SearchResult(
                        query=query,
                        source="github",
                        error=f"HTTP {response.status}",
                    )

                data = await response.json()

            items = data.get("items", [])

            for item in items:
                lib = LibraryInfo(
                    name=item.get("name", ""),
                    source="github",
                    description=item.get("description", "") or "",
                    url=item.get("html_url", ""),
                    metadata={
                        "stars": item.get("stargazers_count", 0),
                        "forks": item.get("forks_count", 0),
                        "full_name": item.get("full_name", ""),
                        "language": item.get("language"),
                        "updated_at": item.get("updated_at"),
                        "created_at": item.get("created_at"),
                    },
                )
                lib.quality_score = self.get_quality_score(lib)
                libraries.append(lib)

            total = data.get("total_count", len(libraries))
            result = SearchResult(
                query=query,
                libraries=libraries,
                total=total,
                source="github",
            )
            self._set_cached("github_search", query, result)
            return result

        except Exception as e:
            self._logger.exception(f"Error searching GitHub: {e}")
            return SearchResult(query=query, source="github", error=str(e))

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 20,
    ) -> SearchResult:
        """
        Search for libraries across multiple sources.

        Args:
            query: Search query string
            sources: List of sources to search ("pypi", "github")
            limit: Maximum results per source

        Returns:
            Combined search results from all sources
        """
        sources = sources or ["pypi", "github"]
        all_libraries: list[LibraryInfo] = []
        errors: list[str] = []

        tasks = []
        if "pypi" in sources:
            tasks.append(self._search_pypi(query, limit))
        if "github" in sources:
            tasks.append(self._search_github(query, limit))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            elif isinstance(result, SearchResult):
                if result.error:
                    errors.append(f"{result.source}: {result.error}")
                all_libraries.extend(result.libraries)

        all_libraries.sort(key=lambda x: x.quality_score, reverse=True)

        return SearchResult(
            query=query,
            libraries=all_libraries[: limit * 2],
            total=len(all_libraries),
            source=",".join(sources),
            error="; ".join(errors) if errors else None,
        )

    async def get_package_info(self, name: str) -> LibraryInfo | None:
        """
        Get detailed information about a PyPI package.

        Args:
            name: Package name

        Returns:
            Library info or None if not found
        """
        cached = self._get_cached("pypi_package", name)
        if cached:
            return cached

        try:
            url = PYPI_JSON_API.format(package=name)
            data = await self._fetch_json(url, source="pypi")

            if not data:
                return None

            info = data.get("info", {})
            releases = data.get("releases", {})

            latest_version = info.get("version", "")

            dependencies: list[str] = []
            requires_dist = info.get("requires_dist") or []
            for req in requires_dist:
                base = req.split(";")[0].split(">")[0].split("<")[0].split("=")[0].strip()
                if base:
                    dependencies.append(base)

            download_url = info.get("project_url", "") or f"https://pypi.org/project/{name}/"

            downloads = 0
            if "downloads" in data:
                downloads = data["downloads"].get("last_month", 0)

            lib = LibraryInfo(
                name=name,
                source="pypi",
                description=info.get("summary", "") or info.get("description", "")[:500],
                version=latest_version,
                url=download_url,
                dependencies=dependencies,
                metadata={
                    "author": info.get("author"),
                    "author_email": info.get("author_email"),
                    "license": info.get("license"),
                    "home_page": info.get("home_page"),
                    "project_urls": info.get("project_urls") or {},
                    "keywords": info.get("keywords"),
                    "classifiers": info.get("classifiers") or [],
                    "downloads_last_month": downloads,
                    "python_requires": info.get("requires_python"),
                },
            )

            lib.quality_score = self.get_quality_score(lib)
            self._set_cached("pypi_package", name, lib)
            return lib

        except Exception as e:
            self._logger.exception(f"Error getting package info for {name}: {e}")
            return None

    def get_quality_score(self, library: LibraryInfo) -> float:
        """
        Calculate a quality score for a library.

        Scoring factors:
        - Downloads/stars (popularity)
        - Recent updates (maintenance)
        - License type (openness)
        - Documentation quality

        Args:
            library: Library to score

        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0

        if library.source == "pypi":
            downloads = library.metadata.get("downloads_last_month", 0)
            if downloads > 0:
                import math

                score += min(0.4, math.log10(downloads + 1) / 7.0 * 0.4)

            if library.description:
                score += min(0.15, len(library.description) / 500 * 0.15)

            classifiers = library.metadata.get("classifiers", [])
            doc_classifiers = [c for c in classifiers if "Documentation" in c or "Status" in c]
            if doc_classifiers:
                score += 0.1

            license_name = (library.metadata.get("license") or "").lower()
            license_score = LICENSE_SCORES.get("unknown", 0.5)
            for key, val in LICENSE_SCORES.items():
                if key in license_name:
                    license_score = val
                    break
            score += license_score * 0.15

            py_requires = library.metadata.get("python_requires")
            if py_requires:
                score += 0.1

        elif library.source == "github":
            stars = library.metadata.get("stars", 0)
            if stars > 0:
                import math

                score += min(0.35, math.log10(stars + 1) / 6.0 * 0.35)

            forks = library.metadata.get("forks", 0)
            if forks > 0:
                import math

                score += min(0.15, math.log10(forks + 1) / 5.0 * 0.15)

            updated_at = library.metadata.get("updated_at")
            if updated_at:
                try:
                    last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    days_since = (datetime.now(timezone.utc) - last_update).days
                    if days_since < 30:
                        score += 0.2
                    elif days_since < 90:
                        score += 0.15
                    elif days_since < 180:
                        score += 0.1
                    elif days_since < 365:
                        score += 0.05
                except Exception:
                    pass

            if library.description:
                score += min(0.1, len(library.description) / 500 * 0.1)

        return min(1.0, max(0.0, score))

    async def recommend_for_task(
        self,
        task_description: str,
        limit: int = 5,
    ) -> list[LibraryInfo]:
        """
        Recommend libraries based on a task description.

        Analyzes the task description to extract keywords and
        searches for relevant libraries, ranking by quality.

        Args:
            task_description: Description of the task
            limit: Maximum number of recommendations

        Returns:
            List of recommended libraries sorted by relevance
        """
        keywords = self._extract_keywords(task_description)

        if not keywords:
            return []

        search_queries = keywords[:3]

        all_results: list[LibraryInfo] = []
        seen_names: set[str] = set()

        for query in search_queries:
            result = await self.search(query, limit=10)
            for lib in result.libraries:
                key = f"{lib.source}:{lib.name}"
                if key not in seen_names:
                    seen_names.add(key)
                    all_results.append(lib)

        all_results.sort(key=lambda x: x.quality_score, reverse=True)

        return all_results[:limit]

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from a task description."""
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "about",
            "against",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "want",
            "need",
            "using",
            "use",
            "python",
            "library",
            "package",
            "module",
            "code",
            "implement",
        }

        tech_keywords = {
            "web": "web framework",
            "http": "http client",
            "api": "api client",
            "database": "database",
            "sql": "sql database",
            "orm": "orm",
            "testing": "testing",
            "test": "testing",
            "async": "async",
            "asyncio": "asyncio",
            "data": "data processing",
            "science": "data science",
            "ml": "machine learning",
            "machine": "machine learning",
            "learning": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "nlp natural language",
            "text": "text processing",
            "image": "image processing",
            "gui": "gui",
            "ui": "user interface",
            "cli": "cli command line",
            "command": "cli",
            "logging": "logging",
            "config": "configuration",
            "security": "security",
            "crypto": "cryptography",
            "auth": "authentication",
            "cache": "caching",
            "queue": "queue message",
            "task": "task queue",
            "celery": "celery",
            "django": "django",
            "flask": "flask",
            "fastapi": "fastapi",
            "requests": "http requests",
            "pandas": "pandas data",
            "numpy": "numpy array",
            "scipy": "scipy scientific",
        }

        words = text.lower().split()
        keywords: list[str] = []

        for word in words:
            clean = "".join(c for c in word if c.isalnum())
            if clean and clean not in stop_words and len(clean) > 2:
                if clean in tech_keywords:
                    keywords.append(tech_keywords[clean])
                else:
                    keywords.append(clean)

        return list(dict.fromkeys(keywords))

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        expired = sum(1 for e in self._cache.values() if e.is_expired())
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired,
            "valid_entries": len(self._cache) - expired,
        }
