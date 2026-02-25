"""
Web Fetcher - Search and Content Retrieval
==========================================

Web search with multiple provider support.
Default: DuckDuckGo (free, no API key required)

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
import urllib.parse
from typing import Any, TYPE_CHECKING

from .types import SearchResult, Source, SourceStatus
from .config import WebFetcherConfig

if TYPE_CHECKING:
    from gaap.providers.async_session import AsyncSessionManager

logger = logging.getLogger("gaap.research.web_fetcher")


class WebFetcher:
    """
    Web search and content fetching with multiple provider support.

    Features:
    - DuckDuckGo search (free, default)
    - Serper/Google search (API key required)
    - Perplexity search (API key required)
    - Content fetching with rate limiting
    - Batch fetching support

    Usage:
        config = WebFetcherConfig(provider="duckduckgo")
        fetcher = WebFetcher(config)

        results = await fetcher.search("FastAPI async")
        content = await fetcher.fetch_content(results[0].url)
    """

    def __init__(
        self,
        config: WebFetcherConfig | None = None,
        session_manager: AsyncSessionManager | None = None,
    ) -> None:
        self.config = config or WebFetcherConfig()
        self._session = session_manager
        self._own_session = session_manager is None

        self._search_count = 0
        self._fetch_count = 0
        self._error_count = 0
        self._total_time_ms = 0.0

        self._rate_limit_lock = asyncio.Lock()
        self._last_request_time = 0.0

        self._logger = logger

    async def _get_session(self) -> Any:
        """Get or create async session."""
        if self._session is None:
            from gaap.providers.async_session import AsyncSessionManager

            self._session = AsyncSessionManager()
        return self._session

    async def _rate_limit(self) -> None:
        """Apply rate limiting."""
        async with self._rate_limit_lock:
            now = time.time()
            min_delay = 1.0 / self.config.rate_limit_per_second
            elapsed = now - self._last_request_time
            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)
            self._last_request_time = time.time()

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """
        Search the web for results.

        Args:
            query: Search query
            max_results: Maximum results (uses config if not specified)

        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        max_results = max_results or self.config.max_results

        try:
            await self._rate_limit()

            if self.config.provider == "duckduckgo":
                results = await self._search_duckduckgo(query, max_results)
            elif self.config.provider == "serper":
                results = await self._search_serper(query, max_results)
            elif self.config.provider == "perplexity":
                results = await self._search_perplexity(query, max_results)
            elif self.config.provider == "brave":
                results = await self._search_brave(query, max_results)
            else:
                results = await self._search_duckduckgo(query, max_results)

            self._search_count += 1
            self._logger.info(f"Search '{query[:50]}': {len(results)} results")

            return results

        except Exception as e:
            self._error_count += 1
            self._logger.error(f"Search failed: {e}")
            return []
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using DuckDuckGo HTML (free, no API key)."""
        results: list[SearchResult] = []

        try:
            session = await self._get_session()

            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html",
                "Accept-Language": "en-US,en;q=0.9",
            }

            response = await session.get(
                url,
                headers=headers,
                timeout=self.config.timeout_seconds,
            )

            html = response.text if hasattr(response, "text") else str(response)

            results = self._parse_ddg_html(html, max_results)

        except Exception as e:
            self._logger.warning(f"DuckDuckGo search error: {e}")

        return results

    def _parse_ddg_html(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse DuckDuckGo HTML response."""
        results: list[SearchResult] = []

        result_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>', re.IGNORECASE
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>', re.IGNORECASE
        )

        result_matches = result_pattern.findall(html)
        snippet_matches = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(result_matches[:max_results]):
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                continue

            snippet = snippet_matches[i] if i < len(snippet_matches) else ""

            clean_url = self._clean_ddg_url(url)

            results.append(
                SearchResult(
                    url=clean_url,
                    title=self._clean_text(title),
                    snippet=self._clean_text(snippet),
                    rank=i + 1,
                    provider="duckduckgo",
                )
            )

        return results

    def _clean_ddg_url(self, url: str) -> str:
        """Clean DuckDuckGo redirect URL."""
        if "duckduckgo.com/l/?uddg=" in url:
            match = re.search(r"uddg=([^&]+)", url)
            if match:
                return urllib.parse.unquote(match.group(1))
        return url

    def _clean_text(self, text: str) -> str:
        """Clean HTML entities and whitespace."""
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text)
        text = urllib.parse.unquote(text)
        return text.strip()

    async def _search_serper(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using Serper (Google) API."""
        if not self.config.api_key:
            self._logger.warning("Serper API key not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, max_results)

        results: list[SearchResult] = []

        try:
            session = await self._get_session()

            response = await session.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.config.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
                timeout=self.config.timeout_seconds,
            )

            data = response.json() if hasattr(response, "json") else {}

            for i, item in enumerate(data.get("organic", [])[:max_results]):
                results.append(
                    SearchResult(
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        rank=i + 1,
                        provider="serper",
                    )
                )

        except Exception as e:
            self._logger.warning(f"Serper search error: {e}")

        return results

    async def _search_perplexity(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using Perplexity API."""
        if not self.config.api_key:
            self._logger.warning("Perplexity API key not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, max_results)

        results: list[SearchResult] = []

        try:
            session = await self._get_session()

            response = await session.post(
                "https://api.perplexity.ai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.1-sonar-small-128k-online",
                    "messages": [{"role": "user", "content": query}],
                },
                timeout=self.config.timeout_seconds,
            )

            data = response.json() if hasattr(response, "json") else {}
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
            urls = url_pattern.findall(content)[:max_results]

            for i, url in enumerate(urls):
                results.append(
                    SearchResult(
                        url=url,
                        title=f"Result {i + 1}",
                        snippet="",
                        rank=i + 1,
                        provider="perplexity",
                    )
                )

        except Exception as e:
            self._logger.warning(f"Perplexity search error: {e}")

        return results

    async def _search_brave(
        self,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        """Search using Brave Search API."""
        if not self.config.api_key:
            self._logger.warning("Brave API key not set, falling back to DuckDuckGo")
            return await self._search_duckduckgo(query, max_results)

        results: list[SearchResult] = []

        try:
            session = await self._get_session()

            response = await session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": self.config.api_key,
                    "Accept": "application/json",
                },
                params={"q": query, "count": max_results},
                timeout=self.config.timeout_seconds,
            )

            data = response.json() if hasattr(response, "json") else {}

            for i, item in enumerate(data.get("web", {}).get("results", [])[:max_results]):
                results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        rank=i + 1,
                        provider="brave",
                    )
                )

        except Exception as e:
            self._logger.warning(f"Brave search error: {e}")

        return results

    async def fetch_content(self, url: str) -> str:
        """
        Fetch content from a URL.

        Args:
            url: URL to fetch

        Returns:
            Raw content (HTML or text)
        """
        start_time = time.time()

        try:
            await self._rate_limit()

            session = await self._get_session()

            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain",
            }

            response = await session.get(
                url,
                headers=headers,
                timeout=self.config.timeout_seconds,
                follow_redirects=True,
            )

            content = response.text if hasattr(response, "text") else str(response)

            self._fetch_count += 1
            self._logger.debug(f"Fetched {url}: {len(content)} bytes")

            return content

        except Exception as e:
            self._error_count += 1
            self._logger.warning(f"Failed to fetch {url}: {e}")
            return ""
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    async def fetch_batch(
        self,
        urls: list[str],
        max_concurrent: int = 5,
    ) -> dict[str, str]:
        """
        Fetch content from multiple URLs in parallel.

        Args:
            urls: URLs to fetch
            max_concurrent: Maximum concurrent fetches

        Returns:
            Dict mapping URL to content
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(url: str) -> tuple[str, str]:
            async with semaphore:
                content = await self.fetch_content(url)
                return url, content

        tasks = [fetch_one(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, str] = {}
        for result in results:
            if isinstance(result, BaseException):
                continue
            url, content = result
            output[url] = content

        return output

    def set_provider(
        self,
        provider: str,
        api_key: str | None = None,
    ) -> None:
        """Switch provider at runtime."""
        self.config.provider = provider  # type: ignore
        if api_key:
            self.config.api_key = api_key
        self._logger.info(f"Switched to {provider} provider")

    def get_stats(self) -> dict[str, Any]:
        """Get fetcher statistics."""
        return {
            "provider": self.config.provider,
            "search_count": self._search_count,
            "fetch_count": self._fetch_count,
            "error_count": self._error_count,
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "avg_time_per_request_ms": f"{self._total_time_ms / max(1, self._search_count + self._fetch_count):.1f}",
        }

    async def close(self) -> None:
        """Close session if we own it."""
        if self._own_session and self._session:
            if hasattr(self._session, "close"):
                await self._session.close()


def create_web_fetcher(
    provider: str = "duckduckgo",
    api_key: str | None = None,
    max_results: int = 10,
) -> WebFetcher:
    """Create a WebFetcher with specified settings."""
    config = WebFetcherConfig(
        provider=provider,  # type: ignore
        api_key=api_key,
        max_results=max_results,
    )
    return WebFetcher(config)
