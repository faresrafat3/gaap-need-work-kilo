"""
Content Extractor - Clean Text Extraction from URLs
===================================================

Extracts clean text content from web pages, handling
HTML, code blocks, metadata, and links.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import date, datetime
from typing import Any
from urllib.parse import urljoin

from .config import ContentExtractorConfig
from .types import ExtractedContent

logger = logging.getLogger("gaap.research.content_extractor")


class ContentExtractor:
    """
    Extract clean text content from web pages.

    Features:
    - HTML to clean text conversion
    - Code block extraction
    - Link extraction
    - Metadata extraction (author, date)
    - Main content detection (ignore nav, ads, etc.)

    Usage:
        extractor = ContentExtractor()
        content = await extractor.extract("https://example.com/article")
        print(content.content)
    """

    HTML_TAGS_TO_REMOVE = [
        "script",
        "style",
        "nav",
        "header",
        "footer",
        "aside",
        "iframe",
        "noscript",
        "svg",
        "form",
        "button",
        "input",
        "select",
        "textarea",
        "advertisement",
        "ad",
        "sidebar",
    ]

    MAIN_CONTENT_SELECTORS = [
        "article",
        "main",
        ".content",
        ".post",
        ".article",
        "#content",
        "#main",
        ".entry-content",
        ".post-content",
        ".article-content",
    ]

    AUTHOR_PATTERNS = [
        re.compile(r"(?:author|by|written by)[:\s]+([A-Za-z\s]+)", re.I),
        re.compile(r'<meta[^>]*name=["\']author["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'rel=["\']author["\'][^>]*>([^<]+)<', re.I),
    ]

    DATE_PATTERNS = [
        re.compile(
            r'<meta[^>]*property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)["\']',
            re.I,
        ),
        re.compile(r'<meta[^>]*name=["\']date["\'][^>]*content=["\']([^"\']+)["\']', re.I),
        re.compile(r'<time[^>]*datetime=["\']([^"\']+)["\']', re.I),
        re.compile(r"(?:published|posted|date)[:\s]+(\d{4}[-/]\d{2}[-/]\d{2})", re.I),
    ]

    def __init__(
        self,
        config: ContentExtractorConfig | None = None,
    ) -> None:
        self.config = config or ContentExtractorConfig()

        self._extracted_count = 0
        self._failed_count = 0
        self._total_time_ms = 0.0
        self._total_bytes = 0

        self._logger = logger

    async def extract(self, url: str, html: str | None = None) -> ExtractedContent:
        """
        Extract content from a URL.

        Args:
            url: URL to extract from
            html: Optional pre-fetched HTML

        Returns:
            ExtractedContent with clean text and metadata
        """
        start_time = time.time()

        try:
            if html is None:
                from .web_fetcher import WebFetcher

                fetcher = WebFetcher()
                html = await fetcher.fetch_content(url)

            if not html:
                return ExtractedContent(
                    url=url,
                    extraction_success=False,
                    error="Failed to fetch content",
                )

            clean_html = self._clean_html(html)

            main_content = self._extract_main_content(clean_html)

            title = self._extract_title(clean_html)

            text = self._html_to_text(main_content)

            author = self._extract_author(html) if self.config.extract_metadata else None
            publish_date = self._extract_date(html) if self.config.extract_metadata else None

            links = self._extract_links(html, url) if self.config.extract_links else []

            code_blocks = self._extract_code_blocks(html) if self.config.extract_code_blocks else []

            extraction_time = (time.time() - start_time) * 1000

            self._extracted_count += 1
            self._total_time_ms += extraction_time
            self._total_bytes += len(text)

            return ExtractedContent(
                url=url,
                title=title,
                content=text[: self.config.max_content_length],
                author=author,
                publish_date=publish_date,
                links=links,
                code_blocks=code_blocks,
                extraction_success=True,
                extraction_time_ms=extraction_time,
            )

        except Exception as e:
            self._failed_count += 1
            self._logger.warning(f"Failed to extract {url}: {e}")
            return ExtractedContent(
                url=url,
                extraction_success=False,
                error=str(e),
            )

    async def extract_batch(
        self,
        urls: list[str],
        html_contents: dict[str, str] | None = None,
        max_concurrent: int = 5,
    ) -> list[ExtractedContent]:
        """
        Extract content from multiple URLs in parallel.

        Args:
            urls: URLs to extract
            html_contents: Optional pre-fetched HTML mapping
            max_concurrent: Maximum concurrent extractions

        Returns:
            List of ExtractedContent
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(url: str) -> ExtractedContent:
            async with semaphore:
                html = html_contents.get(url) if html_contents else None
                return await self.extract(url, html)

        tasks = [extract_one(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return list(results)

    def _clean_html(self, html: str) -> str:
        """Remove unwanted HTML elements."""
        clean = html

        for tag in self.HTML_TAGS_TO_REMOVE:
            clean = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", "", clean, flags=re.DOTALL | re.IGNORECASE)

        if self.config.remove_scripts:
            clean = re.sub(r"<script[^>]*>.*?</script>", "", clean, flags=re.DOTALL | re.I)
        if self.config.remove_styles:
            clean = re.sub(r"<style[^>]*>.*?</style>", "", clean, flags=re.DOTALL | re.I)

        clean = re.sub(r"<!--.*?-->", "", clean, flags=re.DOTALL)

        return clean

    def _extract_main_content(self, html: str) -> str:
        """Extract main content area."""
        for selector in self.MAIN_CONTENT_SELECTORS:
            if selector.startswith("."):
                pattern = (
                    rf'<[^>]*class=["\'][^"\']*\b{selector[1:]}\b[^"\']*["\'][^>]*>(.*?)</[^>]+>'
                )
            elif selector.startswith("#"):
                pattern = rf'<[^>]*id=["\']\b{selector[1:]}\b["\'][^>]*>(.*?)</[^>]+>'
            else:
                pattern = rf"<{selector}[^>]*>(.*?)</{selector}>"

            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1)

        return html

    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        if match:
            return self._clean_text(match.group(1))

        match = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
        if match:
            return self._clean_text(match.group(1))

        return ""

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to clean text."""
        text = html

        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<li[^>]*>", "\n• ", text, flags=re.IGNORECASE)

        text = re.sub(r"<[^>]+>", "", text)

        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&amp;", "&", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&quot;", '"', text)
        text = re.sub(r"&#39;", "'", text)

        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()

    def _extract_author(self, html: str) -> str | None:
        """Extract author from HTML."""
        for pattern in self.AUTHOR_PATTERNS:
            match = pattern.search(html)
            if match:
                return self._clean_text(match.group(1))
        return None

    def _extract_date(self, html: str) -> date | None:
        """Extract publication date from HTML."""
        for pattern in self.DATE_PATTERNS:
            match = pattern.search(html)
            if match:
                date_str = match.group(1)
                try:
                    if "T" in date_str:
                        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
                    date_str = date_str.replace("/", "-")
                    return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_links(self, html: str, base_url: str) -> list[str]:
        """Extract links from HTML."""
        links: list[str] = []

        pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)

        for match in pattern.finditer(html):
            href = match.group(1)

            if href.startswith("#") or href.startswith("javascript:"):
                continue

            absolute_url = urljoin(base_url, href)

            if absolute_url.startswith(("http://", "https://")):
                links.append(absolute_url)

        return list(set(links))

    def _extract_code_blocks(self, html: str) -> list[str]:
        """Extract code blocks from HTML."""
        code_blocks: list[str] = []

        pre_pattern = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
        code_pattern = re.compile(r"<code[^>]*>(.*?)</code>", re.DOTALL | re.IGNORECASE)

        for match in pre_pattern.finditer(html):
            code = re.sub(r"<[^>]+>", "", match.group(1))
            code = code.strip()
            if len(code) > 20:
                code_blocks.append(code)

        for match in code_pattern.finditer(html):
            code = re.sub(r"<[^>]+>", "", match.group(1))
            code = code.strip()
            if len(code) > 20 and code not in code_blocks:
                code_blocks.append(code)

        return code_blocks

    def get_stats(self) -> dict[str, Any]:
        """Get extractor statistics."""
        return {
            "extracted_count": self._extracted_count,
            "failed_count": self._failed_count,
            "success_rate": f"{self._extracted_count / max(1, self._extracted_count + self._failed_count):.1%}",
            "total_bytes_extracted": self._total_bytes,
            "avg_bytes_per_extraction": f"{self._total_bytes / max(1, self._extracted_count):.0f}",
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "avg_time_per_extraction_ms": f"{self._total_time_ms / max(1, self._extracted_count):.1f}",
        }


def create_content_extractor(
    max_content_length: int = 50000,
    extract_code_blocks: bool = True,
    extract_links: bool = True,
) -> ContentExtractor:
    """Create a ContentExtractor with specified settings."""
    config = ContentExtractorConfig(
        max_content_length=max_content_length,
        extract_code_blocks=extract_code_blocks,
        extract_links=extract_links,
    )
    return ContentExtractor(config)
