"""
Async Session Manager for Providers
====================================

Native asyncio support for curl_cffi AsyncSession with:
- Connection pooling
- SSE streaming
- Proxy rotation
- Retry logic
- Rate limiting

Usage:
    from gaap.providers.async_session import AsyncSessionManager

    async with AsyncSessionManager() as session:
        response = await session.post(url, json=data)
        async for chunk in session.stream_sse(url, json=data):
            print(chunk)
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, AsyncGenerator, Callable

logger = logging.getLogger("gaap.providers.async_session")


@dataclass
class SSEEvent:
    """Server-Sent Event."""

    event: str = ""
    data: str = ""
    id: str = ""
    retry: int | None = None

    def is_close(self) -> bool:
        return self.event == "close" or self.data == "[DONE]"


@dataclass
class StreamChunk:
    """Streaming chunk with metadata."""

    content: str
    is_final: bool = False
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncSessionManager:
    """
    Manages curl_cffi AsyncSession with intelligent defaults.

    Features:
    - Lazy initialization
    - Connection pooling
    - Automatic retry on transient errors
    - SSE parsing
    - Browser impersonation

    Attributes:
        impersonate: Browser to impersonate (default: chrome)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        proxy: Optional proxy URL

    Usage:
        >>> async with AsyncSessionManager() as session:
        ...     response = await session.get("https://api.example.com")
        ...     print(response.json())
    """

    def __init__(
        self,
        impersonate: str = "chrome",
        timeout: float = 120.0,
        max_retries: int = 3,
        proxy: str | None = None,
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ):
        self.impersonate = impersonate
        self.timeout = timeout
        self.max_retries = max_retries
        self.proxy = proxy
        self.default_headers = headers or {}
        self.default_cookies = cookies or {}

        self._session: Any = None
        self._closed = False

    async def _ensure_session(self) -> Any:
        """Ensure session is initialized."""
        if self._session is None or self._closed:
            try:
                from curl_cffi.requests import AsyncSession

                self._session = AsyncSession(
                    impersonate=self.impersonate,
                    proxy=self.proxy,
                    timeout=self.timeout,
                    headers=self.default_headers,
                    cookies=self.default_cookies,
                )
                self._closed = False
            except ImportError:
                raise RuntimeError("curl_cffi not installed. Install with: pip install curl_cffi")
        return self._session

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._closed:
            try:
                await self._session.close()
            except Exception as e:
                logger.debug(f"Error closing session: {e}")
            finally:
                self._closed = True
                self._session = None

    async def __aenter__(self) -> "AsyncSessionManager":
        await self._ensure_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """
        Make an async HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request options

        Returns:
            Response object
        """
        session = await self._ensure_session()

        attempt = 0
        last_error: Exception | None = None

        while attempt < self.max_retries:
            try:
                response = await getattr(session, method.lower())(url, **kwargs)
                return response
            except Exception as e:
                last_error = e
                attempt += 1

                if attempt < self.max_retries:
                    delay = min(2**attempt, 30)
                    logger.debug(f"Request failed, retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("Request failed")

    async def get(self, url: str, **kwargs: Any) -> Any:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Any:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> Any:
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> Any:
        return await self.request("DELETE", url, **kwargs)

    async def stream_sse(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> AsyncGenerator[SSEEvent, None]:
        """
        Stream Server-Sent Events.

        Args:
            method: HTTP method (GET or POST)
            url: Request URL
            **kwargs: Additional request options

        Yields:
            SSEEvent objects as they arrive
        """
        session = await self._ensure_session()

        kwargs.setdefault("stream", True)

        response = await getattr(session, method.lower())(url, **kwargs)

        try:
            current_event = SSEEvent()
            buffer = ""

            async for line_bytes in response.aiter_lines():
                if isinstance(line_bytes, bytes):
                    line = line_bytes.decode("utf-8", errors="replace")
                else:
                    line = str(line_bytes)

                line = line.rstrip("\n\r")

                if not line:
                    if current_event.data:
                        yield current_event
                        current_event = SSEEvent()
                    continue

                if line.startswith(":"):
                    continue

                if line.startswith("event:"):
                    current_event.event = line[6:].strip()
                elif line.startswith("data:"):
                    data_content = line[5:]
                    if current_event.data:
                        current_event.data += "\n" + data_content
                    else:
                        current_event.data = data_content
                elif line.startswith("id:"):
                    current_event.id = line[3:].strip()
                elif line.startswith("retry:"):
                    try:
                        current_event.retry = int(line[6:].strip())
                    except ValueError:
                        pass
                else:
                    if current_event.data:
                        current_event.data += "\n" + line
                    else:
                        current_event.data = line

            if current_event.data:
                yield current_event

        finally:
            response.close()

    async def stream_json_chunks(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream JSON chunks from SSE data fields.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional request options

        Yields:
            Parsed JSON objects from SSE data fields
        """
        async for event in self.stream_sse(method, url, **kwargs):
            if event.is_close():
                break

            if not event.data:
                continue

            try:
                data = json.loads(event.data)
                yield data
            except json.JSONDecodeError:
                yield {"_raw": event.data, "event": event.event}

    async def stream_content(
        self,
        method: str,
        url: str,
        chunk_size: int = 1024,
        **kwargs: Any,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream raw content chunks.

        Args:
            method: HTTP method
            url: Request URL
            chunk_size: Size of each chunk in bytes
            **kwargs: Additional request options

        Yields:
            Raw bytes chunks
        """
        session = await self._ensure_session()

        kwargs.setdefault("stream", True)

        response = await getattr(session, method.lower())(url, **kwargs)

        try:
            async for chunk in response.aiter_content(chunk_size):
                yield chunk
        finally:
            response.close()

    def update_headers(self, headers: dict[str, str]) -> None:
        """Update default headers."""
        self.default_headers.update(headers)
        if self._session:
            self._session.headers.update(headers)

    def update_cookies(self, cookies: dict[str, str]) -> None:
        """Update default cookies."""
        self.default_cookies.update(cookies)
        if self._session:
            self._session.cookies.update(cookies)


class AsyncSessionPool:
    """
    Pool of async sessions for different providers.

    Each provider gets its own session with specific configuration.

    Usage:
        >>> pool = AsyncSessionPool()
        >>> session = pool.get_session("deepseek", impersonate="chrome")
        >>> response = await session.get(url)
    """

    def __init__(self, max_sessions: int = 10):
        self.max_sessions = max_sessions
        self._sessions: dict[str, AsyncSessionManager] = {}
        self._lock = asyncio.Lock()

    async def get_session(
        self,
        name: str,
        **kwargs: Any,
    ) -> AsyncSessionManager:
        """
        Get or create a session for a provider.

        Args:
            name: Provider name
            **kwargs: Session configuration

        Returns:
            AsyncSessionManager instance
        """
        async with self._lock:
            if name not in self._sessions:
                if len(self._sessions) >= self.max_sessions:
                    oldest = next(iter(self._sessions))
                    await self._sessions[oldest].close()
                    del self._sessions[oldest]

                self._sessions[name] = AsyncSessionManager(**kwargs)

            return self._sessions[name]

    async def close_all(self) -> None:
        """Close all sessions."""
        async with self._lock:
            for session in self._sessions.values():
                await session.close()
            self._sessions.clear()

    async def __aenter__(self) -> "AsyncSessionPool":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close_all()


_global_pool: AsyncSessionPool | None = None


async def get_global_session_pool() -> AsyncSessionPool:
    """Get the global session pool singleton."""
    global _global_pool
    if _global_pool is None:
        _global_pool = AsyncSessionPool()
    return _global_pool


@asynccontextmanager
async def async_request(
    impersonate: str = "chrome",
    **kwargs: Any,
) -> AsyncGenerator[AsyncSessionManager, None]:
    """Context manager for making async requests."""
    async with AsyncSessionManager(impersonate=impersonate, **kwargs) as session:
        yield session
