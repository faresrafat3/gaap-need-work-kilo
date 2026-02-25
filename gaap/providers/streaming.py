"""
Native Streaming Module for Providers
======================================

True word-by-word streaming with:
- Real-time token yielding
- SSE parsing for multiple providers
- Connect-RPC streaming (Kimi)
- Custom streaming protocols (DeepSeek)
- Callback-based streaming

Usage:
    from gaap.providers.streaming import NativeStreamer, StreamConfig

    streamer = NativeStreamer(config)
    async for chunk in streamer.stream_deepseek(messages, auth):
        print(chunk.content, end="", flush=True)
"""

import asyncio
import base64
import json
import logging
import struct
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable

from .async_session import AsyncSessionManager, SSEEvent

logger = logging.getLogger("gaap.providers.streaming")


class StreamProtocol(Enum):
    """Supported streaming protocols."""

    SSE = "sse"
    CONNECT_RPC = "connect_rpc"
    CUSTOM = "custom"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class StreamConfig:
    """Streaming configuration."""

    protocol: StreamProtocol = StreamProtocol.SSE
    chunk_timeout: float = 30.0
    max_response_bytes: int = 512 * 1024
    yield_delay: float = 0.0
    buffer_size: int = 1
    include_metadata: bool = False


@dataclass
class TokenChunk:
    """A single token chunk."""

    content: str
    is_final: bool = False
    token_count: int = 0
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_thinking(self) -> bool:
        return bool(self.metadata.get("is_thinking", False))


class StreamParser(ABC):
    """Abstract base for stream parsers."""

    @abstractmethod
    async def parse(
        self,
        response: Any,
        config: StreamConfig,
    ) -> AsyncGenerator[TokenChunk, None]:
        """Parse response stream into token chunks."""
        yield TokenChunk("")


class SSEParser(StreamParser):
    """Parse OpenAI-style SSE streams."""

    async def parse(
        self,
        response: Any,
        config: StreamConfig,
    ) -> AsyncGenerator[TokenChunk, None]:
        """Parse SSE stream into token chunks."""
        total_len = 0

        try:
            async for line_bytes in response.aiter_lines():
                if config.max_response_bytes and total_len > config.max_response_bytes:
                    yield TokenChunk("", is_final=True, finish_reason="length")
                    break

                line = (
                    line_bytes.decode("utf-8", errors="replace")
                    if isinstance(line_bytes, bytes)
                    else str(line_bytes)
                ).strip()

                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    yield TokenChunk("", is_final=True)
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    total_len += len(content)
                    yield TokenChunk(
                        content,
                        token_count=1,
                        metadata={"raw": data},
                    )

                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    yield TokenChunk("", is_final=True, finish_reason=finish_reason)

        finally:
            response.close()


class DeepSeekParser(StreamParser):
    """Parse DeepSeek's custom SSE format."""

    async def parse(
        self,
        response: Any,
        config: StreamConfig,
    ) -> AsyncGenerator[TokenChunk, None]:
        """Parse DeepSeek SSE stream."""
        total_len = 0
        thinking_active = False

        try:
            async for line_bytes in response.aiter_lines():
                if config.max_response_bytes and total_len > config.max_response_bytes:
                    yield TokenChunk("", is_final=True, finish_reason="length")
                    break

                line = (
                    line_bytes.decode("utf-8", errors="replace")
                    if isinstance(line_bytes, bytes)
                    else str(line_bytes)
                ).strip()

                if not line:
                    continue

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    if event_type == "close":
                        yield TokenChunk("", is_final=True)
                        break
                    if event_type == "thinking":
                        thinking_active = True
                    elif event_type == "answer":
                        thinking_active = False
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                content = self._extract_deepseek_content(chunk)
                if content:
                    total_len += len(content)
                    yield TokenChunk(
                        content,
                        metadata={"is_thinking": thinking_active, "raw": chunk},
                    )

        finally:
            response.close()

    def _extract_deepseek_content(self, chunk: dict[str, Any]) -> str:
        """Extract content from DeepSeek response format."""
        v = chunk.get("v")
        p = chunk.get("p", "")
        o = chunk.get("o", "")

        if isinstance(v, str) and not p and not o:
            return v

        if isinstance(v, dict) and "response" in v:
            response_obj = v["response"]
            if isinstance(response_obj, dict):
                content = ""
                for frag in response_obj.get("fragments", []):
                    if frag.get("type") == "RESPONSE":
                        content += frag.get("content", "")
                return content

        if o == "APPEND" and "content" in p and isinstance(v, str):
            return v

        return ""


class ConnectRPCParser(StreamParser):
    """Parse Connect-RPC streaming format (Kimi)."""

    @staticmethod
    def parse_envelopes(content: bytes) -> list[tuple[int, dict]]:
        """Parse Connect streaming envelopes."""
        messages = []
        idx = 0

        while idx + 5 <= len(content):
            flags = content[idx]
            length = struct.unpack(">I", content[idx + 1 : idx + 5])[0]
            data = content[idx + 5 : idx + 5 + length]
            idx += 5 + length

            try:
                messages.append((flags, json.loads(data)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                messages.append((flags, {"_raw": data}))

        return messages

    async def parse(
        self,
        response: Any,
        config: StreamConfig,
    ) -> AsyncGenerator[TokenChunk, None]:
        """Parse Connect-RPC response."""
        content = response.content
        envelopes = self.parse_envelopes(content)

        full_text = ""
        total_len = 0

        for flags, msg in envelopes:
            if not isinstance(msg, dict):
                continue

            if flags == 0x02:
                err = msg.get("error", {})
                if err:
                    yield TokenChunk(
                        "",
                        is_final=True,
                        finish_reason="error",
                        metadata={"error": err},
                    )
                    return

            op = msg.get("op", "")
            block = msg.get("block")

            if block and isinstance(block, dict):
                text_block = block.get("text")
                if text_block and isinstance(text_block, dict):
                    text_content = text_block.get("content", "")
                    if text_content:
                        if op == "append":
                            new_text = full_text + text_content
                            delta = text_content
                            full_text = new_text
                        else:
                            delta = text_content[len(full_text) :]
                            full_text = text_content

                        total_len += len(delta)
                        yield TokenChunk(delta, metadata={"raw": msg})

                        if config.max_response_bytes and total_len > config.max_response_bytes:
                            yield TokenChunk("", is_final=True, finish_reason="length")
                            return

        yield TokenChunk("", is_final=True)


class NativeStreamer:
    """
    Native streaming handler for providers.

    Provides unified streaming interface across different protocols.

    Usage:
        >>> streamer = NativeStreamer()
        >>> async for chunk in streamer.stream(
        ...     protocol=StreamProtocol.SSE,
        ...     response=response,
        ... ):
        ...     print(chunk.content, end="")
    """

    _parsers: dict[StreamProtocol, type[StreamParser]] = {
        StreamProtocol.SSE: SSEParser,
        StreamProtocol.OPENAI: SSEParser,
        StreamProtocol.CONNECT_RPC: ConnectRPCParser,
        StreamProtocol.CUSTOM: SSEParser,
    }

    def __init__(self, config: StreamConfig | None = None):
        self.config = config or StreamConfig()

    def get_parser(self, protocol: StreamProtocol) -> StreamParser:
        """Get parser for protocol."""
        parser_class = self._parsers.get(protocol, SSEParser)
        return parser_class()

    async def stream(
        self,
        response: Any,
        protocol: StreamProtocol | None = None,
        config: StreamConfig | None = None,
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Stream response using appropriate parser.

        Args:
            response: HTTP response object
            protocol: Override protocol
            config: Override config

        Yields:
            TokenChunk objects
        """
        cfg = config or self.config
        proto = protocol or cfg.protocol
        parser = self.get_parser(proto)

        async for chunk in parser.parse(response, cfg):
            if cfg.yield_delay > 0:
                await asyncio.sleep(cfg.yield_delay)
            yield chunk

    async def stream_deepseek(
        self,
        session: AsyncSessionManager,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        cookies: dict[str, str] | None = None,
        thinking_enabled: bool = False,
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Stream from DeepSeek API.

        Args:
            session: Async session manager
            url: API endpoint
            payload: Request payload
            headers: Request headers
            cookies: Optional cookies
            thinking_enabled: Enable thinking mode

        Yields:
            TokenChunk objects
        """
        config = StreamConfig(
            protocol=StreamProtocol.CUSTOM,
            max_response_bytes=self.config.max_response_bytes,
        )
        parser = DeepSeekParser()

        response = await session.post(
            url,
            json=payload,
            headers=headers,
            cookies=cookies or {},
            stream=True,
        )

        async for chunk in parser.parse(response, config):
            yield chunk

    async def stream_kimi(
        self,
        session: AsyncSessionManager,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        cookies: dict[str, str] | None = None,
    ) -> AsyncGenerator[TokenChunk, None]:
        """
        Stream from Kimi Connect-RPC API.

        Args:
            session: Async session manager
            url: API endpoint
            payload: Request payload
            headers: Request headers
            cookies: Optional cookies

        Yields:
            TokenChunk objects
        """
        config = StreamConfig(
            protocol=StreamProtocol.CONNECT_RPC,
            max_response_bytes=self.config.max_response_bytes,
        )
        parser = ConnectRPCParser()

        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        body = self._connect_envelope(0, json_bytes)

        response = await session.post(
            url,
            data=body,
            headers=headers,
            cookies=cookies or {},
        )

        async for chunk in parser.parse(response, config):
            yield chunk

    @staticmethod
    def _connect_envelope(flags: int, data: bytes) -> bytes:
        """Create a Connect streaming envelope."""
        return struct.pack(">BI", flags, len(data)) + data


async def collect_stream(
    stream: AsyncGenerator[TokenChunk, None],
    include_metadata: bool = False,
) -> tuple[str, dict[str, Any]]:
    """
    Collect stream into final text.

    Args:
        stream: Token stream
        include_metadata: Include metadata in result

    Returns:
        Tuple of (text, metadata)
    """
    text = ""
    metadata: dict[str, Any] = {
        "token_count": 0,
        "finish_reason": None,
        "chunks": [] if include_metadata else None,
    }

    async for chunk in stream:
        text += chunk.content
        metadata["token_count"] += chunk.token_count

        if chunk.is_final:
            metadata["finish_reason"] = chunk.finish_reason

        if include_metadata and metadata["chunks"] is not None:
            metadata["chunks"].append(
                {
                    "content": chunk.content,
                    "is_thinking": chunk.is_thinking,
                }
            )

    return text, metadata
