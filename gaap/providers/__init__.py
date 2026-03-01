# Providers
from typing import Any

# Spec 38: Async & Native Streaming
from .async_session import AsyncSessionManager, AsyncSessionPool
from .base_provider import BaseProvider, ProviderFactory
from .kilo_gateway import KiloGatewayProvider
from .kilo_multi_account import KiloMultiAccountProvider
from .prompt_caching import CacheConfig, PromptCache

# Provider Cache
from .provider_cache import (
    CacheEvent,
    CacheEventType,
    CacheRefreshStrategy,
    CacheStatistics,
    CircuitBreaker,
    CircuitState,
    ProviderCacheEntry,
    ProviderCacheManager,
    cached_provider_call,
    get_cache_manager,
)
from .streaming import NativeStreamer, StreamConfig, StreamProtocol, TokenChunk
from .tool_calling import (
    ParameterSchema,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    create_tool_from_function,
    tool,
)
from .unified_gaap_provider import UnifiedGAAPProvider


def get_provider(name: str, **kwargs: Any) -> Any:
    """Get provider by name"""
    name = name.lower()

    if name in ("kilo", "kilo-gateway", "kilo_gateway"):
        return KiloGatewayProvider(**kwargs)
    elif name in ("kilo_multi", "kilo-multi", "kilo_multi_account"):
        return KiloMultiAccountProvider(**kwargs)
    elif name in ("unified_gaap", "unified"):
        return UnifiedGAAPProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}")


__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "UnifiedGAAPProvider",
    "KiloGatewayProvider",
    "KiloMultiAccountProvider",
    "get_provider",
    # Spec 38
    "AsyncSessionManager",
    "AsyncSessionPool",
    "NativeStreamer",
    "StreamConfig",
    "StreamProtocol",
    "TokenChunk",
    "PromptCache",
    "CacheConfig",
    "ToolRegistry",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "ParameterSchema",
    "tool",
    "create_tool_from_function",
]
