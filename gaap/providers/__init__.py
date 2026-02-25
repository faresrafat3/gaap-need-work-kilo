# Providers
from typing import Any

from .base_provider import BaseProvider, ProviderFactory
from .chat_based.g4f_provider import G4FProvider
from .kilo_gateway import KiloGatewayProvider
from .kilo_multi_account import KiloMultiAccountProvider
from .unified_gaap_provider import UnifiedGAAPProvider

# Spec 38: Async & Native Streaming
from .async_session import AsyncSessionManager, AsyncSessionPool
from .streaming import NativeStreamer, StreamConfig, StreamProtocol, TokenChunk
from .prompt_caching import PromptCache, CacheConfig
from .tool_calling import (
    ToolRegistry,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ParameterSchema,
    tool,
    create_tool_from_function,
)


def get_provider(name: str, **kwargs: Any) -> Any:
    """Get provider by name"""
    name = name.lower()

    if name in ("kilo", "kilo-gateway", "kilo_gateway"):
        return KiloGatewayProvider(**kwargs)
    elif name in ("kilo_multi", "kilo-multi", "kilo_multi_account"):
        return KiloMultiAccountProvider(**kwargs)
    elif name in ("g4f", "g4f_provider"):
        return G4FProvider(**kwargs)
    elif name in ("unified_gaap", "unified"):
        return UnifiedGAAPProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}")


__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "G4FProvider",
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
