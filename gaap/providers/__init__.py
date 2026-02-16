# Providers
from .base_provider import BaseProvider, ProviderFactory
from .chat_based.g4f_provider import G4FProvider
from .free.additional_providers import (
    HuggingChatProvider,
    MLvocaProvider,
    OllamaFreeAPIProvider,
    OllamaProvider,
    OpenRouterProvider,
    PoeProvider,
    PuterProvider,
    ScitelyProvider,
    YouChatProvider,
)
from .free.llm7_provider import LLM7Provider
from .free_tier.groq_provider import GeminiProvider, GroqProvider
from .unified_gaap_provider import UnifiedGAAPProvider


def get_provider(name: str, **kwargs):
    """Get provider by name"""
    name = name.lower()

    if name in ("llm7", "llm7.io"):
        return LLM7Provider(**kwargs)
    elif name in ("groq",):
        return GroqProvider(**kwargs)
    elif name in ("gemini",):
        return GeminiProvider(**kwargs)
    elif name in ("g4f", "g4f_provider"):
        return G4FProvider(**kwargs)
    elif name in ("openrouter", "openrouter.ai"):
        return OpenRouterProvider(**kwargs)
    elif name in ("huggingchat", "huggingchat.ai", "huggingface"):
        return HuggingChatProvider(**kwargs)
    elif name in ("poe", "poe.com"):
        return PoeProvider(**kwargs)
    elif name in ("youchat", "you.com"):
        return YouChatProvider(**kwargs)
    elif name in ("puter", "puter.com"):
        return PuterProvider(**kwargs)
    elif name in ("scitely",):
        return ScitelyProvider(**kwargs)
    elif name in ("ollama",):
        return OllamaProvider(**kwargs)
    elif name in ("mlvoca",):
        return MLvocaProvider(**kwargs)
    elif name in ("ollamafree", "ollamafreeapi"):
        return OllamaFreeAPIProvider(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {name}")


__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "G4FProvider",
    "GroqProvider",
    "GeminiProvider",
    "LLM7Provider",
    "OpenRouterProvider",
    "HuggingChatProvider",
    "PoeProvider",
    "YouChatProvider",
    "PuterProvider",
    "ScitelyProvider",
    "OllamaProvider",
    "MLvocaProvider",
    "OllamaFreeAPIProvider",
    "UnifiedGAAPProvider",
    "get_provider",
]
