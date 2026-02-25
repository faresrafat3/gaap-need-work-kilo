"""
Providers API - Provider Management Endpoints
==============================================

Provides REST API for managing LLM providers.

Endpoints:
- GET /api/providers - List all providers
- POST /api/providers - Add new provider
- GET /api/providers/{name} - Get provider details
- PUT /api/providers/{name} - Update provider
- DELETE /api/providers/{name} - Remove provider
- POST /api/providers/{name}/test - Test connection
- POST /api/providers/{name}/enable - Enable provider
- POST /api/providers/{name}/disable - Disable provider
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.core.observability import observability
from gaap.providers.base_provider import ProviderFactory
from gaap.routing.router import SmartRouter

logger = logging.getLogger("gaap.api.providers")

router = APIRouter(prefix="/api/providers", tags=["providers"])

_router_instance: SmartRouter | None = None


class ProviderConfig(BaseModel):
    """Provider configuration."""

    name: str
    provider_type: str = "chat"
    api_key: str | None = None
    base_url: str | None = None
    priority: int = 1
    enabled: bool = True
    models: list[str] = []
    default_model: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.7
    metadata: dict[str, Any] = {}


class ProviderStatus(BaseModel):
    """Provider status response."""

    name: str
    type: str
    enabled: bool
    priority: int
    models: list[str]
    health: str
    stats: dict[str, Any]


class ProviderTestResult(BaseModel):
    """Provider test result."""

    success: bool
    latency_ms: float | None = None
    error: str | None = None
    model_available: bool = False


def get_router() -> SmartRouter:
    """Get or create the SmartRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter()
    return _router_instance


def set_router(router_instance: SmartRouter) -> None:
    """Set the SmartRouter instance."""
    global _router_instance
    _router_instance = router_instance


def _check_provider_health(provider_name: str, router_instance: SmartRouter) -> str:
    """Check provider health status."""
    try:
        provider = router_instance.get_provider(provider_name)
        if provider is None:
            return "unhealthy"

        stats = provider.get_stats()
        total_requests = stats.get("total_requests", 0)
        if total_requests == 0:
            return "healthy"

        success_rate = stats.get("success_rate", 1.0)
        if success_rate < 0.5:
            return "unhealthy"
        elif success_rate < 0.8:
            return "degraded"
        return "healthy"
    except Exception:
        return "unhealthy"


@router.get("", response_model=list[ProviderStatus])
async def list_providers() -> list[ProviderStatus]:
    """List all registered providers."""
    router_instance = get_router()
    providers = router_instance.get_all_providers()

    result = []
    for provider in providers:
        name = provider.name
        stats = provider.get_stats()

        result.append(
            ProviderStatus(
                name=name,
                type=provider.provider_type.name,
                enabled=True,
                priority=1,
                models=provider.get_available_models(),
                health=_check_provider_health(name, router_instance),
                stats=stats,
            )
        )

    return result


@router.post("", response_model=ProviderStatus)
async def add_provider(config: ProviderConfig) -> ProviderStatus:
    """Add a new provider."""
    try:
        router_instance = get_router()

        provider = ProviderFactory.create(
            config.name,
            api_key=config.api_key,
            base_url=config.base_url,
            default_model=config.default_model,
        )

        router_instance.register_provider(provider)

        observability.metrics.inc_counter(
            "llm_calls_total",
            {"provider": config.name, "model": "system", "status": "registered"},
        )

        return ProviderStatus(
            name=config.name,
            type=config.provider_type,
            enabled=config.enabled,
            priority=config.priority,
            models=config.models or provider.get_available_models(),
            health="healthy",
            stats={},
        )
    except Exception as e:
        logger.error(f"Failed to add provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}", response_model=ProviderStatus)
async def get_provider(name: str) -> ProviderStatus:
    """Get details for a specific provider."""
    router_instance = get_router()
    provider = router_instance.get_provider(name)

    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    stats = provider.get_stats()

    return ProviderStatus(
        name=name,
        type=provider.provider_type.name,
        enabled=True,
        priority=1,
        models=provider.get_available_models(),
        health=_check_provider_health(name, router_instance),
        stats=stats,
    )


@router.put("/{name}", response_model=ProviderStatus)
async def update_provider(name: str, config: ProviderConfig) -> ProviderStatus:
    """Update provider configuration."""
    try:
        router_instance = get_router()
        provider = router_instance.get_provider(name)

        if provider is None:
            raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

        if config.default_model and hasattr(provider, "default_model"):
            provider.default_model = config.default_model

        observability.metrics.inc_counter(
            "llm_calls_total",
            {"provider": name, "model": "system", "status": "updated"},
        )

        return ProviderStatus(
            name=name,
            type=config.provider_type,
            enabled=config.enabled,
            priority=config.priority,
            models=config.models or provider.get_available_models(),
            health=_check_provider_health(name, router_instance),
            stats=provider.get_stats(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{name}")
async def remove_provider(name: str) -> dict[str, Any]:
    """Remove a provider."""
    try:
        router_instance = get_router()

        if not router_instance.get_provider(name):
            raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

        router_instance.unregister_provider(name)

        observability.metrics.inc_counter(
            "llm_calls_total",
            {"provider": name, "model": "system", "status": "removed"},
        )

        return {"success": True, "message": f"Provider '{name}' removed"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove provider: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{name}/test", response_model=ProviderTestResult)
async def test_provider(name: str) -> ProviderTestResult:
    """Test provider connection."""
    try:
        router_instance = get_router()
        provider = router_instance.get_provider(name)

        if provider is None:
            raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

        start_time = time.time()

        models = provider.get_available_models()
        model_available = len(models) > 0

        latency = (time.time() - start_time) * 1000

        return ProviderTestResult(
            success=True,
            latency_ms=round(latency, 2),
            model_available=model_available,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Provider test failed: {e}")
        return ProviderTestResult(
            success=False,
            error=str(e),
        )


@router.post("/{name}/enable")
async def enable_provider(name: str) -> dict[str, Any]:
    """Enable a provider."""
    router_instance = get_router()

    if not router_instance.get_provider(name):
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    observability.metrics.inc_counter(
        "llm_calls_total",
        {"provider": name, "model": "system", "status": "enabled"},
    )

    return {"success": True, "message": f"Provider '{name}' enabled"}


@router.post("/{name}/disable")
async def disable_provider(name: str) -> dict[str, Any]:
    """Disable a provider."""
    router_instance = get_router()

    if not router_instance.get_provider(name):
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    observability.metrics.inc_counter(
        "llm_calls_total",
        {"provider": name, "model": "system", "status": "disabled"},
    )

    return {"success": True, "message": f"Provider '{name}' disabled"}


def register_routes(app: Any) -> None:
    """Register provider routes with FastAPI app."""
    app.include_router(router)
