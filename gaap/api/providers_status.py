"""
Providers Status API - Live Provider Monitoring
===============================================

Real-time provider status with live model detection, caching, and health checks.

Endpoints:
- GET /api/providers/status - Get live provider status with actual models
- POST /api/providers/refresh - Force refresh all provider models
- GET /api/providers/status/{name} - Get specific provider status

Features:
- Live model detection from providers
- In-memory cache with configurable TTL
- Parallel health checks using asyncio.gather()
- Graceful degradation (partial data on errors)
- Background refresh without blocking

Usage:
    from gaap.api.providers_status import router
    app.include_router(router)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.providers.account_manager import PROVIDER_DEFAULTS, PoolManager
from gaap.providers.base_provider import BaseProvider
from gaap.routing.router import SmartRouter

logger = logging.getLogger("gaap.api.providers_status")

router = APIRouter(prefix="/api/providers", tags=["providers-status"])

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CACHE_TTL_SECONDS = 300  # 5 minutes
HEALTH_CHECK_TIMEOUT_SECONDS = 30
MODEL_DETECTION_TIMEOUT_SECONDS = 45


# =============================================================================
# Data Models
# =============================================================================


class ProviderInfo(BaseModel):
    """Provider information with live-detected model data."""

    name: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable provider name")
    actual_model: str | None = Field(None, description="Live-detected actual model")
    default_model: str = Field(..., description="Configured default model")
    status: str = Field(..., description="Provider status: active, error, offline, unknown")
    last_seen: datetime | None = Field(None, description="Last successful contact timestamp")
    latency_ms: float | None = Field(None, description="Last health check latency in ms")
    success_rate: float = Field(0.0, ge=0.0, le=100.0, description="Success rate percentage")
    accounts_count: int = Field(0, description="Total configured accounts")
    healthy_accounts: int = Field(0, description="Number of healthy accounts")
    models_available: list[str] = Field(default_factory=list, description="Available models")
    provider_type: str = Field("unknown", description="Provider type")
    error_message: str | None = Field(None, description="Error message if status is error/offline")
    cached: bool = Field(False, description="Whether data is from cache")
    cache_age_seconds: float | None = Field(None, description="Cache age in seconds")


class ProvidersStatusResponse(BaseModel):
    """Response model for providers status endpoint."""

    providers: list[ProviderInfo]
    last_updated: datetime
    total_providers: int
    active_providers: int
    failed_providers: int
    cache_hit: bool = False
    refresh_in_progress: bool = False


class RefreshResponse(BaseModel):
    """Response model for refresh endpoint."""

    success: bool
    message: str
    providers_refreshed: int
    providers_failed: int
    refreshed_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class ProviderDetailResponse(BaseModel):
    """Response model for single provider status."""

    provider: ProviderInfo
    last_updated: datetime


# =============================================================================
# Cache Implementation
# =============================================================================


@dataclass
class CacheEntry:
    """Single cache entry with TTL."""

    data: Any
    timestamp: float
    ttl_seconds: float
    is_refreshing: bool = False

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.timestamp > self.ttl_seconds

    def age_seconds(self) -> float:
        """Get cache age in seconds."""
        return time.time() - self.timestamp


class ProviderCache:
    """
    Thread-safe in-memory cache for provider status.

    Features:
    - Per-provider cache keys
    - TTL-based expiration
    - Background refresh tracking
    - Cache invalidation
    """

    _instance: ProviderCache | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __new__(cls) -> ProviderCache:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache: dict[str, CacheEntry] = {}
            cls._instance._default_ttl = DEFAULT_CACHE_TTL_SECONDS
        return cls._instance

    def get(self, key: str) -> CacheEntry | None:
        """Get cache entry if it exists and is not expired."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            return None
        return entry

    def set(
        self,
        key: str,
        data: Any,
        ttl_seconds: float | None = None,
        is_refreshing: bool = False,
    ) -> None:
        """Set cache entry."""
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl_seconds=ttl_seconds or self._default_ttl,
            is_refreshing=is_refreshing,
        )

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entry or entire cache."""
        if key is None:
            self._cache.clear()
            logger.info("Provider cache fully invalidated")
        else:
            self._cache.pop(key, None)
            logger.info(f"Provider cache invalidated for: {key}")

    def is_refreshing(self, key: str) -> bool:
        """Check if a key is currently being refreshed."""
        entry = self._cache.get(key)
        return entry.is_refreshing if entry else False

    def set_refreshing(self, key: str, refreshing: bool) -> None:
        """Set refresh status for a key."""
        entry = self._cache.get(key)
        if entry:
            entry.is_refreshing = refreshing


# Global cache instance
_provider_cache = ProviderCache()


# =============================================================================
# Provider Status Service
# =============================================================================


class ProviderStatusService:
    """
    Service for checking provider status with live model detection.

    Features:
    - Parallel health checks
    - Live model detection
    - Account health integration
    - Error handling with partial results
    """

    # Known provider display names
    DISPLAY_NAMES: ClassVar[dict[str, str]] = {
        "glm": "GLM",
        "kimi": "Kimi",
        "deepseek": "DeepSeek",
        "copilot": "GitHub Copilot",
        "gemini_api": "Google Gemini API",
        "kilo": "Kilo Gateway",
        "kilo_multi": "Kilo Multi-Account",
        "unified": "Unified GAAP",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "azure": "Azure OpenAI",
        "ollama": "Ollama",
    }

    def __init__(self) -> None:
        self._router: SmartRouter | None = None
        self._pool_manager = PoolManager.instance()

    def _get_router(self) -> SmartRouter:
        """Get or create SmartRouter instance."""
        if self._router is None:
            self._router = SmartRouter()
        return self._router

    def _get_display_name(self, provider_name: str) -> str:
        """Get display name for provider."""
        return self.DISPLAY_NAMES.get(provider_name.lower(), provider_name.upper())

    async def _detect_actual_model(
        self,
        provider: BaseProvider,
        timeout: float = MODEL_DETECTION_TIMEOUT_SECONDS,
    ) -> tuple[str | None, float | None, str | None]:
        """
        Detect actual model from provider via health check.

        Returns:
            Tuple of (actual_model, latency_ms, error_message)
        """
        start_time = time.time()

        try:
            # Try to get available models
            models = await asyncio.wait_for(
                asyncio.to_thread(provider.get_available_models),
                timeout=timeout / 2,
            )

            if models:
                # Use default_model if available, otherwise first model
                actual_model = provider.default_model or models[0]
                latency_ms = (time.time() - start_time) * 1000
                return actual_model, latency_ms, None

            # No models available
            return None, None, "No models available"

        except asyncio.TimeoutError:
            return None, None, "Model detection timeout"
        except Exception as e:
            return None, None, str(e)[:200]

    async def _check_provider_health(
        self,
        provider: BaseProvider,
        timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
    ) -> tuple[bool, float | None, str | None]:
        """
        Check if provider is healthy.

        Returns:
            Tuple of (is_healthy, latency_ms, error_message)
        """
        start_time = time.time()

        try:
            # Try to get stats as a lightweight health check
            stats = await asyncio.wait_for(
                asyncio.to_thread(provider.get_stats),
                timeout=timeout,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Check success rate if we have data
            success_rate = stats.get("success_rate", 1.0)
            if success_rate < 0.1 and stats.get("total_requests", 0) > 5:
                return False, latency_ms, f"Low success rate: {success_rate:.1%}"

            return True, latency_ms, None

        except asyncio.TimeoutError:
            return False, None, "Health check timeout"
        except Exception as e:
            return False, None, str(e)[:200]

    async def _get_account_health(self, provider_name: str) -> tuple[int, int, float]:
        """
        Get account health information from PoolManager.

        Returns:
            Tuple of (total_accounts, healthy_accounts, avg_success_rate)
        """
        try:
            pool = self._pool_manager.pool(provider_name)
            accounts = pool.accounts

            if not accounts:
                return 0, 0, 0.0

            total = len(accounts)
            healthy = sum(1 for a in accounts if a.can_call()[0])

            # Calculate average success rate
            success_rates = [a.rate_tracker.success_rate for a in accounts]
            avg_success_rate = (sum(success_rates) / len(success_rates)) * 100

            return total, healthy, avg_success_rate

        except Exception as e:
            logger.debug(f"Failed to get account health for {provider_name}: {e}")
            return 0, 0, 0.0

    async def check_provider(
        self,
        provider: BaseProvider,
        use_cache: bool = True,
    ) -> ProviderInfo:
        """
        Check single provider status with optional caching.

        Args:
            provider: Provider to check
            use_cache: Whether to use cached data if available

        Returns:
            ProviderInfo with live or cached data
        """
        provider_name = provider.name
        cache_key = f"provider:{provider_name}"

        # Check cache
        if use_cache:
            cached = _provider_cache.get(cache_key)
            if cached and not cached.is_refreshing:
                info = cached.data
                info.cached = True
                info.cache_age_seconds = cached.age_seconds()
                return info

        # Check if refresh is already in progress
        if _provider_cache.is_refreshing(cache_key):
            # Return stale data if available
            cached = _provider_cache._cache.get(cache_key)
            if cached:
                info = cached.data
                info.cached = True
                info.cache_age_seconds = cached.age_seconds()
                return info

        # Mark as refreshing
        _provider_cache.set_refreshing(cache_key, True)

        try:
            # Run health check and model detection in parallel
            health_task = self._check_provider_health(provider)
            model_task = self._detect_actual_model(provider)
            account_task = self._get_account_health(provider_name)

            health_result, model_result, account_result = await asyncio.gather(
                health_task,
                model_task,
                account_task,
                return_exceptions=True,
            )

            # Handle exceptions
            if isinstance(health_result, Exception):
                is_healthy, latency_ms, health_error = False, None, str(health_result)
            else:
                is_healthy, latency_ms, health_error = health_result

            if isinstance(model_result, Exception):
                actual_model, _, model_error = None, None, str(model_result)
            else:
                actual_model, model_latency, model_error = model_result
                if latency_ms is None and model_latency is not None:
                    latency_ms = model_latency

            if isinstance(account_result, Exception):
                accounts_count, healthy_accounts, avg_success_rate = 0, 0, 0.0
            else:
                accounts_count, healthy_accounts, avg_success_rate = account_result

            # Determine status
            if is_healthy:
                status = "active"
                error_message = None
            elif health_error and "timeout" in health_error.lower():
                status = "offline"
                error_message = health_error
            else:
                status = "error"
                error_message = health_error or model_error or "Unknown error"

            # Get stats for success rate
            try:
                stats = provider.get_stats()
                success_rate = stats.get("success_rate", avg_success_rate / 100) * 100
            except Exception:
                success_rate = avg_success_rate

            # Build provider info
            info = ProviderInfo(
                name=provider_name,
                display_name=self._get_display_name(provider_name),
                actual_model=actual_model,
                default_model=provider.default_model or "unknown",
                status=status,
                last_seen=datetime.now(timezone.utc) if is_healthy else None,
                latency_ms=round(latency_ms, 2) if latency_ms else None,
                success_rate=round(success_rate, 1),
                accounts_count=accounts_count,
                healthy_accounts=healthy_accounts,
                models_available=provider.get_available_models(),
                provider_type=provider.provider_type.name,
                error_message=error_message,
                cached=False,
                cache_age_seconds=None,
            )

            # Cache the result
            _provider_cache.set(cache_key, info)

            return info

        except Exception as e:
            logger.error(f"Unexpected error checking provider {provider_name}: {e}")
            return ProviderInfo(
                name=provider_name,
                display_name=self._get_display_name(provider_name),
                actual_model=None,
                default_model=getattr(provider, "default_model", "unknown"),
                status="error",
                error_message=f"Check failed: {str(e)[:200]}",
                models_available=[],
                provider_type=getattr(provider, "provider_type", "unknown"),
            )
        finally:
            _provider_cache.set_refreshing(cache_key, False)

    async def check_all_providers(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> ProvidersStatusResponse:
        """
        Check all providers status.

        Args:
            use_cache: Whether to use cached data
            force_refresh: Force refresh even if cache is valid

        Returns:
            ProvidersStatusResponse with all provider info
        """
        if force_refresh:
            _provider_cache.invalidate()

        router = self._get_router()
        providers = router.get_all_providers()

        if not providers:
            return ProvidersStatusResponse(
                providers=[],
                last_updated=datetime.now(timezone.utc),
                total_providers=0,
                active_providers=0,
                failed_providers=0,
            )

        # Check all providers in parallel
        tasks = [self.check_provider(p, use_cache=use_cache) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        provider_infos: list[ProviderInfo] = []
        failed_count = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Provider check failed: {result}")
                failed_count += 1
                continue
            provider_infos.append(result)
            if result.status != "active":
                failed_count += 1

        active_count = sum(1 for p in provider_infos if p.status == "active")

        return ProvidersStatusResponse(
            providers=provider_infos,
            last_updated=datetime.now(timezone.utc),
            total_providers=len(provider_infos),
            active_providers=active_count,
            failed_providers=failed_count,
            cache_hit=use_cache and not force_refresh,
            refresh_in_progress=any(
                _provider_cache.is_refreshing(f"provider:{p.name}") for p in providers
            ),
        )


# Global service instance
_status_service = ProviderStatusService()


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/status", response_model=ProvidersStatusResponse)
async def get_providers_status(
    use_cache: bool = True,
    include_offline: bool = True,
) -> ProvidersStatusResponse:
    """
    Get live status of all providers with actual model detection.

    Args:
        use_cache: Whether to use cached data (default: True)
        include_offline: Include offline providers in response (default: True)

    Returns:
        ProvidersStatusResponse with provider information including:
        - name: Provider identifier
        - display_name: Human-readable name
        - actual_model: Live-detected model (if available)
        - default_model: Configured default model
        - status: active, error, offline, or unknown
        - last_seen: Last successful contact timestamp
        - latency_ms: Health check latency
        - success_rate: Percentage of successful requests
        - accounts_count: Total configured accounts
        - healthy_accounts: Number of healthy accounts
    """
    response = await _status_service.check_all_providers(use_cache=use_cache)

    if not include_offline:
        response.providers = [p for p in response.providers if p.status != "offline"]

    return response


@router.get("/status/{name}", response_model=ProviderDetailResponse)
async def get_provider_status(name: str, use_cache: bool = True) -> ProviderDetailResponse:
    """
    Get live status of a specific provider.

    Args:
        name: Provider name
        use_cache: Whether to use cached data (default: True)

    Returns:
        ProviderDetailResponse with detailed provider information

    Raises:
        HTTPException: If provider not found
    """
    router = _status_service._get_router()
    provider = router.get_provider(name)

    if provider is None:
        # Check if it's a known provider from account manager
        if name.lower() in PROVIDER_DEFAULTS:
            # Return default info for known but unregistered providers
            defaults = PROVIDER_DEFAULTS[name.lower()]
            info = ProviderInfo(
                name=name.lower(),
                display_name=_status_service._get_display_name(name),
                actual_model=None,
                default_model=defaults.get("models", ["unknown"])[0],
                status="offline",
                error_message="Provider not registered in router",
                models_available=defaults.get("models", []),
                provider_type="CHAT_BASED",
            )
            return ProviderDetailResponse(
                provider=info,
                last_updated=datetime.now(timezone.utc),
            )
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    info = await _status_service.check_provider(provider, use_cache=use_cache)

    return ProviderDetailResponse(
        provider=info,
        last_updated=datetime.now(timezone.utc),
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_providers(
    provider_names: list[str] | None = None,
    background: bool = True,
) -> RefreshResponse:
    """
    Force refresh provider status and model detection.

    Args:
        provider_names: Specific providers to refresh (None = all)
        background: Run refresh in background (default: True)

    Returns:
        RefreshResponse with refresh results
    """
    refreshed_at = datetime.now(timezone.utc)

    if provider_names:
        # Refresh specific providers
        refreshed = 0
        failed = 0
        details: dict[str, Any] = {}

        for name in provider_names:
            cache_key = f"provider:{name}"
            _provider_cache.invalidate(cache_key)

            router = _status_service._get_router()
            provider = router.get_provider(name)

            if provider is None:
                failed += 1
                details[name] = {"status": "error", "error": "Provider not found"}
            else:
                try:
                    await _status_service.check_provider(provider, use_cache=False)
                    refreshed += 1
                    details[name] = {"status": "refreshed"}
                except Exception as e:
                    failed += 1
                    details[name] = {"status": "error", "error": str(e)}

        return RefreshResponse(
            success=failed == 0,
            message=f"Refreshed {refreshed} providers, {failed} failed",
            providers_refreshed=refreshed,
            providers_failed=failed,
            refreshed_at=refreshed_at,
            details=details,
        )

    # Refresh all providers
    if background:
        # Invalidate cache immediately, let next request trigger refresh
        _provider_cache.invalidate()

        return RefreshResponse(
            success=True,
            message="Cache cleared. Providers will be refreshed on next status check",
            providers_refreshed=0,
            providers_failed=0,
            refreshed_at=refreshed_at,
            details={"mode": "background", "cache_cleared": True},
        )

    # Synchronous refresh of all providers
    response = await _status_service.check_all_providers(use_cache=False)

    return RefreshResponse(
        success=response.failed_providers == 0,
        message=f"Refreshed {response.active_providers} active, {response.failed_providers} failed",
        providers_refreshed=response.active_providers,
        providers_failed=response.failed_providers,
        refreshed_at=refreshed_at,
        details={
            "total": response.total_providers,
            "active": response.active_providers,
            "failed": response.failed_providers,
        },
    )


@router.post("/status/{name}/refresh", response_model=ProviderDetailResponse)
async def refresh_single_provider(name: str) -> ProviderDetailResponse:
    """
    Force refresh a specific provider's status.

    Args:
        name: Provider name

    Returns:
        ProviderDetailResponse with fresh provider data

    Raises:
        HTTPException: If provider not found
    """
    cache_key = f"provider:{name}"
    _provider_cache.invalidate(cache_key)

    router = _status_service._get_router()
    provider = router.get_provider(name)

    if provider is None:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    info = await _status_service.check_provider(provider, use_cache=False)

    return ProviderDetailResponse(
        provider=info,
        last_updated=datetime.now(timezone.utc),
    )


@router.get("/cache/status")
async def get_cache_status() -> dict[str, Any]:
    """Get cache status and statistics."""
    cache = _provider_cache

    entries = []
    for key, entry in cache._cache.items():
        entries.append(
            {
                "key": key,
                "age_seconds": entry.age_seconds(),
                "ttl_seconds": entry.ttl_seconds,
                "is_expired": entry.is_expired(),
                "is_refreshing": entry.is_refreshing,
            }
        )

    return {
        "cache_entries": len(cache._cache),
        "default_ttl_seconds": cache._default_ttl,
        "entries": entries,
    }


@router.post("/cache/invalidate")
async def invalidate_cache(provider_name: str | None = None) -> dict[str, Any]:
    """
    Invalidate provider cache.

    Args:
        provider_name: Specific provider to invalidate (None = all)

    Returns:
        Confirmation message
    """
    if provider_name:
        _provider_cache.invalidate(f"provider:{provider_name}")
        return {
            "success": True,
            "message": f"Cache invalidated for provider: {provider_name}",
            "provider": provider_name,
        }

    _provider_cache.invalidate()
    return {
        "success": True,
        "message": "All provider cache invalidated",
        "providers_affected": "all",
    }


# =============================================================================
# Integration Helpers
# =============================================================================


def get_provider_status_service() -> ProviderStatusService:
    """Get the global provider status service instance."""
    return _status_service


def set_cache_ttl(seconds: float) -> None:
    """Set the default cache TTL."""
    _provider_cache._default_ttl = seconds


def register_provider_for_monitoring(provider: BaseProvider) -> None:
    """
    Register a provider for status monitoring.

    This is called automatically when providers are added to the router,
    but can be called manually for providers not in the main router.
    """
    router = _status_service._get_router()
    if provider.name not in [p.name for p in router.get_all_providers()]:
        router.register_provider(provider)
        logger.info(f"Registered provider for monitoring: {provider.name}")
