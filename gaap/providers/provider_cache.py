"""
Provider Cache Module for GAAP System

Provides a sophisticated caching system for provider data to optimize
performance and reduce API calls.

Classes:
    - ProviderCacheEntry: Data class for cache entries with TTL
    - CacheRefreshStrategy: Enum for refresh strategies
    - CircuitBreaker: Circuit breaker pattern for failing providers
    - CacheStatistics: Cache metrics and statistics
    - ProviderCacheManager: Singleton cache manager with thread/async safety

Usage:
    from gaap.providers.provider_cache import ProviderCacheManager, ProviderCacheEntry

    # Get singleton instance
    cache = ProviderCacheManager()

    # Try cache first
    data = cache.get("glm")
    if not data:
        # Fetch from provider
        data = await glm_provider.get_provider_info()
        cache.set("glm", data, ttl=300)

Features:
    - Singleton pattern (one cache across the app)
    - Thread-safe and async-safe operations
    - Per-provider cache entries with TTL
    - Background refresh with multiple strategies
    - Circuit breaker pattern for failing providers
    - Event system for cache updates
    - Comprehensive statistics and metrics
"""

from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Protocol, TypeVar

from gaap.core.logging import get_standard_logger

logger = get_standard_logger("gaap.providers.cache")

# =============================================================================
# Constants
# =============================================================================

DEFAULT_TTL_SECONDS = 300  # 5 minutes
PROACTIVE_REFRESH_THRESHOLD = 0.2  # Refresh when 20% of TTL remains
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_RESET_TIMEOUT = 60  # seconds
CIRCUIT_BREAKER_HALF_OPEN_REQUESTS = 3
BACKGROUND_REFRESH_INTERVAL = 10  # seconds

T = TypeVar("T")


# =============================================================================
# Event System
# =============================================================================


class CacheEventType(Enum):
    """Types of cache events."""

    HIT = "hit"
    MISS = "miss"
    SET = "set"
    INVALIDATE = "invalidate"
    EXPIRE = "expire"
    REFRESH_START = "refresh_start"
    REFRESH_SUCCESS = "refresh_success"
    REFRESH_FAILURE = "refresh_failure"
    STALE_SERVING = "stale_serving"
    CIRCUIT_OPEN = "circuit_open"
    CIRCUIT_CLOSE = "circuit_close"


@dataclass(frozen=True)
class CacheEvent:
    """Cache event data."""

    event_type: CacheEventType
    provider_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


class CacheEventListener(Protocol):
    """Protocol for cache event listeners."""

    async def on_cache_event(self, event: CacheEvent) -> None:
        """Handle cache event."""
        ...


class EventEmitter:
    """Event emitter for cache events."""

    def __init__(self) -> None:
        self._listeners: list[CacheEventListener] = []
        self._lock = asyncio.Lock()

    def add_listener(self, listener: CacheEventListener) -> None:
        """Add an event listener."""
        if listener not in self._listeners:
            self._listeners.append(listener)
            logger.debug(f"Added cache event listener: {listener}")

    def remove_listener(self, listener: CacheEventListener) -> None:
        """Remove an event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
            logger.debug(f"Removed cache event listener: {listener}")

    async def emit(self, event: CacheEvent) -> None:
        """Emit event to all listeners."""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener.on_cache_event):
                    await listener.on_cache_event(event)
                else:
                    listener.on_cache_event(event)
            except Exception as e:
                logger.error(f"Error in cache event listener: {e}")


# =============================================================================
# Cache Entry
# =============================================================================


@dataclass
class ProviderCacheEntry:
    """
    Cache entry for provider data.

    Attributes:
        provider_name: Name of the provider
        actual_model: Model being used
        status: Provider status (e.g., 'healthy', 'degraded', 'unhealthy')
        latency_ms: Average latency in milliseconds
        success_rate: Success rate (0.0 to 1.0)
        cached_at: When the entry was cached
        ttl_seconds: Time to live in seconds
        is_stale: Whether the entry is stale
        metadata: Additional provider-specific data

    Usage:
        >>> entry = ProviderCacheEntry(
        ...     provider_name="glm",
        ...     actual_model="glm-4",
        ...     status="healthy",
        ...     latency_ms=150.0,
        ...     success_rate=0.98,
        ...     ttl_seconds=300
        ... )
        >>> if entry.is_valid():
        ...     print(f"Using cached data for {entry.provider_name}")
    """

    provider_name: str
    actual_model: str
    status: str
    latency_ms: float
    success_rate: float
    cached_at: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    is_stale: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if not self.cached_at:
            return True
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        return datetime.utcnow() > expiry_time

    def is_valid(self) -> bool:
        """Check if the entry is valid (not stale and not expired)."""
        return not self.is_stale and not self.is_expired()

    def time_until_expiry(self) -> float:
        """Get time until expiry in seconds."""
        if not self.cached_at:
            return 0.0
        expiry_time = self.cached_at + timedelta(seconds=self.ttl_seconds)
        remaining = (expiry_time - datetime.utcnow()).total_seconds()
        return max(0.0, remaining)

    def should_refresh_proactively(self) -> bool:
        """Check if entry should be refreshed proactively."""
        if self.is_stale or self.is_expired():
            return True
        time_remaining = self.time_until_expiry()
        threshold = self.ttl_seconds * PROACTIVE_REFRESH_THRESHOLD
        return time_remaining <= threshold

    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return (datetime.utcnow() - self.cached_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Convert entry to dictionary."""
        return {
            "provider_name": self.provider_name,
            "actual_model": self.actual_model,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "success_rate": self.success_rate,
            "cached_at": self.cached_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "is_stale": self.is_stale,
            "is_expired": self.is_expired(),
            "is_valid": self.is_valid(),
            "age_seconds": self.age_seconds(),
            "metadata": self.metadata,
        }


# =============================================================================
# Refresh Strategy
# =============================================================================


class CacheRefreshStrategy(Enum):
    """Cache refresh strategies."""

    LAZY = auto()  # Refresh on access if expired
    PROACTIVE = auto()  # Refresh before TTL expires
    STALE_WHILE_REVALIDATE = auto()  # Serve stale, refresh in background


class RefreshStrategy(ABC):
    """Abstract base class for refresh strategies."""

    @abstractmethod
    async def should_refresh(self, entry: ProviderCacheEntry | None, provider_name: str) -> bool:
        """Determine if a refresh is needed."""
        ...

    @abstractmethod
    async def handle_refresh(
        self,
        entry: ProviderCacheEntry | None,
        provider_name: str,
        refresh_func: Callable[[str], Any],
    ) -> ProviderCacheEntry | None:
        """Handle the refresh logic."""
        ...


class LazyRefreshStrategy(RefreshStrategy):
    """Lazy refresh - only refresh on access if expired."""

    async def should_refresh(self, entry: ProviderCacheEntry | None, provider_name: str) -> bool:
        if entry is None:
            return True
        return entry.is_expired()

    async def handle_refresh(
        self,
        entry: ProviderCacheEntry | None,
        provider_name: str,
        refresh_func: Callable[[str], Any],
    ) -> ProviderCacheEntry | None:
        try:
            return await refresh_func(provider_name)
        except Exception as e:
            logger.error(f"Lazy refresh failed for {provider_name}: {e}")
            return None


class ProactiveRefreshStrategy(RefreshStrategy):
    """Proactive refresh - refresh before TTL expires."""

    async def should_refresh(self, entry: ProviderCacheEntry | None, provider_name: str) -> bool:
        if entry is None:
            return True
        return entry.should_refresh_proactively()

    async def handle_refresh(
        self,
        entry: ProviderCacheEntry | None,
        provider_name: str,
        refresh_func: Callable[[str], Any],
    ) -> ProviderCacheEntry | None:
        try:
            return await refresh_func(provider_name)
        except Exception as e:
            logger.error(f"Proactive refresh failed for {provider_name}: {e}")
            return entry  # Return existing entry on failure


class StaleWhileRevalidateStrategy(RefreshStrategy):
    """Stale-while-revalidate - serve stale data while refreshing in background."""

    def __init__(self, background_refresh: bool = True) -> None:
        self.background_refresh = background_refresh

    async def should_refresh(self, entry: ProviderCacheEntry | None, provider_name: str) -> bool:
        if entry is None:
            return True
        return entry.is_expired() or entry.is_stale

    async def handle_refresh(
        self,
        entry: ProviderCacheEntry | None,
        provider_name: str,
        refresh_func: Callable[[str], Any],
    ) -> ProviderCacheEntry | None:
        if entry and self.background_refresh:
            # Serve stale entry and refresh in background
            asyncio.create_task(self._background_refresh(provider_name, refresh_func))
            entry.is_stale = True
            return entry
        else:
            try:
                return await refresh_func(provider_name)
            except Exception as e:
                logger.error(f"Refresh failed for {provider_name}: {e}")
                if entry:
                    entry.is_stale = True
                    return entry
                return None

    async def _background_refresh(
        self, provider_name: str, refresh_func: Callable[[str], Any]
    ) -> None:
        """Perform background refresh."""
        try:
            await refresh_func(provider_name)
        except Exception as e:
            logger.error(f"Background refresh failed for {provider_name}: {e}")


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for failing providers.

    Prevents repeated calls to failing providers.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds to wait before trying again
        half_open_max_requests: Max requests in half-open state

    Usage:
        >>> cb = CircuitBreaker(failure_threshold=5)
        >>> if cb.can_execute():
        ...     try:
        ...         result = await fetch_provider_data()
        ...         cb.record_success()
        ...     except Exception:
        ...         cb.record_failure()
    """

    failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD
    reset_timeout: int = CIRCUIT_BREAKER_RESET_TIMEOUT
    half_open_max_requests: int = CIRCUIT_BREAKER_HALF_OPEN_REQUESTS

    _failure_count: int = field(default=0, init=False)
    _last_failure_time: datetime | None = field(default=None, init=False)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _half_open_requests: int = field(default=0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_requests = 0
                    logger.info("Circuit breaker moved to half-open state")
                    return True
                return False
            else:  # HALF_OPEN
                return self._half_open_requests < self.half_open_max_requests

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return True
        elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
        return elapsed >= self.reset_timeout

    def record_success(self) -> None:
        """Record a successful execution."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_requests += 1
                if self._half_open_requests >= self.half_open_max_requests:
                    self._state = CircuitState.CLOSED
                    self._half_open_requests = 0
                    logger.info("Circuit breaker closed after successful recovery")

    def record_failure(self) -> None:
        """Record a failed execution."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.utcnow()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened for provider after half-open failure")
            elif (
                self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._state

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "last_failure_time": (
                    self._last_failure_time.isoformat() if self._last_failure_time else None
                ),
                "half_open_requests": self._half_open_requests,
            }


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class CacheStatistics:
    """
    Cache statistics and metrics.

    Tracks cache performance metrics.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
        sets: Number of cache sets
        invalidations: Number of invalidations
        refreshes: Number of refreshes
        refresh_failures: Number of refresh failures
        stale_serves: Number of times stale data was served

    Usage:
        >>> stats = CacheStatistics()
        >>> stats.record_hit()
        >>> print(f"Hit rate: {stats.hit_rate():.2%}")
    """

    hits: int = 0
    misses: int = 0
    sets: int = 0
    invalidations: int = 0
    refreshes: int = 0
    refresh_failures: int = 0
    stale_serves: int = 0
    circuit_opens: int = 0
    _total_ttl_seconds: float = field(default=0.0, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def record_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self.misses += 1

    def record_set(self, ttl_seconds: int) -> None:
        """Record a cache set."""
        with self._lock:
            self.sets += 1
            self._total_ttl_seconds += ttl_seconds

    def record_invalidation(self) -> None:
        """Record a cache invalidation."""
        with self._lock:
            self.invalidations += 1

    def record_refresh(self, success: bool = True) -> None:
        """Record a cache refresh."""
        with self._lock:
            self.refreshes += 1
            if not success:
                self.refresh_failures += 1

    def record_stale_serve(self) -> None:
        """Record serving stale data."""
        with self._lock:
            self.stale_serves += 1

    def record_circuit_open(self) -> None:
        """Record circuit breaker opening."""
        with self._lock:
            self.circuit_opens += 1

    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    def miss_rate(self) -> float:
        """Calculate cache miss rate."""
        with self._lock:
            total = self.hits + self.misses
            return self.misses / total if total > 0 else 0.0

    def average_ttl(self) -> float:
        """Calculate average TTL."""
        with self._lock:
            return self._total_ttl_seconds / self.sets if self.sets > 0 else 0.0

    def refresh_success_rate(self) -> float:
        """Calculate refresh success rate."""
        with self._lock:
            return (
                (self.refreshes - self.refresh_failures) / self.refreshes
                if self.refreshes > 0
                else 1.0
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary."""
        with self._lock:
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "invalidations": self.invalidations,
                "refreshes": self.refreshes,
                "refresh_failures": self.refresh_failures,
                "stale_serves": self.stale_serves,
                "circuit_opens": self.circuit_opens,
                "hit_rate": self.hit_rate(),
                "miss_rate": self.miss_rate(),
                "average_ttl": self.average_ttl(),
                "refresh_success_rate": self.refresh_success_rate(),
            }


# =============================================================================
# Cache Manager
# =============================================================================


class ProviderCacheManager:
    """
    Singleton cache manager for provider data.

    Provides thread-safe and async-safe caching with TTL support,
    background refresh, circuit breaker pattern, and event emission.

    Attributes:
        default_ttl: Default TTL for cache entries
        refresh_strategy: Strategy for refreshing cache

    Usage:
        >>> cache = ProviderCacheManager()  # Singleton
        >>> entry = ProviderCacheEntry(...)
        >>> cache.set("glm", entry, ttl=300)
        >>> data = cache.get("glm")
        >>> if data:
        ...     print(f"Latency: {data.latency_ms}ms")
        >>> # Invalidate when needed
        >>> cache.invalidate("glm")
    """

    _instance: ProviderCacheManager | None = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args: Any, **kwargs: Any) -> ProviderCacheManager:
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        default_ttl: int = DEFAULT_TTL_SECONDS,
        refresh_strategy: CacheRefreshStrategy = CacheRefreshStrategy.STALE_WHILE_REVALIDATE,
        enable_background_refresh: bool = True,
    ) -> None:
        """
        Initialize the cache manager.

        Args:
            default_ttl: Default time-to-live in seconds
            refresh_strategy: Strategy for cache refresh
            enable_background_refresh: Whether to enable background refresh
        """
        # Skip if already initialized (singleton)
        if self._initialized:
            return

        self._default_ttl = default_ttl
        self._enable_background_refresh = enable_background_refresh

        # Cache storage
        self._cache: dict[str, ProviderCacheEntry] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = defaultdict(lambda: CircuitBreaker())

        # Threading locks
        self._cache_lock = threading.RLock()
        self._async_lock = asyncio.Lock()

        # Statistics and events
        self._stats = CacheStatistics()
        self._event_emitter = EventEmitter()

        # Refresh strategy
        self._refresh_strategy = self._create_strategy(refresh_strategy)

        # Refresh callbacks
        self._refresh_callbacks: dict[str, Callable[[str], Any]] = {}

        # Background refresh task
        self._background_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()

        # Initialize background refresh
        if self._enable_background_refresh:
            self._start_background_refresh()

        self._initialized = True
        logger.info("ProviderCacheManager initialized")

    def _create_strategy(self, strategy: CacheRefreshStrategy) -> RefreshStrategy:
        """Create refresh strategy instance."""
        if strategy == CacheRefreshStrategy.LAZY:
            return LazyRefreshStrategy()
        elif strategy == CacheRefreshStrategy.PROACTIVE:
            return ProactiveRefreshStrategy()
        else:
            return StaleWhileRevalidateStrategy(background_refresh=self._enable_background_refresh)

    def _start_background_refresh(self) -> None:
        """Start background refresh task."""
        try:
            loop = asyncio.get_running_loop()
            self._background_task = loop.create_task(self._background_refresh_loop())
            logger.debug("Background refresh started")
        except RuntimeError:
            # No running loop, will be started later
            pass

    async def _background_refresh_loop(self) -> None:
        """Background loop for proactive cache refresh."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_proactive_refresh()
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=BACKGROUND_REFRESH_INTERVAL,
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in background refresh loop: {e}")
                await asyncio.sleep(BACKGROUND_REFRESH_INTERVAL)

    async def _perform_proactive_refresh(self) -> None:
        """Perform proactive refresh for entries nearing expiry."""
        entries_to_refresh = []

        with self._cache_lock:
            for provider_name, entry in self._cache.items():
                if entry.should_refresh_proactively() and not entry.is_stale:
                    circuit = self._circuit_breakers[provider_name]
                    if circuit.can_execute():
                        entries_to_refresh.append(provider_name)

        for provider_name in entries_to_refresh:
            await self.refresh_if_needed(provider_name)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get(self, provider_name: str) -> ProviderCacheEntry | None:
        """
        Get cached data for a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Cache entry or None if not found/expired
        """
        with self._cache_lock:
            entry = self._cache.get(provider_name)

            if entry is None:
                self._stats.record_miss()
                asyncio.create_task(
                    self._event_emitter.emit(
                        CacheEvent(
                            event_type=CacheEventType.MISS,
                            provider_name=provider_name,
                        )
                    )
                )
                return None

            if entry.is_valid():
                self._stats.record_hit()
                asyncio.create_task(
                    self._event_emitter.emit(
                        CacheEvent(
                            event_type=CacheEventType.HIT,
                            provider_name=provider_name,
                            metadata={"age_seconds": entry.age_seconds()},
                        )
                    )
                )
                return entry

            # Entry is expired or stale
            if entry.is_stale:
                self._stats.record_stale_serve()
                asyncio.create_task(
                    self._event_emitter.emit(
                        CacheEvent(
                            event_type=CacheEventType.STALE_SERVING,
                            provider_name=provider_name,
                        )
                    )
                )

            self._stats.record_miss()
            asyncio.create_task(
                self._event_emitter.emit(
                    CacheEvent(
                        event_type=CacheEventType.MISS,
                        provider_name=provider_name,
                        metadata={"expired": entry.is_expired(), "stale": entry.is_stale},
                    )
                )
            )
            return None

    def set(
        self,
        provider_name: str,
        entry: ProviderCacheEntry,
        ttl: int | None = None,
    ) -> None:
        """
        Store data in cache.

        Args:
            provider_name: Name of the provider
            entry: Cache entry to store
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        ttl = ttl or self._default_ttl
        entry.ttl_seconds = ttl
        entry.cached_at = datetime.utcnow()
        entry.is_stale = False

        with self._cache_lock:
            self._cache[provider_name] = entry
            self._stats.record_set(ttl)

        asyncio.create_task(
            self._event_emitter.emit(
                CacheEvent(
                    event_type=CacheEventType.SET,
                    provider_name=provider_name,
                    metadata={"ttl": ttl},
                )
            )
        )

        logger.debug(f"Cached data for {provider_name} with TTL {ttl}s")

    def invalidate(self, provider_name: str) -> bool:
        """
        Invalidate cache for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            True if entry was found and removed
        """
        with self._cache_lock:
            if provider_name in self._cache:
                del self._cache[provider_name]
                self._stats.record_invalidation()

                asyncio.create_task(
                    self._event_emitter.emit(
                        CacheEvent(
                            event_type=CacheEventType.INVALIDATE,
                            provider_name=provider_name,
                        )
                    )
                )

                logger.debug(f"Invalidated cache for {provider_name}")
                return True
            return False

    def invalidate_all(self) -> int:
        """
        Invalidate all cache entries.

        Returns:
            Number of entries cleared
        """
        with self._cache_lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.record_invalidation()

            for provider_name in list(self._cache.keys()):
                asyncio.create_task(
                    self._event_emitter.emit(
                        CacheEvent(
                            event_type=CacheEventType.INVALIDATE,
                            provider_name=provider_name,
                        )
                    )
                )

            logger.info(f"Invalidated all cache entries ({count} total)")
            return count

    def get_all_valid(self) -> dict[str, ProviderCacheEntry]:
        """
        Get all non-expired cache entries.

        Returns:
            Dictionary of provider names to valid cache entries
        """
        with self._cache_lock:
            return {name: entry for name, entry in self._cache.items() if entry.is_valid()}

    async def refresh_if_needed(
        self,
        provider_name: str,
        refresh_func: Callable[[str], Any] | None = None,
    ) -> ProviderCacheEntry | None:
        """
        Refresh cache for a provider if needed.

        Args:
            provider_name: Name of the provider
            refresh_func: Optional function to fetch fresh data

        Returns:
            Updated cache entry or None if refresh failed
        """
        entry = self.get(provider_name)

        # Check circuit breaker
        circuit = self._circuit_breakers[provider_name]
        if not circuit.can_execute():
            logger.warning(f"Circuit breaker open for {provider_name}, skipping refresh")
            if entry:
                return entry
            return None

        # Determine if refresh is needed
        if not await self._refresh_strategy.should_refresh(entry, provider_name):
            return entry

        # Use registered callback if no function provided
        if refresh_func is None and provider_name in self._refresh_callbacks:
            refresh_func = self._refresh_callbacks[provider_name]

        if refresh_func is None:
            logger.warning(f"No refresh function for {provider_name}")
            return entry

        # Emit refresh start event
        await self._event_emitter.emit(
            CacheEvent(
                event_type=CacheEventType.REFRESH_START,
                provider_name=provider_name,
            )
        )

        try:
            new_entry = await refresh_func(provider_name)

            if new_entry is not None:
                if isinstance(new_entry, ProviderCacheEntry):
                    self.set(provider_name, new_entry)
                else:
                    # Convert dict to ProviderCacheEntry if needed
                    entry_data = (
                        new_entry
                        if isinstance(new_entry, dict)
                        else {"provider_name": provider_name}
                    )
                    entry = ProviderCacheEntry(
                        provider_name=provider_name,
                        actual_model=entry_data.get("actual_model", ""),
                        status=entry_data.get("status", "unknown"),
                        latency_ms=entry_data.get("latency_ms", 0.0),
                        success_rate=entry_data.get("success_rate", 1.0),
                        metadata=entry_data.get("metadata", {}),
                    )
                    self.set(provider_name, entry)

                circuit.record_success()
                self._stats.record_refresh(success=True)

                await self._event_emitter.emit(
                    CacheEvent(
                        event_type=CacheEventType.REFRESH_SUCCESS,
                        provider_name=provider_name,
                    )
                )

                logger.debug(f"Successfully refreshed cache for {provider_name}")
                return self.get(provider_name)
            else:
                raise ValueError("Refresh function returned None")

        except Exception as e:
            circuit.record_failure()
            self._stats.record_refresh(success=False)

            if circuit.state == CircuitState.OPEN:
                self._stats.record_circuit_open()
                await self._event_emitter.emit(
                    CacheEvent(
                        event_type=CacheEventType.CIRCUIT_OPEN,
                        provider_name=provider_name,
                    )
                )

            await self._event_emitter.emit(
                CacheEvent(
                    event_type=CacheEventType.REFRESH_FAILURE,
                    provider_name=provider_name,
                    metadata={"error": str(e)},
                )
            )

            logger.error(f"Failed to refresh cache for {provider_name}: {e}")
            return entry

    def register_refresh_callback(self, provider_name: str, callback: Callable[[str], Any]) -> None:
        """
        Register a callback for refreshing provider data.

        Args:
            provider_name: Name of the provider
            callback: Async function that fetches fresh data
        """
        self._refresh_callbacks[provider_name] = callback
        logger.debug(f"Registered refresh callback for {provider_name}")

    def unregister_refresh_callback(self, provider_name: str) -> bool:
        """
        Unregister a refresh callback.

        Args:
            provider_name: Name of the provider

        Returns:
            True if callback was found and removed
        """
        if provider_name in self._refresh_callbacks:
            del self._refresh_callbacks[provider_name]
            logger.debug(f"Unregistered refresh callback for {provider_name}")
            return True
        return False

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def add_event_listener(self, listener: CacheEventListener) -> None:
        """Add an event listener."""
        self._event_emitter.add_listener(listener)

    def remove_event_listener(self, listener: CacheEventListener) -> None:
        """Remove an event listener."""
        self._event_emitter.remove_listener(listener)

    # -------------------------------------------------------------------------
    # Statistics and Metrics
    # -------------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics."""
        return self._stats.to_dict()

    def get_provider_stats(self, provider_name: str) -> dict[str, Any] | None:
        """Get statistics for a specific provider."""
        with self._cache_lock:
            entry = self._cache.get(provider_name)
            if entry is None:
                return None

            circuit_stats = self._circuit_breakers[provider_name].get_stats()

            return {
                "cache_entry": entry.to_dict(),
                "circuit_breaker": circuit_stats,
            }

    def get_all_provider_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all providers."""
        with self._cache_lock:
            return {
                name: {
                    "cache_entry": entry.to_dict(),
                    "circuit_breaker": self._circuit_breakers[name].get_stats(),
                }
                for name, entry in self._cache.items()
            }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._stats = CacheStatistics()
        logger.info("Cache statistics reset")

    # -------------------------------------------------------------------------
    # Circuit Breaker Control
    # -------------------------------------------------------------------------

    def get_circuit_breaker(self, provider_name: str) -> CircuitBreaker:
        """Get circuit breaker for a provider."""
        return self._circuit_breakers[provider_name]

    def reset_circuit_breaker(self, provider_name: str) -> None:
        """Reset circuit breaker for a provider."""
        self._circuit_breakers[provider_name] = CircuitBreaker()
        logger.info(f"Reset circuit breaker for {provider_name}")

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Shutdown the cache manager."""
        logger.info("Shutting down ProviderCacheManager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel background task
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

        # Clear cache
        with self._cache_lock:
            self._cache.clear()
            self._refresh_callbacks.clear()

        logger.info("ProviderCacheManager shutdown complete")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        if hasattr(self, "_background_task") and self._background_task:
            if not self._background_task.done():
                self._background_task.cancel()


# =============================================================================
# Convenience Functions
# =============================================================================


def get_cache_manager(
    default_ttl: int = DEFAULT_TTL_SECONDS,
    refresh_strategy: CacheRefreshStrategy = CacheRefreshStrategy.STALE_WHILE_REVALIDATE,
) -> ProviderCacheManager:
    """
    Get the singleton cache manager instance.

    Args:
        default_ttl: Default TTL for cache entries
        refresh_strategy: Strategy for cache refresh

    Returns:
        ProviderCacheManager singleton instance
    """
    return ProviderCacheManager(
        default_ttl=default_ttl,
        refresh_strategy=refresh_strategy,
    )


async def cached_provider_call(
    provider_name: str,
    fetch_func: Callable[[], Any],
    ttl: int = DEFAULT_TTL_SECONDS,
    cache_manager: ProviderCacheManager | None = None,
) -> ProviderCacheEntry:
    """
    Decorator-style function for cached provider calls.

    Args:
        provider_name: Name of the provider
        fetch_func: Function to fetch data if not cached
        ttl: Cache TTL in seconds
        cache_manager: Optional cache manager instance

    Returns:
        ProviderCacheEntry with provider data

    Usage:
        >>> async def fetch_glm_data():
        ...     return await glm_provider.get_info()
        >>> entry = await cached_provider_call("glm", fetch_glm_data, ttl=300)
    """
    cache = cache_manager or ProviderCacheManager()

    # Try cache first
    data = cache.get(provider_name)
    if data and data.is_valid():
        return data

    # Fetch fresh data
    fresh_data = await fetch_func()

    if isinstance(fresh_data, ProviderCacheEntry):
        entry = fresh_data
    else:
        # Create entry from dict or object
        entry_data = fresh_data if isinstance(fresh_data, dict) else {}
        entry = ProviderCacheEntry(
            provider_name=provider_name,
            actual_model=entry_data.get("actual_model", ""),
            status=entry_data.get("status", "unknown"),
            latency_ms=entry_data.get("latency_ms", 0.0),
            success_rate=entry_data.get("success_rate", 1.0),
            metadata=entry_data.get("metadata", {}),
        )

    cache.set(provider_name, entry, ttl=ttl)
    return entry


# =============================================================================
# WebSocket Integration Helper
# =============================================================================


class WebSocketCacheNotifier(CacheEventListener):
    """
    WebSocket notifier for cache events.

    Emits cache events to WebSocket connections.

    Usage:
        >>> notifier = WebSocketCacheNotifier(websocket_manager)
        >>> cache = ProviderCacheManager()
        >>> cache.add_event_listener(notifier)
    """

    def __init__(self, websocket_manager: Any) -> None:
        """
        Initialize the notifier.

        Args:
            websocket_manager: WebSocket manager instance
        """
        self._websocket_manager = websocket_manager

    async def on_cache_event(self, event: CacheEvent) -> None:
        """Handle cache event and broadcast via WebSocket."""
        message = {
            "type": "cache_event",
            "event": event.event_type.value,
            "provider": event.provider_name,
            "timestamp": event.timestamp.isoformat(),
            "metadata": event.metadata,
        }

        # Broadcast to relevant channels
        try:
            # Broadcast to all connected clients
            if hasattr(self._websocket_manager, "broadcast"):
                await self._websocket_manager.broadcast(message)

            # Also send to provider-specific channel
            channel = f"provider:{event.provider_name}"
            if hasattr(self._websocket_manager, "broadcast_to_channel"):
                await self._websocket_manager.broadcast_to_channel(channel, message)
        except Exception as e:
            logger.error(f"Failed to broadcast cache event: {e}")


# =============================================================================
# Export
# =============================================================================

__all__ = [
    # Core classes
    "ProviderCacheManager",
    "ProviderCacheEntry",
    # Event system
    "CacheEvent",
    "CacheEventType",
    "CacheEventListener",
    "EventEmitter",
    # Refresh strategies
    "CacheRefreshStrategy",
    "RefreshStrategy",
    "LazyRefreshStrategy",
    "ProactiveRefreshStrategy",
    "StaleWhileRevalidateStrategy",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    # Statistics
    "CacheStatistics",
    # WebSocket
    "WebSocketCacheNotifier",
    # Constants
    "DEFAULT_TTL_SECONDS",
    "PROACTIVE_REFRESH_THRESHOLD",
    "CIRCUIT_BREAKER_FAILURE_THRESHOLD",
    "CIRCUIT_BREAKER_RESET_TIMEOUT",
    # Functions
    "get_cache_manager",
    "cached_provider_call",
]
