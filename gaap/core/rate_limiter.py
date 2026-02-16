"""
Advanced Rate Limiting Module

Implements multiple rate limiting strategies:
- Token Bucket (smooth rate limiting)
- Sliding Window Log
- Leaky Bucket
- Adaptive Rate Limiter
"""

import asyncio
import contextlib
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""

    TOKEN_BUCKET = auto()
    SLIDING_WINDOW = auto()
    LEAKY_BUCKET = auto()
    FIXED_WINDOW = auto()
    ADAPTIVE = auto()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""

    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    requests_per_second: float = 10.0
    burst_capacity: int = 100
    refill_rate: float | None = None
    window_size_seconds: float = 60.0
    max_requests_per_window: int = 600
    adaptive_increase_factor: float = 1.5
    adaptive_decrease_factor: float = 0.7
    min_rate: float = 1.0
    max_rate: float = 100.0


@dataclass
class RateLimitState:
    """State of the rate limiter"""

    last_refill: float = field(default_factory=time.time)
    tokens: float = 0.0
    request_count: int = 0
    window_start: float = field(default_factory=time.time)
    denied_count: int = 0
    total_requests: int = 0
    adaptive_rate: float = 10.0
    last_success_rate: float = 1.0


@dataclass
class RateLimitResult:
    """Result of a rate limit check"""

    allowed: bool
    tokens_remaining: float
    wait_time_seconds: float
    retry_after: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRateLimiter(ABC):
    """Abstract base class for rate limiters"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._state = RateLimitState()
        self._lock = asyncio.Lock()
        self._callbacks: list[Callable[[RateLimitResult], Awaitable[None]]] = []

    @abstractmethod
    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Try to acquire tokens"""
        pass

    @abstractmethod
    async def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking acquire check"""
        pass

    async def wait_for_token(self, tokens: int = 1, timeout: float | None = None) -> bool:
        """Wait until tokens are available"""
        start = time.time()
        while True:
            result = await self.acquire(tokens)
            if result.allowed:
                return True

            if timeout and (time.time() - start) >= timeout:
                return False

            wait_time = min(result.wait_time_seconds, 1.0)
            await asyncio.sleep(wait_time)

    def on_limit_exceeded(self, callback: Callable[[RateLimitResult], Awaitable[None]]) -> None:
        """Register a callback for limit exceeded events"""
        self._callbacks.append(callback)

    async def _notify_callbacks(self, result: RateLimitResult) -> None:
        """Notify registered callbacks"""
        for callback in self._callbacks:
            with contextlib.suppress(Exception):
                await callback(result)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics"""
        return {
            "strategy": self.config.strategy.name,
            "total_requests": self._state.total_requests,
            "denied_requests": self._state.denied_count,
            "denial_rate": self._state.denied_count / max(self._state.total_requests, 1),
            "tokens_remaining": self._state.tokens,
        }


class TokenBucketRateLimiter(BaseRateLimiter):
    """
    Token Bucket Rate Limiter

    Smooth rate limiting with burst capability:
    - Tokens are added at a constant rate
    - Each request consumes tokens
    - Can burst up to bucket capacity
    """

    def __init__(self, config: RateLimitConfig):
        config.strategy = RateLimitStrategy.TOKEN_BUCKET
        super().__init__(config)
        self._state.tokens = config.burst_capacity
        self._refill_rate = config.refill_rate or config.requests_per_second

    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Try to acquire tokens, returning result with wait time"""
        async with self._lock:
            self._refill()
            self._state.total_requests += 1

            if self._state.tokens >= tokens:
                self._state.tokens -= tokens
                self._state.request_count += 1
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self._state.tokens,
                    wait_time_seconds=0.0,
                )

            needed = tokens - self._state.tokens
            wait_time = needed / self._refill_rate

            self._state.denied_count += 1
            result = RateLimitResult(
                allowed=False,
                tokens_remaining=self._state.tokens,
                wait_time_seconds=wait_time,
                retry_after=int(math.ceil(wait_time)),
            )

            await self._notify_callbacks(result)
            return result

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking check if tokens are available"""
        async with self._lock:
            self._refill()
            if self._state.tokens >= tokens:
                self._state.tokens -= tokens
                self._state.request_count += 1
                self._state.total_requests += 1
                return True
            self._state.denied_count += 1
            self._state.total_requests += 1
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self._state.last_refill

        refill_amount = elapsed * self._refill_rate
        self._state.tokens = min(self.config.burst_capacity, self._state.tokens + refill_amount)
        self._state.last_refill = now


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    Sliding Window Log Rate Limiter

    More accurate than fixed window:
    - Maintains a log of request timestamps
    - Counts requests in sliding window
    - No burst at window boundaries
    """

    def __init__(self, config: RateLimitConfig):
        config.strategy = RateLimitStrategy.SLIDING_WINDOW
        super().__init__(config)
        self._request_log: list[float] = []

    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Try to acquire within the sliding window"""
        async with self._lock:
            self._state.total_requests += 1
            now = time.time()
            window_start = now - self.config.window_size_seconds

            self._request_log = [t for t in self._request_log if t > window_start]

            current_count = len(self._request_log)

            if current_count + tokens <= self.config.max_requests_per_window:
                for _ in range(tokens):
                    self._request_log.append(now)
                self._state.request_count += tokens
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self.config.max_requests_per_window - current_count - tokens,
                    wait_time_seconds=0.0,
                )

            oldest = self._request_log[0] if self._request_log else now
            wait_time = oldest - window_start + 0.001

            self._state.denied_count += 1
            result = RateLimitResult(
                allowed=False,
                tokens_remaining=0,
                wait_time_seconds=wait_time,
                retry_after=int(math.ceil(wait_time)),
            )

            await self._notify_callbacks(result)
            return result

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking check"""
        async with self._lock:
            self._state.total_requests += 1
            now = time.time()
            window_start = now - self.config.window_size_seconds

            self._request_log = [t for t in self._request_log if t > window_start]

            if len(self._request_log) + tokens <= self.config.max_requests_per_window:
                for _ in range(tokens):
                    self._request_log.append(now)
                self._state.request_count += tokens
                return True

            self._state.denied_count += 1
            return False


class LeakyBucketRateLimiter(BaseRateLimiter):
    """
    Leaky Bucket Rate Limiter

    Constant rate output:
    - Requests fill the bucket
    - Bucket leaks at constant rate
    - No burst capability
    """

    def __init__(self, config: RateLimitConfig):
        config.strategy = RateLimitStrategy.LEAKY_BUCKET
        super().__init__(config)
        self._queue: asyncio.Queue[float] = asyncio.Queue(maxsize=config.burst_capacity)
        self._leak_rate = config.requests_per_second
        self._leak_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the leak process"""
        if self._running:
            return
        self._running = True
        self._leak_task = asyncio.create_task(self._leak_loop())

    async def stop(self) -> None:
        """Stop the leak process"""
        self._running = False
        if self._leak_task:
            self._leak_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._leak_task
            self._leak_task = None

    async def _leak_loop(self) -> None:
        """Continuous leak process"""
        while self._running:
            try:
                if not self._queue.empty():
                    await self._queue.get()
                leak_interval = 1.0 / self._leak_rate
                await asyncio.sleep(leak_interval)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.1)

    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Try to add to the bucket"""
        async with self._lock:
            self._state.total_requests += 1

            if self._queue.qsize() + tokens <= self.config.burst_capacity:
                for _ in range(tokens):
                    try:
                        self._queue.put_nowait(time.time())
                    except asyncio.QueueFull:
                        break
                self._state.request_count += tokens
                return RateLimitResult(
                    allowed=True,
                    tokens_remaining=self.config.burst_capacity - self._queue.qsize(),
                    wait_time_seconds=0.0,
                )

            wait_time = (
                self._queue.qsize() + tokens - self.config.burst_capacity
            ) / self._leak_rate

            self._state.denied_count += 1
            result = RateLimitResult(
                allowed=False,
                tokens_remaining=0,
                wait_time_seconds=max(0, wait_time),
                retry_after=int(math.ceil(max(0, wait_time))),
            )

            await self._notify_callbacks(result)
            return result

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking check"""
        async with self._lock:
            self._state.total_requests += 1

            if self._queue.qsize() + tokens <= self.config.burst_capacity:
                for _ in range(tokens):
                    try:
                        self._queue.put_nowait(time.time())
                        self._state.request_count += 1
                    except asyncio.QueueFull:
                        return False
                return True

            self._state.denied_count += 1
            return False


class AdaptiveRateLimiter(BaseRateLimiter):
    """
    Adaptive Rate Limiter

    Adjusts rate based on success/failure:
    - Increases rate on success
    - Decreases rate on failure
    - Smoothly adapts to optimal rate
    """

    def __init__(self, config: RateLimitConfig):
        config.strategy = RateLimitStrategy.ADAPTIVE
        super().__init__(config)
        self._state.adaptive_rate = config.requests_per_second
        self._success_window: list[bool] = []
        self._window_size = 100
        self._underlying: TokenBucketRateLimiter | None = None

    async def _get_underlying(self) -> TokenBucketRateLimiter:
        """Get or create underlying token bucket"""
        if self._underlying is None:
            underlying_config = RateLimitConfig(
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                requests_per_second=self._state.adaptive_rate,
                burst_capacity=self.config.burst_capacity,
            )
            self._underlying = TokenBucketRateLimiter(underlying_config)
        return self._underlying

    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Acquire with adaptive adjustment"""
        underlying = await self._get_underlying()
        result = await underlying.acquire(tokens)

        self._state.total_requests += 1

        self._success_window.append(result.allowed)
        if len(self._success_window) > self._window_size:
            self._success_window.pop(0)

        success_rate = (
            sum(self._success_window) / len(self._success_window) if self._success_window else 1.0
        )
        self._state.last_success_rate = success_rate

        if success_rate > 0.95:
            self._state.adaptive_rate = min(
                self.config.max_rate,
                self._state.adaptive_rate * self.config.adaptive_increase_factor,
            )
        elif success_rate < 0.8:
            self._state.adaptive_rate = max(
                self.config.min_rate,
                self._state.adaptive_rate * self.config.adaptive_decrease_factor,
            )

        underlying._refill_rate = self._state.adaptive_rate

        if not result.allowed:
            self._state.denied_count += 1

        result.metadata["adaptive_rate"] = self._state.adaptive_rate
        result.metadata["success_rate"] = success_rate
        return result

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Non-blocking acquire"""
        underlying = await self._get_underlying()
        result = await underlying.try_acquire(tokens)

        self._success_window.append(result)
        if len(self._success_window) > self._window_size:
            self._success_window.pop(0)

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get statistics including adaptive rate"""
        stats = super().get_stats()
        stats.update(
            {
                "adaptive_rate": self._state.adaptive_rate,
                "success_rate": self._state.last_success_rate,
            }
        )
        return stats


def create_rate_limiter(config: RateLimitConfig) -> BaseRateLimiter:
    """Factory function to create a rate limiter"""
    limiters = {
        RateLimitStrategy.TOKEN_BUCKET: TokenBucketRateLimiter,
        RateLimitStrategy.SLIDING_WINDOW: SlidingWindowRateLimiter,
        RateLimitStrategy.LEAKY_BUCKET: LeakyBucketRateLimiter,
        RateLimitStrategy.ADAPTIVE: AdaptiveRateLimiter,
    }

    limiter_class = limiters.get(config.strategy, TokenBucketRateLimiter)
    return limiter_class(config)


class CompositeRateLimiter:
    """
    Composite Rate Limiter

    Combines multiple rate limiters:
    - Each limiter is checked
    - Most restrictive wins
    """

    def __init__(self, limiters: list[BaseRateLimiter]):
        self._limiters = limiters

    async def acquire(self, tokens: int = 1) -> RateLimitResult:
        """Check all limiters, return most restrictive result"""
        results = await asyncio.gather(*[limiter.acquire(tokens) for limiter in self._limiters])

        denied_results = [r for r in results if not r.allowed]

        if not denied_results:
            return results[0]

        return max(denied_results, key=lambda r: r.wait_time_seconds)

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Check all limiters non-blocking"""
        results = await asyncio.gather(*[limiter.try_acquire(tokens) for limiter in self._limiters])
        return all(results)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics from all limiters"""
        return {f"limiter_{i}": limiter.get_stats() for i, limiter in enumerate(self._limiters)}
