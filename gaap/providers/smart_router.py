"""
Smart Multi-Provider Router
===========================

Intelligent routing across multiple free-tier providers with:
- Automatic failover
- Rate limit awareness
- Load balancing across keys
- Provider health tracking
- Cost optimization
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from multi_provider_config import (
    ProviderConfig,
    ProviderType,
    get_enabled_providers,
)

logger = logging.getLogger(__name__)


@dataclass
class KeyState:
    """Track state of a single API key"""

    key_index: int
    last_used: float = 0.0
    requests_this_minute: int = 0
    requests_today: int = 0
    minute_window_start: float = field(default_factory=time.time)
    day_start: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    is_exhausted: bool = False

    def reset_minute_if_needed(self):
        """Reset minute counter if window passed"""
        now = time.time()
        if now - self.minute_window_start >= 60:
            self.requests_this_minute = 0
            self.minute_window_start = now

    def reset_day_if_needed(self):
        """Reset day counter if day passed"""
        now = time.time()
        if now - self.day_start >= 86400:
            self.requests_today = 0
            self.day_start = now

    def can_use(self, limits) -> bool:
        """Check if key can be used"""
        self.reset_minute_if_needed()
        self.reset_day_if_needed()

        if self.is_exhausted:
            return False

        if (
            limits.requests_per_minute > 0
            and self.requests_this_minute >= limits.requests_per_minute
        ):
            return False

        return not (limits.requests_per_day > 0 and self.requests_today >= limits.requests_per_day)

    def mark_used(self):
        """Mark key as used"""
        self.reset_minute_if_needed()
        self.reset_day_if_needed()
        self.last_used = time.time()
        self.requests_this_minute += 1
        self.requests_today += 1
        self.consecutive_errors = 0

    def mark_error(self):
        """Mark an error occurred"""
        self.consecutive_errors += 1
        if self.consecutive_errors >= 3:
            self.is_exhausted = True

    def mark_rate_limited(self):
        """Mark key as rate limited"""
        self.is_exhausted = True
        # Auto-reset after cooldown period (handled by provider)


@dataclass
class ProviderState:
    """Track state of a provider"""

    config: ProviderConfig
    key_states: list[KeyState] = field(default_factory=list)
    total_requests: int = 0
    total_errors: int = 0
    total_cost: float = 0.0
    is_healthy: bool = True
    last_health_check: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.key_states:
            self.key_states = [KeyState(key_index=i) for i in range(len(self.config.api_keys))]

    def get_available_key(self) -> tuple[int, str] | None:
        """Get an available API key index and value"""
        # Reset exhausted keys if cooldown passed
        self._reset_exhausted_keys()

        # Find keys that can be used
        available = [
            (i, state)
            for i, state in enumerate(self.key_states)
            if state.can_use(self.config.limits)
        ]

        if not available:
            return None

        # Pick the least recently used
        available.sort(key=lambda x: x[1].last_used)
        key_index, state = available[0]

        return key_index, self.config.api_keys[key_index]

    def _reset_exhausted_keys(self):
        """Reset exhausted keys after cooldown"""
        now = time.time()
        cooldown = self.config.limits.cooldown_window

        for state in self.key_states:
            if state.is_exhausted and (now - state.last_used) >= cooldown:
                state.is_exhausted = False
                state.consecutive_errors = 0

    def mark_request(self, key_index: int, success: bool, cost: float = 0.0):
        """Mark a request attempt"""
        self.total_requests += 1

        if success:
            self.key_states[key_index].mark_used()
            self.total_cost += cost
        else:
            self.total_errors += 1
            self.key_states[key_index].mark_error()

    def get_utilization(self) -> float:
        """Get current utilization (0.0 - 1.0)"""
        available = sum(1 for state in self.key_states if state.can_use(self.config.limits))
        total = len(self.key_states)
        return 1.0 - (available / max(total, 1))


class SmartRouter:
    """
    Smart router that manages multiple providers with:
    - Automatic provider selection based on priority and availability
    - Rate limit awareness and key rotation
    - Automatic failover
    - Health tracking
    """

    def __init__(self, enabled_provider_types: list[ProviderType] | None = None):
        """
        Initialize router

        Args:
            enabled_provider_types: List of provider types to enable, or None for all
        """
        self.provider_states: dict[ProviderType, ProviderState] = {}

        # Load configurations
        all_providers = get_enabled_providers()

        for config in all_providers:
            if enabled_provider_types is None or config.provider_type in enabled_provider_types:
                self.provider_states[config.provider_type] = ProviderState(config=config)

        logger.info(f"SmartRouter initialized with {len(self.provider_states)} providers")

    def select_provider(
        self,
        model_preference: str | None = None,
        exclude_providers: list[ProviderType] | None = None,
    ) -> tuple[ProviderType, int, str] | None:
        """
        Select best provider and API key

        Returns:
            Tuple of (provider_type, key_index, api_key) or None if no provider available
        """
        exclude_providers = exclude_providers or []

        # Get eligible providers sorted by priority
        eligible = [
            (ptype, state)
            for ptype, state in self.provider_states.items()
            if ptype not in exclude_providers and state.is_healthy
        ]

        # Sort by priority (from config) and utilization
        eligible.sort(
            key=lambda x: (
                x[1].config.priority,  # Higher priority first
                -x[1].get_utilization(),  # Lower utilization better
            ),
            reverse=True,
        )

        # Try each provider until we find an available key
        for provider_type, state in eligible:
            # Check model compatibility if specified
            if model_preference and model_preference not in state.config.models:
                continue

            # Try to get an available key
            key_result = state.get_available_key()
            if key_result:
                key_index, api_key = key_result
                logger.debug(
                    f"Selected {state.config.name} (key {key_index + 1}/{len(state.config.api_keys)})"
                )
                return provider_type, key_index, api_key

        logger.warning("No available providers!")
        return None

    def mark_request(
        self,
        provider_type: ProviderType,
        key_index: int,
        success: bool,
        cost: float = 0.0,
        rate_limited: bool = False,
    ):
        """Mark request result"""
        if provider_type not in self.provider_states:
            return

        state = self.provider_states[provider_type]
        state.mark_request(key_index, success, cost)

        if rate_limited:
            state.key_states[key_index].mark_rate_limited()

    async def execute_with_retry(self, call_fn, max_provider_attempts: int = 3, **kwargs):
        """
        Execute a function with automatic provider retry

        Args:
            call_fn: Async function that takes (provider_type, api_key) and returns result
            max_provider_attempts: Max number of different providers to try
            **kwargs: Additional args for select_provider

        Returns:
            Result from call_fn

        Raises:
            Exception if all providers exhausted
        """
        attempted_providers = []
        last_error = None

        for attempt in range(max_provider_attempts):
            # Select provider
            selection = self.select_provider(exclude_providers=attempted_providers, **kwargs)

            if not selection:
                raise RuntimeError(
                    f"No available providers after {attempt} attempts. Tried: {attempted_providers}"
                )

            provider_type, key_index, api_key = selection
            attempted_providers.append(provider_type)

            try:
                # Execute the call
                result = await call_fn(provider_type, api_key)

                # Mark success
                self.mark_request(provider_type, key_index, success=True)

                return result

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if rate limited
                is_rate_limited = "rate" in error_str or "quota" in error_str or "429" in error_str

                # Mark failure
                self.mark_request(
                    provider_type, key_index, success=False, rate_limited=is_rate_limited
                )

                logger.warning(
                    f"Provider {provider_type.value} failed (attempt {attempt + 1}): {e}"
                )

                # If not rate limited, this might be a real error - don't retry
                if not is_rate_limited and attempt == 0:
                    # But give it one more chance with different provider
                    continue

                # Wait before retry
                if attempt < max_provider_attempts - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        # All attempts failed
        raise RuntimeError(
            f"All provider attempts failed. Last error: {last_error}"
        ) from last_error

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics"""
        stats = {
            "total_providers": len(self.provider_states),
            "healthy_providers": sum(1 for s in self.provider_states.values() if s.is_healthy),
            "total_requests": sum(s.total_requests for s in self.provider_states.values()),
            "total_errors": sum(s.total_errors for s in self.provider_states.values()),
            "total_cost": sum(s.total_cost for s in self.provider_states.values()),
            "providers": {},
        }

        for ptype, state in self.provider_states.items():
            available_keys = sum(1 for ks in state.key_states if ks.can_use(state.config.limits))

            stats["providers"][ptype.value] = {
                "name": state.config.name,
                "priority": state.config.priority,
                "total_keys": len(state.key_states),
                "available_keys": available_keys,
                "utilization": f"{state.get_utilization() * 100:.1f}%",
                "requests": state.total_requests,
                "errors": state.total_errors,
                "cost": state.total_cost,
                "healthy": state.is_healthy,
            }

        return stats

    def print_stats(self):
        """Print router statistics"""
        stats = self.get_stats()

        print("=" * 80)
        print("SMART ROUTER STATISTICS")
        print("=" * 80)
        print(f"Providers: {stats['healthy_providers']}/{stats['total_providers']} healthy")
        print(f"Total Requests: {stats['total_requests']:,}")
        print(f"Total Errors: {stats['total_errors']:,}")
        print(f"Total Cost: ${stats['total_cost']:.6f}")
        print()
        print(f"{'Provider':<25} {'Pri':>4} {'Keys':>10} {'Util':>8} {'Reqs':>8} {'Errs':>6}")
        print("-" * 80)

        for pdata in sorted(stats["providers"].values(), key=lambda x: x["priority"], reverse=True):
            print(
                f"{pdata['name']:<25} {pdata['priority']:>4} "
                f"{pdata['available_keys']}/{pdata['total_keys']:>1} "
                f"{pdata['utilization']:>8} {pdata['requests']:>8,} {pdata['errors']:>6}"
            )

        print("=" * 80)


# Example usage
async def example_usage():
    """Example of how to use SmartRouter"""

    # Initialize router with specific providers
    router = SmartRouter(
        enabled_provider_types=[
            ProviderType.CEREBRAS,
            ProviderType.GROQ,
            ProviderType.OPENROUTER,
        ]
    )

    # Example call function
    async def make_api_call(provider_type: ProviderType, api_key: str):
        # This would be your actual API call
        print(f"Calling {provider_type.value} with key {api_key[:20]}...")
        await asyncio.sleep(0.1)  # Simulate API call
        return {"response": "Success"}

    # Execute with automatic retry across providers
    try:
        result = await router.execute_with_retry(make_api_call)
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

    # Print stats
    router.print_stats()


if __name__ == "__main__":
    asyncio.run(example_usage())
