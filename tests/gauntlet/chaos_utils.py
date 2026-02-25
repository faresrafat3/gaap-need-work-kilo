"""
Chaos Testing Utilities
======================

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md

Chaos engineering utilities for testing system resilience:
- Network failure injection
- Memory corruption simulation
- Random delays
- Resource exhaustion
"""

from __future__ import annotations

import asyncio
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, TypeVar
from unittest.mock import AsyncMock, patch

T = TypeVar("T")


class ChaosType(Enum):
    NETWORK_FAILURE = auto()
    TIMEOUT = auto()
    RATE_LIMIT = auto()
    MEMORY_PRESSURE = auto()
    RANDOM_DELAY = auto()
    PARTIAL_RESPONSE = auto()
    CORRUPTED_DATA = auto()


@dataclass
class ChaosConfig:
    """Configuration for chaos testing."""

    enabled: bool = True
    failure_rate: float = 0.3
    min_delay_ms: float = 100
    max_delay_ms: float = 2000
    timeout_rate: float = 0.1
    corruption_rate: float = 0.05
    seed: int | None = None

    @classmethod
    def mild(cls) -> ChaosConfig:
        return cls(failure_rate=0.1, timeout_rate=0.05, corruption_rate=0.01)

    @classmethod
    def aggressive(cls) -> ChaosConfig:
        return cls(failure_rate=0.5, timeout_rate=0.2, corruption_rate=0.1)

    @classmethod
    def chaos_monkey(cls) -> ChaosConfig:
        return cls(failure_rate=0.7, timeout_rate=0.3, corruption_rate=0.2)


class ChaosInjector:
    """
    Chaos engineering utility for testing resilience.

    Features:
    - Network failure injection
    - Random delays
    - Timeout simulation
    - Data corruption

    Usage:
        chaos = ChaosInjector(ChaosConfig.aggressive())

        with chaos.inject_failures():
            result = await some_operation()
    """

    def __init__(self, config: ChaosConfig | None = None) -> None:
        self.config = config or ChaosConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)

        self._injected_failures = 0
        self._injected_delays = 0
        self._injected_timeouts = 0

    def _should_fail(self) -> bool:
        return self.config.enabled and random.random() < self.config.failure_rate

    def _should_timeout(self) -> bool:
        return self.config.enabled and random.random() < self.config.timeout_rate

    def _get_delay(self) -> float:
        return random.uniform(self.config.min_delay_ms, self.config.max_delay_ms) / 1000

    @asynccontextmanager
    async def inject_failures(self):
        """Context manager that injects failures into async operations."""
        try:
            if self._should_fail():
                self._injected_failures += 1
                raise ConnectionError("Chaos: Simulated network failure")

            if self._should_timeout():
                self._injected_timeouts += 1
                await asyncio.sleep(self._get_delay() * 2)
                raise TimeoutError("Chaos: Simulated timeout")

            delay = self._get_delay() * 0.1
            if delay > 0 and random.random() < 0.3:
                self._injected_delays += 1
                await asyncio.sleep(delay)

            yield

        except Exception:
            raise

    def wrap_provider(self, provider: Any) -> Any:
        """Wrap a provider with chaos injection."""
        original_complete = getattr(provider, "complete", None) or getattr(
            provider, "chat_completion", None
        )

        if original_complete is None:
            return provider

        async def chaotic_complete(*args, **kwargs):
            async with self.inject_failures():
                return await original_complete(*args, **kwargs)

        if hasattr(provider, "complete"):
            provider.complete = chaotic_complete
        if hasattr(provider, "chat_completion"):
            provider.chat_completion = chaotic_complete

        return provider

    def get_stats(self) -> dict[str, int]:
        """Get chaos injection statistics."""
        return {
            "failures_injected": self._injected_failures,
            "delays_injected": self._injected_delays,
            "timeouts_injected": self._injected_timeouts,
        }

    def reset(self) -> None:
        """Reset statistics."""
        self._injected_failures = 0
        self._injected_delays = 0
        self._injected_timeouts = 0


class FailingProvider:
    """Provider that fails according to chaos configuration."""

    def __init__(
        self,
        failure_rate: float = 0.5,
        timeout_rate: float = 0.1,
        name: str = "chaos-provider",
    ) -> None:
        self.name = name
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.call_count = 0
        self.failure_count = 0
        self.timeout_count = 0

    async def complete(
        self,
        messages: list[Any],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        self.call_count += 1

        if random.random() < self.timeout_rate:
            self.timeout_count += 1
            await asyncio.sleep(10)
            raise TimeoutError("Simulated timeout")

        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ConnectionError("Simulated network failure")

        return "Success response"

    async def chat_completion(
        self,
        messages: list[Any],
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        from gaap.core.types import (
            ChatCompletionChoice,
            ChatCompletionResponse,
            Usage,
            Message,
            MessageRole,
        )

        self.call_count += 1

        if random.random() < self.timeout_rate:
            self.timeout_count += 1
            raise TimeoutError("Simulated timeout")

        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ConnectionError("Simulated network failure")

        return ChatCompletionResponse(
            id="chaos-response",
            model=model or "chaos-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT, content="Success from chaos provider"
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "call_count": self.call_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "failure_rate_actual": self.failure_count / max(1, self.call_count),
        }


@dataclass
class ChaosTestResult:
    """Result of a chaos test run."""

    success: bool
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    recovered_calls: int = 0
    chaos_stats: dict[str, int] = field(default_factory=dict)
    healing_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.successful_calls / max(1, self.total_calls)

    @property
    def recovery_rate(self) -> float:
        return self.recovered_calls / max(1, self.failed_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "recovered_calls": self.recovered_calls,
            "success_rate": f"{self.success_rate:.1%}",
            "recovery_rate": f"{self.recovery_rate:.1%}",
            "chaos_stats": self.chaos_stats,
            "healing_stats": self.healing_stats,
        }


async def run_chaos_test(
    func: Callable[..., Any],
    iterations: int = 10,
    chaos_config: ChaosConfig | None = None,
    healing_system: Any = None,
) -> ChaosTestResult:
    """
    Run a function under chaos conditions.

    Args:
        func: Async function to test
        iterations: Number of iterations
        chaos_config: Chaos configuration
        healing_system: Optional healing system for recovery

    Returns:
        ChaosTestResult with statistics
    """
    chaos = ChaosInjector(chaos_config)
    result = ChaosTestResult(success=True)

    for _ in range(iterations):
        result.total_calls += 1

        try:
            async with chaos.inject_failures():
                await func()
                result.successful_calls += 1
        except Exception as e:
            result.failed_calls += 1

            if healing_system:
                try:
                    from gaap.core.types import Task, TaskType

                    task = Task(
                        id=f"chaos-task-{result.total_calls}",
                        description="Chaos test task",
                        type=TaskType.CODE_GENERATION,
                    )
                    recovery = await healing_system.heal(e, task, func)
                    if recovery.success:
                        result.recovered_calls += 1
                except Exception:
                    pass

    result.chaos_stats = chaos.get_stats()

    if healing_system and hasattr(healing_system, "get_stats"):
        result.healing_stats = healing_system.get_stats()

    result.success = result.successful_calls > 0 or result.recovered_calls > 0

    return result


def create_chaos_provider(failure_rate: float = 0.5) -> FailingProvider:
    """Create a chaos provider for testing."""
    return FailingProvider(failure_rate=failure_rate)
