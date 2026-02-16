"""
Tests for GAAP Fallback System
"""

from datetime import datetime, timedelta

import pytest

from gaap.routing.fallback import FallbackConfig, ProviderHealth


class TestFallbackConfig:
    def test_default_config(self):
        config = FallbackConfig()

        assert config.max_fallbacks == 3
        assert config.retry_delay_base == 1.0
        assert config.retry_delay_max == 30.0
        assert config.exponential_backoff is True
        assert config.jitter is True
        assert config.circuit_breaker_enabled is True
        assert config.circuit_breaker_threshold == 5
        assert config.circuit_breaker_timeout == 60

    def test_custom_config(self):
        config = FallbackConfig(
            max_fallbacks=5,
            retry_delay_base=2.0,
            circuit_breaker_enabled=False,
        )

        assert config.max_fallbacks == 5
        assert config.retry_delay_base == 2.0
        assert config.circuit_breaker_enabled is False


class TestProviderHealth:
    def test_initial_state(self):
        health = ProviderHealth(name="groq")

        assert health.name == "groq"
        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.consecutive_successes == 0
        assert health.last_failure is None
        assert health.last_success is None
        assert health.circuit_open is False

    def test_record_success(self):
        health = ProviderHealth(name="groq")
        health.consecutive_failures = 3
        health.circuit_open = True

        health.record_success()

        assert health.is_healthy is True
        assert health.consecutive_failures == 0
        assert health.consecutive_successes == 1
        assert health.last_success is not None
        assert health.circuit_open is False

    def test_record_failure(self):
        health = ProviderHealth(name="groq")

        health.record_failure()

        assert health.is_healthy is False
        assert health.consecutive_failures == 1
        assert health.consecutive_successes == 0
        assert health.last_failure is not None

    def test_multiple_failures(self):
        health = ProviderHealth(name="groq")

        for _ in range(3):
            health.record_failure()

        assert health.consecutive_failures == 3
        assert health.is_healthy is False

    def test_open_circuit(self):
        health = ProviderHealth(name="groq")

        health.open_circuit()

        assert health.circuit_open is True
        assert health.circuit_open_since is not None

    def test_should_try_circuit_when_open(self):
        health = ProviderHealth(name="groq")
        health.open_circuit()

        # Just opened, should not try
        result = health.should_try_circuit(60)
        assert result is False

    def test_should_try_circuit_after_timeout(self):
        health = ProviderHealth(name="groq")
        health.circuit_open = True
        health.circuit_open_since = datetime.now() - timedelta(seconds=120)

        result = health.should_try_circuit(60)
        assert result is True

    def test_should_try_circuit_when_closed(self):
        health = ProviderHealth(name="groq")

        result = health.should_try_circuit(60)
        assert result is True

    def test_success_resets_circuit(self):
        health = ProviderHealth(name="groq")
        health.open_circuit()

        health.record_success()

        assert health.circuit_open is False
        assert health.circuit_open_since is None


class TestProviderHealthSequence:
    def test_failure_success_sequence(self):
        health = ProviderHealth(name="groq")

        health.record_failure()
        health.record_failure()
        health.record_success()

        assert health.consecutive_failures == 0
        assert health.consecutive_successes == 1
        assert health.is_healthy is True

    def test_success_failure_sequence(self):
        health = ProviderHealth(name="groq")

        health.record_success()
        health.record_success()
        health.record_failure()

        assert health.consecutive_failures == 1
        assert health.consecutive_successes == 0
        assert health.is_healthy is False
