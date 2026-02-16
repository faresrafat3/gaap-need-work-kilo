"""
Tests for GAAP Exceptions
"""

import pytest

from gaap.core.exceptions import (
    GAAPException,
    ConfigurationError,
    InvalidConfigValueError,
    MissingConfigError,
    ProviderError,
    ProviderRateLimitError,
    ProviderNotAvailableError,
    TaskExecutionError,
    SecurityError,
    PromptInjectionError,
    TaskValidationError,
    BudgetExceededError,
    ContextOverflowError,
    HealingFailedError,
    RoutingError,
)


class TestGAAPException:
    def test_basic_exception(self):
        exc = GAAPException("Test error")

        assert exc.message == "Test error"
        assert exc.error_code == "GAAP_000"
        assert exc.recoverable is True
        assert exc.details == {}
        assert exc.suggestions == []

    def test_exception_with_details(self):
        exc = GAAPException(
            "Test error",
            details={"key": "value"},
            suggestions=["Try this", "Or this"],
        )

        assert exc.details["key"] == "value"
        assert len(exc.suggestions) == 2

    def test_exception_with_cause(self):
        cause = ValueError("Original error")
        exc = GAAPException("Wrapped error", cause=cause)

        assert exc.cause == cause
        assert exc.traceback is not None

    def test_to_dict(self):
        exc = GAAPException(
            "Test error",
            details={"count": 5},
            suggestions=["Retry"],
            context={"layer": "L0"},
        )

        result = exc.to_dict()

        assert result["error_code"] == "GAAP_000"
        assert result["message"] == "Test error"
        assert result["details"]["count"] == 5
        assert "Retry" in result["suggestions"]
        assert result["context"]["layer"] == "L0"
        assert "timestamp" in result

    def test_str_representation(self):
        exc = GAAPException("Test error", details={"x": 1})

        s = str(exc)
        assert "GAAP_000" in s
        assert "Test error" in s

    def test_non_recoverable(self):
        exc = GAAPException("Fatal error", recoverable=False)

        assert exc.recoverable is False


class TestConfigurationErrors:
    def test_configuration_error(self):
        exc = ConfigurationError("Config failed")

        assert exc.error_code == "GAAP_CFG_001"
        assert exc.error_category == "configuration"

    def test_invalid_config_value_error(self):
        exc = InvalidConfigValueError("timeout", "abc", "int")

        assert exc.error_code == "GAAP_CFG_002"
        assert "timeout" in exc.message
        assert exc.details["key"] == "timeout"
        assert exc.details["expected_type"] == "int"

    def test_missing_config_error(self):
        exc = MissingConfigError("api_key")

        assert exc.error_code == "GAAP_CFG_003"


class TestProviderErrors:
    def test_provider_error(self):
        exc = ProviderError(
            message="Connection failed for groq",
            details={"provider": "groq"},
        )

        assert exc.error_category == "provider"
        assert "groq" in exc.message

    def test_provider_rate_limit_error(self):
        exc = ProviderRateLimitError("groq", retry_after=60)

        assert exc.recoverable is True
        assert exc.details.get("retry_after_seconds") == 60

    def test_provider_not_available_error(self):
        exc = ProviderNotAvailableError("gemini", "API key not set")

        assert exc.error_category == "provider"
        assert "gemini" in exc.message


class TestTaskErrors:
    def test_task_execution_error(self):
        exc = TaskExecutionError("task-123", "Timeout")

        assert exc.error_category == "task"
        assert "task-123" in exc.message

    def test_security_error(self):
        exc = SecurityError("Security issue detected")

        assert exc.error_category == "security"

    def test_prompt_injection_error(self):
        exc = PromptInjectionError(
            detected_patterns=["ignore instructions"],
            risk_score=0.9,
        )

        assert exc.error_category == "security"
        assert exc.details["risk_score"] == 0.9

    def test_task_validation_error(self):
        exc = TaskValidationError("task-123", ["Invalid field", "Missing required"])

        assert exc.error_category == "task"
        assert "task-123" in exc.message
        assert len(exc.details["errors"]) == 2


class TestResourceErrors:
    def test_budget_exceeded_error(self):
        exc = BudgetExceededError(budget=100.0, required=150.0)

        assert exc.error_category == "routing"
        assert exc.details["budget"] == 100.0
        assert exc.details["required"] == 150.0

    def test_context_overflow_error(self):
        exc = ContextOverflowError(1000000, 2000000)

        assert exc.error_category == "context"


class TestHealingError:
    def test_healing_failed_error(self):
        exc = HealingFailedError(
            level="L3",
            attempts=5,
            last_error="Connection timeout",
        )

        assert exc.error_category == "healing"
        assert exc.details["attempts"] == 5


class TestRoutingError:
    def test_routing_error(self):
        exc = RoutingError("No provider available")

        assert exc.error_category == "routing"


class TestExceptionChaining:
    def test_exception_from_cause(self):
        try:
            raise ValueError("Original")
        except ValueError as e:
            exc = GAAPException("Wrapped", cause=e)

        assert exc.cause is not None
        assert "Original" in str(exc.cause)
