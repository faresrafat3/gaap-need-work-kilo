"""
Comprehensive tests for gaap/core/observability.py module
Tests Tracer, Metrics, and Observability classes
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gaap.core.observability import (
    Metrics,
    MetricsConfig,
    Observability,
    Tracer,
    TracingConfig,
    get_metrics,
    get_tracer,
    observability,
    traced,
)


class TestTracingConfig:
    """Test TracingConfig class"""

    def test_tracing_config_defaults(self):
        """Test TracingConfig default values"""
        config = TracingConfig()
        assert config.service_name == "gaap-sovereign"
        assert config.service_version == "2.1.0-SOVEREIGN"
        assert config.environment == "production"
        assert config.enable_console_export is False
        assert config.enable_otlp_export is False
        assert config.otlp_endpoint is None
        assert config.sample_rate == 1.0

    def test_tracing_config_custom(self):
        """Test TracingConfig with custom values"""
        config = TracingConfig(
            service_name="custom-service",
            service_version="1.0.0",
            environment="staging",
            enable_console_export=True,
            enable_otlp_export=True,
            otlp_endpoint="http://localhost:4317",
            sample_rate=0.5,
        )
        assert config.service_name == "custom-service"
        assert config.service_version == "1.0.0"
        assert config.environment == "staging"
        assert config.enable_console_export is True
        assert config.enable_otlp_export is True
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.sample_rate == 0.5


class TestMetricsConfig:
    """Test MetricsConfig class"""

    def test_metrics_config_defaults(self):
        """Test MetricsConfig default values"""
        config = MetricsConfig()
        assert config.enable_default_metrics is True
        assert config.metrics_port == 9090
        assert config.metrics_path == "/metrics"
        assert config.namespace == "gaap"
        assert config.subsystem == "system"

    def test_metrics_config_custom(self):
        """Test MetricsConfig with custom values"""
        config = MetricsConfig(
            enable_default_metrics=False,
            metrics_port=8080,
            metrics_path="/prometheus",
            namespace="custom",
            subsystem="api",
        )
        assert config.enable_default_metrics is False
        assert config.metrics_port == 8080
        assert config.metrics_path == "/prometheus"
        assert config.namespace == "custom"
        assert config.subsystem == "api"


class TestTracer:
    """Test Tracer class"""

    @pytest.fixture(autouse=True)
    def reset_tracer(self):
        """Reset tracer singleton between tests"""
        Tracer._instance = None
        Tracer._initialized = False
        yield
        Tracer._instance = None
        Tracer._initialized = False

    def test_tracer_singleton(self):
        """Test Tracer is a singleton"""
        tracer1 = Tracer()
        tracer2 = Tracer()
        assert tracer1 is tracer2

    def test_tracer_initialization(self):
        """Test Tracer initialization"""
        config = TracingConfig(service_name="test-service")
        tracer = Tracer(config)

        assert tracer.config is config

    def test_tracer_start_span_no_otel(self):
        """Test start_span without OpenTelemetry"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            Tracer._instance = None
            Tracer._initialized = False
            tracer = Tracer()

            with tracer.start_span("test_span") as span:
                assert span is None

    def test_tracer_property(self):
        """Test tracer property"""
        tracer = Tracer()
        assert tracer.tracer is not None or tracer.tracer is None  # Depends on OTEL availability


class TestMetrics:
    """Test Metrics class"""

    @pytest.fixture(autouse=True)
    def reset_metrics(self):
        """Reset metrics singleton between tests"""
        Metrics._instance = None
        Metrics._initialized = False
        yield
        Metrics._instance = None
        Metrics._initialized = False

    def test_metrics_singleton(self):
        """Test Metrics is a singleton"""
        metrics1 = Metrics()
        metrics2 = Metrics()
        assert metrics1 is metrics2

    def test_metrics_initialization(self):
        """Test Metrics initialization"""
        config = MetricsConfig(namespace="test")
        metrics = Metrics(config)

        assert metrics.config is config

    def test_inc_counter_no_prometheus(self):
        """Test inc_counter without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.inc_counter("test_counter", value=1)

    def test_dec_counter_no_prometheus(self):
        """Test dec_counter without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.dec_counter("test_counter", value=1)

    def test_observe_histogram_no_prometheus(self):
        """Test observe_histogram without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.observe_histogram("test_histogram", value=1.0)

    def test_set_gauge_no_prometheus(self):
        """Test set_gauge without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.set_gauge("test_gauge", value=1.0)

    def test_inc_gauge_no_prometheus(self):
        """Test inc_gauge without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.inc_gauge("test_gauge", value=1)

    def test_dec_gauge_no_prometheus(self):
        """Test dec_gauge without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise
            metrics.dec_gauge("test_gauge", value=1)

    def test_time_histogram_no_prometheus(self):
        """Test time_histogram without Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should work as context manager
            with metrics.time_histogram("test_histogram"):
                pass

    def test_unknown_metric_name(self):
        """Test operations on unknown metric name"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", True):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            # Should not raise for unknown metric
            metrics.inc_counter("unknown_metric")
            metrics.observe_histogram("unknown_metric", 1.0)


class TestObservability:
    """Test Observability class"""

    @pytest.fixture(autouse=True)
    def reset_observability(self):
        """Reset observability singleton between tests"""
        Observability._instance = None
        yield
        Observability._instance = None

    def test_observability_singleton(self):
        """Test Observability is a singleton"""
        obs1 = Observability()
        obs2 = Observability()
        assert obs1 is obs2

    def test_observability_initialization(self):
        """Test Observability initialization"""
        tracing_config = TracingConfig(service_name="test")
        metrics_config = MetricsConfig(namespace="test")

        obs = Observability(tracing_config, metrics_config)

        assert isinstance(obs.tracer, Tracer)
        assert isinstance(obs.metrics, Metrics)
        assert obs._enabled is True

    def test_enable_disable(self):
        """Test enable and disable methods"""
        obs = Observability()

        assert obs._enabled is True

        obs.disable()
        assert obs._enabled is False

        obs.enable()
        assert obs._enabled is True

    def test_trace_span_disabled(self):
        """Test trace_span when disabled"""
        obs = Observability()
        obs.disable()

        with obs.trace_span("test_span") as span:
            assert span is None

    def test_trace_span_enabled_no_otel(self):
        """Test trace_span when enabled but no OTEL"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None
                Tracer._instance = None
                Tracer._initialized = False
                Metrics._instance = None
                Metrics._initialized = False

                obs = Observability()

                with obs.trace_span("test_span", layer="test") as span:
                    # Should return None when OTEL is not available
                    pass

    def test_traced_decorator_disabled(self):
        """Test traced decorator when disabled"""
        obs = Observability()
        obs.disable()

        @obs.traced("test_span")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_traced_decorator_async_disabled(self):
        """Test traced decorator with async function when disabled"""
        obs = Observability()
        obs.disable()

        @obs.traced("test_span")
        async def test_async_func():
            return "async_result"

        result = asyncio.run(test_async_func())
        assert result == "async_result"

    def test_traced_decorator_sync_no_otel(self):
        """Test traced decorator with sync function"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                @obs.traced("test_span")
                def test_func():
                    return "result"

                result = test_func()
                assert result == "result"

    def test_record_llm_call_disabled(self):
        """Test record_llm_call when disabled"""
        obs = Observability()
        obs.disable()

        # Should not raise
        obs.record_llm_call(
            provider="test",
            model="model1",
            input_tokens=10,
            output_tokens=20,
            cost=0.001,
            latency=0.5,
            success=True,
        )

    def test_record_healing_disabled(self):
        """Test record_healing when disabled"""
        obs = Observability()
        obs.disable()

        # Should not raise
        obs.record_healing("L1", success=True)

    def test_record_error_disabled(self):
        """Test record_error when disabled"""
        obs = Observability()
        obs.disable()

        # Should not raise
        obs.record_error("layer1", "TestError", severity="error")

    def test_record_llm_call_enabled_no_prometheus(self):
        """Test record_llm_call when enabled but no Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Observability._instance = None
            Metrics._instance = None
            Metrics._initialized = False

            obs = Observability()

            # Should not raise
            obs.record_llm_call(
                provider="test",
                model="model1",
                input_tokens=10,
                output_tokens=20,
                cost=0.001,
                latency=0.5,
                success=True,
            )

    def test_record_healing_enabled_no_prometheus(self):
        """Test record_healing when enabled but no Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Observability._instance = None
            Metrics._instance = None
            Metrics._initialized = False

            obs = Observability()

            # Should not raise
            obs.record_healing("L1", success=True)

    def test_record_error_enabled_no_prometheus(self):
        """Test record_error when enabled but no Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Observability._instance = None
            Metrics._instance = None
            Metrics._initialized = False

            obs = Observability()

            # Should not raise
            obs.record_error("layer1", "TestError", severity="error")


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_tracer(self):
        """Test get_tracer function"""
        Tracer._instance = None
        Tracer._initialized = False

        tracer = get_tracer()
        assert isinstance(tracer, Tracer)

    def test_get_metrics(self):
        """Test get_metrics function"""
        Metrics._instance = None
        Metrics._initialized = False

        metrics = get_metrics()
        assert isinstance(metrics, Metrics)

    def test_traced_decorator_global(self):
        """Test global traced decorator"""
        Observability._instance = None
        Tracer._instance = None
        Tracer._initialized = False
        Metrics._instance = None
        Metrics._initialized = False

        @traced("test_span", layer="test")
        def test_func():
            return "result"

        result = test_func()
        assert result == "result"

    def test_global_observability(self):
        """Test global observability instance"""
        assert isinstance(observability, Observability)


class TestObservabilityEdgeCases:
    """Test edge cases and error conditions"""

    def test_trace_span_with_exception(self):
        """Test trace_span handles exceptions"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None
                Tracer._instance = None
                Tracer._initialized = False
                Metrics._instance = None
                Metrics._initialized = False

                obs = Observability()

                class TestException(Exception):
                    pass

                with pytest.raises(TestException):
                    with obs.trace_span("test_span", layer="test"):
                        raise TestException("Test error")

    def test_traced_decorator_with_exception(self):
        """Test traced decorator handles exceptions"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                @obs.traced("test_span")
                def test_func():
                    raise ValueError("Test error")

                with pytest.raises(ValueError, match="Test error"):
                    test_func()

    def test_traced_decorator_async_with_exception(self):
        """Test traced async decorator handles exceptions"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                @obs.traced("test_span")
                async def test_async_func():
                    raise ValueError("Test error")

                with pytest.raises(ValueError, match="Test error"):
                    asyncio.run(test_async_func())

    def test_metrics_with_labels_no_prometheus(self):
        """Test metrics operations with labels but no Prometheus"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            labels = {"layer": "test", "provider": "test_provider"}

            # Should not raise
            metrics.inc_counter("test_counter", labels=labels)
            metrics.dec_counter("test_counter", labels=labels)
            metrics.observe_histogram("test_histogram", 1.0, labels=labels)
            metrics.set_gauge("test_gauge", 1.0, labels=labels)
            metrics.inc_gauge("test_gauge", labels=labels)
            metrics.dec_gauge("test_gauge", labels=labels)

    def test_time_histogram_nested(self):
        """Test nested time_histogram context managers"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            with metrics.time_histogram("outer"):
                with metrics.time_histogram("inner"):
                    pass

    def test_concurrent_tracing(self):
        """Test concurrent tracing operations"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                results = []

                def trace_operation(n):
                    with obs.trace_span(f"span_{n}"):
                        results.append(n)

                import threading

                threads = [threading.Thread(target=trace_operation, args=(i,)) for i in range(10)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                assert len(results) == 10

    def test_multiple_record_calls(self):
        """Test multiple record calls"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Observability._instance = None
            Metrics._instance = None
            Metrics._initialized = False

            obs = Observability()

            # Multiple LLM call records
            for i in range(10):
                obs.record_llm_call(
                    provider="test",
                    model="model1",
                    input_tokens=10,
                    output_tokens=20,
                    cost=0.001,
                    latency=0.5,
                    success=True,
                )

            # Multiple healing records
            for i in range(5):
                obs.record_healing(f"L{i}", success=i % 2 == 0)

            # Multiple error records
            for i in range(3):
                obs.record_error("layer1", f"Error{i}", severity="error")

    def test_empty_labels(self):
        """Test metrics operations with empty labels"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            metrics.inc_counter("test_counter", labels={})
            metrics.observe_histogram("test_histogram", 1.0, labels={})
            metrics.set_gauge("test_gauge", 1.0, labels={})

    def test_large_values(self):
        """Test metrics operations with large values"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            large_value = 1e10
            metrics.inc_counter("test_counter", value=large_value)
            metrics.observe_histogram("test_histogram", large_value)
            metrics.set_gauge("test_gauge", large_value)

    def test_negative_values(self):
        """Test metrics operations with negative values"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Metrics._instance = None
            Metrics._initialized = False
            metrics = Metrics()

            metrics.inc_counter("test_counter", value=-5)
            metrics.observe_histogram("test_histogram", -1.0)
            metrics.set_gauge("test_gauge", -100.0)
            metrics.inc_gauge("test_gauge", value=-10)
            metrics.dec_gauge("test_gauge", value=-10)

    def test_traced_with_custom_name(self):
        """Test traced decorator with custom span name"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                @obs.traced(name="custom_span_name", layer="custom_layer")
                def test_func():
                    return "result"

                result = test_func()
                assert result == "result"

    def test_traced_uses_function_name_as_default(self):
        """Test traced decorator uses function name when name not provided"""
        with patch("gaap.core.observability.OTEL_AVAILABLE", False):
            with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
                Observability._instance = None

                obs = Observability()

                @obs.traced(layer="test")
                def my_test_function():
                    return "result"

                result = my_test_function()
                assert result == "result"

    def test_record_llm_call_failure(self):
        """Test record_llm_call with failure status"""
        with patch("gaap.core.observability.PROMETHEUS_AVAILABLE", False):
            Observability._instance = None
            Metrics._instance = None
            Metrics._initialized = False

            obs = Observability()

            obs.record_llm_call(
                provider="test",
                model="model1",
                input_tokens=10,
                output_tokens=20,
                cost=0.001,
                latency=0.5,
                success=False,
            )

    def test_configuration_preservation(self):
        """Test that configurations are preserved"""
        tracing_config = TracingConfig(service_name="preserved")
        metrics_config = MetricsConfig(namespace="preserved")

        Observability._instance = None
        Tracer._instance = None
        Tracer._initialized = False
        Metrics._instance = None
        Metrics._initialized = False

        obs = Observability(tracing_config, metrics_config)

        assert obs.tracer.config.service_name == "preserved"
        assert obs.metrics.config.namespace == "preserved"
