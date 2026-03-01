"""
Comprehensive tests for gaap/api/providers.py module
Tests provider endpoints, configuration, and health checks
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException

pytestmark = pytest.mark.asyncio


class TestProviderConfig:
    """Test ProviderConfig model"""

    def test_default_values(self):
        """Test ProviderConfig default values"""
        from gaap.api.providers import ProviderConfig

        config = ProviderConfig(name="test")
        assert config.name == "test"
        assert config.provider_type == "chat"
        assert config.api_key is None
        assert config.base_url is None
        assert config.priority == 1
        assert config.enabled is True
        assert config.models == []
        assert config.default_model is None
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.metadata == {}

    def test_custom_values(self):
        """Test ProviderConfig with custom values"""
        from gaap.api.providers import ProviderConfig

        config = ProviderConfig(
            name="openai",
            provider_type="completion",
            api_key="secret123",
            base_url="https://api.openai.com",
            priority=5,
            enabled=False,
            models=["gpt-4", "gpt-3.5"],
            default_model="gpt-4",
            max_tokens=8192,
            temperature=0.5,
            metadata={"region": "us-east"},
        )

        assert config.name == "openai"
        assert config.api_key == "secret123"
        assert config.priority == 5
        assert config.enabled is False


class TestProviderStatus:
    """Test ProviderStatus model"""

    def test_provider_status_creation(self):
        """Test creating ProviderStatus"""
        from gaap.api.providers import ProviderStatus

        status = ProviderStatus(
            name="openai",
            type="chat",
            enabled=True,
            priority=1,
            models=["gpt-4"],
            health="healthy",
            stats={"requests": 100},
        )

        assert status.name == "openai"
        assert status.health == "healthy"
        assert status.stats == {"requests": 100}


class TestProviderTestResult:
    """Test ProviderTestResult model"""

    def test_success_result(self):
        """Test successful test result"""
        from gaap.api.providers import ProviderTestResult

        result = ProviderTestResult(
            success=True,
            latency_ms=150.5,
            model_available=True,
        )

        assert result.success is True
        assert result.latency_ms == 150.5
        assert result.model_available is True
        assert result.error is None

    def test_failure_result(self):
        """Test failed test result"""
        from gaap.api.providers import ProviderTestResult

        result = ProviderTestResult(
            success=False,
            error="Connection timeout",
        )

        assert result.success is False
        assert result.error == "Connection timeout"
        assert result.latency_ms is None


class TestGetRouter:
    """Test get_router function"""

    @pytest.fixture(autouse=True)
    def reset_router(self):
        """Reset router singleton between tests"""
        from gaap.api.providers import _router_instance

        global _router_instance
        _router_instance = None
        yield
        _router_instance = None

    def test_get_router_creates_instance(self):
        """Test get_router creates new instance"""
        from gaap.api.providers import get_router, _router_instance

        with patch("gaap.api.providers.SmartRouter") as mock_router_class:
            mock_router = Mock()
            mock_router_class.return_value = mock_router

            router = get_router()

            assert router is mock_router
            mock_router_class.assert_called_once()

    def test_get_router_returns_existing(self):
        """Test get_router returns existing instance"""
        from gaap.api.providers import get_router, set_router

        mock_router = Mock()
        set_router(mock_router)

        router = get_router()

        assert router is mock_router


class TestSetRouter:
    """Test set_router function"""

    def test_set_router(self):
        """Test setting router instance"""
        from gaap.api.providers import set_router, get_router

        mock_router = Mock()
        set_router(mock_router)

        assert get_router() is mock_router


class TestCheckProviderHealth:
    """Test _check_provider_health function"""

    def test_healthy_provider(self):
        """Test healthy provider check"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_provider = Mock()
        mock_provider.get_stats.return_value = {
            "total_requests": 100,
            "success_rate": 0.95,
        }
        mock_router.get_provider.return_value = mock_provider

        health = _check_provider_health("test_provider", mock_router)

        assert health == "healthy"

    def test_degraded_provider(self):
        """Test degraded provider check"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_provider = Mock()
        mock_provider.get_stats.return_value = {
            "total_requests": 100,
            "success_rate": 0.7,  # Between 0.5 and 0.8
        }
        mock_router.get_provider.return_value = mock_provider

        health = _check_provider_health("test_provider", mock_router)

        assert health == "degraded"

    def test_unhealthy_provider(self):
        """Test unhealthy provider check"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_provider = Mock()
        mock_provider.get_stats.return_value = {
            "total_requests": 100,
            "success_rate": 0.3,  # Below 0.5
        }
        mock_router.get_provider.return_value = mock_provider

        health = _check_provider_health("test_provider", mock_router)

        assert health == "unhealthy"

    def test_no_requests_healthy(self):
        """Test provider with no requests is healthy"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_provider = Mock()
        mock_provider.get_stats.return_value = {
            "total_requests": 0,
            "success_rate": 1.0,
        }
        mock_router.get_provider.return_value = mock_provider

        health = _check_provider_health("test_provider", mock_router)

        assert health == "healthy"

    def test_provider_not_found(self):
        """Test provider not found returns unhealthy"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_router.get_provider.return_value = None

        health = _check_provider_health("nonexistent", mock_router)

        assert health == "unhealthy"

    def test_check_health_exception(self):
        """Test exception during health check returns unhealthy"""
        from gaap.api.providers import _check_provider_health

        mock_router = Mock()
        mock_router.get_provider.side_effect = Exception("Error")

        health = _check_provider_health("test_provider", mock_router)

        assert health == "unhealthy"


class TestListProviders:
    """Test list_providers endpoint"""

    async def test_list_providers_success(self):
        """Test listing providers successfully"""
        from gaap.api.providers import list_providers

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["gpt-4", "gpt-3.5"]
            mock_provider.get_stats.return_value = {"requests": 100}

            mock_router.get_all_providers.return_value = [mock_provider]

            mock_get_router.return_value = mock_router

            result = await list_providers()

            assert len(result) == 1
            assert result[0].name == "openai"
            assert result[0].models == ["gpt-4", "gpt-3.5"]

    async def test_list_providers_empty(self):
        """Test listing providers when none exist"""
        from gaap.api.providers import list_providers

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_all_providers.return_value = []

            mock_get_router.return_value = mock_router

            result = await list_providers()

            assert result == []

    async def test_list_providers_multiple(self):
        """Test listing multiple providers"""
        from gaap.api.providers import list_providers

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()

            providers = []
            for name in ["openai", "anthropic", "kimi"]:
                mock_provider = Mock()
                mock_provider.name = name
                mock_provider.provider_type.name = "chat"
                mock_provider.get_available_models.return_value = ["model1"]
                mock_provider.get_stats.return_value = {}
                providers.append(mock_provider)

            mock_router.get_all_providers.return_value = providers
            mock_get_router.return_value = mock_router

            result = await list_providers()

            assert len(result) == 3
            names = [p.name for p in result]
            assert "openai" in names
            assert "anthropic" in names
            assert "kimi" in names


class TestAddProvider:
    """Test add_provider endpoint"""

    async def test_add_provider_success(self):
        """Test adding provider successfully"""
        from gaap.api.providers import add_provider, ProviderConfig

        config = ProviderConfig(name="openai", api_key="secret123")

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.ProviderFactory") as mock_factory,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_get_router.return_value = mock_router

            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["gpt-4"]
            mock_provider.get_stats.return_value = {}

            mock_factory.create.return_value = mock_provider

            result = await add_provider(config)

            assert result.name == "openai"
            mock_router.register_provider.assert_called_once_with(mock_provider)

    async def test_add_provider_failure(self):
        """Test adding provider with failure"""
        from gaap.api.providers import add_provider, ProviderConfig

        config = ProviderConfig(name="invalid")

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.ProviderFactory") as mock_factory,
        ):
            mock_router = Mock()
            mock_get_router.return_value = mock_router
            mock_factory.create.side_effect = Exception("Creation failed")

            with pytest.raises(HTTPException) as exc_info:
                await add_provider(config)

            assert exc_info.value.status_code == 400


class TestGetProvider:
    """Test get_provider endpoint"""

    async def test_get_existing_provider(self):
        """Test getting existing provider"""
        from gaap.api.providers import get_provider as api_get_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["gpt-4"]
            mock_provider.get_stats.return_value = {"requests": 100}

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await api_get_provider("openai")

            assert result.name == "openai"
            assert result.health in ["healthy", "degraded", "unhealthy"]

    async def test_get_nonexistent_provider(self):
        """Test getting non-existent provider"""
        from gaap.api.providers import get_provider as api_get_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await api_get_provider("nonexistent")

            assert exc_info.value.status_code == 404


class TestUpdateProvider:
    """Test update_provider endpoint"""

    async def test_update_existing_provider(self):
        """Test updating existing provider"""
        from gaap.api.providers import update_provider, ProviderConfig

        config = ProviderConfig(name="openai", default_model="gpt-4-turbo")

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["gpt-4", "gpt-4-turbo"]
            mock_provider.get_stats.return_value = {"requests": 100}

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await update_provider("openai", config)

            assert result.name == "openai"
            assert mock_provider.default_model == "gpt-4-turbo"

    async def test_update_nonexistent_provider(self):
        """Test updating non-existent provider"""
        from gaap.api.providers import update_provider, ProviderConfig

        config = ProviderConfig(name="openai")

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await update_provider("nonexistent", config)

            assert exc_info.value.status_code == 404


class TestRemoveProvider:
    """Test remove_provider endpoint"""

    async def test_remove_existing_provider(self):
        """Test removing existing provider"""
        from gaap.api.providers import remove_provider

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_provider = Mock()

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await remove_provider("openai")

            assert result["success"] is True
            mock_router.unregister_provider.assert_called_once_with("openai")

    async def test_remove_nonexistent_provider(self):
        """Test removing non-existent provider"""
        from gaap.api.providers import remove_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await remove_provider("nonexistent")

            assert exc_info.value.status_code == 404


class TestTestProvider:
    """Test test_provider endpoint"""

    async def test_test_provider_success(self):
        """Test testing provider successfully"""
        from gaap.api.providers import test_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.get_available_models.return_value = ["gpt-4"]

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await test_provider("openai")

            assert result.success is True
            assert result.model_available is True
            assert result.latency_ms is not None

    async def test_test_provider_no_models(self):
        """Test provider with no available models"""
        from gaap.api.providers import test_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.get_available_models.return_value = []

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await test_provider("openai")

            assert result.success is True
            assert result.model_available is False

    async def test_test_provider_nonexistent(self):
        """Test testing non-existent provider"""
        from gaap.api.providers import test_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await test_provider("nonexistent")

            assert exc_info.value.status_code == 404

    async def test_test_provider_exception(self):
        """Test provider test with exception"""
        from gaap.api.providers import test_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.get_available_models.side_effect = Exception("Connection failed")

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await test_provider("openai")

            assert result.success is False
            assert result.error is not None


class TestEnableProvider:
    """Test enable_provider endpoint"""

    async def test_enable_existing_provider(self):
        """Test enabling existing provider"""
        from gaap.api.providers import enable_provider

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_provider = Mock()

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await enable_provider("openai")

            assert result["success"] is True
            assert "enabled" in result["message"].lower()

    async def test_enable_nonexistent_provider(self):
        """Test enabling non-existent provider"""
        from gaap.api.providers import enable_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await enable_provider("nonexistent")

            assert exc_info.value.status_code == 404


class TestDisableProvider:
    """Test disable_provider endpoint"""

    async def test_disable_existing_provider(self):
        """Test disabling existing provider"""
        from gaap.api.providers import disable_provider

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_provider = Mock()

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await disable_provider("openai")

            assert result["success"] is True
            assert "disabled" in result["message"].lower()

    async def test_disable_nonexistent_provider(self):
        """Test disabling non-existent provider"""
        from gaap.api.providers import disable_provider

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_router.get_provider.return_value = None
            mock_get_router.return_value = mock_router

            with pytest.raises(HTTPException) as exc_info:
                await disable_provider("nonexistent")

            assert exc_info.value.status_code == 404


class TestRegisterRoutes:
    """Test register_routes function"""

    def test_register_routes(self):
        """Test route registration"""
        from gaap.api.providers import register_routes

        mock_app = Mock()
        register_routes(mock_app)

        mock_app.include_router.assert_called_once()


class TestProviderEdgeCases:
    """Test edge cases and error conditions"""

    async def test_add_provider_no_api_key(self):
        """Test adding provider without API key"""
        from gaap.api.providers import add_provider, ProviderConfig

        config = ProviderConfig(name="openai")  # No api_key

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.ProviderFactory") as mock_factory,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_get_router.return_value = mock_router

            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = []
            mock_provider.get_stats.return_value = {}

            mock_factory.create.return_value = mock_provider

            result = await add_provider(config)

            assert result.name == "openai"
            # API key should be None
            mock_factory.create.assert_called_once()
            call_kwargs = mock_factory.create.call_args[1]
            assert call_kwargs.get("api_key") is None

    async def test_update_provider_without_default_model(self):
        """Test updating provider without changing default model"""
        from gaap.api.providers import update_provider, ProviderConfig

        config = ProviderConfig(name="openai")  # No default_model

        with (
            patch("gaap.api.providers.get_router") as mock_get_router,
            patch("gaap.api.providers.observability") as mock_observability,
        ):
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.name = "openai"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["gpt-4"]
            mock_provider.get_stats.return_value = {}
            mock_provider.default_model = "original_model"

            mock_router.get_provider.return_value = mock_provider
            mock_get_router.return_value = mock_router

            result = await update_provider("openai", config)

            # Default model should not change
            assert mock_provider.default_model == "original_model"

    async def test_list_providers_with_unhealthy(self):
        """Test listing providers with unhealthy provider"""
        from gaap.api.providers import list_providers

        with patch("gaap.api.providers.get_router") as mock_get_router:
            mock_router = Mock()
            mock_provider = Mock()
            mock_provider.name = "problematic"
            mock_provider.provider_type.name = "chat"
            mock_provider.get_available_models.return_value = ["model1"]
            mock_provider.get_stats.side_effect = Exception("Stats error")

            mock_router.get_all_providers.return_value = [mock_provider]
            mock_get_router.return_value = mock_router

            # Should handle exception gracefully
            result = await list_providers()

            assert len(result) == 1
            assert result[0].health == "unhealthy"
