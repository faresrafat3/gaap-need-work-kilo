"""
Comprehensive unit tests for GAAP FastAPI endpoints.

Tests all API modules:
- Health endpoints
- Config API
- Providers API
- Sessions API
- Budget API
- Memory API
- Healing API
- System API
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gaap.api.main import create_app
from gaap.core.config import (
    GAAPConfig,
    BudgetConfig,
    SystemConfig,
    QualityPanelConfig,
    CriticConfig,
)
from gaap.core.events import EventEmitter, EventType


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    config = GAAPConfig()
    config.system = SystemConfig(
        name="test-gaap",
        environment="development",
        version="1.0.0",
        log_level="INFO",
    )
    config.budget = BudgetConfig(
        monthly_limit=1000.0,
        daily_limit=100.0,
        per_task_limit=10.0,
        auto_throttle_at=0.9,
        hard_stop_at=1.0,
        alert_thresholds=[0.5, 0.8, 0.95],
    )
    config.quality_panel = QualityPanelConfig(
        critics=[
            CriticConfig(name="logic", weight=0.35),
            CriticConfig(name="security", weight=0.25),
            CriticConfig(name="performance", weight=0.20),
            CriticConfig(name="style", weight=0.10),
            CriticConfig(name="compliance", weight=0.05),
            CriticConfig(name="ethics", weight=0.05),
        ]
    )
    return config


@pytest.fixture
def mock_config_manager(mock_config):
    """Create a mock configuration manager."""
    manager = MagicMock()
    manager.config = mock_config
    manager._config = mock_config
    manager._config_path = "/tmp/test_config.yaml"

    def mock_dict_to_config(d):
        return mock_config

    def mock_validate_config(c):
        pass

    manager._dict_to_config = mock_dict_to_config
    manager._validate_config = mock_validate_config
    manager.reload = MagicMock()
    return manager


@pytest.fixture
def mock_store():
    """Create a mock SQLite store."""
    store = MagicMock()
    store._data = {}

    def mock_insert(table, data, item_id=None):
        if table not in store._data:
            store._data[table] = {}
        key = item_id or f"item-{len(store._data[table])}"
        store._data[table][key] = {
            "id": key,
            "data": data,
            "created_at": datetime.now().isoformat(),
        }
        return key

    def mock_get(table, item_id):
        return store._data.get(table, {}).get(item_id)

    def mock_update(table, item_id, data):
        if table in store._data and item_id in store._data[table]:
            store._data[table][item_id]["data"].update(data)

    def mock_delete(table, item_id):
        if table in store._data and item_id in store._data[table]:
            del store._data[table][item_id]

    def mock_query(table, where=None, limit=50, offset=0):
        items = list(store._data.get(table, {}).values())
        if where:
            items = [
                item
                for item in items
                if all(item.get("data", {}).get(k) == v for k, v in where.items())
            ]
        return items[offset : offset + limit]

    def mock_count(table, where=None):
        items = list(store._data.get(table, {}).values())
        if where:
            items = [
                item
                for item in items
                if all(item.get("data", {}).get(k) == v for k, v in where.items())
            ]
        return len(items)

    store.insert = mock_insert
    store.get = mock_get
    store.update = mock_update
    store.delete = mock_delete
    store.query = mock_query
    store.count = mock_count
    store.get_stats = MagicMock(return_value={"tables": 0, "size": 0})
    return store


@pytest.fixture
def mock_router():
    """Create a mock SmartRouter."""
    router = MagicMock()
    router._providers = {}

    def mock_get_provider(name):
        return router._providers.get(name)

    def mock_get_all_providers():
        return list(router._providers.values())

    def mock_register_provider(provider):
        router._providers[provider.name] = provider

    def mock_unregister_provider(name):
        if name in router._providers:
            del router._providers[name]

    router.get_provider = mock_get_provider
    router.get_all_providers = mock_get_all_providers
    router.register_provider = mock_register_provider
    router.unregister_provider = mock_unregister_provider
    return router


@pytest.fixture
def mock_provider():
    """Create a mock provider."""
    provider = MagicMock()
    provider.name = "test-provider"
    provider.provider_type = MagicMock()
    provider.provider_type.name = "chat"
    provider.default_model = "test-model"

    provider.get_available_models = MagicMock(return_value=["model-1", "model-2"])
    provider.get_stats = MagicMock(
        return_value={
            "total_requests": 10,
            "successful_requests": 9,
            "success_rate": 0.9,
        }
    )
    return provider


@pytest.fixture
def mock_memory():
    """Create a mock hierarchical memory."""
    memory = MagicMock()
    memory.working = MagicMock()
    memory.working.get_size = MagicMock(return_value=5)
    memory.working.max_size = 100
    memory.working.clear = MagicMock()

    memory.episodic = MagicMock()
    memory.episodic._episodes = []
    memory.episodic._task_index = {}
    memory.episodic.get_episodes = MagicMock(return_value=[])
    memory.episodic.get_recent_lessons = MagicMock(return_value=[])

    memory.semantic = MagicMock()
    memory.semantic._rules = {}
    memory.semantic._pattern_index = {}
    memory.semantic.find_rules = MagicMock(return_value=[])

    memory.procedural = MagicMock()
    memory.procedural._procedures = {}

    memory.get_stats = MagicMock(
        return_value={
            "working": {"size": 5, "max_size": 100},
            "episodic": {"total_episodes": 0},
            "semantic": {"total_rules": 0},
            "procedural": {"total_procedures": 0},
        }
    )
    memory.save = MagicMock(
        return_value={"working": True, "episodic": True, "semantic": True, "procedural": True}
    )
    memory.load = MagicMock(
        return_value={"working": True, "episodic": True, "semantic": True, "procedural": True}
    )
    return memory


@pytest.fixture
def mock_healing():
    """Create a mock healing system."""
    healing = MagicMock()

    healing_config = MagicMock()
    healing_config.max_healing_level = 3
    healing_config.max_retries_per_level = 2
    healing_config.base_delay_seconds = 1.0
    healing_config.max_delay_seconds = 60.0
    healing_config.exponential_backoff = True
    healing_config.jitter = True
    healing_config.enable_learning = True
    healing_config.enable_observability = True
    healing_config.to_dict = MagicMock(
        return_value={
            "max_healing_level": 3,
            "max_retries_per_level": 2,
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 60.0,
            "exponential_backoff": True,
            "jitter": True,
            "enable_learning": True,
            "enable_observability": True,
        }
    )

    healing._config = healing_config
    healing._records = []
    healing._error_history = {}
    healing._pattern_history = {}
    healing._total_healing_attempts = 5
    healing._successful_recoveries = 4
    healing._escalations = 1
    healing._patterns_detected = 2

    healing.get_stats = MagicMock(
        return_value={
            "total_attempts": 5,
            "successful_recoveries": 4,
            "escalations": 1,
            "recovery_rate": 0.8,
            "errors_by_category": {"rate_limit": 2, "timeout": 1},
            "healing_by_level": {
                "L1": {"attempts": 3, "successes": 3},
                "L2": {"attempts": 2, "successes": 1},
            },
        }
    )
    return healing


@pytest.fixture
def mock_budget_tracker():
    """Create a mock budget tracker."""
    tracker = MagicMock()
    tracker.get_monthly_spend = MagicMock(return_value=250.0)
    tracker.get_daily_spend = MagicMock(return_value=25.0)
    tracker.get_usage_history = MagicMock(return_value=[])
    tracker.get_usage_summary = MagicMock(return_value=[])
    tracker.get_alerts = MagicMock(return_value=[])
    tracker.acknowledge_alert = MagicMock()
    return tracker


@pytest.fixture
def mock_event_emitter():
    """Create a fresh EventEmitter for testing."""
    emitter = EventEmitter()
    emitter.clear_history()
    return emitter


@pytest.fixture
def client(
    mock_config,
    mock_config_manager,
    mock_store,
    mock_router,
    mock_memory,
    mock_healing,
    mock_budget_tracker,
    mock_event_emitter,
):
    """Create a test client with all mocks."""
    app = create_app()

    with (
        patch("gaap.api.config.get_config", return_value=mock_config),
        patch("gaap.api.config.get_config_manager", return_value=mock_config_manager),
        patch("gaap.api.sessions.get_store", return_value=mock_store),
        patch("gaap.api.sessions._store", mock_store),
        patch("gaap.api.providers.get_router", return_value=mock_router),
        patch("gaap.api.providers._router_instance", mock_router),
        patch("gaap.api.memory.get_memory_instance", return_value=mock_memory),
        patch("gaap.api.memory._memory_instance", mock_memory),
        patch("gaap.api.healing.get_healing_instance", return_value=mock_healing),
        patch("gaap.api.healing._healing_instance", mock_healing),
        patch("gaap.api.budget.get_budget_tracker", return_value=mock_budget_tracker),
        patch("gaap.api.budget._budget_tracker", mock_budget_tracker),
        patch("gaap.api.budget.get_config", return_value=mock_config),
        patch("gaap.api.budget.get_config_manager", return_value=mock_config_manager),
        patch("gaap.api.system.get_config", return_value=mock_config),
        patch("gaap.core.events.EventEmitter.get_instance", return_value=mock_event_emitter),
    ):
        yield TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data

    def test_health_endpoint(self, client):
        """Test health endpoint returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "connections" in data


class TestConfigAPI:
    """Tests for configuration API endpoints."""

    def test_get_config(self, client, mock_config_manager):
        """Test getting full configuration."""
        response = client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["config"] is not None

    def test_update_config(self, client, mock_config_manager):
        """Test updating configuration."""
        response = client.put(
            "/api/config",
            json={"config": {"system": {"name": "updated-name"}}, "validate": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_module_config(self, client, mock_config_manager):
        """Test getting module configuration."""
        response = client.get("/api/config/budget")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_module_config_not_found(self, client, mock_config_manager):
        """Test getting non-existent module configuration."""
        response = client.get("/api/config/nonexistent_module")
        assert response.status_code == 404

    def test_update_module_config(self, client, mock_config_manager):
        """Test updating module configuration."""
        response = client.put(
            "/api/config/budget",
            json={"config": {"monthly_limit": 2000.0}, "validate": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_validate_config(self, client, mock_config_manager):
        """Test validating configuration."""
        response = client.post(
            "/api/config/validate",
            json={"config": {"system": {"name": "test"}}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "valid" in data
        assert "errors" in data
        assert "warnings" in data

    def test_reload_config(self, client, mock_config_manager):
        """Test reloading configuration."""
        response = client.post("/api/config/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_presets(self, client):
        """Test getting configuration presets."""
        response = client.get("/api/config/presets/list")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for preset in data:
            assert "name" in preset
            assert "description" in preset
            assert "modules" in preset

    def test_get_schema(self, client):
        """Test getting configuration schema."""
        response = client.get("/api/config/schema/all")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "budget" in data
        assert "security" in data


class TestProvidersAPI:
    """Tests for providers API endpoints."""

    def test_list_providers(self, client, mock_router, mock_provider):
        """Test listing all providers."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.get("/api/providers")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_providers_empty(self, client, mock_router):
        """Test listing providers when none exist."""
        mock_router._providers = {}

        response = client.get("/api/providers")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_add_provider(self, client, mock_router):
        """Test adding a new provider."""
        with patch("gaap.api.providers.ProviderFactory") as mock_factory:
            mock_provider = MagicMock()
            mock_provider.name = "new-provider"
            mock_provider.get_available_models = MagicMock(return_value=["model-1"])
            mock_factory.create = MagicMock(return_value=mock_provider)

            response = client.post(
                "/api/providers",
                json={
                    "name": "new-provider",
                    "provider_type": "chat",
                    "api_key": "test-key",
                    "models": ["model-1"],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "new-provider"

    def test_get_provider(self, client, mock_router, mock_provider):
        """Test getting provider details."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.get("/api/providers/test-provider")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-provider"

    def test_get_provider_not_found(self, client, mock_router):
        """Test getting non-existent provider."""
        mock_router._providers = {}

        response = client.get("/api/providers/nonexistent")
        assert response.status_code == 404

    def test_update_provider(self, client, mock_router, mock_provider):
        """Test updating provider configuration."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.put(
            "/api/providers/test-provider",
            json={
                "name": "test-provider",
                "provider_type": "chat",
                "enabled": True,
                "priority": 2,
                "models": ["model-1", "model-2"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-provider"

    def test_delete_provider(self, client, mock_router, mock_provider):
        """Test deleting a provider."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.delete("/api/providers/test-provider")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_provider_not_found(self, client, mock_router):
        """Test deleting non-existent provider."""
        mock_router._providers = {}

        response = client.delete("/api/providers/nonexistent")
        assert response.status_code == 404

    def test_test_provider(self, client, mock_router, mock_provider):
        """Test testing provider connection."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.post("/api/providers/test-provider/test")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "latency_ms" in data

    def test_test_provider_not_found(self, client, mock_router):
        """Test testing non-existent provider."""
        mock_router._providers = {}

        response = client.post("/api/providers/nonexistent/test")
        assert response.status_code == 404

    def test_enable_provider(self, client, mock_router, mock_provider):
        """Test enabling a provider."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.post("/api/providers/test-provider/enable")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_enable_provider_not_found(self, client, mock_router):
        """Test enabling non-existent provider."""
        mock_router._providers = {}

        response = client.post("/api/providers/nonexistent/enable")
        assert response.status_code == 404

    def test_disable_provider(self, client, mock_router, mock_provider):
        """Test disabling a provider."""
        mock_router._providers["test-provider"] = mock_provider

        response = client.post("/api/providers/test-provider/disable")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestSessionsAPI:
    """Tests for sessions API endpoints."""

    def test_list_sessions(self, client, mock_store):
        """Test listing sessions."""
        mock_store._data["sessions"] = {
            "session-1": {
                "id": "session-1",
                "data": {
                    "name": "Test Session",
                    "status": "pending",
                    "priority": "normal",
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data

    def test_list_sessions_with_filter(self, client, mock_store):
        """Test listing sessions with status filter."""
        mock_store._data["sessions"] = {
            "session-1": {
                "id": "session-1",
                "data": {
                    "name": "Running Session",
                    "status": "running",
                    "priority": "high",
                },
                "created_at": datetime.now().isoformat(),
            },
            "session-2": {
                "id": "session-2",
                "data": {
                    "name": "Pending Session",
                    "status": "pending",
                    "priority": "normal",
                },
                "created_at": datetime.now().isoformat(),
            },
        }

        response = client.get("/api/sessions?status=running")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 0

    def test_create_session(self, client, mock_store):
        """Test creating a new session."""
        response = client.post(
            "/api/sessions",
            json={
                "name": "New Session",
                "description": "Test session",
                "priority": "high",
                "tags": ["test"],
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Session"
        assert data["status"] == "pending"

    def test_get_session(self, client, mock_store):
        """Test getting session details."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "running",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                    "progress": 0.5,
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.get("/api/sessions/session-123")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "session-123"

    def test_get_session_not_found(self, client, mock_store):
        """Test getting non-existent session."""
        mock_store._data["sessions"] = {}

        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404

    def test_update_session(self, client, mock_store):
        """Test updating session."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Old Name",
                    "status": "pending",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.put(
            "/api/sessions/session-123",
            json={"name": "Updated Name", "priority": "high"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"

    def test_delete_session(self, client, mock_store):
        """Test deleting a session."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {"name": "Test Session"},
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.delete("/api/sessions/session-123")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_pause_session(self, client, mock_store):
        """Test pausing a running session."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "running",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.post("/api/sessions/session-123/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "paused"

    def test_pause_session_not_running(self, client, mock_store):
        """Test pausing a session that's not running."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "pending",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.post("/api/sessions/session-123/pause")
        assert response.status_code == 400

    def test_resume_session(self, client, mock_store):
        """Test resuming a paused session."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "paused",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                    "started_at": datetime.now().isoformat(),
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.post("/api/sessions/session-123/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"

    def test_resume_session_not_paused(self, client, mock_store):
        """Test resuming a session that's not paused."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "running",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.post("/api/sessions/session-123/resume")
        assert response.status_code == 400

    def test_export_session(self, client, mock_store):
        """Test exporting session data."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Test Session",
                    "status": "completed",
                    "priority": "normal",
                    "tags": [],
                    "config": {},
                    "metadata": {},
                    "progress": 1.0,
                    "tasks_total": 5,
                    "tasks_completed": 5,
                    "tasks_failed": 0,
                    "cost_usd": 1.5,
                    "tokens_used": 5000,
                },
                "created_at": datetime.now().isoformat(),
            }
        }
        mock_store._data["tasks"] = {}
        mock_store._data["logs"] = {}

        response = client.post("/api/sessions/session-123/export")
        assert response.status_code == 200
        data = response.json()
        assert "session" in data
        assert "tasks" in data
        assert "logs" in data
        assert "metrics" in data


class TestBudgetAPI:
    """Tests for budget API endpoints."""

    def test_get_budget_status(self, client, mock_budget_tracker):
        """Test getting budget status."""
        response = client.get("/api/budget")
        assert response.status_code == 200
        data = response.json()
        assert "monthly_limit" in data
        assert "daily_limit" in data
        assert "monthly_spent" in data
        assert "daily_spent" in data
        assert "monthly_remaining" in data
        assert "daily_remaining" in data
        assert "throttling" in data
        assert "hard_stop" in data

    def test_get_budget_usage(self, client, mock_budget_tracker):
        """Test getting budget usage breakdown."""
        response = client.get("/api/budget/usage")
        assert response.status_code == 200
        data = response.json()
        assert "period" in data
        assert "start_date" in data
        assert "end_date" in data
        assert "total_cost" in data
        assert "items" in data
        assert "summary" in data

    def test_get_budget_usage_monthly(self, client, mock_budget_tracker):
        """Test getting monthly budget usage."""
        response = client.get("/api/budget/usage?period=monthly")
        assert response.status_code == 200
        data = response.json()
        assert data["period"] == "monthly"

    def test_get_budget_alerts(self, client, mock_budget_tracker):
        """Test getting budget alerts."""
        response = client.get("/api/budget/alerts")
        assert response.status_code == 200
        data = response.json()
        assert "alerts" in data
        assert "total" in data

    def test_update_budget_limits(self, client, mock_config_manager):
        """Test updating budget limits."""
        response = client.put(
            "/api/budget/limits",
            json={
                "monthly_limit": 2000.0,
                "daily_limit": 200.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["limits"]["monthly_limit"] == 2000.0

    def test_update_budget_limits_invalid_mode(self, client, mock_config_manager):
        """Test updating budget limits with invalid mode."""
        response = client.put(
            "/api/budget/limits",
            json={"cost_optimization_mode": "invalid_mode"},
        )
        assert response.status_code == 400


class TestMemoryAPI:
    """Tests for memory API endpoints."""

    def test_get_memory_stats(self, client, mock_memory):
        """Test getting memory statistics."""
        response = client.get("/api/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "working" in data
        assert "episodic" in data
        assert "semantic" in data
        assert "procedural" in data

    def test_get_memory_tiers(self, client, mock_memory):
        """Test getting memory tier details."""
        response = client.get("/api/memory/tiers")
        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        assert len(data["tiers"]) == 4

    def test_consolidate_memory(self, client, mock_memory):
        """Test triggering memory consolidation."""
        response = client.post(
            "/api/memory/consolidate",
            json={
                "source_tier": "working",
                "target_tier": "episodic",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_consolidate_memory_episodic_to_semantic(self, client, mock_memory):
        """Test consolidating from episodic to semantic."""
        mock_memory.episodic.get_recent_lessons = MagicMock(return_value=["pattern: action"])

        response = client.post(
            "/api/memory/consolidate",
            json={
                "source_tier": "episodic",
                "target_tier": "semantic",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_search_memory(self, client, mock_memory):
        """Test searching memory contents."""
        response = client.get("/api/memory/search?query=test&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total" in data

    def test_save_memory(self, client, mock_memory):
        """Test saving memory to disk."""
        response = client.post("/api/memory/save")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "results" in data

    def test_load_memory(self, client, mock_memory):
        """Test loading memory from disk."""
        response = client.post("/api/memory/load")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "results" in data


class TestHealingAPI:
    """Tests for healing API endpoints."""

    def test_get_healing_config(self, client, mock_healing):
        """Test getting healing configuration."""
        response = client.get("/api/healing/config")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["config"] is not None

    def test_update_healing_config(self, client, mock_healing):
        """Test updating healing configuration."""
        response = client.put(
            "/api/healing/config",
            json={
                "max_healing_level": 4,
                "max_retries_per_level": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_update_healing_config_with_preset(self, client, mock_healing):
        """Test updating healing configuration with preset."""
        with patch("gaap.api.healing.create_healing_config") as mock_create:
            mock_create.return_value = mock_healing._config

            response = client.put(
                "/api/healing/config",
                json={"preset": "conservative"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_get_healing_history(self, client, mock_healing):
        """Test getting healing history."""
        response = client.get("/api/healing/history")
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data

    def test_get_healing_history_with_limit(self, client, mock_healing):
        """Test getting healing history with limit."""
        response = client.get("/api/healing/history?limit=50")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["items"], list)

    def test_get_healing_stats(self, client, mock_healing):
        """Test getting healing statistics."""
        response = client.get("/api/healing/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_attempts" in data
        assert "successful_recoveries" in data
        assert "escalations" in data
        assert "recovery_rate" in data

    def test_reset_healing(self, client, mock_healing):
        """Test resetting healing statistics."""
        response = client.post("/api/healing/reset")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestSystemAPI:
    """Tests for system API endpoints."""

    def test_get_system_health(self, client):
        """Test getting system health status."""
        response = client.get("/api/system/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data
        assert "components" in data

    def test_get_system_metrics(self, client):
        """Test getting system metrics."""
        response = client.get("/api/system/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        assert "memory" in data
        assert "providers" in data
        assert "budget" in data
        assert "healing" in data

    def test_get_system_logs(self, client):
        """Test getting system logs."""
        response = client.get("/api/system/logs")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "total" in data

    def test_get_system_logs_with_filter(self, client):
        """Test getting system logs with level filter."""
        response = client.get("/api/system/logs?level=INFO&limit=50")
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data

    def test_get_system_info(self, client):
        """Test getting system information."""
        response = client.get("/api/system/info")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "environment" in data
        assert "python_version" in data
        assert "platform" in data
        assert "architecture" in data
        assert "uptime_seconds" in data

    def test_restart_system(self, client):
        """Test restarting system components."""
        response = client.post(
            "/api/system/restart",
            json={"components": ["memory"], "graceful": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "restarted" in data
        assert "failed" in data

    def test_clear_cache(self, client):
        """Test clearing system cache."""
        response = client.post("/api/system/clear-cache")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data
        assert "cleared" in data

    def test_get_recent_events(self, client):
        """Test getting recent system events."""
        response = client.get("/api/system/events?limit=20")
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "total" in data


class TestAPIErrorHandling:
    """Tests for API error handling."""

    def test_config_update_error(self, client, mock_config_manager):
        """Test config update handles errors gracefully."""
        response = client.put(
            "/api/config",
            json={"config": {"system": {"name": "test"}}, "validate": False},
        )
        data = response.json()
        assert "success" in data

    def test_session_create_error(self, client, mock_store):
        """Test session creation with error."""
        mock_store.insert = MagicMock(side_effect=Exception("Database error"))

        response = client.post(
            "/api/sessions",
            json={"name": "Test Session"},
        )
        assert response.status_code == 500

    def test_provider_test_error(self, client, mock_router, mock_provider):
        """Test provider test with error."""
        mock_provider.get_available_models = MagicMock(side_effect=Exception("Connection error"))
        mock_router._providers["test-provider"] = mock_provider

        response = client.post("/api/providers/test-provider/test")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False

    def test_memory_consolidate_error(self, client, mock_memory):
        """Test memory consolidation with error."""
        mock_memory.working.get_size = MagicMock(side_effect=Exception("Memory error"))

        response = client.post(
            "/api/memory/consolidate",
            json={"source_tier": "working", "target_tier": "episodic"},
        )
        assert response.status_code == 500


class TestAPIEdgeCases:
    """Tests for API edge cases."""

    def test_empty_session_list(self, client, mock_store):
        """Test empty session list."""
        mock_store._data["sessions"] = {}

        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_session_with_missing_fields(self, client, mock_store):
        """Test getting session with missing optional fields."""
        mock_store._data["sessions"] = {
            "session-123": {
                "id": "session-123",
                "data": {
                    "name": "Minimal Session",
                    "status": "pending",
                    "priority": "normal",
                },
                "created_at": datetime.now().isoformat(),
            }
        }

        response = client.get("/api/sessions/session-123")
        assert response.status_code == 200

    def test_budget_status_no_tracker(
        self, client, mock_budget_tracker, mock_config, mock_config_manager
    ):
        """Test budget status when no tracker is available."""
        with (
            patch("gaap.api.budget.get_budget_tracker", return_value=None),
            patch("gaap.api.budget.get_config", return_value=mock_config),
            patch("gaap.api.budget.get_config_manager", return_value=mock_config_manager),
        ):
            response = client.get("/api/budget")

        assert response.status_code == 200
        data = response.json()
        assert data["monthly_spent"] == 0.0
        assert data["daily_spent"] == 0.0

    def test_healing_config_no_instance(self, client):
        """Test healing config when no instance is available."""
        with patch("gaap.api.healing.get_healing_instance", return_value=None):
            response = client.get("/api/healing/config")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_search_memory_empty_results(self, client, mock_memory):
        """Test memory search with no results."""
        mock_memory.semantic.find_rules = MagicMock(return_value=[])
        mock_memory.episodic.get_episodes = MagicMock(return_value=[])
        mock_memory.episodic.get_recent_lessons = MagicMock(return_value=[])

        response = client.get("/api/memory/search?query=nonexistent")
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total"] == 0

    def test_provider_health_check_degraded(self, client, mock_router, mock_provider):
        """Test provider health check with degraded status."""
        mock_provider.get_stats = MagicMock(
            return_value={
                "total_requests": 10,
                "success_rate": 0.6,
            }
        )
        mock_router._providers["test-provider"] = mock_provider

        response = client.get("/api/providers")
        assert response.status_code == 200
        data = response.json()
        if data:
            assert data[0]["health"] in ["healthy", "degraded", "unhealthy"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
