"""
Tests for GAAP Core Config Module
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from gaap.core.config import (
    BudgetConfig,
    ConfigManager,
    ContextManagementConfig,
    CriticConfig,
    ExecutionConfig,
    ExternalConnectorsConfig,
    FirewallConfig,
    GAAPConfig,
    ParserConfig,
    ProviderSettings,
    QualityPanelConfig,
    ResourceAllocatorConfig,
    SecurityConfig,
    SovereignConfig,
    StrategicPlannerConfig,
    SystemConfig,
    TacticalDecomposerConfig,
    ConfigLoadError,
    ConfigurationError,
)


class TestFirewallConfig:
    def test_defaults(self):
        config = FirewallConfig()
        assert config.strictness == "high"
        assert config.adversarial_test_enabled is True
        assert config.max_input_length == 100000
        assert config.layer1_surface_enabled is True

    def test_custom_values(self):
        config = FirewallConfig(strictness="low", max_input_length=5000)
        assert config.strictness == "low"
        assert config.max_input_length == 5000


class TestParserConfig:
    def test_defaults(self):
        config = ParserConfig()
        assert config.default_model == "gpt-4o-mini"
        assert config.routing_strategy == "quality_first"
        assert config.confidence_threshold == 0.85


class TestStrategicPlannerConfig:
    def test_defaults(self):
        config = StrategicPlannerConfig()
        assert config.tot_depth == 5
        assert config.branching_factor == 4
        assert config.consensus_threshold == 0.85


class TestResourceAllocatorConfig:
    def test_defaults(self):
        config = ResourceAllocatorConfig()
        assert config.tier_1_model == "claude-3-5-opus"
        assert config.allow_local_fallbacks is True
        assert config.cost_tracking_enabled is True


class TestTacticalDecomposerConfig:
    def test_defaults(self):
        config = TacticalDecomposerConfig()
        assert config.max_subtasks == 50
        assert config.max_task_size_lines == 500
        assert config.enable_smart_batching is True


class TestSovereignConfig:
    def test_defaults(self):
        config = SovereignConfig()
        assert config.enable_deep_research is True
        assert config.enable_tool_synthesis is True
        assert config.dlp_strictness == "high"


class TestExecutionConfig:
    def test_defaults(self):
        config = ExecutionConfig()
        assert config.max_parallel_tasks == 10
        assert config.self_healing_enabled is True
        assert config.sandbox_type == "gvisor"


class TestCriticConfig:
    def test_defaults(self):
        critic = CriticConfig(name="logic")
        assert critic.name == "logic"
        assert critic.weight == 1.0
        assert critic.enabled is True

    def test_custom_weight(self):
        critic = CriticConfig(name="security", weight=0.5)
        assert critic.weight == 0.5


class TestQualityPanelConfig:
    def test_defaults(self):
        config = QualityPanelConfig()
        assert len(config.critics) == 6
        assert config.min_approval_score == 70.0

    def test_custom_critics(self):
        critics = [
            CriticConfig(name="logic", weight=0.4),
            CriticConfig(name="security", weight=0.6),
        ]
        config = QualityPanelConfig(critics=critics)
        assert len(config.critics) == 2


class TestSecurityConfig:
    def test_defaults(self):
        config = SecurityConfig()
        assert config.sandbox_type == "gvisor"
        assert config.encryption_enabled is True
        assert config.encryption_algorithm == "AES-256-GCM"


class TestBudgetConfig:
    def test_defaults(self):
        config = BudgetConfig()
        assert config.monthly_limit == 5000.0
        assert config.daily_limit == 200.0
        assert 0.5 in config.alert_thresholds

    def test_alert_thresholds_validation(self):
        config = BudgetConfig(alert_thresholds=[0.3, 0.7, 0.9])
        assert len(config.alert_thresholds) == 3


class TestContextManagementConfig:
    def test_defaults(self):
        config = ContextManagementConfig()
        assert config.default_budget_level == "medium"
        assert config.smart_chunking_enabled is True
        assert config.context_cache_size_mb == 100


class TestProviderSettings:
    def test_defaults(self):
        provider = ProviderSettings(name="openai")
        assert provider.name == "openai"
        assert provider.enabled is True
        assert provider.priority == 0

    def test_with_api_key(self):
        provider = ProviderSettings(name="openai", api_key="sk-test123")
        assert provider.api_key == "sk-test123"


class TestExternalConnectorsConfig:
    def test_defaults(self):
        config = ExternalConnectorsConfig()
        assert config.enable_web_search is False
        assert config.timeout_seconds == 30


class TestSystemConfig:
    def test_defaults(self):
        config = SystemConfig()
        assert config.name == "GAAP-Production-Alpha"
        assert config.environment == "production"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"


class TestGAAPConfig:
    def test_defaults(self):
        config = GAAPConfig()
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.firewall, FirewallConfig)
        assert isinstance(config.budget, BudgetConfig)

    def test_to_dict(self):
        config = GAAPConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "system" in d
        assert "budget" in d
        assert d["system"]["name"] == "GAAP-Production-Alpha"


class TestConfigManager:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager._instance = None
        ConfigManager._initialized = False
        yield
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_singleton(self, reset_singleton):
        manager1 = ConfigManager.__new__(ConfigManager)
        manager1._config = GAAPConfig()
        manager1._initialized = True
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_default_config(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        assert manager.config is not None
        assert manager.config.system.name == "GAAP-Production-Alpha"

    def test_get_path(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        result = manager.get("system.name")
        assert result == "GAAP-Production-Alpha"

    def test_get_nested_path(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        result = manager.get("budget.monthly_limit")
        assert result == 5000.0

    def test_get_default_value(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        result = manager.get("nonexistent.key", default="default_value")
        assert result == "default_value"

    def test_get_layer_config(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        manager._layer_configs = {}
        manager.set_layer_config("layer1", {"key": "value"})
        config = manager.get_layer_config("layer1")
        assert config == {"key": "value"}

    def test_get_layer_config_default(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        manager._layer_configs = {}
        config = manager.get_layer_config("nonexistent")
        assert config == {}

    def test_add_watcher(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        manager._watchers = []
        called = []

        def callback(cfg):
            called.append(cfg)

        manager.add_watcher(callback)
        assert callback in manager._watchers

    def test_remove_watcher(self, reset_singleton):
        manager = ConfigManager.__new__(ConfigManager)
        manager._config = GAAPConfig()
        manager._initialized = True
        manager._watchers = []

        def callback(cfg):
            pass

        manager.add_watcher(callback)
        manager.remove_watcher(callback)
        assert callback not in manager._watchers


class TestConfigManagerFromFile:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager._instance = None
        ConfigManager._initialized = False
        yield
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_load_json_config(self, tmp_path):
        config_data = {
            "system": {"name": "TestSystem", "environment": "development"},
            "budget": {"monthly_limit": 1000.0},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        manager = ConfigManager(config_path=str(config_file))
        assert manager.config.system.name == "TestSystem"
        assert manager.config.budget.monthly_limit == 1000.0

    def test_load_yaml_config(self, tmp_path):
        config_data = {
            "system": {"name": "YamlSystem", "environment": "staging"},
            "budget": {"daily_limit": 50.0},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump(config_data))

        manager = ConfigManager(config_path=str(config_file))
        assert manager.config.system.name == "YamlSystem"
        assert manager.config.budget.daily_limit == 50.0

    def test_load_nonexistent_file(self):
        with pytest.raises(ConfigLoadError):
            ConfigManager(config_path="/nonexistent/path/config.json")

    def test_load_invalid_json(self, tmp_path):
        config_file = tmp_path / "invalid.json"
        config_file.write_text("{ invalid json }")

        with pytest.raises(ConfigLoadError):
            ConfigManager(config_path=str(config_file))


class TestConfigValidation:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager._instance = None
        ConfigManager._initialized = False
        yield
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_invalid_environment(self, tmp_path):
        config_data = {
            "system": {"environment": "invalid_env"},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_invalid_log_level(self, tmp_path):
        config_data = {
            "system": {"log_level": "INVALID"},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_negative_budget(self, tmp_path):
        config_data = {
            "budget": {"monthly_limit": -100},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_daily_exceeds_monthly(self, tmp_path):
        config_data = {
            "budget": {"monthly_limit": 100, "daily_limit": 200},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_invalid_alert_thresholds(self, tmp_path):
        config_data = {
            "budget": {"alert_thresholds": [1.5]},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_invalid_critic_weights(self, tmp_path):
        config_data = {
            "quality_panel": {
                "critics": [
                    {"name": "logic", "weight": 0.3},
                    {"name": "security", "weight": 0.3},
                ]
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_invalid_sandbox_type(self, tmp_path):
        config_data = {
            "security": {"sandbox_type": "invalid"},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))


class TestConfigFromEnv:
    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        ConfigManager._instance = None
        ConfigManager._initialized = False
        yield
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_env_override(self, monkeypatch, tmp_path):
        config_data = {
            "system": {"environment": "production", "name": "BaseName"},
            "budget": {"monthly_limit": 100, "daily_limit": 10},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("GAAP_SYSTEM_NAME", "EnvSystem")

        manager = ConfigManager(config_path=str(config_file))
        assert manager.config.system.name == "EnvSystem"

    def test_env_model_override(self, monkeypatch, tmp_path):
        config_data = {
            "budget": {"monthly_limit": 100, "daily_limit": 10},
            "quality_panel": {"critics": [{"name": "logic", "weight": 1.0}]},
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        monkeypatch.setenv("GAAP_TIER_1_MODEL", "gpt-5")

        manager = ConfigManager(config_path=str(config_file))
        assert manager.config.resource_allocator.tier_1_model == "gpt-5"
