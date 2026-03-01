"""
Comprehensive tests for gaap/core/config.py module
Tests ConfigManager, ConfigLoader, ConfigValidator, and all configuration classes
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from gaap.core.config import (
    BudgetConfig,
    ConfigBuilder,
    ConfigConverter,
    ConfigLoader,
    ConfigManager,
    ConfigValidator,
    CriticConfig,
    ExecutionConfig,
    ExternalConnectorsConfig,
    FirewallConfig,
    GAAPConfig,
    ParserConfig,
    PerformanceMonitorConfig,
    ProviderSettings,
    QualityPanelConfig,
    ResourceAllocatorConfig,
    SecurityConfig,
    SovereignConfig,
    StrategicPlannerConfig,
    SystemConfig,
    TacticalDecomposerConfig,
    get_config,
    get_config_manager,
    get_default_config,
    init_config,
    load_config,
)
from gaap.core.exceptions import ConfigLoadError, ConfigurationError


class TestConfigDataclasses:
    """Test configuration dataclasses"""

    def test_system_config_defaults(self):
        """Test SystemConfig default values"""
        config = SystemConfig()
        assert config.name == "GAAP-Production-Alpha"
        assert config.environment == "production"
        assert config.version == "1.0.0"
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.metrics_port == 9090
        assert config.health_check_port == 8080

    def test_firewall_config_defaults(self):
        """Test FirewallConfig default values"""
        config = FirewallConfig()
        assert config.strictness == "high"
        assert config.adversarial_test_enabled is True
        assert config.max_input_length == 100000
        assert config.enable_behavioral_analysis is True
        assert config.enable_semantic_analysis is True

    def test_firewall_config_all_layers_enabled(self):
        """Test that all firewall layers are enabled by default"""
        config = FirewallConfig()
        assert config.layer1_surface_enabled is True
        assert config.layer2_lexical_enabled is True
        assert config.layer3_syntactic_enabled is True
        assert config.layer4_semantic_enabled is True
        assert config.layer5_contextual_enabled is True
        assert config.layer6_behavioral_enabled is True
        assert config.layer7_adversarial_enabled is True

    def test_parser_config_defaults(self):
        """Test ParserConfig default values"""
        config = ParserConfig()
        assert config.default_model == "gpt-4o-mini"
        assert config.routing_strategy == "quality_first"
        assert config.extract_implicit_requirements is True
        assert config.max_classification_retries == 2
        assert config.confidence_threshold == 0.85

    def test_strategic_planner_config_defaults(self):
        """Test StrategicPlannerConfig default values"""
        config = StrategicPlannerConfig()
        assert config.tot_depth == 5
        assert config.branching_factor == 4
        assert config.mad_debate_rounds == 3
        assert config.consensus_threshold == 0.85
        assert config.enable_external_research is True
        assert config.max_parallel_options == 5

    def test_resource_allocator_config_defaults(self):
        """Test ResourceAllocatorConfig default values"""
        config = ResourceAllocatorConfig()
        assert config.tier_1_model == "claude-3-5-opus"
        assert config.tier_2_model == "gpt-4o"
        assert config.tier_3_model == "gpt-4o-mini"
        assert config.tier_4_model == "llama-3-70b"
        assert config.allow_local_fallbacks is True
        assert config.cost_tracking_enabled is True
        assert config.auto_downgrade_on_budget is True

    def test_tactical_decomposer_config_defaults(self):
        """Test TacticalDecomposerConfig default values"""
        config = TacticalDecomposerConfig()
        assert config.max_subtasks == 50
        assert config.max_task_size_lines == 500
        assert config.max_task_time_minutes == 10
        assert config.enable_smart_batching is True
        assert config.dependency_analysis_enabled is True
        assert config.critical_path_analysis is True

    def test_execution_config_defaults(self):
        """Test ExecutionConfig default values"""
        config = ExecutionConfig()
        assert config.max_parallel_tasks == 10
        assert config.genetic_twin_enabled is True
        assert config.genetic_twin_for_critical_only is True
        assert config.self_healing_enabled is True
        assert config.self_healing_max_retries == 3
        assert config.consciousness_migration_enabled is True
        assert config.checkpoint_interval_seconds == 30
        assert config.sandbox_type == "gvisor"

    def test_security_config_defaults(self):
        """Test SecurityConfig default values"""
        config = SecurityConfig()
        assert config.sandbox_type == "gvisor"
        assert config.capability_tokens_ttl == 300
        assert config.blockchain_audit_enabled is True
        assert config.encryption_enabled is True
        assert config.encryption_algorithm == "AES-256-GCM"
        assert config.seccomp_enabled is True

    def test_budget_config_defaults(self):
        """Test BudgetConfig default values"""
        config = BudgetConfig()
        assert config.monthly_limit == 5000.0
        assert config.daily_limit == 200.0
        assert config.per_task_limit == 10.0
        assert config.alert_thresholds == [0.5, 0.8, 0.95]
        assert config.auto_throttle_at == 0.9
        assert config.hard_stop_at == 1.0

    def test_sovereign_config_defaults(self):
        """Test SovereignConfig default values"""
        config = SovereignConfig()
        assert config.enable_deep_research is True
        assert config.enable_tool_synthesis is True
        assert config.enable_ghost_fs is True
        assert config.enable_dlp_shield is True
        assert config.enable_semantic_firewall is True
        assert config.dlp_strictness == "high"
        assert config.research_depth == 3

    def test_performance_monitor_config_defaults(self):
        """Test PerformanceMonitorConfig default values"""
        config = PerformanceMonitorConfig()
        assert config.enabled is True
        assert config.sampling_strategy == "adaptive"
        assert config.sampling_rate == 0.1
        assert config.max_samples_per_metric == 10000
        assert config.max_age_minutes == 60

    def test_critic_config(self):
        """Test CriticConfig creation"""
        critic = CriticConfig(name="logic", weight=0.35)
        assert critic.name == "logic"
        assert critic.weight == 0.35
        assert critic.enabled is True
        assert critic.custom_rules == []

    def test_quality_panel_config_defaults(self):
        """Test QualityPanelConfig default values"""
        config = QualityPanelConfig()
        assert len(config.critics) == 6
        assert config.min_approval_score == 70.0
        assert config.unanimous_required_for_critical is True
        assert config.max_debate_rounds == 5

    def test_provider_settings(self):
        """Test ProviderSettings creation"""
        provider = ProviderSettings(
            name="test",
            api_key="secret123",
            models=["model1", "model2"],
            default_model="model1",
        )
        assert provider.name == "test"
        assert provider.api_key == "secret123"
        assert provider.models == ["model1", "model2"]
        assert provider.default_model == "model1"
        assert provider.enabled is True

    def test_gaap_config_defaults(self):
        """Test GAAPConfig default values"""
        config = GAAPConfig()
        assert isinstance(config.system, SystemConfig)
        assert isinstance(config.firewall, FirewallConfig)
        assert isinstance(config.parser, ParserConfig)
        assert isinstance(config.strategic_planner, StrategicPlannerConfig)
        assert isinstance(config.resource_allocator, ResourceAllocatorConfig)
        assert isinstance(config.tactical_decomposer, TacticalDecomposerConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.quality_panel, QualityPanelConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.budget, BudgetConfig)
        assert isinstance(config.sovereign, SovereignConfig)

    def test_gaap_config_to_dict(self):
        """Test GAAPConfig.to_dict() method"""
        config = GAAPConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "system" in config_dict
        assert "firewall" in config_dict
        assert "providers" in config_dict

    def test_gaap_config_secrets_none_without_init(self):
        """Test that secrets is None without initialization"""
        config = GAAPConfig()
        assert config.secrets is None


class TestConfigLoader:
    """Test ConfigLoader class"""

    def test_load_from_yaml_file(self, tmp_path: Path):
        """Test loading config from YAML file"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "TestApp", "environment": "development"}}
        config_file.write_text(yaml.dump(config_data))

        loader = ConfigLoader()
        result = loader.load_from_file(str(config_file))
        assert result["system"]["name"] == "TestApp"
        assert result["system"]["environment"] == "development"

    def test_load_from_json_file(self, tmp_path: Path):
        """Test loading config from JSON file"""
        import json

        config_file = tmp_path / "config.json"
        config_data = {"system": {"name": "TestApp", "environment": "development"}}
        config_file.write_text(json.dumps(config_data))

        loader = ConfigLoader()
        result = loader.load_from_file(str(config_file))
        assert result["system"]["name"] == "TestApp"

    def test_load_from_file_not_exists(self):
        """Test loading from non-existent file raises error"""
        loader = ConfigLoader()
        with pytest.raises(ConfigLoadError):
            loader.load_from_file("/nonexistent/config.yaml")

    def test_load_from_invalid_yaml(self, tmp_path: Path):
        """Test loading invalid YAML raises error"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: [")

        loader = ConfigLoader()
        with pytest.raises(ConfigLoadError):
            loader.load_from_file(str(config_file))

    def test_load_from_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON raises error"""
        import json

        config_file = tmp_path / "config.json"
        config_file.write_text("{invalid json}")

        loader = ConfigLoader()
        with pytest.raises(ConfigLoadError):
            loader.load_from_file(str(config_file))

    def test_load_from_unsupported_format(self, tmp_path: Path):
        """Test loading unsupported file format raises error"""
        config_file = tmp_path / "config.txt"
        config_file.write_text("some text")

        loader = ConfigLoader()
        with pytest.raises(ConfigLoadError):
            loader.load_from_file(str(config_file))

    def test_load_from_env(self, monkeypatch):
        """Test loading configuration from environment variables"""
        monkeypatch.setenv("GAAP_SYSTEM_NAME", "TestApp")
        monkeypatch.setenv("GAAP_ENVIRONMENT", "staging")
        monkeypatch.setenv("GAAP_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("GAAP_BUDGET_MONTHLY_LIMIT", "10000")

        loader = ConfigLoader()
        result = loader.load_from_env()

        assert result["system"]["name"] == "TestApp"
        assert result["system"]["environment"] == "staging"
        assert result["system"]["log_level"] == "DEBUG"
        assert result["budget"]["monthly_limit"] == "10000"

    def test_deep_merge(self):
        """Test deep merging of configuration dictionaries"""
        loader = ConfigLoader()
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}
        result = loader.deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 4
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_deep_merge_preserves_nested_structure(self):
        """Test that deep merge preserves nested structure"""
        loader = ConfigLoader()
        base = {"level1": {"level2": {"level3": "value"}}}
        override = {"level1": {"new_key": "new_value"}}
        result = loader.deep_merge(base, override)

        assert result["level1"]["level2"]["level3"] == "value"
        assert result["level1"]["new_key"] == "new_value"

    def test_calculate_file_hash(self, tmp_path: Path):
        """Test file hash calculation"""
        import hashlib

        test_file = tmp_path / "test.txt"
        content = "test content"
        test_file.write_text(content)

        loader = ConfigLoader()
        hash1 = loader.calculate_file_hash(str(test_file))
        hash2 = hashlib.md5(content.encode()).hexdigest()
        assert hash1 == hash2


class TestConfigValidator:
    """Test ConfigValidator class"""

    def test_validate_valid_config(self):
        """Test validation of valid configuration"""
        validator = ConfigValidator()
        config = GAAPConfig()
        warnings, errors = validator.validate(config)

        assert len(errors) == 0
        assert len(warnings) == 0

    def test_validate_invalid_environment(self):
        """Test validation catches invalid environment"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.system.environment = "invalid_env"

        warnings, errors = validator.validate(config)
        assert len(errors) == 1
        assert "Invalid environment" in errors[0]

    def test_validate_invalid_log_level(self):
        """Test validation fixes invalid log level"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.system.log_level = "INVALID"

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert config.system.log_level == "INFO"

    def test_validate_invalid_budget(self):
        """Test validation fixes invalid budget values"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.budget.monthly_limit = -100
        config.budget.daily_limit = -50

        warnings, errors = validator.validate(config)
        assert len(warnings) == 2
        assert config.budget.monthly_limit == 5000.0
        assert config.budget.daily_limit == 200.0

    def test_validate_daily_budget_exceeds_monthly(self):
        """Test validation fixes daily budget exceeding monthly"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.budget.monthly_limit = 100
        config.budget.daily_limit = 200

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert config.budget.daily_limit == 100

    def test_validate_invalid_alert_thresholds(self):
        """Test validation fixes invalid alert thresholds"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.budget.alert_thresholds = [0.5, 1.5, -0.1]

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert config.budget.alert_thresholds == [0.5, 0.8, 0.95]

    def test_validate_and_normalize_critics(self):
        """Test validation normalizes critic weights"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.quality_panel.critics = [
            CriticConfig(name="logic", weight=0.5),
            CriticConfig(name="security", weight=0.5),
        ]

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert "normalizing to 1.0" in warnings[0]

    def test_validate_no_enabled_critics(self):
        """Test validation handles no enabled critics"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.quality_panel.critics = [CriticConfig(name="test", weight=1.0, enabled=False)]

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert "No enabled critics found" in warnings[0]
        assert len(config.quality_panel.critics) == 6

    def test_normalize_weights_zero_sum(self):
        """Test normalization with zero weight sum"""
        validator = ConfigValidator()
        critics = [
            CriticConfig(name="test1", weight=0.0),
            CriticConfig(name="test2", weight=0.0),
        ]
        result = validator.normalize_weights(critics)
        assert len(result) == 6  # Should return defaults

    def test_validate_invalid_sandbox_type(self):
        """Test validation fixes invalid sandbox type"""
        validator = ConfigValidator()
        config = GAAPConfig()
        config.security.sandbox_type = "invalid_sandbox"

        warnings, errors = validator.validate(config)
        assert len(warnings) == 1
        assert config.security.sandbox_type == "gvisor"


class TestConfigConverter:
    """Test ConfigConverter class"""

    def test_dict_to_config(self):
        """Test converting dictionary to GAAPConfig"""
        config_dict = {
            "system": {"name": "TestApp", "environment": "staging"},
            "firewall": {"strictness": "medium"},
            "budget": {"monthly_limit": 1000.0},
        }

        converter = ConfigConverter()
        config = converter.dict_to_config(config_dict)

        assert config.system.name == "TestApp"
        assert config.system.environment == "staging"
        assert config.firewall.strictness == "medium"
        assert config.budget.monthly_limit == 1000.0

    def test_dict_to_config_empty(self):
        """Test converting empty dictionary to GAAPConfig"""
        converter = ConfigConverter()
        config = converter.dict_to_config({})

        assert isinstance(config, GAAPConfig)
        assert config.system.name == "GAAP-Production-Alpha"

    def test_process_quality_panel_with_custom_critics(self):
        """Test processing quality panel with custom critics"""
        converter = ConfigConverter()
        quality_dict = {
            "critics": [
                {"name": "custom1", "weight": 0.6},
                {"name": "custom2", "weight": 0.4},
            ],
            "min_approval_score": 80.0,
        }

        panel = converter._process_quality_panel(quality_dict)
        assert len(panel.critics) == 2
        assert panel.critics[0].name == "custom1"
        assert panel.min_approval_score == 80.0

    def test_process_quality_panel_empty_uses_defaults(self):
        """Test that empty quality panel uses default critics"""
        converter = ConfigConverter()
        panel = converter._process_quality_panel({})

        assert len(panel.critics) == 6
        assert panel.critics[0].name == "logic"


class TestConfigManager:
    """Test ConfigManager class"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        yield
        ConfigManager._instance = None
        ConfigManager._initialized = False

    def test_singleton_pattern(self):
        """Test that ConfigManager is a singleton"""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_get_config_property(self):
        """Test getting configuration"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()
        config = manager.config
        assert isinstance(config, GAAPConfig)

    def test_get_with_dot_notation(self):
        """Test getting config values with dot notation"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        assert manager.get("system.name") == "GAAP-Production-Alpha"
        assert manager.get("system.log_level") == "INFO"
        assert manager.get("firewall.strictness") == "high"

    def test_get_with_default(self):
        """Test getting config with default value"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        assert manager.get("nonexistent.key", "default") == "default"
        assert manager.get("system.nonexistent", 123) == 123

    def test_get_nested_invalid_path(self):
        """Test getting with invalid path returns default"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        assert manager.get("system.nonexistent.nested", "default") == "default"

    def test_is_production(self):
        """Test is_production check"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()
        assert manager.is_production() is True

    def test_is_development(self):
        """Test is_development check"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()
        assert manager.is_development() is False

    def test_get_enabled_providers(self):
        """Test getting enabled providers"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()
        providers = manager.get_enabled_providers()
        assert isinstance(providers, list)

    def test_layer_config(self):
        """Test layer-specific configuration"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        manager.set_layer_config("execution", {"timeout": 30})
        assert manager.get_layer_config("execution") == {"timeout": 30}
        assert manager.get_layer_config("nonexistent") == {}

    def test_get_api_key_from_env(self, monkeypatch):
        """Test getting API key from environment"""
        monkeypatch.setenv("GAAP_OPENAI_API_KEY", "test-api-key")

        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        assert manager.get_api_key("openai") == "test-api-key"

    def test_get_api_key_unknown_provider(self):
        """Test getting API key for unknown provider"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        assert manager.get_api_key("unknown_provider") is None

    def test_watcher_functionality(self):
        """Test config watcher functionality"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        watcher_called = []

        def watcher(config):
            watcher_called.append(True)

        manager.add_watcher(watcher)
        manager.reload()

        assert len(watcher_called) == 1

        manager.remove_watcher(watcher)
        manager.reload()

        assert len(watcher_called) == 1  # Should not be called again

    def test_watcher_remove_nonexistent(self):
        """Test removing non-existent watcher doesn't error"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()

        def watcher(config):
            pass

        manager.remove_watcher(watcher)  # Should not raise

    def test_check_and_reload_no_file(self):
        """Test check_and_reload when no file is configured"""
        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager()
        assert manager.check_and_reload() is False

    def test_load_from_file(self, tmp_path: Path):
        """Test loading configuration from file"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "TestFromFile"}}
        config_file.write_text(yaml.dump(config_data))

        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager(config_path=str(config_file))

        assert manager.config.system.name == "TestFromFile"

    def test_config_validation_error(self, tmp_path: Path):
        """Test configuration validation raises error"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"environment": "invalid_env"}}
        config_file.write_text(yaml.dump(config_data))

        ConfigManager._instance = None
        ConfigManager._initialized = False

        with pytest.raises(ConfigurationError):
            ConfigManager(config_path=str(config_file))

    def test_config_merge_with_env(self, tmp_path: Path, monkeypatch):
        """Test config loading merges with environment variables"""
        monkeypatch.setenv("GAAP_SYSTEM_NAME", "EnvOverride")

        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "FileValue"}}
        config_file.write_text(yaml.dump(config_data))

        ConfigManager._instance = None
        ConfigManager._initialized = False
        manager = ConfigManager(config_path=str(config_file))

        # Environment should override file
        assert manager.config.system.name == "EnvOverride"


class TestConfigBuilder:
    """Test ConfigBuilder class"""

    def test_basic_builder(self):
        """Test basic config building"""
        config = (
            ConfigBuilder()
            .with_system(name="TestApp", environment="staging")
            .with_budget(monthly=1000, daily=50)
            .build()
        )

        assert config.system.name == "TestApp"
        assert config.system.environment == "staging"
        assert config.budget.monthly_limit == 1000.0
        assert config.budget.daily_limit == 50.0

    def test_builder_with_provider(self):
        """Test builder with provider configuration"""
        config = (
            ConfigBuilder()
            .with_system(name="TestApp")
            .with_provider(
                name="openai",
                api_key="test-key",
                models=["gpt-4", "gpt-3.5"],
                default_model="gpt-4",
                priority=1,
            )
            .build()
        )

        assert len(config.providers) == 1
        assert config.providers[0].name == "openai"
        assert config.providers[0].api_key == "test-key"

    def test_builder_with_security(self):
        """Test builder with security configuration"""
        config = (
            ConfigBuilder()
            .with_system(name="TestApp")
            .with_security(sandbox_type="docker", audit_enabled=False)
            .build()
        )

        assert config.security.sandbox_type == "docker"
        assert config.security.blockchain_audit_enabled is False

    def test_builder_with_execution(self):
        """Test builder with execution configuration"""
        config = (
            ConfigBuilder()
            .with_system(name="TestApp")
            .with_execution(max_parallel=20, genetic_twin=False, self_healing=False)
            .build()
        )

        assert config.execution.max_parallel_tasks == 20
        assert config.execution.genetic_twin_enabled is False
        assert config.execution.self_healing_enabled is False

    def test_builder_with_custom(self):
        """Test builder with custom configuration"""
        config = (
            ConfigBuilder()
            .with_system(name="TestApp")
            .with_custom("custom_key", "custom_value")
            .with_custom("nested", {"key": "value"})
            .build()
        )

        assert config.custom["custom_key"] == "custom_value"
        assert config.custom["nested"] == {"key": "value"}

    def test_builder_from_file(self, tmp_path: Path):
        """Test builder loading from file"""
        config_file = tmp_path / "base.yaml"
        config_data = {"system": {"name": "BaseName", "log_level": "DEBUG"}}
        config_file.write_text(yaml.dump(config_data))

        config = (
            ConfigBuilder().from_file(str(config_file)).with_system(environment="staging").build()
        )

        assert config.system.name == "BaseName"
        assert config.system.log_level == "DEBUG"
        assert config.system.environment == "staging"

    def test_builder_to_yaml(self, tmp_path: Path):
        """Test saving config to YAML file"""
        output_file = tmp_path / "output.yaml"

        ConfigBuilder().with_system(name="TestApp").to_yaml(str(output_file))

        assert output_file.exists()
        content = output_file.read_text()
        assert "name: TestApp" in content

    def test_builder_to_json(self, tmp_path: Path):
        """Test saving config to JSON file"""
        import json

        output_file = tmp_path / "output.json"

        ConfigBuilder().with_system(name="TestApp").to_json(str(output_file))

        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["system"]["name"] == "TestApp"

    def test_builder_validation_error(self):
        """Test builder raises error on validation failure"""
        with pytest.raises(ConfigurationError):
            # This will fail because invalid environment
            ConfigBuilder().with_system(environment="invalid_env").build()


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_default_config(self):
        """Test get_default_config function"""
        config = get_default_config()
        assert isinstance(config, GAAPConfig)
        assert config.system.name == "GAAP-Production-Alpha"

    def test_load_config(self):
        """Test load_config function"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False

        config = load_config()
        assert isinstance(config, GAAPConfig)

    def test_init_config(self):
        """Test init_config function"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False

        manager = init_config()
        assert isinstance(manager, ConfigManager)

    def test_get_config(self):
        """Test get_config function"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False

        config = get_config()
        assert isinstance(config, GAAPConfig)

    def test_get_config_manager(self):
        """Test get_config_manager function"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._initialized = False

        manager = get_config_manager()
        assert isinstance(manager, ConfigManager)


class TestConfigEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_provider_list(self):
        """Test configuration with empty provider list"""
        config = GAAPConfig(providers=[])
        assert config.providers == []

    def test_config_with_none_values(self):
        """Test configuration handles None values"""
        config = GAAPConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

    def test_config_serialization_roundtrip(self):
        """Test config can be serialized and deserialized"""
        config = GAAPConfig(system=SystemConfig(name="Test", environment="staging"))
        config_dict = config.to_dict()

        converter = ConfigConverter()
        restored = converter.dict_to_config(config_dict)

        assert restored.system.name == "Test"
        assert restored.system.environment == "staging"

    def test_thread_safety(self):
        """Test ConfigManager is thread-safe"""
        ConfigManager._instance = None
        ConfigManager._initialized = False

        results = []

        def create_manager():
            manager = ConfigManager()
            results.append(manager)

        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All managers should be the same instance
        assert all(m is results[0] for m in results)

    def test_config_with_special_characters(self, tmp_path: Path):
        """Test config handles special characters"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "App with special chars: <>&\"'"}}
        config_file.write_text(yaml.dump(config_data))

        loader = ConfigLoader()
        result = loader.load_from_file(str(config_file))
        assert "<>&\"'" in result["system"]["name"]

    def test_very_long_config_values(self):
        """Test config handles very long values"""
        long_name = "x" * 10000
        config = GAAPConfig(system=SystemConfig(name=long_name))
        assert config.system.name == long_name

    def test_unicode_in_config(self, tmp_path: Path):
        """Test config handles unicode characters"""
        config_file = tmp_path / "config.yaml"
        config_data = {"system": {"name": "测试应用 🚀 café"}}
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        loader = ConfigLoader()
        result = loader.load_from_file(str(config_file))
        assert result["system"]["name"] == "测试应用 🚀 café"

    def test_nested_dict_merge_preserves_types(self):
        """Test that deep merge preserves value types"""
        loader = ConfigLoader()
        base = {"int_val": 42, "float_val": 3.14, "bool_val": True, "list_val": [1, 2, 3]}
        override = {"int_val": 100}
        result = loader.deep_merge(base, override)

        assert result["int_val"] == 100
        assert result["float_val"] == 3.14
        assert result["bool_val"] is True
        assert result["list_val"] == [1, 2, 3]
