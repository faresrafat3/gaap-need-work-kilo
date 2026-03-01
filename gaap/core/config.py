"""
Configuration Management for GAAP
==================================

Refactored from a God Class (ConfigManager) into specialized components:
- ConfigLoader: Handles loading from files and environment
- ConfigValidator: Validates configuration values
- ConfigManager: Coordinates loading and validation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gaap.storage.atomic import atomic_write

from .exceptions import ConfigLoadError, ConfigurationError

# Import secrets module (optional - will work without it)
try:
    from .secrets import SecretsManager, SecretsProvider

    _HAS_SECRETS = True
except ImportError:
    _HAS_SECRETS = False
    SecretsManager = None  # type: ignore
    SecretsProvider = None  # type: ignore


# =============================================================================
# Configuration Dataclasses (unchanged)
# =============================================================================


@dataclass
class FirewallConfig:
    """Firewall configuration (Layer 0)"""

    strictness: str = "high"
    adversarial_test_enabled: bool = True
    blocked_patterns_update_interval: str = "6h"
    max_input_length: int = 100000
    enable_behavioral_analysis: bool = True
    enable_semantic_analysis: bool = True
    layer1_surface_enabled: bool = True
    layer2_lexical_enabled: bool = True
    layer3_syntactic_enabled: bool = True
    layer4_semantic_enabled: bool = True
    layer5_contextual_enabled: bool = True
    layer6_behavioral_enabled: bool = True
    layer7_adversarial_enabled: bool = True


@dataclass
class ParserConfig:
    """Request parser configuration"""

    default_model: str = "gpt-4o-mini"
    routing_strategy: str = "quality_first"
    extract_implicit_requirements: bool = True
    max_classification_retries: int = 2
    confidence_threshold: float = 0.85


@dataclass
class StrategicPlannerConfig:
    """Strategic planner configuration (Layer 1)"""

    tot_depth: int = 5
    branching_factor: int = 4
    mad_debate_rounds: int = 3
    consensus_threshold: float = 0.85
    enable_external_research: bool = True
    max_parallel_options: int = 5


@dataclass
class ResourceAllocatorConfig:
    """Resource allocator configuration"""

    tier_1_model: str = "claude-3-5-opus"
    tier_2_model: str = "gpt-4o"
    tier_3_model: str = "gpt-4o-mini"
    tier_4_model: str = "llama-3-70b"
    allow_local_fallbacks: bool = True
    cost_tracking_enabled: bool = True
    auto_downgrade_on_budget: bool = True


@dataclass
class TacticalDecomposerConfig:
    """Tactical decomposer configuration (Layer 2)"""

    max_subtasks: int = 50
    max_task_size_lines: int = 500
    max_task_time_minutes: int = 10
    enable_smart_batching: bool = True
    dependency_analysis_enabled: bool = True
    critical_path_analysis: bool = True


@dataclass
class SovereignConfig:
    """Sovereign intelligence configuration (v2.1)"""

    enable_deep_research: bool = True
    enable_tool_synthesis: bool = True
    enable_ghost_fs: bool = True
    enable_dlp_shield: bool = True
    enable_semantic_firewall: bool = True
    dlp_strictness: str = "high"
    research_depth: int = 3


@dataclass
class PerformanceMonitorConfig:
    """Performance monitoring configuration"""

    enabled: bool = True
    sampling_strategy: str = "adaptive"
    sampling_rate: float = 0.1
    max_samples_per_metric: int = 10000
    max_age_minutes: int = 60
    enable_memory_tracking: bool = True
    enable_throughput: bool = True
    enable_latency_percentiles: bool = True
    gc_during_cleanup: bool = True
    export_format: str = "json"


@dataclass
class ExecutionConfig:
    """Execution configuration (Layer 3)"""

    max_parallel_tasks: int = 10
    genetic_twin_enabled: bool = True
    genetic_twin_for_critical_only: bool = True
    self_healing_enabled: bool = True
    self_healing_max_retries: int = 3
    consciousness_migration_enabled: bool = True
    checkpoint_interval_seconds: int = 30
    sandbox_type: str = "gvisor"


@dataclass
class CriticConfig:
    """Individual critic configuration"""

    name: str
    weight: float = 1.0
    enabled: bool = True
    custom_rules: list[str] = field(default_factory=list)


@dataclass
class QualityPanelConfig:
    """Quality panel configuration"""

    critics: list[CriticConfig] = field(
        default_factory=lambda: [
            CriticConfig(name="logic", weight=0.35),
            CriticConfig(name="security", weight=0.25),
            CriticConfig(name="performance", weight=0.20),
            CriticConfig(name="style", weight=0.10),
            CriticConfig(name="compliance", weight=0.05),
            CriticConfig(name="ethics", weight=0.05),
        ]
    )
    min_approval_score: float = 70.0
    unanimous_required_for_critical: bool = True
    max_debate_rounds: int = 5


@dataclass
class SecurityConfig:
    """Security configuration"""

    sandbox_type: str = "gvisor"
    capability_tokens_ttl: int = 300
    blockchain_audit_enabled: bool = True
    audit_storage_provider: str = "local"
    audit_storage_path: str = "./audit_logs"
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    seccomp_enabled: bool = True
    seccomp_default_action: str = "kill"
    cgroup_cpu_max: float = 0.5
    cgroup_memory_max: str = "512M"
    cgroup_pids_max: int = 100


@dataclass
class BudgetConfig:
    """Budget configuration"""

    monthly_limit: float = 5000.0
    daily_limit: float = 200.0
    per_task_limit: float = 10.0
    alert_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    auto_throttle_at: float = 0.9
    hard_stop_at: float = 1.0
    cost_optimization_mode: str = "balanced"


@dataclass
class ContextManagementConfig:
    """Context management configuration"""

    default_budget_level: str = "medium"
    hierarchical_loading_enabled: bool = True
    pkg_agent_enabled: bool = False
    smart_chunking_enabled: bool = True
    external_brain_enabled: bool = True
    context_cache_enabled: bool = True
    context_cache_size_mb: int = 100
    parallel_loading_enabled: bool = True
    garbage_collection_enabled: bool = True
    gc_interval_seconds: int = 300


@dataclass
class ProviderSettings:
    """Individual provider settings"""

    name: str
    enabled: bool = True
    priority: int = 0
    api_key: str | None = None
    base_url: str | None = None
    rate_limit_per_minute: int = 60
    timeout_seconds: int = 120
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    models: list[str] = field(default_factory=list)
    default_model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExternalConnectorsConfig:
    """External connectors configuration"""

    enable_web_search: bool = False
    enable_perplexity: bool = False
    enable_github: bool = False
    perplexity_api_key: str | None = None
    github_token: str | None = None
    web_search_provider: str = "duckduckgo"
    timeout_seconds: int = 30
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """Core system configuration"""

    name: str = "GAAP-Production-Alpha"
    environment: str = "production"
    version: str = "1.0.0"
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    health_check_port: int = 8080


@dataclass
class GAAPConfig:
    """Complete GAAP system configuration"""

    system: SystemConfig = field(default_factory=SystemConfig)
    firewall: FirewallConfig = field(default_factory=FirewallConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    strategic_planner: StrategicPlannerConfig = field(default_factory=StrategicPlannerConfig)
    resource_allocator: ResourceAllocatorConfig = field(default_factory=ResourceAllocatorConfig)
    tactical_decomposer: TacticalDecomposerConfig = field(default_factory=TacticalDecomposerConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    quality_panel: QualityPanelConfig = field(default_factory=QualityPanelConfig)
    external_connectors: ExternalConnectorsConfig = field(default_factory=ExternalConnectorsConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    context_management: ContextManagementConfig = field(default_factory=ContextManagementConfig)
    sovereign: SovereignConfig = field(default_factory=SovereignConfig)
    performance_monitor: PerformanceMonitorConfig = field(default_factory=PerformanceMonitorConfig)
    providers: list[ProviderSettings] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)
    _secrets_manager: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""

        def dataclass_to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            else:
                return obj

        result = dataclass_to_dict(self)
        return result if isinstance(result, dict) else {}

    @property
    def secrets(self) -> Any:
        """Get the secrets provider associated with this configuration"""
        if self._secrets_manager is not None:
            return self._secrets_manager.secrets
        return None

    def init_secrets(
        self,
        env_prefix: str = "GAAP_",
        env_file: str | None = ".env",
        auto_load: bool = True,
    ) -> Any:
        """Initialize the secrets manager for this configuration"""
        if not _HAS_SECRETS:
            import logging

            logging.getLogger(__name__).warning(
                "Secrets module not available. Install with: pip install python-dotenv"
            )
            return None

        if _HAS_SECRETS:
            from .secrets import SecretsManager as _SecretsManagerClass

            self._secrets_manager = _SecretsManagerClass(
                env_prefix=env_prefix,
                env_file=env_file,
                auto_load=auto_load,
            )
            return self._secrets_manager
        return None


# =============================================================================
# ConfigLoader - Handles loading configuration from various sources
# =============================================================================


class ConfigLoader:
    """
    Loads configuration from files (YAML/JSON) and environment variables.

    Responsibilities:
    - File I/O operations
    - Environment variable parsing
    - Deep merging of configuration sources
    """

    def __init__(self, env_prefix: str = "GAAP_"):
        self.env_prefix = env_prefix
        self._logger = logging.getLogger("gaap.config.loader")

    def load_from_file(self, path: str) -> dict[str, Any]:
        """Load configuration from a YAML or JSON file"""
        file_path = Path(path)

        if not file_path.exists():
            raise ConfigLoadError(config_path=path, reason="File does not exist")

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            if file_path.suffix.lower() in (".yaml", ".yml"):
                return yaml.safe_load(content) or {}
            elif file_path.suffix.lower() == ".json":
                return dict(json.loads(content))
            else:
                raise ConfigLoadError(
                    config_path=path, reason=f"Unsupported file format: {file_path.suffix}"
                )
        except yaml.YAMLError as e:
            raise ConfigLoadError(config_path=path, reason=f"YAML error: {e}")
        except json.JSONDecodeError as e:
            raise ConfigLoadError(config_path=path, reason=f"JSON error: {e}")
        except Exception as e:
            raise ConfigLoadError(config_path=path, reason=str(e))

    def load_from_env(self) -> dict[str, Any]:
        """Load configuration from environment variables"""
        config: dict[str, Any] = {}

        env_mappings = {
            f"{self.env_prefix}SYSTEM_NAME": ("system", "name"),
            f"{self.env_prefix}ENVIRONMENT": ("system", "environment"),
            f"{self.env_prefix}LOG_LEVEL": ("system", "log_level"),
            f"{self.env_prefix}BUDGET_MONTHLY_LIMIT": ("budget", "monthly_limit"),
            f"{self.env_prefix}BUDGET_DAILY_LIMIT": ("budget", "daily_limit"),
            f"{self.env_prefix}TIER_1_MODEL": ("resource_allocator", "tier_1_model"),
            f"{self.env_prefix}TIER_2_MODEL": ("resource_allocator", "tier_2_model"),
            f"{self.env_prefix}TIER_3_MODEL": ("resource_allocator", "tier_3_model"),
            f"{self.env_prefix}SANDBOX_TYPE": ("security", "sandbox_type"),
            f"{self.env_prefix}AUDIT_ENABLED": ("security", "blockchain_audit_enabled"),
            f"{self.env_prefix}MAX_PARALLEL_TASKS": ("execution", "max_parallel_tasks"),
            f"{self.env_prefix}GENETIC_TWIN_ENABLED": ("execution", "genetic_twin_enabled"),
            f"{self.env_prefix}PERFORMANCE_MONITOR_ENABLED": ("performance_monitor", "enabled"),
            f"{self.env_prefix}PERFORMANCE_SAMPLING_RATE": ("performance_monitor", "sampling_rate"),
            f"{self.env_prefix}OPENAI_API_KEY": ("_api_keys", "openai"),
            f"{self.env_prefix}ANTHROPIC_API_KEY": ("_api_keys", "anthropic"),
            f"{self.env_prefix}GROQ_API_KEY": ("_api_keys", "kimi"),
            f"{self.env_prefix}GEMINI_API_KEY": ("_api_keys", "gemini"),
            f"{self.env_prefix}PERPLEXITY_API_KEY": ("_api_keys", "perplexity"),
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config, config_path, value)

        return config

    def _set_nested(self, config: dict[str, Any], path: tuple, value: Any) -> None:
        """Set a value in a nested dictionary path"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries"""
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def calculate_file_hash(self, path: str) -> str:
        """Calculate MD5 hash of a file"""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()


# =============================================================================
# ConfigValidator - Validates configuration values
# =============================================================================


class ConfigValidator:
    """
    Validates configuration values and provides defaults.

    Responsibilities:
    - Validate configuration values
    - Normalize weights and thresholds
    - Provide default values for invalid inputs
    """

    def __init__(self):
        self._logger = logging.getLogger("gaap.config.validator")

    # Default critics configuration - used for weight normalization
    DEFAULT_CRITICS = [
        CriticConfig(name="logic", weight=0.35),
        CriticConfig(name="security", weight=0.25),
        CriticConfig(name="performance", weight=0.20),
        CriticConfig(name="style", weight=0.10),
        CriticConfig(name="compliance", weight=0.05),
        CriticConfig(name="ethics", weight=0.05),
    ]

    def validate(self, config: GAAPConfig) -> tuple[list[str], list[str]]:
        """
        Validate configuration and return warnings and errors.

        Returns:
            Tuple of (warnings, errors)
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Validate environment
        valid_environments = {"development", "staging", "production"}
        if config.system.environment not in valid_environments:
            errors.append(
                f"Invalid environment: {config.system.environment}. "
                f"Must be one of: {valid_environments}"
            )

        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.system.log_level.upper() not in valid_log_levels:
            warnings.append(f"Invalid log_level: {config.system.log_level}, defaulting to INFO")
            config.system.log_level = "INFO"

        # Validate budget
        self._validate_budget(config, warnings)

        # Validate and normalize critics
        self._validate_and_normalize_critics(config, warnings)

        # Validate sandbox type
        valid_sandbox_types = {"gvisor", "docker", "process"}
        if config.security.sandbox_type not in valid_sandbox_types:
            warnings.append(
                f"Invalid sandbox_type: {config.security.sandbox_type}, defaulting to gvisor"
            )
            config.security.sandbox_type = "gvisor"

        return warnings, errors

    def _validate_budget(self, config: GAAPConfig, warnings: list[str]) -> None:
        """Validate budget configuration"""
        if config.budget.monthly_limit <= 0:
            warnings.append("budget.monthly_limit must be positive, using default 5000.0")
            config.budget.monthly_limit = 5000.0
        if config.budget.daily_limit <= 0:
            warnings.append("budget.daily_limit must be positive, using default 200.0")
            config.budget.daily_limit = 200.0
        if config.budget.daily_limit > config.budget.monthly_limit:
            warnings.append("budget.daily_limit cannot exceed budget.monthly_limit, adjusting")
            config.budget.daily_limit = config.budget.monthly_limit

        # Validate alert thresholds
        if config.budget.alert_thresholds:
            if not all(0 <= t <= 1 for t in config.budget.alert_thresholds):
                warnings.append("budget.alert_thresholds must be between 0 and 1, using defaults")
                config.budget.alert_thresholds = [0.5, 0.8, 0.95]

    def _validate_and_normalize_critics(self, config: GAAPConfig, warnings: list[str]) -> None:
        """Validate and normalize critic weights"""
        enabled_critics = [c for c in config.quality_panel.critics if c.enabled]

        if not enabled_critics:
            warnings.append("No enabled critics found, using defaults")
            config.quality_panel.critics = list(self.DEFAULT_CRITICS)
            return

        total_weight = sum(c.weight for c in enabled_critics)

        if abs(total_weight - 1.0) > 0.01:
            if total_weight > 0:
                warnings.append(
                    f"quality_panel critics weights sum to {total_weight:.2f}, normalizing to 1.0"
                )
                # Normalize weights
                for c in enabled_critics:
                    c.weight = c.weight / total_weight
            else:
                warnings.append("quality_panel critics weights sum to 0, using defaults")
                config.quality_panel.critics = list(self.DEFAULT_CRITICS)

    def normalize_weights(self, critics: list[CriticConfig]) -> list[CriticConfig]:
        """
        Normalize critic weights to sum to 1.0.

        This is a utility method that can be used independently
        of the full validation process.
        """
        enabled = [c for c in critics if c.enabled]
        if not enabled:
            return list(self.DEFAULT_CRITICS)

        total = sum(c.weight for c in enabled)
        if total <= 0:
            return list(self.DEFAULT_CRITICS)

        if abs(total - 1.0) > 0.01:
            for c in enabled:
                c.weight = round(c.weight / total, 4)

        return critics


# =============================================================================
# ConfigConverter - Converts dictionaries to GAAPConfig objects
# =============================================================================


class ConfigConverter:
    """Converts configuration dictionaries to GAAPConfig objects"""

    @staticmethod
    def dict_to_config(config_dict: dict[str, Any]) -> GAAPConfig:
        """Convert a configuration dictionary to a GAAPConfig object"""
        # Create sub-configurations
        system = SystemConfig(**config_dict.get("system", {}))
        firewall = FirewallConfig(**config_dict.get("firewall", {}))
        parser = ParserConfig(**config_dict.get("parser", {}))
        strategic_planner = StrategicPlannerConfig(**config_dict.get("strategic_planner", {}))
        resource_allocator = ResourceAllocatorConfig(**config_dict.get("resource_allocator", {}))
        tactical_decomposer = TacticalDecomposerConfig(**config_dict.get("tactical_decomposer", {}))
        execution = ExecutionConfig(**config_dict.get("execution", {}))

        # Process quality panel
        quality_panel = ConfigConverter._process_quality_panel(config_dict.get("quality_panel", {}))

        external_connectors = ExternalConnectorsConfig(**config_dict.get("external_connectors", {}))
        security = SecurityConfig(**config_dict.get("security", {}))
        budget = BudgetConfig(**config_dict.get("budget", {}))
        context_management = ContextManagementConfig(**config_dict.get("context_management", {}))
        sovereign = SovereignConfig(**config_dict.get("sovereign", {}))
        performance_monitor = PerformanceMonitorConfig(**config_dict.get("performance_monitor", {}))

        # Process providers
        providers_list = [ProviderSettings(**p) for p in config_dict.get("providers", [])]

        return GAAPConfig(
            system=system,
            firewall=firewall,
            parser=parser,
            strategic_planner=strategic_planner,
            resource_allocator=resource_allocator,
            tactical_decomposer=tactical_decomposer,
            execution=execution,
            quality_panel=quality_panel,
            external_connectors=external_connectors,
            security=security,
            budget=budget,
            context_management=context_management,
            sovereign=sovereign,
            performance_monitor=performance_monitor,
            providers=providers_list,
            custom=config_dict.get("custom", {}),
        )

    @staticmethod
    def _process_quality_panel(quality_dict: dict) -> QualityPanelConfig:
        """Process quality panel configuration with defaults"""
        critics_list = []
        for c in quality_dict.get("critics", []):
            critics_list.append(CriticConfig(**c))

        # Use defaults if empty
        if not critics_list:
            critics_list = [
                CriticConfig(name="logic", weight=0.35),
                CriticConfig(name="security", weight=0.25),
                CriticConfig(name="performance", weight=0.20),
                CriticConfig(name="style", weight=0.10),
                CriticConfig(name="compliance", weight=0.05),
                CriticConfig(name="ethics", weight=0.05),
            ]

        quality_panel = QualityPanelConfig(critics=critics_list)
        if "min_approval_score" in quality_dict:
            quality_panel.min_approval_score = quality_dict["min_approval_score"]
        if "unanimous_required_for_critical" in quality_dict:
            quality_panel.unanimous_required_for_critical = quality_dict[
                "unanimous_required_for_critical"
            ]

        return quality_panel


# =============================================================================
# ConfigManager - Coordinates configuration loading and management
# =============================================================================


class ConfigManager:
    """
    Configuration manager - coordinates loading, validation, and access.

    Refactored from a God Class to use specialized components:
    - ConfigLoader: Handles file I/O and environment variables
    - ConfigValidator: Validates and normalizes values
    - ConfigConverter: Converts dict to GAAPConfig objects

    Features:
    - Thread-safe singleton
    - Hot reload support
    - Change watchers
    """

    _instance: "ConfigManager | None" = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        """Singleton pattern to ensure single ConfigManager instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, config_path: str | None = None, env_prefix: str = "GAAP_", auto_reload: bool = False
    ):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._config: GAAPConfig | None = None
        self._config_path = config_path
        self._env_prefix = env_prefix
        self._auto_reload = auto_reload
        self._file_hash: str | None = None
        self._watchers: list[Callable[[GAAPConfig], None]] = []
        self._layer_configs: dict[str, dict[str, Any]] = {}

        # Initialize components
        self._loader = ConfigLoader(env_prefix=env_prefix)
        self._validator = ConfigValidator()
        self._converter = ConfigConverter()

        # Load configuration
        self._load_config()
        self._initialized = True

    def _load_config(self) -> None:
        """Load configuration from all sources"""
        # 1. Start with empty config
        config_dict: dict[str, Any] = {}

        # 2. Load from file if provided
        if self._config_path:
            file_config = self._loader.load_from_file(self._config_path)
            config_dict = self._loader.deep_merge(config_dict, file_config)
            self._file_hash = self._loader.calculate_file_hash(self._config_path)

        # 3. Merge environment variables
        env_config = self._loader.load_from_env()
        config_dict = self._loader.deep_merge(config_dict, env_config)

        # 4. Convert to GAAPConfig
        self._config = self._converter.dict_to_config(config_dict)

        # 5. Validate
        self._validate_config(self._config)

    def _validate_config(self, config: GAAPConfig) -> None:
        """Validate configuration and raise on errors"""
        warnings, errors = self._validator.validate(config)

        for warning in warnings:
            logging.getLogger("gaap.config").warning(warning)

        if errors:
            raise ConfigurationError(
                message="Configuration validation failed",
                details={"errors": errors},
                suggestions=["Fix the configuration errors listed above"],
            )

    def reload(self) -> None:
        """Reload configuration from sources"""
        old_config = self._config
        self._load_config()

        # Notify watchers
        if old_config != self._config and self._config is not None:
            for watcher in self._watchers:
                watcher(self._config)

    def check_and_reload(self) -> bool:
        """Check for changes and reload if necessary"""
        if not self._config_path or not self._auto_reload:
            return False

        current_hash = self._loader.calculate_file_hash(self._config_path)
        if current_hash != self._file_hash:
            self.reload()
            return True
        return False

    def add_watcher(self, callback: Callable[[GAAPConfig], None]) -> None:
        """Add a configuration change watcher"""
        self._watchers.append(callback)

    def remove_watcher(self, callback: Callable[[GAAPConfig], None]) -> None:
        """Remove a configuration change watcher"""
        if callback in self._watchers:
            self._watchers.remove(callback)

    @property
    def config(self) -> GAAPConfig:
        """Get current configuration"""
        if self._config is None:
            self._load_config()
        if self._config is None:
            raise ConfigLoadError(
                config_path=self._config_path or "", reason="Failed to load config"
            )
        return self._config

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get a value from configuration using dot notation.

        Args:
            path: Dot-separated path (e.g., "system.log_level")
            default: Default value if not found

        Returns:
            The requested value or default
        """
        config_dict = self.config.to_dict()
        keys = path.split(".")
        current = config_dict

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set_layer_config(self, layer: str, config: dict[str, Any]) -> None:
        """Set layer-specific configuration"""
        self._layer_configs[layer] = config

    def get_layer_config(self, layer: str) -> dict[str, Any]:
        """Get layer-specific configuration"""
        return self._layer_configs.get(layer, {})

    def get_api_key(self, provider: str) -> str | None:
        """Get API key for a provider"""
        # Search in providers first
        for p in self.config.providers:
            if p.name.lower() == provider.lower():
                return p.api_key

        # Search in environment variables
        env_mappings = {
            "openai": f"{self._env_prefix}OPENAI_API_KEY",
            "anthropic": f"{self._env_prefix}ANTHROPIC_API_KEY",
            "kimi": f"{self._env_prefix}GROQ_API_KEY",
            "gemini": f"{self._env_prefix}GEMINI_API_KEY",
            "perplexity": f"{self._env_prefix}PERPLEXITY_API_KEY",
            "deepinfra": f"{self._env_prefix}DEEPINFRA_API_KEY",
            "together": f"{self._env_prefix}TOGETHER_API_KEY",
        }

        env_var = env_mappings.get(provider.lower())
        if env_var:
            return os.environ.get(env_var)

        return None

    def get_enabled_providers(self) -> list[ProviderSettings]:
        """Get list of enabled providers"""
        return [p for p in self.config.providers if p.enabled]

    def is_production(self) -> bool:
        """Check if environment is production"""
        return self.config.system.environment == "production"

    def is_development(self) -> bool:
        """Check if environment is development"""
        return self.config.system.environment == "development"


# =============================================================================
# ConfigBuilder - Fluent interface for building configuration
# =============================================================================


class ConfigBuilder:
    """
    Configuration builder - fluent interface for building configuration programmatically.

    Example:
        config = (ConfigBuilder()
            .with_system(name="MyApp", environment="production")
            .with_budget(monthly=1000)
            .with_provider("kimi", api_key="...")
            .build())
    """

    def __init__(self) -> None:
        self._config_dict: dict[str, Any] = {}

    def with_system(
        self, name: str = "GAAP-App", environment: str = "development", log_level: str = "INFO"
    ) -> "ConfigBuilder":
        """Add system configuration"""
        self._config_dict["system"] = {
            "name": name,
            "environment": environment,
            "log_level": log_level,
        }
        return self

    def with_budget(
        self, monthly: float = 1000.0, daily: float = 50.0, per_task: float = 5.0
    ) -> "ConfigBuilder":
        """Add budget configuration"""
        self._config_dict["budget"] = {
            "monthly_limit": monthly,
            "daily_limit": daily,
            "per_task_limit": per_task,
        }
        return self

    def with_provider(
        self,
        name: str,
        api_key: str | None = None,
        models: list[str] | None = None,
        default_model: str = "",
        priority: int = 0,
    ) -> "ConfigBuilder":
        """Add a provider"""
        if "providers" not in self._config_dict:
            self._config_dict["providers"] = []

        provider = {
            "name": name,
            "api_key": api_key,
            "models": models or [],
            "default_model": default_model,
            "priority": priority,
            "enabled": True,
        }
        self._config_dict["providers"].append(provider)
        return self

    def with_security(
        self, sandbox_type: str = "gvisor", audit_enabled: bool = True
    ) -> "ConfigBuilder":
        """Add security configuration"""
        self._config_dict["security"] = {
            "sandbox_type": sandbox_type,
            "blockchain_audit_enabled": audit_enabled,
        }
        return self

    def with_execution(
        self, max_parallel: int = 10, genetic_twin: bool = True, self_healing: bool = True
    ) -> "ConfigBuilder":
        """Add execution configuration"""
        self._config_dict["execution"] = {
            "max_parallel_tasks": max_parallel,
            "genetic_twin_enabled": genetic_twin,
            "self_healing_enabled": self_healing,
        }
        return self

    def with_custom(self, key: str, value: Any) -> "ConfigBuilder":
        """Add custom configuration"""
        if "custom" not in self._config_dict:
            self._config_dict["custom"] = {}
        self._config_dict["custom"][key] = value
        return self

    def from_file(self, path: str) -> "ConfigBuilder":
        """Load from file as base"""
        loader = ConfigLoader()
        file_config = loader.load_from_file(path)
        self._config_dict = loader.deep_merge(file_config, self._config_dict)
        return self

    def with_secrets(
        self,
        env_prefix: str = "GAAP_",
        env_file: str | None = ".env",
        auto_load: bool = True,
    ) -> "ConfigBuilder":
        """Initialize secrets management"""
        self._config_dict["_secrets_config"] = {
            "env_prefix": env_prefix,
            "env_file": env_file,
            "auto_load": auto_load,
        }
        return self

    def build(self) -> GAAPConfig:
        """Build final configuration"""
        converter = ConfigConverter()
        config = converter.dict_to_config(self._config_dict)

        # Initialize secrets if configured
        secrets_config = self._config_dict.get("_secrets_config")
        if secrets_config and _HAS_SECRETS:
            config.init_secrets(
                env_prefix=secrets_config.get("env_prefix", "GAAP_"),
                env_file=secrets_config.get("env_file", ".env"),
                auto_load=secrets_config.get("auto_load", True),
            )

        # Validate
        validator = ConfigValidator()
        warnings, errors = validator.validate(config)
        for warning in warnings:
            logging.getLogger("gaap.config").warning(warning)

        if errors:
            raise ConfigurationError(
                message="Configuration validation failed",
                details={"errors": errors},
            )

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration as YAML file"""
        config = self.build()
        content = yaml.dump(config.to_dict(), default_flow_style=False)
        atomic_write(path, content)

    def to_json(self, path: str) -> None:
        """Save configuration as JSON file"""
        config = self.build()
        content = json.dumps(config.to_dict(), indent=2)
        atomic_write(path, content)


# =============================================================================
# Convenience Functions
# =============================================================================


def get_default_config() -> GAAPConfig:
    """Get default configuration"""
    return GAAPConfig()


def load_config(config_path: str | None = None, env_prefix: str = "GAAP_") -> GAAPConfig:
    """
    Load configuration from available sources.

    Args:
        config_path: Path to configuration file (optional)
        env_prefix: Environment variable prefix

    Returns:
        Configuration object
    """
    manager = ConfigManager(config_path=config_path, env_prefix=env_prefix)
    return manager.config


# =============================================================================
# Global Config Instance
# =============================================================================

_config_manager: ConfigManager | None = None


def init_config(
    config_path: str | None = None, env_prefix: str = "GAAP_", auto_reload: bool = False
) -> ConfigManager:
    """Initialize global configuration manager"""
    global _config_manager
    _config_manager = ConfigManager(
        config_path=config_path, env_prefix=env_prefix, auto_reload=auto_reload
    )
    return _config_manager


def get_config() -> GAAPConfig:
    """Get global configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
