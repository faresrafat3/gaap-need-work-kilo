import hashlib
import json
import os
import threading
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .exceptions import (
    ConfigLoadError,
    ConfigurationError,
)

# =============================================================================
# Configuration Dataclasses
# =============================================================================


@dataclass
class FirewallConfig:
    """تكوين جدار الحماية (Layer 0)"""

    strictness: str = "high"  # low, medium, high, paranoid
    adversarial_test_enabled: bool = True
    blocked_patterns_update_interval: str = "6h"
    max_input_length: int = 100000
    enable_behavioral_analysis: bool = True
    enable_semantic_analysis: bool = True

    # طبقات الفحص السبعة
    layer1_surface_enabled: bool = True
    layer2_lexical_enabled: bool = True
    layer3_syntactic_enabled: bool = True
    layer4_semantic_enabled: bool = True
    layer5_contextual_enabled: bool = True
    layer6_behavioral_enabled: bool = True
    layer7_adversarial_enabled: bool = True


@dataclass
class ParserConfig:
    """تكوين محلل الطلبات"""

    default_model: str = "gpt-4o-mini"
    routing_strategy: str = "quality_first"  # speed_first, cost_optimized, quality_first
    extract_implicit_requirements: bool = True
    max_classification_retries: int = 2
    confidence_threshold: float = 0.85


@dataclass
class StrategicPlannerConfig:
    """تكوين المخطط الاستراتيجي (Layer 1)"""

    tot_depth: int = 5
    branching_factor: int = 4
    mad_debate_rounds: int = 3
    consensus_threshold: float = 0.85
    enable_external_research: bool = True
    max_parallel_options: int = 5


@dataclass
class ResourceAllocatorConfig:
    """تكوين موجه الموارد"""

    tier_1_model: str = "claude-3-5-opus"
    tier_2_model: str = "gpt-4o"
    tier_3_model: str = "gpt-4o-mini"
    tier_4_model: str = "llama-3-70b"
    allow_local_fallbacks: bool = True
    cost_tracking_enabled: bool = True
    auto_downgrade_on_budget: bool = True


@dataclass
class TacticalDecomposerConfig:
    """تكوين المحلل التكتيكي (Layer 2)"""

    max_subtasks: int = 50
    max_task_size_lines: int = 500
    max_task_time_minutes: int = 10
    enable_smart_batching: bool = True
    dependency_analysis_enabled: bool = True
    critical_path_analysis: bool = True


@dataclass
class SovereignConfig:
    """تكوين السيادة والذكاء المتقدم (v2.1)"""

    # Cognitive Features
    enable_deep_research: bool = True  # تفعيل البحث العميق
    enable_tool_synthesis: bool = True  # تفعيل تصنيع الأدوات
    enable_ghost_fs: bool = True  # تفعيل محاكاة نظام الملفات

    # Security Features
    enable_dlp_shield: bool = True  # تفعيل درع منع التسريب
    enable_semantic_firewall: bool = True  # تفعيل الجدار الدلالي

    # Thresholds
    dlp_strictness: str = "high"  # low, medium, high
    research_depth: int = 3  # عمق البحث (عدد الخطوات)


@dataclass
class ExecutionConfig:
    """تكوين التنفيذ (Layer 3)"""

    max_parallel_tasks: int = 10
    genetic_twin_enabled: bool = True
    genetic_twin_for_critical_only: bool = True
    self_healing_enabled: bool = True
    self_healing_max_retries: int = 3
    consciousness_migration_enabled: bool = True
    checkpoint_interval_seconds: int = 30
    sandbox_type: str = "gvisor"  # gvisor, docker, process


@dataclass
class CriticConfig:
    """تكوين ناقد واحد"""

    name: str
    weight: float = 1.0
    enabled: bool = True
    custom_rules: list[str] = field(default_factory=list)


@dataclass
class QualityPanelConfig:
    """تكوين لجنة الجودة"""

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
    """تكوين الأمان"""

    sandbox_type: str = "gvisor"
    capability_tokens_ttl: int = 300  # seconds
    blockchain_audit_enabled: bool = True
    audit_storage_provider: str = "local"  # local, s3, gcs
    audit_storage_path: str = "./audit_logs"
    encryption_enabled: bool = True
    encryption_algorithm: str = "AES-256-GCM"

    # Seccomp filters
    seccomp_enabled: bool = True
    seccomp_default_action: str = "kill"

    # Cgroups limits
    cgroup_cpu_max: float = 0.5  # 50% of one core
    cgroup_memory_max: str = "512M"
    cgroup_pids_max: int = 100


@dataclass
class BudgetConfig:
    """تكوين الميزانية"""

    monthly_limit: float = 5000.0  # USD
    daily_limit: float = 200.0
    per_task_limit: float = 10.0
    alert_thresholds: list[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])
    auto_throttle_at: float = 0.9
    hard_stop_at: float = 1.0
    cost_optimization_mode: str = "balanced"  # aggressive, balanced, quality_first


@dataclass
class ContextManagementConfig:
    """تكوين إدارة السياق"""

    default_budget_level: str = "medium"
    hierarchical_loading_enabled: bool = True
    pkg_agent_enabled: bool = False  # للمشاريع الضخمة فقط
    smart_chunking_enabled: bool = True
    external_brain_enabled: bool = True
    context_cache_enabled: bool = True
    context_cache_size_mb: int = 100
    parallel_loading_enabled: bool = True
    garbage_collection_enabled: bool = True
    gc_interval_seconds: int = 300


@dataclass
class ProviderSettings:
    """إعدادات مزود واحد"""

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
    """تكوين الموصلات الخارجية"""

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
    """التكوين الأساسي للنظام"""

    name: str = "GAAP-Production-Alpha"
    environment: str = "production"  # development, staging, production
    version: str = "1.0.0"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "json"  # json, text
    metrics_enabled: bool = True
    metrics_port: int = 9090
    health_check_enabled: bool = True
    health_check_port: int = 8080


@dataclass
class GAAPConfig:
    """التكوين الشامل لنظام GAAP"""

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
    providers: list[ProviderSettings] = field(default_factory=list)

    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """تحويل التكوين إلى قاموس"""

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


# =============================================================================
# Configuration Manager
# =============================================================================


class ConfigManager:
    """
    مدير التكوين - يوفر وصولاً موحداً للتكوين

    الميزات:
    - تحميل من ملفات YAML/JSON
    - دعم متغيرات البيئة
    - تكوين هرمي (توريث)
    - تحقق من الصحة
    - إعادة تحميل ساخنة
    - خيط آمن (Thread-safe)
    """

    _instance: "ConfigManager | None" = None
    _lock = threading.Lock()
    _initialized: bool = False

    def __new__(cls, *args: Any, **kwargs: Any) -> "ConfigManager":
        """نمط Singleton للتأكد من وجود مدير تكوين واحد"""
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

        # تحميل التكوين
        self._load_config()
        self._initialized = True

    def _load_config(self) -> None:
        """تحميل التكوين من المصادر المتاحة"""
        # 1. تحميل التكوين الافتراضي
        config_dict: dict[str, Any] = {}

        # 2. تحميل من الملف إذا وُجد
        if self._config_path:
            file_config = self._load_from_file(self._config_path)
            config_dict = self._deep_merge(config_dict, file_config)
            self._file_hash = self._calculate_file_hash(self._config_path)

        # 3. تحميل من متغيرات البيئة
        env_config = self._load_from_env()
        config_dict = self._deep_merge(config_dict, env_config)

        # 4. إنشاء كائن التكوين
        self._config = self._dict_to_config(config_dict)

        # 5. التحقق من الصحة
        self._validate_config(self._config)

    def _load_from_file(self, path: str) -> dict[str, Any]:
        """تحميل التكوين من ملف"""
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

    def _load_from_env(self) -> dict[str, Any]:
        """تحميل التكوين من متغيرات البيئة"""
        config: dict[str, Any] = {}

        # تعيينات متغيرات البيئة
        env_mappings = {
            # النظام
            f"{self._env_prefix}SYSTEM_NAME": ("system", "name"),
            f"{self._env_prefix}ENVIRONMENT": ("system", "environment"),
            f"{self._env_prefix}LOG_LEVEL": ("system", "log_level"),
            # الميزانية
            f"{self._env_prefix}BUDGET_MONTHLY_LIMIT": ("budget", "monthly_limit"),
            f"{self._env_prefix}BUDGET_DAILY_LIMIT": ("budget", "daily_limit"),
            # الموارد
            f"{self._env_prefix}TIER_1_MODEL": ("resource_allocator", "tier_1_model"),
            f"{self._env_prefix}TIER_2_MODEL": ("resource_allocator", "tier_2_model"),
            f"{self._env_prefix}TIER_3_MODEL": ("resource_allocator", "tier_3_model"),
            # الأمان
            f"{self._env_prefix}SANDBOX_TYPE": ("security", "sandbox_type"),
            f"{self._env_prefix}AUDIT_ENABLED": ("security", "blockchain_audit_enabled"),
            # التنفيذ
            f"{self._env_prefix}MAX_PARALLEL_TASKS": ("execution", "max_parallel_tasks"),
            f"{self._env_prefix}GENETIC_TWIN_ENABLED": ("execution", "genetic_twin_enabled"),
            # مفاتيح API
            f"{self._env_prefix}OPENAI_API_KEY": ("_api_keys", "openai"),
            f"{self._env_prefix}ANTHROPIC_API_KEY": ("_api_keys", "anthropic"),
            f"{self._env_prefix}GROQ_API_KEY": ("_api_keys", "groq"),
            f"{self._env_prefix}GEMINI_API_KEY": ("_api_keys", "gemini"),
            f"{self._env_prefix}PERPLEXITY_API_KEY": ("_api_keys", "perplexity"),
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config, config_path, value)

        return config

    def _set_nested(self, config: dict[str, Any], path: tuple, value: Any) -> None:
        """تعيين قيمة في مسار متداخل"""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """دمج عميق لقاموسين"""
        result = deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def _dict_to_config(self, config_dict: dict[str, Any]) -> GAAPConfig:
        """تحويل القاموس إلى كائن تكوين"""
        # إنشاء كائنات فرعية
        system = SystemConfig(**config_dict.get("system", {}))
        firewall = FirewallConfig(**config_dict.get("firewall", {}))
        parser = ParserConfig(**config_dict.get("parser", {}))
        strategic_planner = StrategicPlannerConfig(**config_dict.get("strategic_planner", {}))
        resource_allocator = ResourceAllocatorConfig(**config_dict.get("resource_allocator", {}))
        tactical_decomposer = TacticalDecomposerConfig(**config_dict.get("tactical_decomposer", {}))
        execution = ExecutionConfig(**config_dict.get("execution", {}))

        # معالجة لجنة الجودة
        quality_dict = config_dict.get("quality_panel", {})
        critics_list = []
        for c in quality_dict.get("critics", []):
            critics_list.append(CriticConfig(**c))
        quality_panel = QualityPanelConfig(critics=critics_list)
        if "min_approval_score" in quality_dict:
            quality_panel.min_approval_score = quality_dict["min_approval_score"]
        if "unanimous_required_for_critical" in quality_dict:
            quality_panel.unanimous_required_for_critical = quality_dict[
                "unanimous_required_for_critical"
            ]

        external_connectors = ExternalConnectorsConfig(**config_dict.get("external_connectors", {}))
        security = SecurityConfig(**config_dict.get("security", {}))
        budget = BudgetConfig(**config_dict.get("budget", {}))
        context_management = ContextManagementConfig(**config_dict.get("context_management", {}))

        # معالجة المزودين
        providers_list = []
        for p in config_dict.get("providers", []):
            providers_list.append(ProviderSettings(**p))

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
            providers=providers_list,
            custom=config_dict.get("custom", {}),
        )

    def _validate_config(self, config: GAAPConfig) -> None:
        """التحقق من صحة التكوين"""
        errors: list[str] = []

        # التحقق من البيئة
        valid_environments = {"development", "staging", "production"}
        if config.system.environment not in valid_environments:
            errors.append(
                f"Invalid environment: {config.system.environment}. "
                f"Must be one of: {valid_environments}"
            )

        # التحقق من مستوى السجل
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if config.system.log_level.upper() not in valid_log_levels:
            errors.append(
                f"Invalid log_level: {config.system.log_level}. Must be one of: {valid_log_levels}"
            )

        # التحقق من الميزانية
        if config.budget.monthly_limit <= 0:
            errors.append("budget.monthly_limit must be positive")
        if config.budget.daily_limit <= 0:
            errors.append("budget.daily_limit must be positive")
        if config.budget.daily_limit > config.budget.monthly_limit:
            errors.append("budget.daily_limit cannot exceed budget.monthly_limit")

        # التحقق من عتبات التنبيه
        if not all(0 <= t <= 1 for t in config.budget.alert_thresholds):
            errors.append("budget.alert_thresholds must be between 0 and 1")

        # التحقق من صلاحيات النقاد
        total_weight = sum(c.weight for c in config.quality_panel.critics if c.enabled)
        if abs(total_weight - 1.0) > 0.01:
            errors.append(f"quality_panel critics weights must sum to 1.0, got {total_weight}")

        # التحقق من نوع الـ sandbox
        valid_sandbox_types = {"gvisor", "docker", "process"}
        if config.security.sandbox_type not in valid_sandbox_types:
            errors.append(
                f"Invalid sandbox_type: {config.security.sandbox_type}. "
                f"Must be one of: {valid_sandbox_types}"
            )

        if errors:
            raise ConfigurationError(
                message="Configuration validation failed",
                details={"errors": errors},
                suggestions=["Fix the configuration errors listed above"],
            )

    def _calculate_file_hash(self, path: str) -> str:
        """حساب hash للملف للتحقق من التغييرات"""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def reload(self) -> None:
        """إعادة تحميل التكوين"""
        old_config = self._config
        self._load_config()

        # إشعار المراقبين
        if old_config != self._config and self._config is not None:
            for watcher in self._watchers:
                watcher(self._config)

    def check_and_reload(self) -> bool:
        """التحقق من التغييرات وإعادة التحميل إذا لزم الأمر"""
        if not self._config_path or not self._auto_reload:
            return False

        current_hash = self._calculate_file_hash(self._config_path)
        if current_hash != self._file_hash:
            self.reload()
            return True
        return False

    def add_watcher(self, callback: Callable[[GAAPConfig], None]) -> None:
        """إضافة مراقب للتغييرات في التكوين"""
        self._watchers.append(callback)

    def remove_watcher(self, callback: Callable[[GAAPConfig], None]) -> None:
        """إزالة مراقب"""
        if callback in self._watchers:
            self._watchers.remove(callback)

    @property
    def config(self) -> GAAPConfig:
        """الحصول على التكوين الحالي"""
        if self._config is None:
            self._load_config()
        if self._config is None:
            raise ConfigLoadError(
                config_path=self._config_path or "", reason="Failed to load config"
            )
        return self._config

    def get(self, path: str, default: Any = None) -> Any:
        """
        الحصول على قيمة من التكوين باستخدام مسار

        Args:
            path: المسار مفصول بنقاط (مثل: "system.log_level")
            default: القيمة الافتراضية إذا لم يوجد

        Returns:
            القيمة المطلوبة أو القيمة الافتراضية
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
        """تعيين تكوين خاص بطبقة معينة"""
        self._layer_configs[layer] = config

    def get_layer_config(self, layer: str) -> dict[str, Any]:
        """الحصول على تكوين طبقة معينة"""
        return self._layer_configs.get(layer, {})

    def get_api_key(self, provider: str) -> str | None:
        """الحصول على مفتاح API لمزود معين"""
        # البحث أولاً في المزودين
        for p in self.config.providers:
            if p.name.lower() == provider.lower():
                return p.api_key

        # البحث في متغيرات البيئة
        env_mappings = {
            "openai": f"{self._env_prefix}OPENAI_API_KEY",
            "anthropic": f"{self._env_prefix}ANTHROPIC_API_KEY",
            "groq": f"{self._env_prefix}GROQ_API_KEY",
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
        """الحصول على قائمة المزودين المفعّلين"""
        return [p for p in self.config.providers if p.enabled]

    def is_production(self) -> bool:
        """هل البيئة إنتاجية؟"""
        return self.config.system.environment == "production"

    def is_development(self) -> bool:
        """هل البيئة تطويرية؟"""
        return self.config.system.environment == "development"


# =============================================================================
# Configuration Builder (Fluent Interface)
# =============================================================================


class ConfigBuilder:
    """
    منشئ التكوين - واجهة fluent لبناء التكوين برمجياً

    Example:
        config = (ConfigBuilder()
            .with_system(name="MyApp", environment="production")
            .with_budget(monthly=1000)
            .with_provider("groq", api_key="...")
            .build())
    """

    def __init__(self) -> None:
        self._config_dict: dict[str, Any] = {}

    def with_system(
        self, name: str = "GAAP-App", environment: str = "development", log_level: str = "INFO"
    ) -> "ConfigBuilder":
        """إضافة تكوين النظام"""
        self._config_dict["system"] = {
            "name": name,
            "environment": environment,
            "log_level": log_level,
        }
        return self

    def with_budget(
        self, monthly: float = 1000.0, daily: float = 50.0, per_task: float = 5.0
    ) -> "ConfigBuilder":
        """إضافة تكوين الميزانية"""
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
        """إضافة مزود"""
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
        """إضافة تكوين الأمان"""
        self._config_dict["security"] = {
            "sandbox_type": sandbox_type,
            "blockchain_audit_enabled": audit_enabled,
        }
        return self

    def with_execution(
        self, max_parallel: int = 10, genetic_twin: bool = True, self_healing: bool = True
    ) -> "ConfigBuilder":
        """إضافة تكوين التنفيذ"""
        self._config_dict["execution"] = {
            "max_parallel_tasks": max_parallel,
            "genetic_twin_enabled": genetic_twin,
            "self_healing_enabled": self_healing,
        }
        return self

    def with_custom(self, key: str, value: Any) -> "ConfigBuilder":
        """إضافة تكوين مخصص"""
        if "custom" not in self._config_dict:
            self._config_dict["custom"] = {}
        self._config_dict["custom"][key] = value
        return self

    def from_file(self, path: str) -> "ConfigBuilder":
        """تحميل من ملف كأساس"""
        if path.endswith((".yaml", ".yml")):
            with open(path) as f:
                file_config = yaml.safe_load(f) or {}
        elif path.endswith(".json"):
            with open(path) as f:
                file_config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        # دمج عميق
        self._config_dict = self._deep_merge(file_config, self._config_dict)
        return self

    def _deep_merge(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """دمج عميق"""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    def build(self) -> GAAPConfig:
        """بناء التكوين النهائي"""
        # إنشاء ConfigManager مؤقت للتحويل والتحقق
        manager = ConfigManager.__new__(ConfigManager)
        manager._env_prefix = "GAAP_"
        config = manager._dict_to_config(self._config_dict)
        manager._validate_config(config)
        return config

    def to_yaml(self, path: str) -> None:
        """حفظ التكوين كملف YAML"""
        config = self.build()
        with open(path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)

    def to_json(self, path: str) -> None:
        """حفظ التكوين كملف JSON"""
        config = self.build()
        with open(path, "w") as f:
            json.dump(config.to_dict(), f, indent=2)


# =============================================================================
# Default Configuration
# =============================================================================


def get_default_config() -> GAAPConfig:
    """الحصول على التكوين الافتراضي"""
    return GAAPConfig()


def load_config(config_path: str | None = None, env_prefix: str = "GAAP_") -> GAAPConfig:
    """
    تحميل التكوين من المصادر المتاحة

    Args:
        config_path: مسار ملف التكوين (اختياري)
        env_prefix: بادئة متغيرات البيئة

    Returns:
        كائن التكوين
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
    """تهيئة مدير التكوين العام"""
    global _config_manager
    _config_manager = ConfigManager(
        config_path=config_path, env_prefix=env_prefix, auto_reload=auto_reload
    )
    return _config_manager


def get_config() -> GAAPConfig:
    """الحصول على التكوين العام"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """الحصول على مدير التكوين العام"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
