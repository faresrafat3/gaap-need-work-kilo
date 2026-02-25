"""
Core Base Classes for GAAP System

Provides foundational base classes for all GAAP components:

Classes:
    - BaseComponent: Base class for all components
    - BaseAgent: Base class for all agents
    - BaseLayer: Base class for all layers
    - BaseProvider: Base class for all providers

Usage:
    from gaap.core.base import BaseComponent, BaseAgent

    class MyComponent(BaseComponent):
        def initialize(self) -> None:
            self._is_initialized = True

        def shutdown(self) -> None:
            pass
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Generic,
    Optional,
    ParamSpec,
    TypeVar,
    cast,
)

from .exceptions import (
    GAAPException,
    MaxRetriesExceededError,
    TaskTimeoutError,
)
from .types import (
    AgentCapabilities,
    AgentIdentity,
    ChatCompletionResponse,
    ContextBudget,
    ContextLevel,
    ContextWindow,
    CriticEvaluation,
    CriticType,
    HealingLevel,
    IACPHeader,
    IACPMessage,
    IACPPayload,
    IACPSecurity,
    LayerType,
    Message,
    Task,
    TaskResult,
    TaskType,
)

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LOG_LEVEL = "INFO"
COMPONENT_ID_LENGTH = 8  # Length of truncated component ID in repr


# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


# =============================================================================
# Logging Setup
# =============================================================================


def setup_logger(name: str, level: str = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with standard configuration.

    Args:
        name: Logger name (typically module or component name)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance with stream handler

    Example:
        >>> logger = setup_logger("gaap.my_component")
        >>> logger.info("Component initialized")
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# Base Component
# =============================================================================


class BaseComponent(ABC):
    """
    Abstract base class for all GAAP components.

    Provides foundational functionality for all components including:
    - Unique component ID
    - Unified logging
    - Lifecycle management (initialize/shutdown)
    - Performance metrics tracking

    Attributes:
        id: Unique component identifier (UUID)
        name: Component name
        layer: Component layer in the architecture
        created_at: Component creation timestamp
        _logger: Logger instance for the component
        _metrics: Dictionary of recorded metrics
        _is_initialized: Whether component is initialized
        _is_running: Whether component is currently running

    Usage:
        >>> class MyComponent(BaseComponent):
        ...     def initialize(self) -> None:
        ...         self._is_initialized = True
        ...
        ...     def shutdown(self) -> None:
        ...         pass
        >>> component = MyComponent(name="my_component")
        >>> component.initialize()
    """

    def __init__(
        self,
        component_id: str | None = None,
        name: str | None = None,
        layer: LayerType = LayerType.EXECUTION,
    ) -> None:
        """
        Initialize base component.

        Args:
            component_id: Optional custom ID (auto-generated if not provided)
            name: Optional custom name (class name if not provided)
            layer: Architecture layer (default: EXECUTION)
        """
        self.id = component_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.layer = layer
        self.created_at = datetime.now()
        self._logger = setup_logger(f"gaap.{self.name}")
        self._metrics: dict[str, Any] = {}
        self._is_initialized = False
        self._is_running = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the component.

        Should be called before using the component.
        Sets up resources, connections, and state.

        Raises:
            Exception: If initialization fails
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """
        Shutdown the component.

        Should be called when component is no longer needed.
        Cleans up resources and closes connections.

        Raises:
            Exception: If shutdown fails
        """
        pass

    def record_metric(self, key: str, value: Any) -> None:
        """
        Record a metric with timestamp.

        Args:
            key: Metric name (e.g., "latency_ms", "tokens_used")
            value: Metric value (any JSON-serializable type)

        Example:
            >>> component.record_metric("latency_ms", 250.5)
            >>> component.record_metric("tokens_used", 100)
        """
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(
            {
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_metrics(self) -> dict[str, Any]:
        """
        Get all recorded metrics.

        Returns:
            Dictionary of metric names to lists of (value, timestamp) pairs

        Example:
            >>> metrics = component.get_metrics()
            >>> print(metrics["latency_ms"])
            [{'value': 250.5, 'timestamp': '2026-02-17T10:30:00'}]
        """
        return self._metrics.copy()

    def log(self, level: str, message: str, **kwargs: Any) -> None:
        """
        Log a message with component ID prefix.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message string
            **kwargs: Additional context data to include

        Example:
            >>> component.log("info", "Task started", task_id="123")
            >>> component.log("error", "Task failed", error="timeout")
        """
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(f"[{self.id}] {message}", extra=kwargs)

    def __repr__(self) -> str:
        """
        Get string representation of component.

        Returns:
            String with component name, ID, and layer
        """
        return f"<{self.name}(id={self.id[:COMPONENT_ID_LENGTH]}..., layer={self.layer.name})>"


# =============================================================================
# Base Agent
# =============================================================================
# Base Agent
# =============================================================================


class BaseAgent(BaseComponent, Generic[T, R]):
    """
    Abstract base class for all GAAP agents.

    Provides unified interface for agent execution with:
    - Memory management
    - Context tracking
    - Error handling
    - Self-healing capabilities
    - Checkpoint support

    Attributes:
        identity: Agent identity configuration
        capabilities: Agent capabilities
        _memory: Agent memory storage
        _context: Current execution context
        _current_task: Currently executing task
        _execution_history: History of executed tasks

    Type Parameters:
        T: Input type for agent operations
        R: Output type for agent operations

    Usage:
        >>> class MyAgent(BaseAgent):
        ...     async def execute(self, task: Task) -> TaskResult:
        ...         # Implementation here
        ...         pass
        >>> agent = MyAgent(identity, capabilities)
    """

    def __init__(
        self,
        identity: AgentIdentity,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        """
        Initialize base agent.

        Args:
            identity: Agent identity with name, ID, and layer
            capabilities: Optional agent capabilities (uses identity capabilities if not provided)
        """
        super().__init__(
            component_id=identity.id,
            name=identity.name,
            layer=identity.layer,
        )
        self.identity = identity
        self.capabilities = capabilities or identity.capabilities
        self._memory: list[dict[str, Any]] = []
        self._context: dict[str, Any] = {}
        self._current_task: Task | None = None
        self._execution_history: list[TaskResult] = []

    @abstractmethod
    async def execute(self, task: Task) -> TaskResult:
        """
        Execute a task asynchronously.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution outcome

        Raises:
            TaskError: If task execution fails
            TaskTimeoutError: If execution times out
        """
        pass

    async def validate_task(self, task: Task) -> bool:
        """
        Validate if agent can handle the task.

        Checks:
        1. Task type is supported
        2. Complexity level is manageable

        Args:
            task: Task to validate

        Returns:
            True if task can be handled, False otherwise
        """
        # Check type
        if task.type not in self._get_supported_task_types():
            self.log("warning", f"Unsupported task type: {task.type}")
            return False

        # Check complexity
        if not self._can_handle_complexity(task.complexity):
            self.log("warning", f"Cannot handle complexity: {task.complexity}")
            return False

        return True

    def _get_supported_task_types(self) -> list[TaskType]:
        """
        Get list of supported task types.

        Returns:
            List of TaskType values based on agent capabilities
        """
        types: list[TaskType] = []
        if self.capabilities.code_generation:
            types.append(TaskType.CODE_GENERATION)
        if self.capabilities.code_review:
            types.append(TaskType.CODE_REVIEW)
        if self.capabilities.debugging:
            types.append(TaskType.DEBUGGING)
        if self.capabilities.research:
            types.append(TaskType.RESEARCH)
        if self.capabilities.analysis:
            types.append(TaskType.ANALYSIS)
        if self.capabilities.planning:
            types.append(TaskType.PLANNING)
        if self.capabilities.testing:
            types.append(TaskType.TESTING)
        return types

    def _can_handle_complexity(self, complexity: Any) -> bool:
        """
        Check if agent can handle given complexity level.

        Args:
            complexity: Task complexity level

        Returns:
            True if complexity can be handled (override in subclasses)
        """
        return True  # Can be overridden in subclasses

    def add_to_memory(self, item: dict[str, Any]) -> None:
        """
        Add item to agent memory.

        Args:
            item: Memory item to store (will be timestamped)

        Example:
            >>> agent.add_to_memory({"event": "task_completed", "task_id": "123"})
        """
        item["timestamp"] = datetime.now().isoformat()
        item["agent_id"] = self.id
        self._memory.append(item)

    def get_memory(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent memory items.

        Args:
            limit: Maximum number of items to return (default: 100)

        Returns:
            List of most recent memory items
        """
        return self._memory[-limit:]

    def clear_memory(self) -> None:
        """Clear all agent memory."""
        self._memory.clear()

    def set_context(self, key: str, value: Any) -> None:
        """
        Set context value.

        Args:
            key: Context key
            value: Context value

        Example:
            >>> agent.set_context("user_id", "123")
        """
        self._context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get context value.

        Args:
            key: Context key to retrieve
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self._context.get(key, default)

    async def execute_with_timeout(
        self,
        task: Task,
        timeout_seconds: float = 300,
    ) -> TaskResult:
        """
        Execute task with timeout.

        Args:
            task: Task to execute
            timeout_seconds: Maximum execution time in seconds (default: 300)

        Returns:
            TaskResult with execution outcome

        Raises:
            TaskTimeoutError: If execution exceeds timeout
        """
        try:
            return await asyncio.wait_for(self.execute(task), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise TaskTimeoutError(
                task_id=task.id,
                timeout_seconds=timeout_seconds,
            )

    def create_checkpoint(self) -> dict[str, Any]:
        """
        Create agent checkpoint for recovery.

        Returns:
            Dictionary with agent state for recovery

        Example:
            >>> checkpoint = agent.create_checkpoint()
            >>> # Later...
            >>> agent.restore_from_checkpoint(checkpoint)
        """
        return {
            "agent_id": self.id,
            "timestamp": datetime.now().isoformat(),
            "memory": self._memory.copy(),
            "context": self._context.copy(),
            "current_task_id": self._current_task.id if self._current_task else None,
            "execution_history_count": len(self._execution_history),
        }

    def restore_from_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """
        Restore agent state from checkpoint.

        Args:
            checkpoint: Previously created checkpoint

        Example:
            >>> agent.restore_from_checkpoint(saved_checkpoint)
        """
        self._memory = checkpoint.get("memory", []).copy()
        self._context = checkpoint.get("context", {}).copy()
        self.log(
            "info",
            f"Restored from checkpoint at {checkpoint.get('timestamp')}",
        )


# =============================================================================
# Base Provider
# =============================================================================


class BaseProvider(BaseComponent):
    """
    الفئة الأساسية لمزودي النماذج

    توفر:
    - واجهة موحدة للاتصال بالنماذج
    - إدارة المهلات وإعادة المحاولة
    - تتبع الاستخدام
    """

    def __init__(
        self,
        name: str,
        models: list[str],
        provider_type: Any = None,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limit: int = 60,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        super().__init__(name=name, layer=LayerType.EXTERNAL)
        self.models = models
        self.provider_type = provider_type
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.max_retries = max_retries
        self._request_count = 0
        self._error_count = 0
        self._total_tokens_used = 0
        self._last_request_time: datetime | None = None

    @abstractmethod
    async def chat_completion(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> ChatCompletionResponse:
        """إكمال محادثة"""
        pass

    @abstractmethod
    async def stream_chat_completion(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """إكمال محادثة بتدفق"""
        pass

    def is_model_available(self, model: str) -> bool:
        """التحقق من توفر النموذج"""
        return model in self.models

    def get_available_models(self) -> list[str]:
        """الحصول على النماذج المتاحة"""
        return self.models.copy()

    def record_usage(self, tokens: int) -> None:
        """تسجيل الاستخدام"""
        self._total_tokens_used += tokens
        self._request_count += 1
        self._last_request_time = datetime.now()

    def record_error(self) -> None:
        """تسجيل خطأ"""
        self._error_count += 1

    def get_stats(self) -> dict[str, Any]:
        """الحصول على الإحصائيات"""
        return {
            "provider": self.name,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "total_tokens": self._total_tokens_used,
            "last_request": (
                self._last_request_time.isoformat() if self._last_request_time else None
            ),
            "models": self.models,
        }


# =============================================================================
# Base Layer
# =============================================================================


class BaseLayer(BaseComponent):
    """
    الفئة الأساسية لطبقات GAAP

    توفر:
    - معالجة متسلسلة
    - اتصال بين الطبقات
    - إدارة السياق
    """

    def __init__(self, layer_type: LayerType):
        super().__init__(name=f"Layer{layer_type.value}", layer=layer_type)
        self._upper_layer: BaseLayer | None = None
        self._lower_layer: BaseLayer | None = None
        self._context_manager: Any | None = None

    def set_upper_layer(self, layer: "BaseLayer") -> None:
        """تعيين الطبقة العليا"""
        self._upper_layer = layer

    def set_lower_layer(self, layer: "BaseLayer") -> None:
        """تعيين الطبقة السفلى"""
        self._lower_layer = layer

    def initialize(self) -> None:
        """تهيئة الطبقة"""
        self._is_initialized = True
        self._is_running = True

    def shutdown(self) -> None:
        """إيقاف الطبقة"""
        self._is_running = False
        self._is_initialized = False

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """معالجة البيانات"""
        pass

    async def send_to_lower(self, data: Any) -> Any:
        """إرسال للطبقة السفلى"""
        if self._lower_layer:
            return await self._lower_layer.process(data)
        return None

    async def send_to_upper(self, data: Any) -> Any:
        """إرسال للطبقة العليا"""
        if self._upper_layer:
            return await self._upper_layer.process(data)
        return None


# =============================================================================
# Base Critic
# =============================================================================


class BaseCritic(BaseComponent):
    """
    الفئة الأساسية للنقاد في MAD Panel

    توفر:
    - تقييم موحد
    - اقتراحات للتحسين
    - تتبع القرارات
    """

    def __init__(self, critic_type: CriticType, weight: float = 1.0, threshold: float = 70.0):
        super().__init__(name=f"{critic_type.name}_Critic", layer=LayerType.EXECUTION)
        self.critic_type = critic_type
        self.weight = weight
        self.threshold = threshold
        self._evaluations: list[CriticEvaluation] = []

    @abstractmethod
    async def evaluate(self, artifact: Any, context: dict[str, Any]) -> CriticEvaluation:
        """تقييم المخرجات"""
        pass

    def is_approved(self, evaluation: CriticEvaluation) -> bool:
        """هل التقييم مقبول؟"""
        return evaluation.score >= self.threshold and evaluation.approved

    def get_recent_evaluations(self, limit: int = 50) -> list[CriticEvaluation]:
        """الحصول على التقييمات الأخيرة"""
        return self._evaluations[-limit:]


# =============================================================================
# Base Healer (Self-Healing)
# =============================================================================


class BaseHealer(BaseComponent):
    """
    الفئة الأساسية للتعافي الذاتي

    توفر:
    - مستويات التعافي الخمسة
    - تتبع المحاولات
    - تصعيد عند الضرورة
    """

    def __init__(self) -> None:
        super().__init__(name="Healer", layer=LayerType.EXECUTION)
        self._healing_history: dict[str, list[dict[str, Any]]] = {}
        self._max_level = HealingLevel.L5_HUMAN_ESCALATION

    @abstractmethod
    async def heal(
        self,
        error: Exception,
        context: dict[str, Any],
        current_level: HealingLevel = HealingLevel.L1_RETRY,
    ) -> dict[str, Any]:
        """
        محاولة التعافي

        Args:
            error: الخطأ الذي حدث
            context: سياق الخطأ
            current_level: مستوى التعافي الحالي

        Returns:
            نتيجة التعافي
        """
        pass

    def record_healing_attempt(
        self, task_id: str, level: HealingLevel, success: bool, details: dict[str, Any]
    ) -> None:
        """تسجيل محاولة تعافي"""
        if task_id not in self._healing_history:
            self._healing_history[task_id] = []

        self._healing_history[task_id].append(
            {
                "level": level.name,
                "success": success,
                "details": details,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_next_level(self, current: HealingLevel) -> HealingLevel | None:
        """الحصول على المستوى التالي"""
        levels = list(HealingLevel)
        current_idx = levels.index(current)
        if current_idx < len(levels) - 1:
            return levels[current_idx + 1]
        return None


# =============================================================================
# Base Plugin
# =============================================================================


class BasePlugin(ABC):
    """
    الفئة الأساسية للإضافات

    توفر واجهة موحدة لتوسيع نظام GAAP
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._is_initialized = False
        self._logger = setup_logger(f"gaap.plugin.{self.__class__.__name__}")

    @property
    @abstractmethod
    def name(self) -> str:
        """اسم الإضافة"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """إصدار الإضافة"""
        pass

    def initialize(self) -> None:
        """تهيئة الإضافة"""
        self._logger.info(f"Initializing plugin: {self.name} v{self.version}")
        self._is_initialized = True

    @abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """تنفيذ الإضافة"""
        pass

    def shutdown(self) -> None:
        """إيقاف الإضافة"""
        self._logger.info(f"Shutting down plugin: {self.name}")
        self._is_initialized = False


# =============================================================================
# Base Memory
# =============================================================================


class BaseMemory(BaseComponent):
    """
    الفئة الأساسية لإدارة الذاكرة

    توفر:
    - تخزين واسترجاع
    - ضغط تلقائي
    - تنظيف دوري
    """

    def __init__(self, max_size: int = 10000):
        super().__init__(name="Memory", layer=LayerType.EXECUTION)
        self.max_size = max_size
        self._storage: dict[str, Any] = {}
        self._access_log: list[dict[str, Any]] = []

    @abstractmethod
    async def store(self, key: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        """تخزين قيمة"""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Any | None:
        """استرجاع قيمة"""
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[Any]:
        """بحث"""
        pass

    def record_access(self, key: str, operation: str) -> None:
        """تسجيل وصول"""
        self._access_log.append(
            {"key": key, "operation": operation, "timestamp": datetime.now().isoformat()}
        )

    def get_size(self) -> int:
        """الحصول على حجم التخزين"""
        return len(self._storage)

    def needs_cleanup(self) -> bool:
        """هل يحتاج تنظيف؟"""
        return self.get_size() > self.max_size * 0.9


# =============================================================================
# Context Manager
# =============================================================================


class ContextManager:
    """
    مدير السياق - يدير سياق المشروع والمهمة

    الميزات:
    - تحميل هرمي
    - ضغط ذكي
    - تتبع الاستخدام
    """

    def __init__(self, budget: ContextBudget):
        self.budget = budget
        self._windows: dict[str, ContextWindow] = {}
        self._priority_queue: list[str] = []
        self._access_patterns: dict[str, int] = {}

    def add_window(self, window: ContextWindow) -> bool:
        """إضافة نافذة سياق"""
        if self.budget.remaining < window.token_count:
            return False

        self._windows[window.id] = window
        self.budget.used += window.token_count
        self._priority_queue.append(window.id)
        self._priority_queue.sort(key=lambda x: self._windows[x].priority, reverse=True)
        return True

    def remove_window(self, window_id: str) -> None:
        """إزالة نافذة سياق"""
        if window_id in self._windows:
            window = self._windows.pop(window_id)
            self.budget.used -= window.token_count
            if window_id in self._priority_queue:
                self._priority_queue.remove(window_id)

    def get_window(self, window_id: str) -> ContextWindow | None:
        """الحصول على نافذة سياق"""
        window = self._windows.get(window_id)
        if window:
            window.last_accessed = datetime.now()
            window.access_count += 1
            self._access_patterns[window_id] = self._access_patterns.get(window_id, 0) + 1
        return window

    def get_all_content(self) -> str:
        """الحصول على كل المحتوى"""
        contents = []
        for window_id in self._priority_queue:
            if window_id in self._windows:
                contents.append(self._windows[window_id].content)
        return "\n\n".join(contents)

    def optimize(self) -> int:
        """
        تحسين استخدام السياق
        Returns: عدد الرموز المحررة
        """
        freed_tokens = 0
        to_remove = []

        # إزالة النوافذ منخفضة الأولوية وغير المستخدمة
        for window_id, window in self._windows.items():
            if window.priority < 0 and window.access_count == 0 and self.budget.utilization > 0.8:
                to_remove.append(window_id)

        for window_id in to_remove:
            self.remove_window(window_id)
            freed_tokens += (
                self._windows.get(
                    window_id,
                    ContextWindow(
                        id="", content="", token_count=0, level=ContextLevel.LEVEL_0_OVERVIEW
                    ),
                ).token_count
                if window_id in self._windows
                else 0
            )

        return freed_tokens


# =============================================================================
# Execution Context
# =============================================================================


@dataclass
class ExecutionContext:
    """سياق التنفيذ - يُمرر عبر جميع المكونات"""

    task: Task
    agent_id: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    budget_remaining: float = 0.0
    parent_context: Optional["ExecutionContext"] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_child(self, task: Task, agent_id: str) -> "ExecutionContext":
        """إنشاء سياق فرعي"""
        return ExecutionContext(
            task=task,
            agent_id=agent_id,
            session_id=self.session_id,
            trace_id=self.trace_id,
            parent_context=self,
            budget_remaining=self.budget_remaining,
        )

    def get_elapsed_time(self) -> float:
        """الحصول على الوقت المنقضي بالثواني"""
        return (datetime.now() - self.start_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """تحويل إلى قاموس"""
        return {
            "task_id": self.task.id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "elapsed_time": self.get_elapsed_time(),
            "budget_remaining": self.budget_remaining,
            "metadata": self.metadata,
        }


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class ExecutionResult(Generic[T]):
    """نتيجة تنفيذ عامة"""

    success: bool
    data: T | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls, data: T, **kwargs: Any) -> "ExecutionResult[T]":
        """إنشاء نتيجة ناجحة"""
        return cls(success=True, data=data, **kwargs)

    @classmethod
    def fail(cls, error: str, **kwargs: Any) -> "ExecutionResult[T]":
        """إنشاء نتيجة فاشلة"""
        return cls(success=False, error=error, **kwargs)


# =============================================================================
# Decorators
# =============================================================================


def measure_time(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """مُزخرف لقياس وقت التنفيذ"""

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.time()
        try:
            result: T = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            if args and hasattr(args[0], "record_metric"):
                args[0].record_metric(f"{func.__name__}_time", elapsed)

    return wrapper


def with_retry(
    max_retries: int = 3, delay: float = 1.0
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """مُزخرف لإعادة المحاولة"""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    result: T = await func(*args, **kwargs)
                    return result
                except GAAPException as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise MaxRetriesExceededError(
                task_id="unknown", max_retries=max_retries, last_error=str(last_error)
            )

        return wrapper

    return decorator


def validate_input(
    validator: Callable[[Any], bool],
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """مُزخرف للتحقق من المدخلات"""

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not validator(args[1] if len(args) > 1 else kwargs):
                raise ValueError("Input validation failed")
            result: T = await func(*args, **kwargs)
            return result

        return wrapper

    return decorator


# =============================================================================
# Utility Functions
# =============================================================================


async def run_with_timeout(
    coro: Awaitable[T], timeout: float, default: T | None = None
) -> T | None:
    """تشغيل مع مهلة زمنية"""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return default


async def gather_with_concurrency(*coros: Awaitable[T], limit: int = 10) -> list[T]:
    """تشغيل متوازي مع حد للتزامن"""
    semaphore = asyncio.Semaphore(limit)

    async def run_with_semaphore(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[run_with_semaphore(c) for c in coros])


def create_iacp_message(
    sender_id: str,
    sender_layer: LayerType,
    sender_role: str,
    recipient_id: str,
    recipient_layer: LayerType,
    intent: str,
    content: dict[str, Any],
    trace_id: str | None = None,
) -> IACPMessage:
    """إنشاء رسالة IACP"""
    return IACPMessage(
        header=IACPHeader(
            sender_id=sender_id,
            sender_layer=sender_layer,
            sender_role=sender_role,
            recipient_id=recipient_id,
            recipient_layer=recipient_layer,
            trace_id=trace_id or str(uuid.uuid4()),
        ),
        payload=IACPPayload(intent=intent, content=content),
        security=IACPSecurity(),
    )
