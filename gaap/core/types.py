"""
Core Types for GAAP System

Provides fundamental type definitions used throughout GAAP:

Classes:
    - TaskPriority: Task priority levels
    - TaskComplexity: Task complexity levels
    - TaskType: Task type enumeration
    - LayerType: GAAP layer types
    - ModelTier: LLM model tiers
    - ProviderType: Provider types
    - MessageRole: Message roles in conversations
    - CriticType: MAD panel critic types
    - HealingLevel: Self-healing levels
    - ExecutionStatus: Task execution status
    - SecurityRiskLevel: Security risk levels
    - ContextLevel: Context loading levels
    - MemoryType: Memory system types
    - Message: Message data class
    - Task: Task data class
    - TaskResult: Task execution result
    - And many more...

Usage:
    from gaap.core.types import (
        Task, TaskPriority, TaskType, TaskComplexity,
        Message, MessageRole,
        GAAPRequest, GAAPResponse
    )

    # Create a task
    task = Task(
        id="task-123",
        description="Write a function",
        type=TaskType.CODE_GENERATION,
        priority=TaskPriority.NORMAL,
        complexity=TaskComplexity.SIMPLE
    )
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any,
    Final,
    TypedDict,
    TypeVar,
)

# =============================================================================
# Type Variables
# =============================================================================

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)


# =============================================================================
# Basic Enums
# =============================================================================


class TaskPriority(Enum):
    """
    Task priority levels.

    Determines task scheduling and resource allocation.

    Members:
        CRITICAL: Critical - requires maximum resources
        HIGH: High priority
        NORMAL: Normal priority (default)
        LOW: Low priority
        BACKGROUND: Background - does not affect user experience

    Usage:
        >>> priority = TaskPriority.HIGH
        >>> print(priority.name)
        'HIGH'
    """

    CRITICAL = auto()  # حرجة - تتطلب موارد قصوى
    HIGH = auto()  # عالية
    NORMAL = auto()  # عادية (افتراضي)
    LOW = auto()  # منخفضة
    BACKGROUND = auto()  # خلفية - لا تؤثر على المستخدم


class TaskComplexity(Enum):
    """
    Task complexity levels.

    Estimates task difficulty and resource requirements.

    Members:
        TRIVIAL: Trivial - single line
        SIMPLE: Simple - single function
        MODERATE: Moderate - single component
        COMPLEX: Complex - multiple components
        ARCHITECTURAL: Architectural - full system

    Usage:
        >>> complexity = TaskComplexity.MODERATE
        >>> estimated_tokens = get_token_estimate(complexity)
    """

    TRIVIAL = auto()  # بسيط جداً - سطر واحد
    SIMPLE = auto()  # بسيط - دالة واحدة
    MODERATE = auto()  # متوسط - مكون واحد
    COMPLEX = auto()  # معقد - مكونات متعددة
    ARCHITECTURAL = auto()  # معماري - نظام كامل


class TaskType(Enum):
    """
    Task type enumeration.

    Defines the nature of work to be performed.

    Members:
        CODE_GENERATION: Generate new code
        CODE_REVIEW: Review existing code
        DEBUGGING: Fix bugs
        REFACTORING: Improve code structure
        DOCUMENTATION: Write documentation
        TESTING: Write/run tests
        RESEARCH: Research solutions
        ANALYSIS: Analyze requirements
        PLANNING: Plan architecture
        ORCHESTRATION: Orchestrate multiple tasks

    Usage:
        >>> task_type = TaskType.CODE_GENERATION
        >>> if task_type == TaskType.CODE_GENERATION:
        ...     generate_code()
    """

    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    RESEARCH = auto()
    ANALYSIS = auto()
    PLANNING = auto()
    ORCHESTRATION = auto()


class LayerType(Enum):
    """
    GAAP layer types.

    Represents the active layer architecture.

    Members:
        INTERFACE: Layer 0 - Interface and security
        STRATEGIC: Layer 1 - Strategic planning
        TACTICAL: Layer 2 - Tactical organization
        EXECUTION: Layer 3 - Execution and quality
        EXTERNAL: Layer 5 - External intelligence

    Usage:
        >>> layer = LayerType.STRATEGIC
        >>> print(layer.value)
        1
    """

    INTERFACE = 0  # Layer 0: الواجهة والحماية
    STRATEGIC = 1  # Layer 1: التخطيط الاستراتيجي
    TACTICAL = 2  # Layer 2: التنظيم التكتيكي
    EXECUTION = 3  # Layer 3: التنفيذ والجودة
    EXTERNAL = 5  # Layer 5: الذكاء الخارجي


class ModelTier(Enum):
    """
    LLM model tiers.

    Categorizes models by capability and cost.

    Members:
        TIER_1_STRATEGIC: Smartest models - for strategic planning
        TIER_2_TACTICAL: Balanced intelligence/speed - for execution
        TIER_3_EFFICIENT: Fast and cheap - for simple tasks
        TIER_4_PRIVATE: Local - for sensitive data

    Usage:
        >>> tier = ModelTier.TIER_1_STRATEGIC
        >>> model = get_model_for_tier(tier)
    """

    TIER_1_STRATEGIC = auto()  # أذكى النماذج - للتخطيط الاستراتيجي
    TIER_2_TACTICAL = auto()  # توازن ذكاء/سرعة - للتنفيذ
    TIER_3_EFFICIENT = auto()  # سريع ورخيص - للمهام البسيطة
    TIER_4_PRIVATE = auto()  # محلي - للبيانات الحساسة


class ProviderType(Enum):
    """
    Provider type enumeration.

    Categorizes LLM providers by service model.

    Members:
        CHAT_BASED: Chat-based provider (g4f, etc.)
        FREE_TIER: Free tier (Groq, Gemini)
        PAID: Paid provider (OpenAI, Anthropic)
        LOCAL: Local provider (Ollama, vLLM)

    Usage:
        >>> provider_type = ProviderType.FREE_TIER
        >>> if provider_type == ProviderType.FREE_TIER:
        ...     use_free_tier()
    """

    CHAT_BASED = auto()  # مزود قائم على Chat (g4f, etc.)
    FREE_TIER = auto()  # طبقة مجانية (Groq, Gemini)
    PAID = auto()  # مدفوع (OpenAI, Anthropic)
    LOCAL = auto()  # محلي (Ollama, vLLM)


class MessageRole(Enum):
    """
    Message roles in conversations.

    Defines the sender of a message.

    Members:
        SYSTEM: System message
        USER: User message
        ASSISTANT: Assistant response
        FUNCTION: Function call result
        TOOL: Tool call result

    Usage:
        >>> role = MessageRole.USER
        >>> message = Message(role=role, content="Hello")
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class CriticType(Enum):
    """
    أنواع النقاد في لجنة MAD
    
    تحدد مجالات الخبرة لكل ناقد.
    """

    # Software & Logic (Existing)
    LOGIC = auto()  # ناقد المنطق
    SECURITY = auto()  # ناقد الأمان
    PERFORMANCE = auto()  # ناقد الأداء
    STYLE = auto()  # ناقد الأسلوب
    COMPLIANCE = auto()  # ناقد الامتثال
    ETHICS = auto()  # ناقد الأخلاق
    
    # Research & Intelligence (New)
    ACCURACY = auto()  # ناقد الدقة العلمية
    SOURCE_CREDIBILITY = auto()  # ناقد موثوقية المصادر
    COMPLETENESS = auto()  # ناقد كمال البحث
    
    # Diagnostics (New)
    ROOT_CAUSE = auto()  # ناقد السبب الجذري
    RELIABILITY = auto()  # ناقد الموثوقية التشخيصية
    
    # Analysis
    CRITICAL_THINKING = auto()  # ناقد التفكير النقدي
    BIAS_DETECTION = auto()  # ناقد كشف التحيز



class HealingLevel(Enum):
    """
    Self-healing levels.

    Escalating levels of recovery attempts.

    Members:
        L1_RETRY: Retry (transient errors)
        L2_REFINE: Refine prompt (syntax/logic errors)
        L3_PIVOT: Change model (capability limits)
        L4_STRATEGY_SHIFT: Change strategy (complexity)
        L5_HUMAN_ESCALATION: Human escalation (unrecoverable)

    Usage:
        >>> level = HealingLevel.L1_RETRY
        >>> if should_retry(level):
        ...     retry_operation()
    """

    L1_RETRY = auto()  # إعادة المحاولة
    L2_REFINE = auto()  # تحسين الصيغة
    L3_PIVOT = auto()  # تغيير النموذج
    L4_STRATEGY_SHIFT = auto()  # تغيير الاستراتيجية
    L5_HUMAN_ESCALATION = auto()  # تصعيد بشري


class ExecutionStatus(Enum):
    """
    Task execution status.

    Tracks task lifecycle.

    Members:
        PENDING: Task created, not started
        QUEUED: Task in queue
        RUNNING: Task executing
        COMPLETED: Task completed successfully
        FAILED: Task failed
        RETRYING: Task being retried
        ESCALATED: Task escalated to human
        CANCELLED: Task cancelled

    Usage:
        >>> status = ExecutionStatus.RUNNING
        >>> if status == ExecutionStatus.COMPLETED:
        ...     handle_success()
    """

    PENDING = auto()
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    RETRYING = auto()
    ESCALATED = auto()
    CANCELLED = auto()


class SecurityRiskLevel(Enum):
    """
    Security risk levels.

    Assesses security threat severity.

    Members:
        SAFE: Safe - no threat
        LOW: Low risk
        MEDIUM: Medium risk
        HIGH: High risk
        CRITICAL: Critical risk
        BLOCKED: Blocked - immediate threat

    Usage:
        >>> risk = SecurityRiskLevel.HIGH
        >>> if risk >= SecurityRiskLevel.HIGH:
        ...     block_request()
    """

    SAFE = auto()  # آمن
    LOW = auto()  # خطر منخفض
    MEDIUM = auto()  # خطر متوسط
    HIGH = auto()  # خطر عالي
    CRITICAL = auto()  # خطر حرج
    BLOCKED = auto()  # محظور


class ContextLevel(Enum):
    """
    Context loading levels.

    Determines how much context to load.

    Members:
        LEVEL_0_OVERVIEW: Overview (~100 tokens)
        LEVEL_1_MODULE: Module view (~500 tokens)
        LEVEL_2_FILE: File view (~2k tokens)
        LEVEL_3_FULL: Full content (~20k+ tokens)
        LEVEL_4_DEPENDENCIES: With dependencies

    Usage:
        >>> level = ContextLevel.LEVEL_2_FILE
        >>> context = load_context(level)
    """

    LEVEL_0_OVERVIEW = auto()  # نظرة عامة (~100 tokens)
    LEVEL_1_MODULE = auto()  # نظرة وحدة (~500 tokens)
    LEVEL_2_FILE = auto()  # نظرة ملف (~2k tokens)
    LEVEL_3_FULL = auto()  # محتوى كامل (~20k+ tokens)
    LEVEL_4_DEPENDENCIES = auto()  # مع التبعيات


class MemoryType(Enum):
    """أنواع الذاكرة"""

    WORKING = auto()  # ذاكرة العمل (L1)
    EPISODIC = auto()  # ذاكرة الحلقات (L2)
    SEMANTIC = auto()  # ذاكرة دلالية (L3)
    PROCEDURAL = auto()  # ذاكرة إجرائية (L4)


# =============================================================================
# Constants
# =============================================================================

# Default values for task properties
DEFAULT_TASK_PRIORITY = TaskPriority.NORMAL
DEFAULT_TASK_COMPLEXITY = TaskComplexity.SIMPLE
DEFAULT_TASK_TYPE = TaskType.CODE_GENERATION


# =============================================================================
# Message Types
# =============================================================================


@dataclass
class Message:
    """رسالة في المحادثة"""

    role: MessageRole
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """تحويل الرسالة إلى قاموس"""
        result: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result


@dataclass
class ChatCompletionRequest:
    """طلب إكمال محادثة"""

    messages: list[Message]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    stop: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatCompletionChoice:
    """خيار إكمال"""

    index: int
    message: Message
    finish_reason: str
    logprobs: dict[str, Any] | None = None


@dataclass
class Usage:
    """استخدام الرموز"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletionResponse:
    """استجابة إكمال المحادثة"""

    id: str
    object: str = "chat.completion"
    created: datetime = field(default_factory=datetime.now)
    model: str = ""
    choices: list[ChatCompletionChoice] = field(default_factory=list)
    usage: Usage | None = None
    provider: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Task Types
# =============================================================================


@dataclass
class TaskContext:
    """سياق المهمة"""

    project_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    previous_tasks: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """نتيجة المهمة"""

    success: bool
    output: Any
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    """مهمة في نظام GAAP"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    type: TaskType = TaskType.CODE_GENERATION
    priority: TaskPriority = TaskPriority.NORMAL
    complexity: TaskComplexity = TaskComplexity.MODERATE
    context: TaskContext = field(default_factory=TaskContext)
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    estimated_tokens: int = 2000
    max_retries: int = 3
    retry_count: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: TaskResult | None = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Agent Types
# =============================================================================


@dataclass
class AgentCapabilities:
    """قدرات الوكيل"""

    code_generation: bool = False
    code_review: bool = False
    debugging: bool = False
    research: bool = False
    analysis: bool = False
    planning: bool = False
    testing: bool = False
    security_audit: bool = False
    max_context_tokens: int = 128000
    supported_languages: list[str] = field(default_factory=list)
    supported_frameworks: list[str] = field(default_factory=list)


@dataclass
class AgentIdentity:
    """هوية الوكيل"""

    id: str
    name: str
    role: str
    layer: LayerType
    description: str = ""
    capabilities: AgentCapabilities = field(default_factory=AgentCapabilities)
    created_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# Provider Types
# =============================================================================


@dataclass
class ProviderConfig:
    """تكوين المزود"""

    name: str
    provider_type: ProviderType
    api_key: str | None = None
    base_url: str | None = None
    models: list[str] = field(default_factory=list)
    rate_limit: int = 60  # طلبات في الدقيقة
    timeout: int = 120  # ثانية
    max_retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True
    priority: int = 0  # أولوية الاختيار
    cost_per_1k_tokens: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelInfo:
    """معلومات النموذج"""

    name: str
    provider: str
    tier: ModelTier
    context_window: int
    max_output_tokens: int
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_vision: bool = False
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    capabilities: list[str] = field(default_factory=list)


# =============================================================================
# Routing Types
# =============================================================================


@dataclass
class RoutingDecision:
    """قرار التوجيه"""

    selected_provider: str
    selected_model: str
    reasoning: str
    alternatives: list[str] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingContext:
    """سياق التوجيه"""

    task: Task
    available_providers: list[str]
    budget_remaining: float
    latency_requirement: float | None = None
    quality_requirement: float = 0.8
    privacy_required: bool = False
    preferred_models: list[str] = field(default_factory=list)
    excluded_models: list[str] = field(default_factory=list)


# =============================================================================
# Security Types
# =============================================================================


@dataclass
class SecurityScanResult:
    """نتيجة الفحص الأمني"""

    is_safe: bool
    risk_level: SecurityRiskLevel
    detected_patterns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 1.0
    scan_duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityToken:
    """توكن القدرة - للتحكم في الصلاحيات"""

    subject: str  # معرف الوكيل
    resource: str  # المورد المطلوب
    action: str  # الإجراء المطلوب
    issued_at: datetime
    expires_at: datetime
    constraints: dict[str, Any] = field(default_factory=dict)
    nonce: str = field(default_factory=lambda: str(uuid.uuid4()))
    signature: str | None = None


# =============================================================================
# Context Management Types
# =============================================================================


@dataclass
class ContextBudget:
    """ميزانية السياق"""

    total: int  # إجمالي الرموز المتاحة
    used: int = 0  # المستخدم
    reserved: int = 0  # المحجوز
    level: ContextLevel = ContextLevel.LEVEL_2_FILE

    @property
    def remaining(self) -> int:
        """الرموز المتبقية"""
        return self.total - self.used - self.reserved

    @property
    def utilization(self) -> float:
        """نسبة الاستخدام"""
        if self.total == 0:
            return 0.0
        return (self.used + self.reserved) / self.total


@dataclass
class ContextWindow:
    """نافذة السياق"""

    id: str
    content: str
    token_count: int
    level: ContextLevel
    priority: int = 0
    loaded_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MAD (Multi-Agent Debate) Types
# =============================================================================


@dataclass
class CriticEvaluation:
    """تقييم الناقد"""

    critic_type: CriticType
    score: float  # 0-100
    approved: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MADDecision:
    """قرار لجنة MAD"""

    consensus: bool
    final_score: float
    evaluations: list[CriticEvaluation] = field(default_factory=list)
    required_changes: list[str] = field(default_factory=list)
    debate_rounds: int = 0
    decision_reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# IACP (Inter-Agent Communication Protocol) Types
# =============================================================================


@dataclass
class IACPHeader:
    """ترويسة رسالة IACP"""

    protocol: str = "IACP/1.0"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: str = ""
    sender_layer: LayerType = LayerType.EXECUTION
    sender_role: str = ""
    recipient_id: str = ""
    recipient_layer: LayerType = LayerType.EXECUTION
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str | None = None
    reply_to: str | None = None


@dataclass
class IACPPayload:
    """حمولة رسالة IACP"""

    intent: str
    priority: TaskPriority = TaskPriority.NORMAL
    content: dict[str, Any] = field(default_factory=dict)
    context_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class IACPSecurity:
    """أمان رسالة IACP"""

    signature: str = ""
    access_level: str = "read"
    encryption: str | None = None
    checksum: str = ""


@dataclass
class IACPMessage:
    """رسالة IACP كاملة"""

    header: IACPHeader
    payload: IACPPayload
    security: IACPSecurity
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """تحويل الرسالة إلى قاموس"""
        return {
            "header": {
                "protocol": self.header.protocol,
                "message_id": self.header.message_id,
                "timestamp": self.header.timestamp.isoformat(),
                "sender": {
                    "id": self.header.sender_id,
                    "layer": self.header.sender_layer.name,
                    "role": self.header.sender_role,
                },
                "recipient": {
                    "id": self.header.recipient_id,
                    "layer": self.header.recipient_layer.name,
                },
                "trace_id": self.header.trace_id,
            },
            "payload": {
                "intent": self.payload.intent,
                "priority": self.payload.priority.name,
                "content": self.payload.content,
                "context_snapshot": self.payload.context_snapshot,
            },
            "security": {
                "signature": self.security.signature,
                "access_level": self.security.access_level,
            },
            "metadata": self.metadata,
        }


# =============================================================================
# Metrics Types
# =============================================================================


@dataclass
class ExecutionMetrics:
    """مقاييس التنفيذ"""

    task_id: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    retries: int = 0
    healing_level: HealingLevel | None = None
    mad_score: float | None = None
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """مقاييس النظام"""

    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_tokens_used: int = 0
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    success_rate: float = 0.0
    avg_mad_score: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime | None = None


# =============================================================================
# TypedDict Definitions for JSON Schemas
# =============================================================================


class StructuredIntentDict(TypedDict):
    """الطلب المهيكل من Layer 0"""

    request_id: str
    timestamp: str
    security_scan: dict[str, Any]
    classification: dict[str, Any]
    enriched_context: dict[str, Any]
    routing_directives: dict[str, Any]


class ArchitectureSpecDict(TypedDict):
    """مواصفات معمارية من Layer 1"""

    spec_id: str
    paradigm: str
    data_strategy: str
    communication_pattern: str
    infrastructure: str
    observability: str
    decisions: list[dict[str, Any]]
    risks: list[dict[str, Any]]
    estimated_resources: dict[str, Any]


class AtomicTaskDict(TypedDict):
    """مهمة ذرية من Layer 2"""

    task_id: str
    component: str
    type: str
    priority: str
    description: str
    constraints: dict[str, Any]
    acceptance_criteria: list[str]
    dependencies: list[str]
    estimated_resources: dict[str, Any]


# =============================================================================
# Constants
# =============================================================================

# حدود السياق للنماذج المختلفة
CONTEXT_LIMITS: Final[dict[str, int]] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "claude-3-5-sonnet": 200000,
    "claude-3-5-opus": 200000,
    "claude-3-opus": 200000,
    "gemini-1.5-pro": 1000000,
    "gemini-1.5-flash": 1000000,
    "llama-3-70b": 128000,
    "llama-3-8b": 128000,
    "mixtral-8x7b": 32000,
}

# تكاليف النماذج (لكل 1M tokens)
MODEL_COSTS: Final[dict[str, dict[str, float]]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-opus": {"input": 15.00, "output": 75.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "llama-3-70b": {"input": 0.59, "output": 0.79},  # Groq
    "mixtral-8x7b": {"input": 0.27, "output": 0.27},  # Groq
}

# ميزانية السياق الافتراضية
DEFAULT_CONTEXT_BUDGETS: Final[dict[str, int]] = {
    "minimal": 10_000,
    "low": 20_000,
    "medium": 50_000,
    "high": 80_000,
    "critical": 120_000,
    "unlimited": 500_000,
}

# أوزان النقاد في MAD Panel
CRITIC_WEIGHTS: Final[dict[CriticType, float]] = {
    CriticType.LOGIC: 0.35,
    CriticType.SECURITY: 0.25,
    CriticType.PERFORMANCE: 0.20,
    CriticType.STYLE: 0.10,
    CriticType.COMPLIANCE: 0.05,
    CriticType.ETHICS: 0.05,
}

# حدود إعادة المحاولة
MAX_RETRIES_PER_LEVEL: Final[dict[HealingLevel, int]] = {
    HealingLevel.L1_RETRY: 3,
    HealingLevel.L2_REFINE: 2,
    HealingLevel.L3_PIVOT: 2,
    HealingLevel.L4_STRATEGY_SHIFT: 1,
    HealingLevel.L5_HUMAN_ESCALATION: 0,
}


# =============================================================================
# OODA Loop Types
# =============================================================================


class OODAPhase(Enum):
    """مراحل دورة OODA"""

    OBSERVE = auto()
    ORIENT = auto()
    DECIDE = auto()
    ACT = auto()
    LEARN = auto()


class ReplanTrigger(Enum):
    """محفزات إعادة التخطيط"""

    NONE = auto()
    L3_CRITICAL_FAILURE = auto()
    AXIOM_VIOLATION = auto()
    RESOURCE_EXHAUSTED = auto()
    GOAL_DRIFT = auto()
    USER_INTERRUPT = auto()


@dataclass
class OODAState:
    """حالة دورة OODA"""

    request_id: str
    current_phase: OODAPhase = OODAPhase.OBSERVE
    iteration: int = 0
    max_iterations: int = 10

    goal_achieved: bool = False
    needs_replanning: bool = False
    replan_trigger: ReplanTrigger = ReplanTrigger.NONE
    replan_count: int = 0

    completed_tasks: set[str] = field(default_factory=set)
    failed_tasks: set[str] = field(default_factory=set)
    in_progress_tasks: set[str] = field(default_factory=set)

    axiom_violations: list[dict[str, Any]] = field(default_factory=list)
    lessons_learned: list[str] = field(default_factory=list)

    current_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0

    last_observation: dict[str, Any] = field(default_factory=dict)
    last_decision: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def advance_phase(self) -> OODAPhase:
        """الانتقال للمرحلة التالية"""
        phases = list(OODAPhase)
        current_idx = phases.index(self.current_phase)
        next_idx = (current_idx + 1) % len(phases)
        self.current_phase = phases[next_idx]
        if self.current_phase == OODAPhase.OBSERVE:
            self.iteration += 1
        return self.current_phase

    def trigger_replan(self, trigger: ReplanTrigger) -> None:
        """تفعيل إعادة التخطيط"""
        self.needs_replanning = True
        self.replan_trigger = trigger
        self.replan_count += 1

    def record_axiom_violation(self, violation: dict[str, Any]) -> None:
        """تسجيل انتهاك بديهية"""
        self.axiom_violations.append(violation)

    def get_stupidity_rate(self) -> float:
        """حساب معدل الأخطاء المنخفضة المستوى"""
        if not self.axiom_violations:
            return 0.0
        low_level = [v for v in self.axiom_violations if v.get("level") == "low"]
        return len(low_level) / max(len(self.axiom_violations), 1)
