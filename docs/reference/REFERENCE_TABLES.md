# GAAP Comprehensive Reference Tables

> **جداول المرجع الشاملة**  
> **آخر تحديث:** February 17, 2026  
> **الغرض:** Quick lookup لجميع الـ APIs والمكونات

---

## 📦 Table 1: Core Types Reference

| Type | Module | Description | Common Values |
|------|--------|-------------|---------------|
| `TaskPriority` | `gaap.core.types` | أولوية المهمة | `CRITICAL`, `HIGH`, `NORMAL`, `LOW`, `BACKGROUND` |
| `TaskComplexity` | `gaap.core.types` | تعقيد المهمة | `TRIVIAL`, `SIMPLE`, `MODERATE`, `COMPLEX`, `ARCHITECTURAL` |
| `TaskType` | `gaap.core.types` | نوع المهمة | `CODE_GENERATION`, `CODE_REVIEW`, `DEBUGGING`, `REFACTORING`, `TESTING`, `RESEARCH`, `ANALYSIS`, `PLANNING` |
| `LayerType` | `gaap.core.types` | نوع الطبقة | `INTERFACE(0)`, `STRATEGIC(1)`, `TACTICAL(2)`, `EXECUTION(3)` |
| `ModelTier` | `gaap.core.types` | مستوى المودل | `TIER_1_STRATEGIC`, `TIER_2_TACTICAL`, `TIER_3_EFFICIENT` |
| `ProviderType` | `gaap.core.types` | نوع المزود | `CHAT_BASED`, `FREE_TIER`, `PAID`, `LOCAL` |
| `MessageRole` | `gaap.core.types` | دور الرسالة | `SYSTEM`, `USER`, `ASSISTANT`, `FUNCTION`, `TOOL` |
| `CriticType` | `gaap.core.types` | نوع الناقد | `LOGIC`, `SECURITY`, `PERFORMANCE`, `STYLE`, `COMPLIANCE`, `ETHICS` |
| `HealingLevel` | `gaap.core.types` | مستوى التعافي | `L1_RETRY`, `L2_REFINE`, `L3_PIVOT`, `L4_STRATEGY_SHIFT`, `L5_HUMAN_ESCALATION` |
| `ExecutionStatus` | `gaap.core.types` | حالة التنفيذ | `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `RETRYING`, `ESCALATED` |
| `MemoryTier` | `gaap.memory` | طبقة الذاكرة | `WORKING`, `EPISODIC`, `SEMANTIC`, `PROCEDURAL` |
| `RiskLevel` | `gaap.security` | مستوى الخطر | `SAFE`, `LOW`, `MEDIUM`, `HIGH`, `CRITICAL`, `BLOCKED` |
| `RoutingStrategy` | `gaap.routing` | استراتيجية التوجيه | `QUALITY_FIRST`, `COST_OPTIMIZED`, `SPEED_FIRST`, `BALANCED`, `SMART` |

---

## 🔧 Table 2: GAAPEngine API

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `__init__` | `providers`, `budget`, `enable_context`, `enable_healing`, `enable_memory`, `enable_security`, `enable_axiom_enforcement`, `project_path` | `GAAPEngine` | إنشاء محرك جديد |
| `process` | `request: GAAPRequest` | `GAAPResponse` | معالجة طلب كامل |
| `chat` | `message: str`, `context: dict` | `str` | محادثة بسيطة |
| `get_stats` | - | `dict` | إحصائيات النظام |
| `get_ooda_stats` | - | `dict` | إحصائيات OODA |
| `shutdown` | - | `None` | إيقاف المحرك |

---

## 📋 Table 3: GAAPRequest Fields

| Field | Type | Default | Required | Description |
|-------|------|---------|----------|-------------|
| `text` | `str` | - | ✅ | نص الطلب |
| `context` | `dict \| None` | `None` | ❌ | سياق إضافي |
| `priority` | `TaskPriority` | `NORMAL` | ❌ | أولوية المهمة |
| `budget_limit` | `float \| None` | `None` | ❌ | حد الميزانية |
| `metadata` | `dict` | `{}` | ❌ | بيانات إضافية |

---

## 📊 Table 4: GAAPResponse Fields

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | `str` | معرف الطلب الفريد |
| `success` | `bool` | هل نجح الطلب؟ |
| `output` | `Any` | المخرجات (كود، نص، إلخ) |
| `error` | `str \| None` | رسالة الخطأ (إن وجدت) |
| `intent` | `StructuredIntent` | النية المصنفة |
| `architecture_spec` | `ArchitectureSpec` | المواصفات المعمارية |
| `task_graph` | `TaskGraph` | رسم المهام |
| `execution_results` | `list[ExecutionResult]` | نتائج التنفيذ |
| `total_time_ms` | `float` | الوقت الإجمالي (ms) |
| `total_cost_usd` | `float` | التكلفة الإجمالية ($) |
| `total_tokens` | `int` | إجمالي التوكنز |
| `quality_score` | `float` | درجة الجودة (0-1) |
| `ooda_iterations` | `int` | عدد دورات OODA |
| `strategic_replan_count` | `int` | عدد إعادة التخطيط |
| `axiom_violation_count` | `int` | عدد انتهاكات البديهيات |
| `metadata` | `dict` | بيانات إضافية |

---

## 🏗️ Table 5: Layer APIs

### L0: Interface

| Method | Input | Output | Time |
|--------|-------|--------|------|
| `process` | `text: str` | `StructuredIntent` | 50-120ms |
| `classify_intent` | `text: str` | `IntentType` | 10-400ms |
| `estimate_complexity` | `text: str` | `TaskComplexity` | 15-30ms |
| `get_stats` | - | `dict` | <1ms |

### L1: Strategic

| Method | Input | Output | Time |
|--------|-------|--------|------|
| `process` | `intent: StructuredIntent` | `ArchitectureSpec` | 4-8s |
| `tree_of_thoughts` | `problem: str` | `list[Solutions]` | 2-4s |
| `mad_panel` | `spec: ArchitectureSpec` | `MADDecision` | 2-4s |
| `get_stats` | - | `dict` | <1ms |

### L2: Tactical

| Method | Input | Output | Time |
|--------|-------|--------|------|
| `process` | `spec: ArchitectureSpec` | `TaskGraph` | 1-4s |
| `decompose_task` | `task: Task` | `list[AtomicTask]` | 500ms-2s |
| `build_dag` | `tasks: list[AtomicTask]` | `TaskGraph` | 300-700ms |
| `get_stats` | - | `dict` | <1ms |

### L3: Execution

| Method | Input | Output | Time |
|--------|-------|--------|------|
| `process` | `task: AtomicTask` | `ExecutionResult` | 1-3s |
| `execute_parallel` | `tasks: list[AtomicTask]` | `list[ExecutionResult]` | 2-5s |
| `quality_check` | `output: Any` | `MADDecision` | 1-2s |
| `get_stats` | - | `dict` | <1ms |

---

## 🛡️ Table 6: Security API

### PromptFirewall

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `scan` | `input_text: str`, `context: dict` | `FirewallResult` | فحص النص |
| `get_stats` | - | `dict` | إحصائيات الفحص |
| `sanitize` | `input_text: str` | `str` | تنقية النص |

### FirewallResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_safe` | `bool` | هل النص آمن؟ |
| `risk_level` | `RiskLevel` | مستوى الخطر |
| `detected_patterns` | `list[str]` | الأنماط المكتشفة |
| `sanitized_input` | `str` | النص المنقى |
| `recommendations` | `list[str]` | التوصيات |
| `scan_time_ms` | `float` | وقت الفحص |
| `layer_scores` | `dict[str, float]` | درجات الطبقات |

---

## 🧠 Table 7: Memory API

### HierarchicalMemory

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `record_episode` | `episode: EpisodicMemory` | `str` | تسجيل حدث |
| `search_lessons` | `query: str`, `top_k: int` | `list[dict]` | بحث عن دروس |
| `retrieve_relevant` | `context: str`, `min_strength: float` | `dict` | استرجاع سياق |
| `clear_tier` | `tier: MemoryTier` | `int` | مسح طبقة |
| `get_stats` | - | `dict` | إحصائيات الذاكرة |

### EpisodicMemory Fields

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | معرف المهمة |
| `action` | `str` | الإجراء |
| `result` | `str` | النتيجة |
| `success` | `bool` | هل نجح؟ |
| `duration_ms` | `float` | المدة (ms) |
| `tokens_used` | `int` | التوكنز المستخدمة |
| `cost_usd` | `float` | التكلفة ($) |
| `model` | `str` | المودل المستخدم |
| `provider` | `str` | المزود |
| `lessons` | `list[str]` | الدروس المستفادة |

---

## 🔄 Table 8: Healing API

### SelfHealingSystem

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `heal` | `error: Exception`, `task: Task`, `execute_func: Callable` | `HealingResult` | محاولة التعافي |
| `classify_error` | `error: Exception` | `ErrorCategory` | تصنيف الخطأ |
| `get_healing_history` | - | `list[HealingRecord]` | سجل التعافي |
| `get_stats` | - | `dict` | إحصائيات التعافي |

### HealingLevel Actions

| Level | Action | When to Use | Success Rate |
|-------|--------|-------------|--------------|
| `L1_RETRY` | إعادة محاولة | أخطاء عابرة (شبكة، timeout) | 60% |
| `L2_REFINE` | تحسين الـ Prompt | أخطاء صيغة/منطق | 40% |
| `L3_PIVOT` | تغيير المزود | حدود قدرة المودل | 30% |
| `L4_STRATEGY_SHIFT` | تبسيط المهمة | مهام معقدة جداً | 15% |
| `L5_HUMAN_ESCALATION` | تدخل بشري | أخطاء حرجة | 5% |

---

## 🎯 Table 9: Routing API

### SmartRouter

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `route` | `request: RoutingRequest` | `RoutingDecision` | اتخاذ قرار التوجيه |
| `score_providers` | `requirements: dict` | `list[ProviderScore]` | تقييم المزودين |
| `get_routing_stats` | - | `dict` | إحصائيات التوجيه |

### RoutingStrategy Comparison

| Strategy | Quality | Cost | Speed | Best For |
|----------|---------|------|-------|----------|
| `QUALITY_FIRST` | 95/100 | $$$$ | Slow | Critical tasks |
| `COST_OPTIMIZED` | 75/100 | $ | Medium | Budget tasks |
| `SPEED_FIRST` | 70/100 | $$ | Fast | Urgent tasks |
| `BALANCED` | 88/100 | $$ | Medium | General use |
| `SMART` | 90/100 | $$ | Medium | Most tasks |

---

## 📊 Table 10: Provider Comparison

| Provider | Models | Rate Limit | Latency | Cost | Tier |
|----------|--------|------------|---------|------|------|
| **Groq** | Llama-3.3-70b | 30 RPM/key | 227ms | Free | TIER_2 |
| **Gemini** | 1.5-Flash/Pro | 5 RPM/key | 384ms | Free | TIER_2 |
| **Cerebras** | Llama-3.1-70b | 30 RPM/key | 511ms | Free | TIER_2 |
| **Mistral** | Mistral-Large | 60 RPM/key | 603ms | Free | TIER_2 |
| **G4F** | Multi-provider | ~5 RPM | 1-3s | Free | TIER_3 |
| **WebChat** | Kimi/DeepSeek/GLM | Varies | 2-3s | Free | TIER_1 |

---

## 🐛 Table 11: Exception Reference

| Exception | Code | Recoverable | Action |
|-----------|------|-------------|--------|
| `ProviderRateLimitError` | `GAAP_PRV_004` | ✅ Yes | Wait + Retry |
| `ProviderTimeoutError` | `GAAP_PRV_006` | ✅ Yes | Increase timeout |
| `ProviderAuthenticationError` | `GAAP_PRV_005` | ❌ No | Check API key |
| `ProviderNotFoundError` | `GAAP_PRV_002` | ❌ No | Use available provider |
| `BudgetExceededError` | `GAAP_ROT_003` | ❌ No | Increase budget |
| `NoAvailableProviderError` | `GAAP_ROT_002` | ❌ No | Add providers |
| `MaxRetriesExceededError` | `GAAP_TSK_007` | ❌ No | Human escalate |
| `TaskTimeoutError` | `GAAP_TSK_005` | ✅ Yes | Increase timeout |
| `CircularDependencyError` | `GAAP_TSK_004` | ❌ No | Restructure tasks |
| `PromptInjectionError` | `GAAP_SEC_002` | ❌ No | Sanitize input |
| `SandboxEscapeError` | `GAAP_SEC_006` | ❌ No | Security alert |
| `ContextOverflowError` | `GAAP_CTX_002` | ✅ Yes | Reduce context |
| `ConsensusNotReachedError` | `GAAP_MAD_002` | ✅ Yes | Add rounds |
| `HealingFailedError` | `GAAP_HLH_002` | ❌ No | Human escalate |
| `AxiomViolationError` | `GAAP_AXM_002` | ⚠️ Maybe | Fix violation |

---

## 🔧 Table 12: Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `budget` | `float` | `100.0` | ميزانية التشغيل ($) |
| `enable_context` | `bool` | `True` | تفعيل إدارة السياق |
| `enable_healing` | `bool` | `True` | تفعيل التعافي الذاتي |
| `enable_memory` | `bool` | `True` | تفعيل الذاكرة |
| `enable_security` | `bool` | `True` | تفعيل الأمان |
| `enable_axiom_enforcement` | `bool` | `True` | تفعيل البديهيات |
| `project_path` | `str \| None` | `None` | مسار المشروع |
| `firewall_strictness` | `str` | `"high"` | صرامة الجدار الناري |
| `max_ooda_iterations` | `int` | `15` | حد دورات OODA |
| `max_task_retries` | `int` | `2` | حد إعادة المحاولات |

---

## 📦 Table 13: Cache API

| Cache Type | Max Size | TTL | Use Case |
|------------|----------|-----|----------|
| `MemoryCache` | 1000 | Configurable | Fast, volatile |
| `DiskCache` | Unlimited | Configurable | Persistent, slower |
| `ResponseCache` | 5000 | 3600s | LLM responses |
| `SemanticCache` | 2000 | 7200s | Similar queries |

### Cache Operations

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `get` | `key: str` | `Any \| None` | جلب من الكاش |
| `set` | `key: str`, `value: Any`, `ttl: int` | `bool` | تخزين في الكاش |
| `delete` | `key: str` | `bool` | حذف من الكاش |
| `clear` | - | `int` | مسح الكاش |
| `get_stats` | - | `dict` | إحصائيات الكاش |

---

## 🛠️ Table 14: Tool Registry

### Built-in Tools

| Tool | Parameters | Returns | Description |
|------|------------|---------|-------------|
| `list_dir` | `path: str` | `str` | سرد الملفات |
| `read_file` | `path: str` | `str` | قراءة ملف |
| `write_file` | `path: str`, `content: str` | `str` | كتابة ملف |
| `run_command` | `command: str` | `str` | تنفيذ أمر |
| `search_codebase` | `query: str` | `str` | بحث في الكود |

### ToolRegistry API

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `register` | `name`, `description`, `parameters`, `func` | `None` | تسجيل أداة |
| `execute` | `name: str`, `**kwargs` | `str` | تنفيذ أداة |
| `get_instructions` | - | `str` | تعليمات LLM |
| `list_tools` | - | `list[str]` | سرد الأدوات |

---

## 📈 Table 15: Performance Targets

| Component | Target | Acceptable | Critical |
|-----------|--------|------------|----------|
| L0 Processing | <100ms | <200ms | >500ms |
| L1 Strategic | <5s | <10s | >20s |
| L2 Tactical | <2s | <5s | >10s |
| L3 Execution | <2s | <5s | >10s |
| Full OODA | <10s | <20s | >60s |
| Memory Usage | <2GB | <4GB | >8GB |
| Success Rate | >95% | >90% | <80% |

---

*GAAP Comprehensive Reference Tables - Last Updated: February 17, 2026*
