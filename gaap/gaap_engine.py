# GAAP Engine
# 400+ lines
import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from gaap.context.orchestrator import ContextOrchestrator
from gaap.core.memory_guard import MemoryGuard, get_rss_mb
from gaap.core.types import TaskPriority, TaskType
from gaap.healing.healer import SelfHealingSystem
from gaap.layers.layer0_interface import Layer0Interface, StructuredIntent
from gaap.layers.layer1_strategic import ArchitectureSpec, Layer1Strategic
from gaap.layers.layer2_tactical import AtomicTask, Layer2Tactical, TaskCategory, TaskGraph
from gaap.layers.layer3_execution import ExecutionResult, Layer3Execution
from gaap.memory.hierarchical import HierarchicalMemory
from gaap.providers.free_tier import GeminiProvider, GroqProvider
from gaap.providers.unified_gaap_provider import UnifiedGAAPProvider
from gaap.routing.fallback import FallbackManager
from gaap.routing.router import RoutingStrategy, SmartRouter
from gaap.security.firewall import AuditTrail, PromptFirewall

# =============================================================================
# Logger Setup
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class GAAPRequest:
    """طلب GAAP"""

    text: str
    context: dict[str, Any] | None = None
    priority: TaskPriority = TaskPriority.NORMAL
    budget_limit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GAAPResponse:
    """استجابة GAAP"""

    request_id: str
    success: bool
    output: Any = None
    error: str | None = None

    # رحلة الطلب
    intent: StructuredIntent | None = None
    architecture_spec: ArchitectureSpec | None = None
    task_graph: TaskGraph | None = None
    execution_results: list[ExecutionResult] = field(default_factory=list)

    # المقاييس
    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0

    # الجودة
    quality_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# GAAP Engine
# =============================================================================


class GAAPEngine:
    """
    محرك GAAP الرئيسي

    يجمع كل الطبقات:
    - Layer 0: Interface (فحص أمني، تصنيف)
    - Layer 1: Strategic (تخطيط معماري)
    - Layer 2: Tactical (تفكيك المهام)
    - Layer 3: Execution (تنفيذ وجودة)
    """

    def __init__(
        self,
        providers: list | None = None,
        budget: float = 100.0,
        enable_context: bool = True,
        enable_healing: bool = True,
        enable_memory: bool = True,
        enable_security: bool = True,
        project_path: str | None = None,
    ):
        self._logger = get_logger("gaap.engine")

        # إعداد المزودين (الافتراضي: المزود الموحد الذي يبدأ بـ Kimi)
        if providers is None:
            providers = [UnifiedGAAPProvider()]

        self.providers = providers

        # إعداد التوجيه
        self.router = SmartRouter(
            providers=providers, strategy=RoutingStrategy.SMART, budget_limit=budget
        )

        # إعداد الـ Fallback
        self.fallback = FallbackManager(router=self.router)

        # الأنظمة الداعمة
        self.context_orchestrator = None
        self.healing_system = None
        self.memory = None
        self.firewall = None
        self.audit_trail = None

        if enable_context and project_path:
            self.context_orchestrator = ContextOrchestrator(project_path=project_path)

        if enable_healing:
            self.healing_system = SelfHealingSystem()

        if enable_memory:
            self.memory = HierarchicalMemory()

        if enable_security:
            self.firewall = PromptFirewall(strictness="high")
            self.audit_trail = AuditTrail()

        # الطبقات
        self.layer0 = Layer0Interface(firewall_strictness="high", enable_behavioral_analysis=True)

        self.layer1 = Layer1Strategic(
            tot_depth=5, tot_branching=4, mad_rounds=3, provider=providers[0] if providers else None
        )

        self.layer2 = Layer2Tactical(
            max_subtasks=5, max_parallel=3, provider=providers[0] if providers else None
        )

        self.layer3 = Layer3Execution(
            router=self.router, fallback=self.fallback, enable_twin=False, max_parallel=3
        )

        # === Memory Safety ===
        # RLIMIT_AS removed — it limits virtual address space (not RSS),
        # which conflicts with curl_cffi/asyncio thread pool virtual mappings.
        # Protection is now: MemoryGuard (RSS monitoring) + systemd-run MemoryMax (OS cgroup limit).
        self._memory_guard = MemoryGuard(max_rss_mb=4096, warn_rss_mb=2048)

        # الإحصائيات
        self._requests_processed = 0
        self._successful_requests = 0
        self._failed_requests = 0

    async def process(self, request: GAAPRequest) -> GAAPResponse:
        """معالجة طلب كامل"""
        start_time = time.time()
        request_id = f"req_{int(time.time()*1000)}"

        self._logger.info(f"Processing request {request_id}")
        self._requests_processed += 1

        # Reset provider health at start of each request so previous
        # failures (e.g. rate-limits) don't permanently block providers
        if self.fallback:
            self.fallback.reset_health()

        response = GAAPResponse(request_id=request_id, success=False)

        try:
            # ========== Layer 0: Interface ==========
            self._logger.debug("Layer 0: Interface")

            # فحص أمني
            if self.firewall:
                scan_result = self.firewall.scan(request.text, request.context)
                if not scan_result.is_safe:
                    response.error = f"Security risk detected: {scan_result.risk_level.name}"
                    response.metadata["security_scan"] = scan_result.to_dict()
                    return response

            # تصنيف النية
            intent = await self.layer0.process(request.text)
            response.intent = intent

            # توجيه بناءً على نوع الطلب
            if intent.routing_target.value == "layer3_execution":
                # طلب بسيط - تنفيذ مباشر
                result = await self._direct_execution(request, intent)
                response.output = result.output
                response.success = result.success
                response.execution_results.append(result)

            else:
                # ========== Layer 1: Strategic ==========
                self._logger.info(f"Layer 1: Strategic — RSS={get_rss_mb():.0f}MB")

                spec = await self.layer1.process(intent)
                response.architecture_spec = spec

                self._logger.info(f"Layer 1 done — RSS={get_rss_mb():.0f}MB")
                gc.collect()

                # ========== Layer 2: Tactical ==========
                self._logger.info(f"Layer 2: Tactical — RSS={get_rss_mb():.0f}MB")

                graph = await self.layer2.process(spec)
                response.task_graph = graph

                self._logger.info(
                    f"Layer 2 done — RSS={get_rss_mb():.0f}MB, tasks={graph.total_tasks}"
                )
                gc.collect()

                # ========== Layer 3: Execution ==========
                self._logger.info(f"Layer 3: Execution — RSS={get_rss_mb():.0f}MB")

                # الحصول على المهام الجاهزة
                completed: set = set()
                failed: set = set()
                in_progress: set = set()
                task_attempts: dict = {}  # track retry count per task
                MAX_TASK_RETRIES = 2

                while True:
                    ready_tasks = graph.get_ready_tasks(completed | failed, in_progress)

                    if not ready_tasks:
                        break

                    # تنفيذ الدفعة (واحدة واحدة لتقليل الضغط)
                    for task in ready_tasks:
                        in_progress.add(task.id)

                        # Memory check before each task
                        self._memory_guard.check(context=f"before task {task.id}")

                        # التنفيذ مع تعافي
                        if self.healing_system:
                            result = await self._execute_with_healing(task)
                        else:
                            result = await self.layer3.process(task)

                        response.execution_results.append(result)

                        if result.success:
                            completed.add(task.id)
                        else:
                            task_attempts[task.id] = task_attempts.get(task.id, 0) + 1
                            if task_attempts[task.id] >= MAX_TASK_RETRIES:
                                self._logger.warning(
                                    f"Task {task.id} failed after {MAX_TASK_RETRIES} attempts, skipping"
                                )
                                failed.add(task.id)
                        in_progress.discard(task.id)

                        # تنظيف ذاكرة + cooldown بين المهام
                        gc.collect()
                        self._memory_guard.check(context=f"after task {task.id}")
                        await asyncio.sleep(2)

                # تجميع النتائج
                outputs = [r.output for r in response.execution_results if r.success and r.output]
                response.output = "\n\n".join(str(o) for o in outputs[:5])  # أول 5 نتائج
                response.success = any(r.success for r in response.execution_results)

            # تسجيل في الـ Audit Trail
            if self.audit_trail:
                self.audit_trail.record(
                    action="process_request",
                    agent_id="gaap_engine",
                    resource=request_id,
                    result="success" if response.success else "failed",
                )

            # تحديث الإحصائيات
            if response.success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

        except Exception as e:
            self._logger.error(f"Request {request_id} failed: {e}")
            import traceback

            self._logger.error(f"Traceback:\n{traceback.format_exc()}")
            response.error = str(e)
            self._failed_requests += 1

        # حساب المقاييس
        response.total_time_ms = (time.time() - start_time) * 1000
        response.total_cost_usd = sum(r.cost_usd for r in response.execution_results)
        response.total_tokens = sum(r.tokens_used for r in response.execution_results)
        response.quality_score = sum(r.quality_score for r in response.execution_results) / max(
            len(response.execution_results), 1
        )

        self._logger.info(
            f"Request {request_id} completed: "
            f"success={response.success}, "
            f"time={response.total_time_ms:.0f}ms, "
            f"cost=${response.total_cost_usd:.4f}"
        )

        return response

    async def _direct_execution(
        self, request: GAAPRequest, intent: StructuredIntent
    ) -> ExecutionResult:
        """تنفيذ مباشر للطلبات البسيطة"""
        # إنشاء مهمة بسيطة
        task = AtomicTask(
            id=f"direct_{int(time.time()*1000)}",
            name="Direct Execution",
            description=request.text,
            category=TaskCategory.SETUP,
            type=intent.intent_type.name.lower().startswith("code")
            and TaskType.CODE_GENERATION
            or TaskType.ANALYSIS,
        )

        return await self.layer3.process(task)

    async def _execute_with_healing(self, task: AtomicTask) -> ExecutionResult:
        """تنفيذ مع تعافي ذاتي"""

        async def execute_func(t):
            return await self.layer3.process(t)

        # محاولة التنفيذ
        try:
            result = await execute_func(task)

            if result.success:
                return result

            # محاولة التعافي
            if self.healing_system and result.error:
                healing_result = await self.healing_system.heal(
                    error=Exception(result.error),
                    task=task.to_task(),
                    execute_func=lambda t: self.layer3.process(
                        AtomicTask(
                            id=t.id,
                            name=t.description[:50],
                            description=t.description,
                            category=TaskCategory.SETUP,
                            type=t.type,
                        )
                    ),
                )

                if healing_result.success:
                    return ExecutionResult(
                        task_id=task.id,
                        success=True,
                        output=healing_result.result,
                        healing_level=healing_result.level.value,
                        retries=healing_result.attempts,
                    )

            return result

        except Exception as e:
            return ExecutionResult(task_id=task.id, success=False, error=str(e))

    async def chat(self, message: str, context: dict[str, Any] | None = None) -> str:
        """محادثة بسيطة"""
        request = GAAPRequest(text=message, context=context)

        response = await self.process(request)

        return response.output or response.error or "No response"

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات النظام"""
        return {
            "requests_processed": self._requests_processed,
            "successful": self._successful_requests,
            "failed": self._failed_requests,
            "success_rate": self._successful_requests / max(self._requests_processed, 1),
            "layer0_stats": self.layer0.get_stats(),
            "layer1_stats": self.layer1.get_stats(),
            "layer2_stats": self.layer2.get_stats(),
            "layer3_stats": self.layer3.get_stats(),
            "router_stats": self.router.get_routing_stats(),
        }

    def shutdown(self) -> None:
        """إيقاف المحرك وإغلاق المزودين"""
        for provider in self.providers:
            try:
                provider.shutdown()
            except Exception as e:
                self._logger.warning(f"Provider shutdown warning: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================


def create_engine(
    groq_api_key: str | None = None,
    gemini_api_key: str | None = None,
    gemini_api_keys: list[str] | None = None,
    budget: float = 100.0,
    project_path: str | None = None,
    enable_all: bool = True,
) -> GAAPEngine:
    """إنشاء محرك GAAP"""
    providers = []

    # Groq
    if groq_api_key:
        providers.append(GroqProvider(api_key=groq_api_key))

    # Gemini (single key + key pool)
    if gemini_api_keys is None:
        env_keys_raw = os.environ.get("GEMINI_API_KEYS", "")
        gemini_api_keys = [k.strip() for k in env_keys_raw.split(",") if k.strip()]

    if gemini_api_key:
        gemini_api_keys = [gemini_api_key] + [k for k in gemini_api_keys if k != gemini_api_key]

    if gemini_api_keys:
        providers.append(
            GeminiProvider(
                api_key=gemini_api_keys[0],
                api_keys=gemini_api_keys,
            )
        )

    # Fallback مجاني (Kimi-first) فقط إذا لا توجد مفاتيح مزودات حقيقية
    if not providers:
        providers.append(UnifiedGAAPProvider())

    return GAAPEngine(
        providers=providers,
        budget=budget,
        enable_context=enable_all,
        enable_healing=enable_all,
        enable_memory=enable_all,
        enable_security=enable_all,
        project_path=project_path,
    )


async def quick_chat(message: str, groq_api_key: str | None = None, budget: float = 10.0) -> str:
    """محادثة سريعة"""
    engine = create_engine(groq_api_key=groq_api_key, budget=budget, enable_all=False)

    return await engine.chat(message)
