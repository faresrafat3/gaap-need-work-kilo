# Layer 3: Execution Layer
import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from gaap.core.base import BaseLayer
from gaap.core.types import (
    CriticEvaluation,
    CriticType,
    LayerType,
    MADDecision,
    Message,
    MessageRole,
    TaskPriority,
    TaskType,
)
from gaap.layers.layer2_tactical import AtomicTask
from gaap.providers.base_provider import BaseProvider
from gaap.routing.fallback import FallbackManager
from gaap.routing.router import SmartRouter

# =============================================================================
# Logger Setup
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Enums
# =============================================================================

class ExecutionMode(Enum):
    """أوضاع التنفيذ"""
    SINGLE = "single"          # تنفيذ واحد
    GENETIC_TWIN = "twin"       # توأم جيني
    CONSENSUS = "consensus"     # إجماع متعدد


class QualityGate(Enum):
    """بوابات الجودة"""
    SYNTAX_CHECK = auto()
    LOGIC_REVIEW = auto()
    SECURITY_SCAN = auto()
    PERFORMANCE_CHECK = auto()
    STYLE_CHECK = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExecutionResult:
    """نتيجة تنفيذ"""
    task_id: str
    success: bool
    output: Any = None
    error: str | None = None

    # المقاييس
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # الجودة
    quality_score: float = 0.0
    critic_evaluations: list[CriticEvaluation] = field(default_factory=list)

    # الـ twin
    twin_used: bool = False
    twin_agreement: float = 1.0

    # التعافي
    healing_level: int = 0
    retries: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentExecutor:
    """منفذ وكيل"""
    id: str
    specialization: TaskType
    provider: BaseProvider
    model: str

    # الإحصائيات
    tasks_completed: int = 0
    success_rate: float = 1.0
    avg_latency_ms: float = 0.0


# =============================================================================
# Genetic Twin System
# =============================================================================

class GeneticTwin:
    """
    نظام التوأم الجيني
    
    يشغل نسختين من الحل ويقارن بينهما:
    - Model Diversity: نماذج مختلفة
    - Prompt Diversity: صياغات مختلفة
    - Temperature Diversity: درجات حرارة مختلفة
    """

    def __init__(
        self,
        enabled: bool = True,
        similarity_threshold: float = 0.95,
        for_critical_only: bool = True
    ):
        self.enabled = enabled
        self.similarity_threshold = similarity_threshold
        self.for_critical_only = for_critical_only
        self._logger = get_logger("gaap.layer3.twin")

        # الإحصائيات
        self._twins_spawned = 0
        self._agreements = 0
        self._disagreements = 0

    async def execute_with_twin(
        self,
        task: AtomicTask,
        execute_func: Callable[[AtomicTask, dict[str, Any]], Awaitable[Any]],
        is_critical: bool = False
    ) -> tuple[Any, float, bool]:
        """
        تنفيذ مع توأم
        
        Returns:
            (result, agreement_score, twin_was_used)
        """
        # التحقق من الحاجة للتوأم
        if not self.enabled:
            return await execute_func(task, {}), 1.0, False

        if self.for_critical_only and not is_critical:
            return await execute_func(task, {}), 1.0, False

        self._twins_spawned += 1

        # تشغيل نسختين متوازيتين
        primary_config = {"temperature": 0.7}
        twin_config = {"temperature": 0.3}  # أكثر دقة

        # تنفيذ متوازي
        results = await asyncio.gather(
            execute_func(task, primary_config),
            execute_func(task, twin_config),
            return_exceptions=True
        )

        primary_result = results[0] if not isinstance(results[0], Exception) else None
        twin_result = results[1] if not isinstance(results[1], Exception) else None

        # المقارنة
        if primary_result and twin_result:
            agreement = self._calculate_similarity(primary_result, twin_result)

            if agreement >= self.similarity_threshold:
                self._agreements += 1
                self._logger.debug(f"Twin agreement: {agreement:.2%}")
                return primary_result, agreement, True
            else:
                self._disagreements += 1
                self._logger.warning(f"Twin disagreement: {agreement:.2%}")

                # اختيار الأفضل (يمكن تحسينه)
                return primary_result, agreement, True

        # إذا فشل أحد التوأمين
        return primary_result or twin_result, 0.5, True

    def _calculate_similarity(self, result1: Any, result2: Any) -> float:
        """حساب التشابه"""
        # تحويل لنص
        text1 = str(result1) if result1 else ""
        text2 = str(result2) if result2 else ""

        # حساب Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "twins_spawned": self._twins_spawned,
            "agreements": self._agreements,
            "disagreements": self._disagreements,
            "agreement_rate": self._agreements / max(self._twins_spawned, 1),
        }


# =============================================================================
# MAD Quality Panel
# =============================================================================

class MADQualityPanel:
    """
    لجنة جودة MAD
    
    تتكون من نقاد متخصصين:
    - Logic Critic: صحة المنطق
    - Security Critic: الأمان
    - Performance Critic: الأداء
    - Style Critic: الأسلوب
    """

    CRITIC_WEIGHTS = {
        CriticType.LOGIC: 0.35,
        CriticType.SECURITY: 0.25,
        CriticType.PERFORMANCE: 0.20,
        CriticType.STYLE: 0.10,
        CriticType.COMPLIANCE: 0.05,
        CriticType.ETHICS: 0.05,
    }

    def __init__(
        self,
        min_score: float = 70.0,
        unanimous_for_critical: bool = True
    ):
        self.min_score = min_score
        self.unanimous_for_critical = unanimous_for_critical
        self._logger = get_logger("gaap.layer3.mad")

        # الإحصائيات
        self._evaluations_count = 0
        self._rejections = 0

    async def evaluate(
        self,
        artifact: Any,
        task: AtomicTask,
        critics: list[CriticType] | None = None
    ) -> MADDecision:
        """تقييم المخرجات"""
        self._evaluations_count += 1

        # تحديد النقاد
        if critics is None:
            critics = [CriticType.LOGIC, CriticType.SECURITY,
                      CriticType.PERFORMANCE, CriticType.STYLE]

        evaluations = []

        for critic_type in critics:
            evaluation = await self._evaluate_with_critic(
                artifact, task, critic_type
            )
            evaluations.append(evaluation)

        # حساب الدرجة النهائية
        final_score = self._calculate_final_score(evaluations)

        # تحديد القبول
        approved = self._determine_approval(
            final_score,
            evaluations,
            task.priority == TaskPriority.CRITICAL
        )

        if not approved:
            self._rejections += 1

        # جمع التغييرات المطلوبة
        required_changes = []
        for eval_result in evaluations:
            if not eval_result.approved:
                required_changes.extend(eval_result.suggestions)

        return MADDecision(
            consensus=approved,
            final_score=final_score,
            evaluations=evaluations,
            required_changes=required_changes,
            decision_reasoning=f"Score: {final_score:.1f}, Critics: {len(evaluations)}"
        )

    async def _evaluate_with_critic(
        self,
        artifact: Any,
        task: AtomicTask,
        critic_type: CriticType
    ) -> CriticEvaluation:
        """تقييم بناقد محدد"""
        # تحويل artifact لنص
        artifact_text = str(artifact) if artifact else ""

        # تقييم مبسط (في الإنتاج، سيستخدم LLM)
        score = self._simple_evaluate(artifact_text, task, critic_type)

        approved = score >= self.min_score

        issues = []
        suggestions = []

        if not approved:
            if critic_type == CriticType.SECURITY:
                issues.append("Potential security concern detected")
                suggestions.append("Review for injection vulnerabilities")
            elif critic_type == CriticType.LOGIC:
                issues.append("Logic issues detected")
                suggestions.append("Verify edge cases")
            elif critic_type == CriticType.PERFORMANCE:
                issues.append("Performance concerns")
                suggestions.append("Consider optimization")

        return CriticEvaluation(
            critic_type=critic_type,
            score=score,
            approved=approved,
            issues=issues,
            suggestions=suggestions,
            reasoning=f"Evaluated by {critic_type.name}"
        )

    def _simple_evaluate(
        self,
        artifact: str,
        task: AtomicTask,
        critic_type: CriticType
    ) -> float:
        """تقييم بسيط"""
        score = 70.0  # درجة أساسية

        # تعديلات بناءً على النوع
        if critic_type == CriticType.LOGIC:
            # فحص الكلمات المفتاحية
            if "error" in artifact.lower():
                score -= 20
            if "return" in artifact.lower():
                score += 10

        elif critic_type == CriticType.SECURITY:
            # فحص الأمان
            dangerous_patterns = ["eval(", "exec(", "sql", "password"]
            for pattern in dangerous_patterns:
                if pattern in artifact.lower():
                    score -= 15

        elif critic_type == CriticType.PERFORMANCE:
            # فحص الأداء
            if "O(n" in artifact:
                score += 5

        elif critic_type == CriticType.STYLE:
            # فحص الأسلوب
            if len(artifact) > 0:
                score += 5

        return max(min(score, 100), 0)

    def _calculate_final_score(
        self,
        evaluations: list[CriticEvaluation]
    ) -> float:
        """حساب الدرجة النهائية"""
        total_weight = 0.0
        weighted_sum = 0.0

        for evaluation in evaluations:
            weight = self.CRITIC_WEIGHTS.get(evaluation.critic_type, 0.1)
            weighted_sum += evaluation.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0

    def _determine_approval(
        self,
        final_score: float,
        evaluations: list[CriticEvaluation],
        is_critical: bool
    ) -> bool:
        """تحديد القبول"""
        # فحص الدرجة الأساسية
        if final_score < self.min_score:
            return False

        # للمهام الحرجة، نحتاج إجماع
        if is_critical and self.unanimous_for_critical:
            return all(e.approved for e in evaluations)

        return True

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "evaluations": self._evaluations_count,
            "rejections": self._rejections,
            "rejection_rate": self._rejections / max(self._evaluations_count, 1),
        }


# =============================================================================
# Executor Pool
# =============================================================================

class ExecutorPool:
    """مجمع المنفذين"""

    def __init__(
        self,
        router: SmartRouter,
        fallback: FallbackManager,
        max_parallel: int = 10
    ):
        self.router = router
        self.fallback = fallback
        self.max_parallel = max_parallel

        self._executors: dict[str, AgentExecutor] = {}
        self._logger = get_logger("gaap.layer3.executor")

        # الإحصائيات
        self._tasks_executed = 0
        self._successful = 0
        self._failed = 0

    def register_executor(self, executor: AgentExecutor) -> None:
        """تسجيل منفذ"""
        self._executors[executor.id] = executor

    async def execute(
        self,
        task: AtomicTask,
        context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """تنفيذ مهمة"""
        start_time = time.time()
        context = context or {}

        self._tasks_executed += 1

        try:
            # تحضير الرسائل
            messages = self._prepare_messages(task)

            # الحصول على قرار التوجيه
            decision = await self.router.route(
                messages=messages,
                task=task.to_task()
            )

            # تنفيذ مع fallback
            response = await self.fallback.execute_with_fallback(
                messages=messages,
                primary_provider=decision.selected_provider,
                primary_model=decision.selected_model
            )

            # استخراج النتيجة
            output = response.choices[0].message.content if response.choices else ""

            # حساب التكلفة
            cost = decision.estimated_cost
            if response.usage:
                cost = (
                    response.usage.prompt_tokens * 0.00001 +
                    response.usage.completion_tokens * 0.00003
                )

            self._successful += 1

            return ExecutionResult(
                task_id=task.id,
                success=True,
                output=output,
                latency_ms=response.latency_ms,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                cost_usd=cost,
                metadata={
                    "provider": decision.selected_provider,
                    "model": decision.selected_model,
                }
            )

        except Exception as e:
            self._failed += 1

            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
                retries=task.retry_count,
            )

    def _prepare_messages(self, task: AtomicTask) -> list[Message]:
        """تحضير الرسائل"""
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are an expert software developer. Provide clean, efficient, and well-documented code."
            ),
            Message(
                role=MessageRole.USER,
                content=task.description
            )
        ]

        # إضافة القيود
        if task.constraints:
            constraints_text = "Constraints:\n"
            for key, value in task.constraints.items():
                constraints_text += f"- {key}: {value}\n"
            messages.insert(1, Message(
                role=MessageRole.SYSTEM,
                content=constraints_text
            ))

        # إضافة معايير القبول
        if task.acceptance_criteria:
            criteria_text = "Acceptance Criteria:\n"
            for criterion in task.acceptance_criteria:
                criteria_text += f"- {criterion}\n"
            messages.append(Message(
                role=MessageRole.USER,
                content=criteria_text
            ))

        return messages

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "tasks_executed": self._tasks_executed,
            "successful": self._successful,
            "failed": self._failed,
            "success_rate": self._successful / max(self._tasks_executed, 1),
        }


# =============================================================================
# Quality Pipeline
# =============================================================================

class QualityPipeline:
    """خط أنابيب الجودة"""

    def __init__(
        self,
        mad_panel: MADQualityPanel,
        twin_system: GeneticTwin
    ):
        self.mad_panel = mad_panel
        self.twin_system = twin_system
        self._logger = get_logger("gaap.layer3.quality")

    async def process(
        self,
        artifact: Any,
        task: AtomicTask,
        is_critical: bool = False
    ) -> tuple[Any, float, list[CriticEvaluation]]:
        """معالجة الجودة"""
        # تقييم MAD
        decision = await self.mad_panel.evaluate(artifact, task)

        return artifact, decision.final_score, decision.evaluations


# =============================================================================
# Layer 3 Execution
# =============================================================================

class Layer3Execution(BaseLayer):
    """
    طبقة التنفيذ والجودة
    
    المسؤوليات:
    - تنفيذ المهام الذرية
    - تطبيق التوأم الجيني
    - تقييم الجودة (MAD)
    - التعافي الذاتي
    """

    def __init__(
        self,
        router: SmartRouter,
        fallback: FallbackManager,
        enable_twin: bool = True,
        max_parallel: int = 10,
        quality_threshold: float = 70.0
    ):
        super().__init__(LayerType.EXECUTION)

        self.executor_pool = ExecutorPool(router, fallback, max_parallel)
        self.twin_system = GeneticTwin(enabled=enable_twin)
        self.mad_panel = MADQualityPanel(min_score=quality_threshold)
        self.quality_pipeline = QualityPipeline(self.mad_panel, self.twin_system)

        self._logger = get_logger("gaap.layer3")

        # الإحصائيات
        self._artifacts_produced = 0

    async def process(self, input_data: Any) -> ExecutionResult:
        """معالجة المدخل"""
        start_time = time.time()

        # استخراج المهمة
        if isinstance(input_data, AtomicTask):
            task = input_data
        else:
            raise ValueError("Expected AtomicTask")

        self._logger.info(f"Executing task {task.id}: {task.name}")

        # تحديد ما إذا كانت المهمة حرجة
        is_critical = task.priority in (TaskPriority.CRITICAL, TaskPriority.HIGH)

        # تنفيذ مع توأم جيني
        if self.twin_system.enabled and (is_critical or not self.twin_system.for_critical_only):
            result, agreement, twin_used = await self.twin_system.execute_with_twin(
                task,
                self.executor_pool.execute
            )
        else:
            result = await self.executor_pool.execute(task)
            agreement = 1.0
            twin_used = False

        # تقييم الجودة
        if result.success:
            artifact, quality_score, evaluations = await self.quality_pipeline.process(
                result.output, task, is_critical
            )

            result.quality_score = quality_score
            result.critic_evaluations = evaluations
            result.twin_used = twin_used
            result.twin_agreement = agreement

            self._artifacts_produced += 1

        result.latency_ms = (time.time() - start_time) * 1000

        return result

    async def execute_batch(
        self,
        tasks: list[AtomicTask]
    ) -> list[ExecutionResult]:
        """تنفيذ دفعة مهام"""
        results = []

        # تنفيذ متوازي
        for task in tasks:
            result = await self.process(task)
            results.append(result)

        return results

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "layer": "L3_Execution",
            "artifacts_produced": self._artifacts_produced,
            "executor_stats": self.executor_pool.get_stats(),
            "twin_stats": self.twin_system.get_stats(),
            "mad_stats": self.mad_panel.get_stats(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_execution_layer(
    router: SmartRouter,
    fallback: FallbackManager,
    enable_twin: bool = True,
    max_parallel: int = 10
) -> Layer3Execution:
    """إنشاء طبقة التنفيذ"""
    return Layer3Execution(
        router=router,
        fallback=fallback,
        enable_twin=enable_twin,
        max_parallel=max_parallel
    )
