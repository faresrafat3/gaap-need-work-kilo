# Layer 3: Execution Layer
import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from gaap.core.base import BaseLayer
from gaap.core.streaming_auditor import AuditResult, StreamingAuditor
from gaap.tools import NativeToolCaller, ToolCall, MCPClient, MCPToolRegistry
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
from gaap.mad.critic_prompts import SYSTEM_PROMPTS, build_user_prompt
from gaap.mad.response_parser import (
    CriticParseError,
    fallback_evaluation,
    parse_critic_response,
)
from gaap.memory import VECTOR_MEMORY_AVAILABLE, LessonStore
from gaap.providers.base_provider import BaseProvider
from gaap.routing.fallback import FallbackManager
from gaap.routing.router import SmartRouter
from gaap.tools.synthesizer import ToolSynthesizer
from gaap.layers.sop_mixin import SOPExecutionMixin

from gaap.layers.layer3_config import Layer3Config
from gaap.layers.native_function_caller import create_native_caller
from gaap.layers.active_lesson_injector import create_lesson_injector
from gaap.layers.code_auditor import create_code_auditor

# =============================================================================
# Logger Setup
# =============================================================================


from gaap.core.logging import get_standard_logger as get_logger

# =============================================================================
# Helper Functions
# =============================================================================


def parse_tool_args(s: str) -> dict[str, str]:
    """
    Parse tool arguments from string format.

    Parses arguments like: param1='value1', param2='value2'
    Handles escaped quotes and whitespace.
    """
    result: dict[str, str] = {}
    i = 0
    while i < len(s):
        if s[i].isalpha() or s[i] == "_":
            key_start = i
            while i < len(s) and (s[i].isalnum() or s[i] == "_"):
                i += 1
            key = s[key_start:i]
            while i < len(s) and s[i] in " \t\n":
                i += 1
            if i < len(s) and s[i] == "=":
                i += 1
                while i < len(s) and s[i] in " \t\n":
                    i += 1
                if i < len(s) and s[i] in "\"'":
                    quote = s[i]
                    i += 1
                    value_start = i
                    while i < len(s):
                        if s[i] == "\\" and i + 1 < len(s):
                            i += 2
                        elif s[i] == quote:
                            break
                        else:
                            i += 1
                    value = s[value_start:i]
                    if i < len(s):
                        i += 1
                    result[key] = value
        else:
            i += 1
    return result


# =============================================================================
# Enums
# =============================================================================


class ExecutionMode(Enum):
    """أوضاع التنفيذ"""

    SINGLE = "single"  # تنفيذ واحد
    GENETIC_TWIN = "twin"  # توأم جيني
    CONSENSUS = "consensus"  # إجماع متعدد


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
        for_critical_only: bool = True,
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
        is_critical: bool = False,
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
            return_exceptions=True,
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
        # Software Engineering Weights
        CriticType.LOGIC: 0.35,
        CriticType.SECURITY: 0.25,
        CriticType.PERFORMANCE: 0.20,
        CriticType.STYLE: 0.10,
        CriticType.COMPLIANCE: 0.05,
        CriticType.ETHICS: 0.05,
        # Research & Intelligence Weights
        CriticType.ACCURACY: 0.40,
        CriticType.SOURCE_CREDIBILITY: 0.30,
        CriticType.COMPLETENESS: 0.20,
        # Diagnostics Weights
        CriticType.ROOT_CAUSE: 0.40,
        CriticType.RELIABILITY: 0.30,
    }

    def __init__(
        self,
        min_score: float = 70.0,
        unanimous_for_critical: bool = True,
        provider: BaseProvider | None = None,
        critic_model: str | None = None,
    ):
        self.min_score = min_score
        self.unanimous_for_critical = unanimous_for_critical
        self.provider = provider
        self.critic_model = critic_model
        self._logger = get_logger("gaap.layer3.mad")

        # الإحصائيات
        self._evaluations_count = 0
        self._rejections = 0
        self._llm_failures = 0

    async def evaluate(
        self, artifact: Any, task: AtomicTask, critics: list[CriticType] | None = None
    ) -> MADDecision:
        """تقييم المخرجات بناءً على نوع المهمة"""
        self._evaluations_count += 1

        from gaap.layers.layer2_tactical import TaskCategory

        # تحديد النقاد ديناميكياً حسب نوع المهمة
        if critics is None:
            # 1. Research & Analysis Panel
            if task.category in (
                TaskCategory.INFORMATION_GATHERING,
                TaskCategory.SOURCE_VERIFICATION,
                TaskCategory.DATA_SYNTHESIS,
                TaskCategory.LITERATURE_REVIEW,
                TaskCategory.ANALYSIS,
            ):
                critics = [
                    CriticType.ACCURACY,
                    CriticType.SOURCE_CREDIBILITY,
                    CriticType.COMPLETENESS,
                    CriticType.STYLE,
                ]

            # 2. Diagnostic & Troubleshooting Panel
            elif task.category in (
                TaskCategory.REPRODUCTION,
                TaskCategory.LOG_ANALYSIS,
                TaskCategory.ROOT_CAUSE_ANALYSIS,
                TaskCategory.DIAGNOSTIC_ACTION,
            ):
                critics = [
                    CriticType.ROOT_CAUSE,
                    CriticType.RELIABILITY,
                    CriticType.LOGIC,
                    CriticType.SECURITY,
                ]

            # 3. Standard Software Panel (Default)
            else:
                critics = [
                    CriticType.LOGIC,
                    CriticType.SECURITY,
                    CriticType.PERFORMANCE,
                    CriticType.STYLE,
                ]

        evaluations = []

        for critic_type in critics:
            evaluation = await self._evaluate_with_critic(artifact, task, critic_type)
            evaluations.append(evaluation)

        # حساب الدرجة النهائية
        final_score = self._calculate_final_score(evaluations)

        # تحديد القبول
        approved = self._determine_approval(
            final_score, evaluations, task.priority == TaskPriority.CRITICAL
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
            decision_reasoning=f"Score: {final_score:.1f}, Critics: {len(evaluations)}, Category: {task.category.name}",
        )

    async def _evaluate_with_critic(
        self, artifact: Any, task: AtomicTask, critic_type: CriticType
    ) -> CriticEvaluation:
        """تقييم بناقد محدد باستخدام LLM"""
        artifact_text = str(artifact) if artifact else ""

        if self.provider is None:
            self._logger.warning("No provider configured, using fallback evaluation")
            return self._fallback_critic_eval(artifact_text, task, critic_type)

        try:
            # Get specialized prompt for research/diagnostic critics if available
            system_prompt = (
                SYSTEM_PROMPTS.get(critic_type) or SYSTEM_PROMPTS.get(CriticType.LOGIC) or ""
            )
            user_prompt = build_user_prompt(artifact_text, task)

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.critic_model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices or not response.choices[0].message.content:
                self._logger.warning(f"LLM call failed for {critic_type.name}, using fallback")
                self._llm_failures += 1
                return self._fallback_critic_eval(artifact_text, task, critic_type)

            parsed = parse_critic_response(response.choices[0].message.content, critic_type)

            return CriticEvaluation(
                critic_type=critic_type,
                score=parsed["score"],
                approved=parsed["approved"],
                issues=parsed["issues"],
                suggestions=parsed["suggestions"],
                reasoning=parsed["reasoning"],
            )

        except CriticParseError as e:
            self._logger.warning(f"Parse error for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._fallback_critic_eval(artifact_text, task, critic_type)
        except Exception as e:
            self._logger.warning(f"LLM evaluation failed for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._fallback_critic_eval(artifact_text, task, critic_type)

    def _fallback_critic_eval(
        self, artifact: str, task: AtomicTask, critic_type: CriticType
    ) -> CriticEvaluation:
        """تقييم احتياطي باستخدام heuristics"""
        result = fallback_evaluation(critic_type, artifact)
        return CriticEvaluation(
            critic_type=critic_type,
            score=result["score"],
            approved=result["approved"],
            issues=result["issues"],
            suggestions=result["suggestions"],
            reasoning=result["reasoning"],
        )

    def _simple_evaluate(self, artifact: str, task: AtomicTask, critic_type: CriticType) -> float:
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

    def _calculate_final_score(self, evaluations: list[CriticEvaluation]) -> float:
        """حساب الدرجة النهائية الموزونة بناءً على نوع المهمة"""
        if not evaluations:
            return 0.0

        total_weighted_score = 0.0
        total_weight = 0.0

        for evaluation in evaluations:
            weight = self.CRITIC_WEIGHTS.get(evaluation.critic_type, 0.1)
            total_weighted_score += evaluation.score * weight
            total_weight += weight

        if total_weight == 0:
            return sum(e.score for e in evaluations) / len(evaluations)

        return total_weighted_score / total_weight

    def _determine_approval(
        self, final_score: float, evaluations: list[CriticEvaluation], is_critical: bool
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
            "llm_failures": self._llm_failures,
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
        max_parallel: int = 10,
        mcp_client: "MCPClient | None" = None,
    ):
        self.router = router
        self.fallback = fallback
        self.max_parallel = max_parallel

        self.native_tools = NativeToolCaller()
        self.mcp_registry = MCPToolRegistry(mcp_client) if mcp_client else None

        self._executors: dict[str, AgentExecutor] = {}
        self._logger = get_logger("gaap.layer3.executor")

        # الإحصائيات
        self._tasks_executed = 0
        self._successful = 0
        self._failed = 0

    def register_executor(self, executor: AgentExecutor) -> None:
        """تسجيل منفذ"""
        self._executors[executor.id] = executor

    def _get_persona_and_tools(self, task: AtomicTask) -> tuple[str, list[Any]]:
        """تحديد الشخصية والأدوات المناسبة حسب نوع المهمة"""
        from gaap.layers.layer2_tactical import TaskCategory

        if task.category in (
            TaskCategory.INFORMATION_GATHERING,
            TaskCategory.SOURCE_VERIFICATION,
            TaskCategory.DATA_SYNTHESIS,
            TaskCategory.LITERATURE_REVIEW,
            TaskCategory.ANALYSIS,
        ):
            persona = "You are an expert Research & Intelligence Agent. Focus on gathering high-quality information, verifying sources, and synthesizing data into actionable insights."
            tools = self.native_tools.get_tools_by_tags(
                ["research", "web", "retrieval", "analysis"]
            )
        elif task.category in (
            TaskCategory.REPRODUCTION,
            TaskCategory.LOG_ANALYSIS,
            TaskCategory.ROOT_CAUSE_ANALYSIS,
            TaskCategory.DIAGNOSTIC_ACTION,
        ):
            persona = "You are an expert Diagnostic Engineer. Focus on reproducing issues, analyzing logs, and identifying root causes through systematic isolation."
            tools = self.native_tools.get_tools_by_tags(
                ["diagnostic", "logs", "monitor", "tracing"]
            )
        else:
            persona = "You are an expert Software Engineer. Focus on writing clean, efficient, and secure code following best practices and architectural specs."
            tools = self.native_tools.get_tools_by_tags(
                ["coding", "file", "terminal", "testing", "database"]
            )

        return persona, tools

    async def _execute_tool_call(self, tool_name: str, args: dict[str, Any]) -> str:
        """تنفيذ استدعاء أداة"""
        try:
            if tool_name in self.native_tools._tools:
                return self.native_tools.execute_call(
                    ToolCall(name=tool_name, arguments=args)
                ).output
            elif self.mcp_registry:
                mcp_tools = await self.mcp_registry.get_mcp_tools()
                if tool_name in mcp_tools:
                    return await self.mcp_registry.execute_mcp_tool(tool_name, **args)
            return f"Error: Tool '{tool_name}' not found."
        except Exception as e:
            return f"Error executing tool: {e}"

    async def _execute_python_code(self, code: str) -> str:
        """تنفيذ كود Python في sandbox"""
        try:
            from gaap.security.sandbox import get_sandbox

            sandbox = get_sandbox(use_docker=True)
            result = await sandbox.execute(code, language="python")

            exec_result = f"STDOUT:\n{result.output}\nSTDERR:\n{result.error}"
            if not result.success:
                exec_result = f"FAILED (Exit {result.exit_code}):\n{exec_result}"

            self._logger.info(f"Sandbox execution complete ({result.execution_time_ms:.0f}ms)")
            return exec_result
        except Exception as e:
            return f"Execution Error: {e}"

    async def execute(
        self, task: AtomicTask, context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        """تنفيذ مهمة مع دعم الأدوات المتخصصة والمنظور المناسب"""
        start_time = time.time()
        context = context or {}

        self._tasks_executed += 1

        try:
            # 1. تحضير الرسائل الأولية
            messages = self._prepare_messages(task)

            # 2. تخصيص الشخصية وفلترة الأدوات
            persona, relevant_tools = self._get_persona_and_tools(task)

            # Inject Tool Instructions
            tool_instructions = self.native_tools.get_instructions(tools=relevant_tools)
            if self.mcp_registry:
                tool_instructions += "\n" + self.mcp_registry.get_tool_instructions()

            messages.insert(
                0, Message(role=MessageRole.SYSTEM, content=f"{persona}\n\n{tool_instructions}")
            )

            iteration = 0
            MAX_ITERATIONS = 7
            total_tokens = 0
            final_output = ""
            last_decision = None

            while iteration < MAX_ITERATIONS:
                iteration += 1

                # الحصول على قرار التوجيه
                decision = await self.router.route(messages=messages, task=task.to_task())
                last_decision = decision

                # تنفيذ مع fallback
                response = await self.fallback.execute_with_fallback(
                    messages=messages,
                    primary_provider=decision.selected_provider,
                    primary_model=decision.selected_model,
                )

                # استخراج النتيجة
                assistant_output = response.choices[0].message.content if response.choices else ""
                total_tokens += response.usage.total_tokens if response.usage else 0

                # إضافة رد المساعد للسجل
                messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_output))

                # 2. البحث عن استدعاء أداة CALL: tool_name(param='val') أو كود Python تلقائي
                tool_call_match = re.search(r"CALL:\s*(\w+)\((.*?)\)", assistant_output, re.DOTALL)
                python_code_match = re.search(r"```python\n(.*?)```", assistant_output, re.DOTALL)

                if tool_call_match:
                    tool_name = tool_call_match.group(1).strip()
                    args_str = tool_call_match.group(2).strip()

                    try:
                        args = parse_tool_args(args_str)
                    except Exception as e:
                        self._logger.warning(f"Failed to parse tool args: {e}")
                        args = {}

                    tool_result = await self._execute_tool_call(tool_name, args)

                    feedback = f"System: TOOL RESULT ({tool_name}): {tool_result}"
                    messages.append(Message(role=MessageRole.USER, content=feedback))

                    self._logger.info(f"Tool {tool_name} executed. Continuing loop with feedback.")
                    continue

                elif python_code_match:
                    code = python_code_match.group(1).strip()
                    self._logger.info("Detected Python code block, executing in sandbox...")

                    exec_result = await self._execute_python_code(code)
                    messages.append(
                        Message(
                            role=MessageRole.USER,
                            content=f"System: SANDBOX EXECUTION RESULT:\n{exec_result}",
                        )
                    )
                    continue

                else:
                    # لا توجد أدوات أخرى للاستدعاء، هذا هو الرد النهائي
                    final_output = assistant_output
                    break

            self._successful += 1

            return ExecutionResult(
                task_id=task.id,
                success=True,
                output=final_output,
                latency_ms=(time.time() - start_time) * 1000,
                tokens_used=total_tokens,
                cost_usd=0.0,  # WebChat is free
                metadata={
                    "iterations": iteration,
                    "provider": last_decision.selected_provider if last_decision else "unknown",
                    "model": last_decision.selected_model if last_decision else "unknown",
                },
            )

        except Exception as e:
            self._failed += 1
            self._logger.error(f"Execution failed for task {task.id}: {e}")
            import traceback

            self._logger.error(traceback.format_exc())

            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                latency_ms=(time.time() - start_time) * 1000,
                retries=task.retry_count,
            )

    def _prepare_messages(self, task: AtomicTask) -> list[Message]:
        """تحضير الرسائل الأولية للمهمة"""
        messages = [
            Message(role=MessageRole.USER, content=task.description),
        ]

        # إضافة القيود
        if task.constraints:
            constraints_text = "Constraints:\n"
            for key, value in task.constraints.items():
                constraints_text += f"- {key}: {value}\n"
            messages.insert(1, Message(role=MessageRole.SYSTEM, content=constraints_text))

        # إضافة معايير القبول
        if task.acceptance_criteria:
            criteria_text = "Acceptance Criteria:\n"
            for criterion in task.acceptance_criteria:
                criteria_text += f"- {criterion}\n"
            messages.append(Message(role=MessageRole.USER, content=criteria_text))

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
    """خط أنابيب الجودة مع لجنة النقاد"""

    def __init__(
        self,
        mad_panel: MADQualityPanel,
        twin_system: GeneticTwin,
    ):
        self.mad_panel = mad_panel
        self.twin_system = twin_system
        self._logger = get_logger("gaap.layer3.quality")

        self._artifacts: dict[str, Any] = {}
        self._validation_count = 0

    def register_artifact(self, name: str, content: Any) -> None:
        """Register an artifact for SOP validation"""
        self._artifacts[name] = content
        self._logger.debug(f"Registered artifact: {name}")

    def get_artifact(self, name: str) -> Any | None:
        """Get a registered artifact"""
        return self._artifacts.get(name)

    def get_all_artifacts(self) -> dict[str, Any]:
        """Get all registered artifacts"""
        return self._artifacts.copy()

    def clear_artifacts(self) -> None:
        """Clear all registered artifacts"""
        self._artifacts.clear()

    async def process(
        self, artifact: Any, task: AtomicTask, is_critical: bool = False
    ) -> tuple[Any, float, list[CriticEvaluation]]:
        """معالجة الجودة"""
        self._validation_count += 1
        decision = await self.mad_panel.evaluate(artifact, task)
        final_score = max(0, min(100, decision.final_score))
        return artifact, final_score, decision.evaluations

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "validations": self._validation_count,
        }


# =============================================================================
# Layer 3 Execution
# =============================================================================


class Layer3Execution(BaseLayer, SOPExecutionMixin):
    """
    طبقة التنفيذ والجودة

    المسؤوليات:
    - تنفيذ المهام الذرية
    - تطبيق التوأم الجيني
    - تقييم الجودة (MAD)
    - التعافي الذاتي

    Evolution 2026:
    - Native Function Calling
    - Active Lesson Injection
    - Code Auditing
    - Zero-Trust Execution
    """

    def __init__(
        self,
        router: SmartRouter,
        fallback: FallbackManager,
        enable_twin: bool = True,
        max_parallel: int = 10,
        quality_threshold: float = 70.0,
        provider: BaseProvider | None = None,
        enable_vector_memory: bool = True,
        enable_preflight: bool = True,
        enable_sop: bool = True,
        config: Layer3Config | None = None,
    ):
        super().__init__(LayerType.EXECUTION)

        self._config = config or Layer3Config()

        # Override config with explicit parameters
        if enable_sop != self._config.enable_sop:
            self._config.enable_sop = enable_sop

        self._init_sop(sop_enabled=self._config.enable_sop)

        self.max_parallel = self._config.max_parallel
        self.executor_pool = ExecutorPool(router, fallback, self.max_parallel)
        self.twin_system = GeneticTwin(
            enabled=self._config.enable_twin,
            for_critical_only=self._config.twin_for_critical_only,
        )
        self.mad_panel = MADQualityPanel(
            min_score=self._config.quality_threshold,
            provider=provider,
        )
        self.quality_pipeline = QualityPipeline(self.mad_panel, self.twin_system)
        self.synthesizer = ToolSynthesizer()

        self._native_caller = create_native_caller(self._config)
        self._lesson_injector = create_lesson_injector(self._config)
        self._code_auditor = create_code_auditor(self._config)
        self._auditor = StreamingAuditor(
            enabled=self._config.enable_streaming_audit,
            max_repetition=self._config.max_repetition,
        )

        self._logger = get_logger("gaap.layer3")

        if self._config.lesson_injection.enabled and VECTOR_MEMORY_AVAILABLE:
            self._lesson_store: LessonStore | None = LessonStore()
            self._lesson_injector._memory = self._lesson_store
            self._logger.debug("Vector memory enabled for lesson learning")
        else:
            self._lesson_store = None
            if self._config.lesson_injection.enabled:
                self._logger.debug("Vector memory requested but chromadb not available")

        self._preflight = None
        if self._config.audit.enabled:
            from gaap.security.preflight import PreFlightCheck

            self._preflight = PreFlightCheck(memory=self._lesson_store)
            self._logger.info("Pre-flight check enabled")

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

        self._set_audit_context(
            goal=task.description,
            keywords=[task.name, task.id, getattr(task, "task_type", "general")],
        )

        # Start SOP tracking
        self._start_sop_tracking(task)

        # Pre-flight check for code tasks (Existing logic)
        code_match = None
        task_code = getattr(task, "code", None)
        if task_code:
            code_match = task_code
        elif "```" in task.description:
            import re

            code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", task.description)
            if code_blocks:
                code_match = code_blocks[0]

        if code_match and self._preflight:
            preflight_report = self._preflight.check(
                code=code_match,
                task_id=task.id,
                task_description=task.description,
            )
            if not preflight_report.overall_passed:
                self._logger.warning(
                    f"Pre-flight check failed for {task.id}: {preflight_report.errors}"
                )
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=f"Pre-flight check failed: {'; '.join(preflight_report.errors[:3])}",
                    metadata={"preflight_report": preflight_report.to_dict()},
                )
            if preflight_report.lessons_injected:
                self._logger.debug(f"Injected {len(preflight_report.lessons_injected)} lessons")

        # تحديد ما إذا كانت المهمة حرجة
        is_critical = task.priority in (TaskPriority.CRITICAL, TaskPriority.HIGH)

        # تنفيذ مع توأم جيني
        if self.twin_system.enabled and (is_critical or not self.twin_system.for_critical_only):
            twin_result, agreement, twin_used = await self.twin_system.execute_with_twin(
                task, self.executor_pool.execute
            )
            if isinstance(twin_result, ExecutionResult):
                result = twin_result
            else:
                result = ExecutionResult(
                    task_id=task.id, success=False, error="Invalid twin result"
                )
        else:
            result = await self.executor_pool.execute(task)
            agreement = 1.0
            twin_used = False

        if result.success and result.output:
            audit_result = self._audit_execution_step(str(result.output))
            if audit_result.should_interrupt:
                self._logger.warning(f"Audit interrupt triggered: {audit_result.interrupt_message}")
                result.metadata["audit_interrupt"] = audit_result.to_dict()

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

            # Register artifact for SOP validation
            self._register_artifact("source_code", artifact)

            if self._lesson_store:
                await self._learn_from_execution(task, result, evaluations)

        result.latency_ms = (time.time() - start_time) * 1000

        # Handle Missing Tools via Synthesis (v2 Sovereignty)
        if not result.success and (
            "Tool" in str(result.error) and "not found" in str(result.error)
        ):
            self._logger.info(
                f"Missing tool detected for task '{task.id}'. Triggering Autonomous Synthesis..."
            )
            new_tool = await self.synthesizer.synthesize(
                intent=f"Tool for {task.name}",
                context={"code_proposal": self._generate_tool_logic_proposal(task)},
            )
            if new_tool:
                self._logger.info(
                    f"Successfully synthesized tool: {new_tool.name}. Re-executing..."
                )
                self.executor_pool.native_tools.register(
                    name=new_tool.name,
                    description=new_tool.description,
                    parameters={"args": "dict"},
                    func=getattr(new_tool.module, "run", None)
                    or getattr(new_tool.module, "execute", None),
                )
                return await self.process(task)  # Recursive retry with new capability

        # SOP completion validation
        if result.success and self._sop_enabled:
            sop_result = self._validate_sop_completion(task.id)
            if not sop_result["complete"]:
                self._logger.warning(
                    f"SOP incomplete for {task.id}: {sop_result.get('reason', 'unknown')}"
                )
                result.metadata["sop_status"] = sop_result

            # Handle reflexion if required
            if self._requires_reflexion():
                reflexion_prompt = self._get_reflexion_prompt()
                if reflexion_prompt:
                    result.metadata["reflexion_required"] = True
                    result.metadata["reflexion_prompt"] = reflexion_prompt
                    self._logger.info(f"Reflexion triggered for task {task.id}")

            result.metadata["sop_summary"] = self._get_sop_summary()

        return result

    def _generate_tool_logic_proposal(self, task: AtomicTask) -> str:
        """توليد مقترح كود للأداة المفقودة"""
        return f"""
def run(**kwargs):
    # Auto-generated logic for {task.name}
    print("Executing autonomous tool logic for {task.id}")
    return "Result synthesized from description: {task.description[:100]}"
"""

    async def _learn_from_execution(
        self,
        task: AtomicTask,
        result: "ExecutionResult",
        evaluations: list[CriticEvaluation],
    ) -> None:
        """التعلم من التنفيذ وتخزين الدروس حسب المجال (Research, Diagnostic, or Code)"""
        try:
            issues = []
            suggestions = []
            for eval_result in evaluations:
                if eval_result.issues:
                    issues.extend(eval_result.issues)
                if eval_result.suggestions:
                    suggestions.extend(eval_result.suggestions)

            if issues or suggestions:
                lesson = f"Task '{task.name}': "
                if issues:
                    lesson += f"Issues to avoid: {'; '.join(issues[:3])}. "
                if suggestions:
                    lesson += f"Improvements: {'; '.join(suggestions[:3])}."

                # Determine domain-specific category
                category = "execution"
                from gaap.layers.layer2_tactical import TaskCategory

                if task.category in (
                    TaskCategory.INFORMATION_GATHERING,
                    TaskCategory.LITERATURE_REVIEW,
                    TaskCategory.DATA_SYNTHESIS,
                ):
                    category = "research"
                elif task.category in (
                    TaskCategory.REPRODUCTION,
                    TaskCategory.ROOT_CAUSE_ANALYSIS,
                    TaskCategory.LOG_ANALYSIS,
                ):
                    category = "diagnostic"
                elif task.category in (
                    TaskCategory.API,
                    TaskCategory.DATABASE,
                    TaskCategory.FRONTEND,
                    TaskCategory.SECURITY,
                ):
                    category = "code"

                if self._lesson_store:
                    self._lesson_store.store_lesson(
                        lesson=lesson,
                        context="execution",
                        category=category,
                        task_type=getattr(task, "task_type", "general"),
                        success=result.success,
                    )
                self._logger.info(f"Learned specialized lesson in category: {category}")
        except Exception as e:
            self._logger.debug(f"Failed to store lesson: {e}")

    async def execute_batch(self, tasks: list[AtomicTask]) -> list[ExecutionResult]:
        """تنفيذ دفعة مهام بشكل متوازي فعلي"""
        if not tasks:
            return []

        semaphore = asyncio.Semaphore(self.max_parallel)

        async def _bounded_process(task: AtomicTask) -> ExecutionResult:
            async with semaphore:
                return await self.process(task)

        results = await asyncio.gather(
            *[_bounded_process(task) for task in tasks],
            return_exceptions=True,
        )

        final_results: list[ExecutionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                final_results.append(
                    ExecutionResult(
                        task_id=tasks[i].id,
                        success=False,
                        output="",
                        error=str(result),
                        quality_score=0.0,
                    )
                )
            elif isinstance(result, ExecutionResult):
                final_results.append(result)

        return final_results

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "layer": "L3_Execution",
            "artifacts_produced": self._artifacts_produced,
            "executor_stats": self.executor_pool.get_stats(),
            "twin_stats": self.twin_system.get_stats(),
            "mad_stats": self.mad_panel.get_stats(),
            "native_caller_stats": self._native_caller.get_stats(),
            "lesson_injector_stats": self._lesson_injector.get_stats(),
            "code_auditor_stats": self._code_auditor.get_stats(),
            "streaming_audit_stats": self.get_audit_stats(),
        }

    def _audit_execution_step(self, thought: str) -> AuditResult:
        """Audit a thought during execution"""
        return self._auditor.audit_thought(thought)

    def _set_audit_context(self, goal: str, keywords: list[str]) -> None:
        """Set context for audit drift detection"""
        self._auditor.set_context(goal, keywords)

    def get_audit_stats(self) -> dict[str, Any]:
        """Get streaming auditor statistics"""
        return self._auditor.get_stats()


def create_execution_layer(
    router: SmartRouter,
    fallback: FallbackManager,
    enable_twin: bool = True,
    max_parallel: int = 10,
    config: Layer3Config | None = None,
) -> Layer3Execution:
    """إنشاء طبقة التنفيذ"""
    return Layer3Execution(
        router=router,
        fallback=fallback,
        enable_twin=enable_twin,
        max_parallel=max_parallel,
        config=config,
    )
