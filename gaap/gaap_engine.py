# GAAP Engine - OODA Loop Implementation (Evolution 2026)
"""
Enhanced OODA Loop with:
- Constitutional Gatekeeper: Blocks INVARIANT violations
- Dynamic Few-Shot Injection: Fetches examples from VectorMemory
- Lessons Injection: Passes lessons to task context
- Enhanced Back-propagation: AXIOM_VIOLATION triggers replanning
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gaap.core.memory_guard import MemoryGuard, get_rss_mb
from gaap.core.types import OODAPhase, OODAState, ReplanTrigger, Task, TaskPriority, TaskType
from gaap.healing.healer import SelfHealingSystem
from gaap.layers.layer0_interface import Layer0Interface, StructuredIntent
from gaap.layers.layer1_strategic import ArchitectureSpec, Layer1Strategic
from gaap.layers.layer2_tactical import AtomicTask, Layer2Tactical, TaskCategory, TaskGraph
from gaap.layers.layer3_execution import ExecutionResult, Layer3Execution
from gaap.memory.hierarchical import HierarchicalMemory
from gaap.providers.base_provider import BaseProvider
from gaap.providers.unified_gaap_provider import UnifiedGAAPProvider
from gaap.routing.fallback import FallbackManager
from gaap.routing.router import RoutingStrategy, SmartRouter
from gaap.security.firewall import AuditTrail, PromptFirewall
from gaap.security.dlp import DLPScanner
from gaap.core.observer import create_observer
from gaap.core.observability import Observability
from gaap.core.governance import SOPGatekeeper
from gaap.core.exceptions import AxiomViolationError

if TYPE_CHECKING:
    from gaap.core.axioms import AxiomValidator, AxiomCheckResult
    from gaap.core.observer import EnvironmentState


from gaap.core.logging import get_standard_logger as get_logger


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

    intent: StructuredIntent | None = None
    architecture_spec: ArchitectureSpec | None = None
    task_graph: TaskGraph | None = None
    execution_results: list[ExecutionResult] = field(default_factory=list)

    total_time_ms: float = 0.0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    quality_score: float = 0.0

    ooda_iterations: int = 0
    strategic_replan_count: int = 0
    axiom_violation_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


class GAAPEngine:
    """
    محرك GAAP الرئيسي - OODA Loop Implementation

    يجمع كل الطبقات في دورة OODA:
    - OBSERVE: مراقبة البيئة والذاكرة
    - ORIENT: تحديث التخطيط بناءً على الملاحظات
    - DECIDE: اختيار المهمة التالية
    - ACT: تنفيذ مع Axiom Enforcement
    - LEARN: انعكاس فوري وتخزين الدروس
    """

    MAX_OODA_ITERATIONS = 15
    MAX_TASK_RETRIES = 2

    def __init__(
        self,
        providers: list | None = None,
        budget: float = 100.0,
        enable_healing: bool = True,
        enable_memory: bool = True,
        enable_security: bool = True,
        enable_axiom_enforcement: bool = True,
        enable_metacognition: bool = True,
        enable_mcp: bool = True,
        project_path: str | None = None,
    ) -> None:
        from gaap.core.axioms import create_validator
        from gaap.core.observer import create_observer

        self._logger = get_logger("gaap.engine")

        if providers is None:
            providers = [UnifiedGAAPProvider()]

        self.providers = providers

        self.router = SmartRouter(
            providers=providers, strategy=RoutingStrategy.SMART, budget_limit=budget
        )

        self.fallback = FallbackManager(router=self.router)

        self.healing_system: SelfHealingSystem | None = None
        self.memory: HierarchicalMemory | None = None
        self.firewall: PromptFirewall | None = None
        self.audit_trail: AuditTrail | None = None
        self.metacognition_engine: Any | None = None

        if enable_healing:
            self.healing_system = SelfHealingSystem()

        if enable_memory:
            self.memory = HierarchicalMemory()

        if enable_security:
            self.firewall = PromptFirewall(strictness="high")
            self.audit_trail = AuditTrail()
            self.dlp = DLPScanner()  # v2: Security Shield

        self.layer0 = Layer0Interface(
            firewall_strictness="high",
            enable_behavioral_analysis=True,
            provider=providers[0] if providers else None,
        )
        self.layer1 = Layer1Strategic(
            tot_depth=5, tot_branching=4, mad_rounds=3, provider=providers[0] if providers else None
        )

        self.layer2 = Layer2Tactical(
            max_subtasks=5, max_parallel=3, provider=providers[0] if providers else None
        )

        # v2: Optional MCP Support
        self.mcp_client = None
        if enable_mcp:
            from gaap.tools import MCPClient, MCP_AVAILABLE

            if MCP_AVAILABLE:
                self.mcp_client = MCPClient(timeout=30)

        self.layer3 = Layer3Execution(
            router=self.router,
            fallback=self.fallback,
            enable_twin=False,
            max_parallel=3,
            provider=providers[0] if providers else None,
        )

        self._memory_guard = MemoryGuard(max_rss_mb=4096, warn_rss_mb=2048)

        self.axiom_validator: AxiomValidator | None = None
        if enable_axiom_enforcement:
            self.axiom_validator = create_validator(strict=True)

        self.observer = create_observer(memory_system=self.memory)

        self.reflector: Any = None
        if enable_memory:
            from gaap.core.reflection import RealTimeReflector

            self.reflector = RealTimeReflector()

        # v2: Observability (tracing + Prometheus metrics)
        self.observability = Observability()

        # v2: SOP Gatekeeper — ensures tasks are completed per role SOPs
        self.sop_gatekeeper = SOPGatekeeper()

        self._requests_processed = 0
        self._successful_requests = 0
        self._failed_requests = 0

    def __repr__(self) -> str:
        return f"GAAPEngine(providers={len(self.providers)}, budget={self.router._budget_limit})"

    async def process(self, request: GAAPRequest) -> GAAPResponse:
        """
        معالجة طلب كامل - OODA Loop
        """
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"

        self._logger.info(f"[OODA] Processing request {request_id}")
        self._requests_processed += 1

        if self.fallback:
            self.fallback.reset_health()

        response = GAAPResponse(request_id=request_id, success=False)

        ooda = OODAState(request_id=request_id, max_iterations=self.MAX_OODA_ITERATIONS)

        try:
            intent = await self._initial_observe(request, ooda, response)
            if intent is None:
                return response

            if intent.routing_target.value == "layer3_execution":
                result = await self._direct_execution(request, intent)
                response.output = result.output
                response.success = result.success
                response.execution_results.append(result)
                return response

            spec = await self._initial_orient(intent, ooda, response)
            if spec is None:
                return response

            graph = await self._initial_tactical(spec, ooda, response)
            if graph is None:
                return response

            goal_achieved = await self._ooda_loop(ooda, graph, response)

            outputs = [r.output for r in response.execution_results if r.success and r.output]
            final_output = "\n\n".join(str(o) for o in outputs[:5])

            # Final Security Pass (DLP Shield)
            if self.firewall and self.dlp and final_output:
                self._logger.info("Applying DLP Security Shield to output...")
                original_len = len(final_output)
                final_output = self.dlp.scan_and_redact(final_output)
                if len(final_output) != original_len:
                    response.metadata["dlp_triggered"] = True
                    self._logger.warning("DLP: Potential leak blocked in final response.")

            response.output = final_output
            response.success = goal_achieved or any(r.success for r in response.execution_results)

            if self.audit_trail:
                self.audit_trail.record(
                    action="process_request",
                    agent_id="gaap_engine",
                    resource=request_id,
                    result="success" if response.success else "failed",
                )

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

        response.total_time_ms = (time.time() - start_time) * 1000
        response.total_cost_usd = sum(r.cost_usd for r in response.execution_results)
        response.total_tokens = sum(r.tokens_used for r in response.execution_results)
        response.quality_score = sum(r.quality_score for r in response.execution_results) / max(
            len(response.execution_results), 1
        )

        response.ooda_iterations = ooda.iteration
        response.strategic_replan_count = ooda.replan_count
        response.axiom_violation_count = len(ooda.axiom_violations)

        # v2: Record to observability
        self.observability.record_llm_call(
            provider=self.providers[0].name if self.providers else "unknown",
            model=self.providers[0].default_model if self.providers else "unknown",
            input_tokens=response.total_tokens // 2,
            output_tokens=response.total_tokens - response.total_tokens // 2,
            cost=response.total_cost_usd,
            latency=response.total_time_ms / 1000,
            success=response.success,
        )

        self._logger.info(
            f"[OODA] Request {request_id} completed: "
            f"success={response.success}, "
            f"iterations={ooda.iteration}, "
            f"replans={ooda.replan_count}, "
            f"axioms={len(ooda.axiom_violations)}"
        )

        return response

    async def _initial_observe(
        self, request: GAAPRequest, ooda: OODAState, response: GAAPResponse
    ) -> StructuredIntent | None:
        """المرحلة الأولى: ملاحظة وتصنيف الطلب"""
        ooda.current_phase = OODAPhase.OBSERVE

        if self.firewall:
            scan_result = self.firewall.scan(request.text, request.context)
            if not scan_result.is_safe:
                response.error = f"Security risk detected: {scan_result.risk_level.name}"
                response.metadata["security_scan"] = scan_result.to_dict()
                return None

        intent = await self.layer0.process(request.text)
        response.intent = intent

        ooda.last_observation = {
            "intent_type": intent.intent_type.name,
            "routing_target": intent.routing_target.value,
            "goals": intent.explicit_goals[:5],
        }

        return intent

    async def _initial_orient(
        self, intent: StructuredIntent, ooda: OODAState, response: GAAPResponse
    ) -> ArchitectureSpec | None:
        """المرحلة الأولى: توجيه استراتيجي"""
        ooda.current_phase = OODAPhase.ORIENT

        self._logger.info(f"[OODA] Layer 1: Strategic — RSS={get_rss_mb():.0f}MB")

        spec = await self.layer1.process(intent)
        response.architecture_spec = spec

        self._logger.info(f"[OODA] Layer 1 done — RSS={get_rss_mb():.0f}MB")
        gc.collect()

        return spec

    async def _initial_tactical(
        self, spec: ArchitectureSpec, ooda: OODAState, response: GAAPResponse
    ) -> TaskGraph | None:
        """المرحلة الأولى: تنظيم تكتيكي"""
        ooda.current_phase = OODAPhase.DECIDE

        self._logger.info(f"[OODA] Layer 2: Tactical — RSS={get_rss_mb():.0f}MB")

        graph = await self.layer2.process(spec)
        response.task_graph = graph

        self._logger.info(
            f"[OODA] Layer 2 done — RSS={get_rss_mb():.0f}MB, tasks={graph.total_tasks}"
        )
        gc.collect()

        return graph

    async def _ooda_loop(self, ooda: OODAState, graph: TaskGraph, response: GAAPResponse) -> bool:
        """
        Main OODA Loop - Parallel Execution Enabled (Wavefront)
        """
        task_attempts: dict[str, int] = {}

        while not ooda.goal_achieved and ooda.iteration < ooda.max_iterations:
            ooda.advance_phase()

            # Observe
            state = await self._observe_phase(ooda, graph)
            ooda.last_observation = state.to_dict()

            if state.needs_replanning:
                await self._orient_replan(ooda, state, graph, response)

            # Decide (Parallel Batch)
            next_tasks = self._decide_phase(ooda, graph)

            if not next_tasks:
                if not ooda.in_progress_tasks:
                    self._logger.info("[OODA] No tasks remaining. Goal achieved/exhausted.")
                    ooda.goal_achieved = True
                    break
                else:
                    self._logger.debug(
                        f"[OODA] Waiting for {len(ooda.in_progress_tasks)} active tasks..."
                    )
                    await asyncio.sleep(0.5)
                    continue

            # Act (Parallel Execution)
            self._logger.info(f"[OODA] Executing batch of {len(next_tasks)} tasks")

            results = await asyncio.gather(
                *[self._act_phase(task, ooda, response) for task in next_tasks]
            )

            # Process Results
            for i, result in enumerate(results):
                task = next_tasks[i]
                ooda.in_progress_tasks.discard(task.id)  # Important: Clear in-progress

                if result.success:
                    ooda.completed_tasks.add(task.id)
                    self.layer2.complete_task(task.id)
                    task_attempts.pop(task.id, None)
                else:
                    attempts = task_attempts.get(task.id, 0) + 1
                    task_attempts[task.id] = attempts

                    if attempts >= self.MAX_TASK_RETRIES:
                        self._logger.warning(
                            f"[OODA] Task {task.id} failed after {self.MAX_TASK_RETRIES} attempts (Permanent)"
                        )
                        ooda.failed_tasks.add(task.id)
                        self.layer2.fail_task(task.id)
                    else:
                        self._logger.info(
                            f"[OODA] Task {task.id} failed (Attempt {attempts}/{self.MAX_TASK_RETRIES}) - Retrying..."
                        )

                self._learn_phase(result, ooda)

            self._memory_guard.check(context=f"OODA iteration {ooda.iteration}")
            gc.collect()
            await asyncio.sleep(0.1)  # Yield

        return ooda.goal_achieved or len(ooda.completed_tasks) > 0

    async def _observe_phase(self, ooda: OODAState, graph: TaskGraph) -> EnvironmentState:
        """OODA Phase: Observe"""
        ooda.current_phase = OODAPhase.OBSERVE

        original_goals = []
        response = getattr(self, "_current_response", None)
        if response and response.intent:
            original_goals = response.intent.explicit_goals

        state = await self.observer.scan(ooda, task_graph=graph, original_goals=original_goals)

        if self.memory:
            lessons = state.lessons_available[:3]
            if lessons:
                self._logger.info(f"[OODA] Lessons available: {len(lessons)}")

        return state

    async def _orient_replan(
        self, ooda: OODAState, state: EnvironmentState, graph: TaskGraph, response: GAAPResponse
    ) -> None:
        """
        OODA Phase: Orient - Enhanced Back-propagation

        Evolution 2026:
        - Handles L3_CRITICAL_FAILURE (existing)
        - Handles AXIOM_VIOLATION (new) - strategic replanning
        - Handles RESOURCE_EXHAUSTED (existing)
        - Handles GOAL_DRIFT (new) - realignment
        """
        ooda.current_phase = OODAPhase.ORIENT

        trigger = state.replan_trigger
        ooda.trigger_replan(trigger)

        self._logger.warning(
            f"[OODA] Replanning triggered: {trigger.name}, count={ooda.replan_count}"
        )

        if trigger == ReplanTrigger.L3_CRITICAL_FAILURE:
            if response.architecture_spec and response.intent:
                new_spec = await self.layer1.process(response.intent)
                response.architecture_spec = new_spec

                new_graph = await self.layer2.process(new_spec)
                response.task_graph = new_graph

                self._logger.info(
                    f"[OODA] Replanned: new spec, new graph with {new_graph.total_tasks} tasks"
                )

        elif trigger == ReplanTrigger.AXIOM_VIOLATION:
            self._logger.warning(f"[OODA] Constitutional violations detected - adjusting strategy")

            if response.architecture_spec and response.intent:
                violated_axioms = set(v.get("axiom", "unknown") for v in ooda.axiom_violations)

                self._logger.info(
                    f"[OODA] Violated axioms: {violated_axioms} - requesting alternative approach"
                )

                response.intent.metadata["avoid_axiom_violations"] = list(violated_axioms)

                new_spec = await self.layer1.process(response.intent)
                response.architecture_spec = new_spec

                new_graph = await self.layer2.process(new_spec)
                response.task_graph = new_graph

                ooda.axiom_violations.clear()

        elif trigger == ReplanTrigger.RESOURCE_EXHAUSTED:
            gc.collect()
            await asyncio.sleep(3)

        elif trigger == ReplanTrigger.GOAL_DRIFT:
            self._logger.warning("[OODA] Goal drift detected - realigning")
            if response.intent:
                response.architecture_spec = await self.layer1.process(response.intent)
                response.task_graph = await self.layer2.process(response.architecture_spec)

        ooda.needs_replanning = False

    def _decide_phase(self, ooda: OODAState, graph: TaskGraph) -> list[AtomicTask]:
        """OODA Phase: Decide - Return Batch for Parallel Processing"""
        ooda.current_phase = OODAPhase.DECIDE

        # Get all tasks ready for execution
        ready_tasks = graph.get_ready_tasks(ooda.completed_tasks, ooda.in_progress_tasks)

        if not ready_tasks:
            self._logger.debug("[OODA] No ready tasks available")
            return []

        # Limit batch size to prevent overloading
        MAX_BATCH_SIZE = 5
        batch_tasks = ready_tasks[:MAX_BATCH_SIZE]

        # Mark all as in-progress
        for task in batch_tasks:
            ooda.in_progress_tasks.add(task.id)
            self._logger.info(f"[OODA] Decided: task={task.id} '{task.name}'")

        if batch_tasks:
            # Preserve partial compat with 'task_id' expectation of observers
            ooda.last_decision = {
                "task_id": batch_tasks[0].id,
                "task_name": batch_tasks[0].name,
                "batch_size": len(batch_tasks),
                "tasks": [t.name for t in batch_tasks],
            }

        return batch_tasks

    async def _act_phase(
        self, task: AtomicTask, ooda: OODAState, response: GAAPResponse
    ) -> ExecutionResult:
        """
        OODA Phase: Act - Execute with Constitutional Enforcement

        Evolution 2026:
        - Fetches few-shot examples from VectorMemory
        - Injects lessons learned into task context
        - Blocks execution on INVARIANT axiom violations
        - Triggers replanning on repeated violations
        """
        ooda.current_phase = OODAPhase.ACT

        self._logger.info(f"[OODA] Executing task {task.id}")

        self._memory_guard.check(context=f"before task {task.id}")

        enriched_task = await self._enrich_task_context(task, ooda)

        result: ExecutionResult

        if self.healing_system:
            result = await self._execute_with_healing(enriched_task)
        else:
            result = await self.layer3.process(enriched_task)

        if self.axiom_validator:
            result = self._enforce_axioms(result, task, ooda, response)

        response.execution_results.append(result)
        ooda.in_progress_tasks.discard(task.id)

        return result

    async def _enrich_task_context(self, task: AtomicTask, ooda: OODAState) -> AtomicTask:
        """
        Enrich task context with few-shot examples and lessons.

        Dynamic Few-Shot Injection (Spec 2.3):
        - Fetches relevant examples from VectorMemory
        - Injects lessons from previous iterations
        """
        enriched_metadata = dict(task.metadata) if task.metadata else {}

        if self.memory:
            try:
                from gaap.memory import LessonStore

                lesson_store = LessonStore()
                lessons = lesson_store.retrieve_lessons(task.description, k=3)
                if lessons:
                    enriched_metadata["relevant_lessons"] = [
                        lesson.content if hasattr(lesson, "content") else str(lesson)
                        for lesson in lessons
                    ]
                    self._logger.info(f"[OODA] Injected {len(lessons)} lessons for task {task.id}")
            except Exception as e:
                self._logger.warning(f"[OODA] Failed to enrich task context: {e}")

        if ooda.lessons_learned:
            enriched_metadata["session_lessons"] = ooda.lessons_learned[-5:]

        return AtomicTask(
            id=task.id,
            name=task.name,
            description=task.description,
            category=task.category,
            type=task.type,
            priority=task.priority,
            complexity=task.complexity,
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            dependencies=task.dependencies,
            dependency_type=task.dependency_type,
            estimated_tokens=task.estimated_tokens,
            estimated_time_minutes=task.estimated_time_minutes,
            estimated_cost_usd=task.estimated_cost_usd,
            status=task.status,
            result=task.result,
            assigned_agent=task.assigned_agent,
            retry_count=task.retry_count,
            metadata=enriched_metadata,
        )

    def _enforce_axioms(
        self,
        result: ExecutionResult,
        task: AtomicTask,
        ooda: OODAState,
        response: GAAPResponse,
    ) -> ExecutionResult:
        """
        Constitutional Gatekeeper (Spec 2.2)

        Enforces axiom validation:
        - INVARIANT level violations BLOCK execution
        - GUIDELINE violations trigger warnings
        - Repeated violations trigger replanning
        """
        if not self.axiom_validator or not result.success:
            return result

        axiom_results = self.axiom_validator.validate(
            code=result.output if isinstance(result.output, str) else None,
            task_id=task.id,
        )

        violations = [r for r in axiom_results if not r.passed]

        if not violations:
            return result

        from gaap.core.axioms import AxiomLevel

        invariant_violations = []
        for violation in violations:
            ooda.record_axiom_violation(violation.to_dict())

            axiom_info = self.axiom_validator.axioms.get(violation.axiom_name)
            is_invariant = axiom_info and axiom_info.level == AxiomLevel.INVARIANT

            self._logger.warning(
                f"[OODA] Axiom violation: {violation.axiom_name} "
                f"(level={'INVARIANT' if is_invariant else 'GUIDELINE'}) - {violation.message}"
            )

            if is_invariant:
                invariant_violations.append(violation)

        if invariant_violations:
            response.axiom_violation_count += len(invariant_violations)

            if len(ooda.axiom_violations) >= 3:
                self._logger.error(
                    f"[OODA] Multiple axiom violations detected - triggering replanning"
                )
                ooda.trigger_replan(ReplanTrigger.AXIOM_VIOLATION)

            return ExecutionResult(
                task_id=task.id,
                success=False,
                output=result.output,
                error=f"Constitutional violation: {invariant_violations[0].message}",
                quality_score=0.0,
            )

        return result

    def _learn_phase(self, result: ExecutionResult, ooda: OODAState) -> None:
        """OODA Phase: Learn"""
        ooda.current_phase = OODAPhase.LEARN

        if result.success:
            if self.memory and result.output:
                from gaap.memory.hierarchical import EpisodicMemory

                episode = EpisodicMemory(
                    task_id=result.task_id,
                    action="execution",
                    result=str(result.output)[:200],
                    success=True,
                    duration_ms=result.latency_ms,
                    tokens_used=result.tokens_used,
                    cost_usd=result.cost_usd,
                    model=result.metadata.get("model", "unknown"),
                    provider=result.metadata.get("provider", "unknown"),
                )
                self.memory.record_episode(episode)
                ooda.lessons_learned.append(f"Success: {result.task_id}")

        if self.reflector:
            reflections = self.reflector.reflect(
                task_id=result.task_id,
                success=result.success,
                duration_ms=result.latency_ms,
                tokens_used=result.tokens_used,
                cost_usd=result.cost_usd,
                model=result.metadata.get("model", "unknown"),
                provider=result.metadata.get("provider", "unknown"),
                error=result.error,
                output=str(result.output)[:200] if result.output else None,
                quality_score=result.quality_score,
            )
            for r in reflections:
                ooda.lessons_learned.append(r.lesson[:100])

        if not result.success:
            lesson = (
                f"Failure: {result.task_id} - {result.error[:100] if result.error else 'unknown'}"
            )
            ooda.lessons_learned.append(lesson)
            if self.observer:
                self.observer.record_error(lesson)

    async def _direct_execution(
        self, request: GAAPRequest, intent: StructuredIntent
    ) -> ExecutionResult:
        """تنفيذ مباشر للطلبات البسيطة"""
        task = AtomicTask(
            id=f"direct_{int(time.time() * 1000)}",
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

        async def execute_func(t: AtomicTask) -> ExecutionResult:
            return await self.layer3.process(t)

        try:
            result = await execute_func(task)

            if result.success:
                return result

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
        stats: dict[str, Any] = {
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

        if self.axiom_validator:
            stats["axiom_stats"] = self.axiom_validator.get_stats()

        stats["observer_stats"] = self.observer.get_stats()

        if self.reflector:
            stats["reflector_stats"] = self.reflector.get_stats()

        return stats

    def get_ooda_stats(self) -> dict[str, Any]:
        """إحصائيات OODA"""
        return {
            "observer_stats": self.observer.get_stats(),
            "axiom_violation_rate": (
                self.axiom_validator.get_stats()["violation_rate"] if self.axiom_validator else 0.0
            ),
            "strategic_replans": (
                self.axiom_validator.get_stats()["checks_run"] if self.axiom_validator else 0
            ),
            "reflector_stats": self.reflector.get_stats() if self.reflector else {},
        }

    def shutdown(self) -> None:
        """إيقاف المحرك وإغلاق المزودين"""
        for provider in self.providers:
            try:
                provider.shutdown()
            except Exception as e:
                self._logger.warning(f"Provider shutdown warning: {e}")


def create_engine(
    budget: float = 100.0,
    project_path: str | None = None,
    enable_all: bool = True,
    enable_mcp: bool = True,
    provider: BaseProvider | None = None,
) -> GAAPEngine:
    """
    Create a GAAP Engine instance.

    Args:
        budget: Budget limit in USD
        project_path: Optional project path
        enable_all: Enable all optional features
        enable_mcp: Enable MCP tool support
        provider: Optional custom provider (uses UnifiedGAAPProvider if not provided)

    Returns:
        Configured GAAPEngine instance
    """
    providers: list[BaseProvider] = []

    if provider is not None:
        providers.append(provider)
    else:
        providers.append(UnifiedGAAPProvider())

    return GAAPEngine(
        providers=providers,
        budget=budget,
        enable_healing=enable_all,
        enable_memory=enable_all,
        enable_security=enable_all,
        enable_axiom_enforcement=enable_all,
        enable_mcp=enable_mcp,
        project_path=project_path,
    )


async def quick_chat(
    message: str, budget: float = 10.0, provider: BaseProvider | None = None
) -> str:
    """محادثة سريعة"""
    engine = create_engine(budget=budget, enable_all=False, provider=provider)

    return await engine.chat(message)
