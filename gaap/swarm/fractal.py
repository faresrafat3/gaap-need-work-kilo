"""
Fractal Agent - Intelligent Task Executor

A Fractal is a specialized sub-agent that:
- Has domain expertise (python, sql, security, etc.)
- Bids on tasks based on capability and load
- Learns from successes and failures
- Practices epistemic humility (predicting own failures)

Smart Behaviors:
1. **Self-Assessment**: Evaluates own capability before bidding
2. **Load Awareness**: Tracks current workload
3. **Memory Integration**: Uses episodic memory for task estimation
4. **Confidence Modeling**: Adjusts confidence based on task similarity
5. **Failure Prediction**: Can predict and decline unsuitable tasks
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any
import uuid

from gaap.core.types import Task, TaskResult, Message, MessageRole
from gaap.swarm.reputation import ReputationStore
from gaap.swarm.gisp_protocol import (
    TaskAuction,
    TaskBid,
    TaskAward,
    TaskResult as GISPResult,
    TaskDomain,
    TaskPriority,
    CapabilityAnnounce,
)


class FractalState(Enum):
    """حالة Fractal"""

    IDLE = auto()  # متاح للمهام
    BIDDING = auto()  # يشارك في مزاد
    EXECUTING = auto()  # ينفذ مهمة
    COOLDOWN = auto()  # فترة راحة بعد فشل
    OFFLINE = auto()  # غير متصل


@dataclass
class FractalCapability:
    """
    قدرات Fractal.
    """

    domain: str
    skill_level: float  # 0.0 to 1.0
    tasks_completed: int = 0
    success_rate: float = 0.5
    tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "skill_level": round(self.skill_level, 4),
            "tasks_completed": self.tasks_completed,
            "success_rate": round(self.success_rate, 4),
            "tools": self.tools,
        }


@dataclass
class TaskEstimate:
    """
    تقدير Fractal لمهمة.
    """

    can_execute: bool
    estimated_success: float
    estimated_cost: float
    estimated_time: float
    confidence: float
    rationale: str
    similar_tasks: int = 0
    risk_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_execute": self.can_execute,
            "estimated_success": round(self.estimated_success, 4),
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_time": round(self.estimated_time, 2),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "similar_tasks": self.similar_tasks,
            "risk_factors": self.risk_factors,
        }


class FractalAgent:
    """
    وكيل فرعي ذكي.

    Features:
    - Domain specialization with skill levels
    - Self-assessment before bidding
    - Episodic memory for estimation
    - Load tracking and management
    - Epistemic humility (failure prediction)

    Usage:
        fractal = FractalAgent(
            fractal_id="coder_01",
            domains=["python", "testing"],
            provider=my_provider,
            memory=my_memory,
            reputation_store=reputation_store,
        )

        # Register
        await fractal.announce_capabilities()

        # Evaluate task
        estimate = fractal.estimate_task(auction)
        if estimate.can_execute:
            bid = fractal.create_bid(auction, estimate)

        # Execute
        result = await fractal.execute_task(task, award)
    """

    def __init__(
        self,
        fractal_id: str,
        domains: list[str],
        provider: Any,  # BaseProvider
        memory: Any,  # HierarchicalMemory
        reputation_store: ReputationStore,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.fractal_id = fractal_id
        self.domains = domains
        self._provider = provider
        self._memory = memory
        self._reputation = reputation_store
        self._config = config or {}
        self._logger = logging.getLogger(f"gaap.swarm.fractal.{fractal_id}")

        # State
        self._state = FractalState.IDLE
        self._current_task: Task | None = None
        self._task_history: list[str] = []

        # Capabilities
        self._capabilities: dict[str, FractalCapability] = {}
        for domain in domains:
            self._capabilities[domain] = FractalCapability(
                domain=domain,
                skill_level=0.5,  # Start neutral
            )

        # Load management
        self._current_load = 0.0
        self._max_concurrent_tasks = self._config.get("max_concurrent_tasks", 1)
        self._active_tasks: dict[str, Task] = {}

        # Performance tracking
        self._total_tasks = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._predicted_failures = 0

        # Cooldown after failures
        self._cooldown_until: datetime | None = None
        self._cooldown_duration = self._config.get("cooldown_seconds", 60)

    @property
    def state(self) -> FractalState:
        """حالة Fractal الحالية"""
        # Check cooldown
        if self._cooldown_until and datetime.now() < self._cooldown_until:
            return FractalState.COOLDOWN
        if self._cooldown_until and datetime.now() >= self._cooldown_until:
            self._cooldown_until = None
            self._state = FractalState.IDLE

        if self._active_tasks:
            return FractalState.EXECUTING

        return self._state

    @property
    def current_load(self) -> float:
        """التحميل الحالي (0.0 to 1.0)"""
        return float(len(self._active_tasks)) / float(self._max_concurrent_tasks)

    async def announce_capabilities(self) -> CapabilityAnnounce:
        """
        إعلان القدرات للـ Orchestrator.
        """
        return CapabilityAnnounce(
            fractal_id=self.fractal_id,
            domains=self.domains,
            skills=list(self._capabilities.keys()),
            tools=[],  # TODO: Get from provider
            max_concurrent_tasks=self._max_concurrent_tasks,
            preferred_task_types=self.domains,
        )

    def can_bid_on(self, auction: TaskAuction) -> bool:
        """
        التحقق من إمكانية المشاركة في المزاد.
        """
        # Check state
        if self.state not in (FractalState.IDLE, FractalState.BIDDING):
            return False

        # Check load
        if self.current_load >= 1.0:
            return False

        # Check domain match
        domain = auction.domain.value
        if domain not in self.domains and domain != TaskDomain.GENERAL.value:
            return False

        # Check reputation threshold
        reputation = self._reputation.get_domain_reputation(self.fractal_id, domain)
        if reputation < auction.min_reputation:
            return False

        return True

    def estimate_task(self, auction: TaskAuction) -> TaskEstimate:
        """
        تقدير قدرة Fractal على تنفيذ المهمة.

        This is where epistemic humility kicks in:
        - Fractal assesses its own capability
        - Checks memory for similar tasks
        - Identifies risk factors
        - May decline if uncertain
        """
        domain = auction.domain.value

        # Get capability
        capability = self._capabilities.get(domain)
        if not capability:
            capability = FractalCapability(domain=domain, skill_level=0.3)

        # Base estimates
        base_success = capability.success_rate if capability.tasks_completed > 0 else 0.5

        # Adjust for complexity
        complexity_factor = 1.0 - (auction.complexity / 20)  # complexity 1-10

        # Check memory for similar tasks
        similar_tasks = 0
        memory_boost = 0.0
        if self._memory:
            try:
                similar = self._memory.retrieve(
                    query=auction.task_description,
                    domain=domain,
                    k=5,
                )
                similar_tasks = len(similar)
                if similar_tasks > 0:
                    memory_boost = min(0.1 * similar_tasks, 0.3)
            except Exception as e:
                self._logger.warning(f"Memory query failed: {e}")

        # Compute estimates
        estimated_success = min(1.0, base_success * complexity_factor + memory_boost)

        # Cost estimate (based on complexity and domain)
        base_cost = 50.0  # tokens
        estimated_cost = (
            base_cost * auction.complexity * (1.5 if domain not in self.domains else 1.0)
        )

        # Time estimate
        base_time = 10.0  # seconds
        estimated_time = (
            base_time * auction.complexity * (1.5 if domain not in self.domains else 1.0)
        )

        # Confidence
        confidence = 0.5 + 0.05 * min(similar_tasks, 5)
        if capability.tasks_completed > 0:
            confidence = min(0.9, confidence + 0.1)

        # Risk factors
        risk_factors = []
        if domain not in self.domains:
            risk_factors.append("domain_mismatch")
        if auction.complexity > 7:
            risk_factors.append("high_complexity")
        if auction.priority == TaskPriority.CRITICAL:
            risk_factors.append("critical_priority")
        if self.current_load > 0.5:
            risk_factors.append("high_load")

        # Can execute decision (epistemic humility)
        can_execute = estimated_success >= 0.3 and len(risk_factors) < 3

        if not can_execute:
            self._logger.info(
                f"Declining task {auction.task_id}: success={estimated_success:.2f}, "
                f"risks={risk_factors}"
            )

        return TaskEstimate(
            can_execute=can_execute,
            estimated_success=estimated_success,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            confidence=confidence,
            rationale=f"Domain: {domain}, Similar tasks: {similar_tasks}, "
            f"Capability: {capability.skill_level:.2f}",
            similar_tasks=similar_tasks,
            risk_factors=risk_factors,
        )

    def create_bid(self, auction: TaskAuction, estimate: TaskEstimate) -> TaskBid:
        """
        إنشاء عرض للمهمة.
        """
        domain = auction.domain.value
        reputation = self._reputation.get_domain_reputation(self.fractal_id, domain)

        bid = TaskBid(
            task_id=auction.task_id,
            bidder_id=self.fractal_id,
            estimated_success_rate=estimate.estimated_success,
            estimated_cost_tokens=estimate.estimated_cost,
            estimated_time_seconds=estimate.estimated_time,
            confidence_in_estimate=estimate.confidence,
            rationale=estimate.rationale,
            similar_tasks_completed=estimate.similar_tasks,
            current_load=self.current_load,
        )

        # Compute utility
        bid.compute_utility_score(
            reputation=reputation,
            max_cost=auction.constraints.get("max_cost", estimate.estimated_cost * 2),
            max_time=auction.constraints.get("max_time", estimate.estimated_time * 2),
        )

        self._logger.info(
            f"Created bid for {auction.task_id}: utility={bid.utility_score:.4f}, "
            f"success={bid.estimated_success_rate:.2f}"
        )

        return bid

    async def execute_task(
        self,
        task: Task,
        award: TaskAward,
    ) -> GISPResult:
        """
        تنفيذ المهمة.

        This is where the actual work happens:
        1. Mark as executing
        2. Execute via provider
        3. Assess result quality
        4. Update reputation
        5. Record in memory
        """
        self._state = FractalState.EXECUTING
        self._active_tasks[task.id] = task
        self._current_task = task

        start_time = datetime.now()
        predicted_success = True  # We bid, so we predict success

        # Get pre-execution confidence
        domain = getattr(task, "domain", "general")
        confidence_before = self._reputation.get_domain_reputation(self.fractal_id, domain)

        try:
            # Execute via provider
            result = await self._execute_via_provider(task)

            # Assess result
            success = result.success
            quality_score = self._assess_quality(result)

            # Calculate metrics
            duration = (datetime.now() - start_time).total_seconds()
            cost = self._estimate_tokens_used(result)

            # Update reputation
            if success:
                self._reputation.record_success(self.fractal_id, domain)
                self._successful_tasks += 1
            else:
                self._reputation.record_failure(self.fractal_id, domain, predicted=False)
                self._failed_tasks += 1
                predicted_success = False

            # Update capability
            self._update_capability(domain, success, quality_score)

            # Record in memory
            if self._memory:
                await self._record_episode(task, result, success)

            return GISPResult(
                task_id=task.id,
                fractal_id=self.fractal_id,
                success=success,
                output=result.output,
                error=result.error,
                actual_cost_tokens=cost,
                actual_time_seconds=duration,
                quality_score=quality_score,
                predicted_success=predicted_success,
                confidence_before=confidence_before,
                confidence_after=self._reputation.get_domain_reputation(self.fractal_id, domain),
            )

        except Exception as e:
            self._logger.error(f"Task execution failed: {e}")

            # Record failure
            self._reputation.record_failure(self.fractal_id, domain, predicted=False)
            self._failed_tasks += 1

            # Start cooldown
            self._start_cooldown()

            return GISPResult(
                task_id=task.id,
                fractal_id=self.fractal_id,
                success=False,
                error=str(e),
                predicted_success=predicted_success,
                confidence_before=confidence_before,
                confidence_after=self._reputation.get_domain_reputation(self.fractal_id, domain),
            )

        finally:
            # Cleanup
            del self._active_tasks[task.id]
            self._current_task = None
            self._state = FractalState.IDLE
            self._total_tasks += 1
            self._task_history.append(task.id)

    async def _execute_via_provider(self, task: Task) -> TaskResult:
        """تنفيذ عبر الـ Provider"""
        messages = [
            Message(
                role=MessageRole.USER,
                content=task.description,
            )
        ]

        # Use provider to generate response
        if hasattr(self._provider, "chat_completion"):
            response = await self._provider.chat_completion(
                messages=messages,
                model=getattr(self._provider, "default_model", "default"),
            )

            return TaskResult(
                success=True,
                output=response.choices[0].message.content
                if hasattr(response, "choices")
                else str(response),
            )
        else:
            raise RuntimeError("Provider does not support chat_completion")

    def _assess_quality(self, result: TaskResult) -> float:
        """تقييم جودة النتيجة"""
        if not result.success:
            return 0.0

        # Basic quality assessment
        output = result.output or ""

        # Length check
        if len(output) < 10:
            return 0.3

        # Content indicators
        quality = 0.5

        # Good indicators
        if "error" not in output.lower():
            quality += 0.1
        if "todo" not in output.lower():
            quality += 0.1
        if len(output) > 100:
            quality += 0.1

        # Bad indicators
        if "failed" in output.lower():
            quality -= 0.2
        if "exception" in output.lower():
            quality -= 0.1

        return max(0.0, min(1.0, quality))

    def _estimate_tokens_used(self, result: TaskResult) -> float:
        """تقدير الـ tokens المستخدمة"""
        output = result.output or ""
        return len(output.split()) * 1.3  # Rough estimate

    def _update_capability(self, domain: str, success: bool, quality: float) -> None:
        """تحديث القدرة"""
        if domain not in self._capabilities:
            self._capabilities[domain] = FractalCapability(domain=domain, skill_level=0.3)

        cap = self._capabilities[domain]
        cap.tasks_completed += 1

        # Update success rate (running average)
        n = cap.tasks_completed
        if success:
            cap.success_rate = (cap.success_rate * (n - 1) + 1.0) / n
        else:
            cap.success_rate = (cap.success_rate * (n - 1) + 0.0) / n

        # Update skill level
        cap.skill_level = cap.success_rate * 0.7 + quality * 0.3

    async def _record_episode(
        self,
        task: Task,
        result: TaskResult,
        success: bool,
    ) -> None:
        """تسجيل حلقة في الذاكرة"""
        try:
            episode = {
                "task_id": task.id,
                "description": task.description,
                "success": success,
                "output_preview": (result.output or "")[:500],
                "timestamp": datetime.now().isoformat(),
            }

            if hasattr(self._memory, "record_episode"):
                await self._memory.record_episode(episode)
        except Exception as e:
            self._logger.warning(f"Failed to record episode: {e}")

    def _start_cooldown(self) -> None:
        """بدء فترة راحة بعد فشل"""
        self._cooldown_until = datetime.now() + timedelta(seconds=self._cooldown_duration)
        self._state = FractalState.COOLDOWN
        self._logger.info(f"Starting cooldown for {self._cooldown_duration}s")

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات Fractal"""
        return {
            "fractal_id": self.fractal_id,
            "state": self.state.name,
            "domains": self.domains,
            "current_load": round(self.current_load, 2),
            "total_tasks": self._total_tasks,
            "successful_tasks": self._successful_tasks,
            "failed_tasks": self._failed_tasks,
            "success_rate": (
                self._successful_tasks / self._total_tasks if self._total_tasks > 0 else 0.0
            ),
            "capabilities": {d: c.to_dict() for d, c in self._capabilities.items()},
            "cooldown_until": (self._cooldown_until.isoformat() if self._cooldown_until else None),
        }
