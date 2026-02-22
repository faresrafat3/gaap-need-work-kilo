"""
Swarm Orchestrator - Central Coordinator

The Orchestrator manages the entire swarm:
- Registers and tracks Fractals
- Runs task auctions
- Monitors execution
- Forms Guilds
- Maintains system health

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    SwarmOrchestrator                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
    │  │  Registry   │  │  Auctioneer │  │   Monitor   │         │
    │  │  (Fractals) │  │  (Auctions) │  │  (Health)   │         │
    │  └─────────────┘  └─────────────┘  └─────────────┘         │
    │         │                │                │                 │
    │  ┌──────────────────────────────────────────────────────┐  │
    │  │                    Guild Manager                      │  │
    │  │   [Python Guild]  [SQL Guild]  [Security Guild]       │  │
    │  └──────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Flow:
    1. Task received from Layer 2
    2. Broadcast auction to eligible Fractals
    3. Collect bids
    4. Award task to highest utility
    5. Monitor execution
    6. Update reputations
    7. Form Guilds when thresholds met
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any
import uuid

from gaap.core.types import Task, TaskResult
from gaap.swarm.reputation import ReputationStore
from gaap.swarm.gisp_protocol import (
    TaskAuction,
    TaskBid,
    TaskAward,
    TaskResult as GISPResult,
    TaskDomain,
    TaskPriority,
    MessageType,
)
from gaap.swarm.auction import (
    TaskAuctioneer,
    AuctionConfig,
    AuctionResult,
    AuctionState,
)
from gaap.swarm.fractal import FractalAgent, FractalState
from gaap.swarm.guild import Guild, GuildState


class OrchestratorState(Enum):
    """حالة Orchestrator"""

    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()


@dataclass
class SwarmConfig:
    """إعدادات Swarm"""

    # Auction settings
    default_auction_timeout: float = 30.0
    min_fractals_for_auction: int = 1
    max_pending_auctions: int = 10

    # Guild settings
    enable_guild_formation: bool = True
    guild_formation_threshold: int = 3  # Fractals needed

    # Monitoring
    health_check_interval: float = 60.0
    fractal_timeout: float = 300.0  # 5 minutes

    # Resource limits
    max_concurrent_tasks: int = 100
    max_fractals: int = 50


@dataclass
class SwarmMetrics:
    """إحصائيات Swarm"""

    total_tasks_processed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_auctions: int = 0
    successful_auctions: int = 0
    avg_auction_duration_ms: float = 0.0
    avg_task_duration_ms: float = 0.0
    active_fractals: int = 0
    active_guilds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks_processed,
            "success_rate": (
                self.successful_tasks / self.total_tasks_processed
                if self.total_tasks_processed > 0
                else 0.0
            ),
            "total_auctions": self.total_auctions,
            "auction_success_rate": (
                self.successful_auctions / self.total_auctions if self.total_auctions > 0 else 0.0
            ),
            "avg_auction_ms": round(self.avg_auction_duration_ms, 2),
            "avg_task_ms": round(self.avg_task_duration_ms, 2),
            "active_fractals": self.active_fractals,
            "active_guilds": self.active_guilds,
        }


class SwarmOrchestrator:
    """
    منسق Swarm الذكي.

    Features:
    - Dynamic fractal registration
    - Reputation-based task auctions
    - Automatic guild formation
    - Health monitoring
    - Load balancing

    Usage:
        orchestrator = SwarmOrchestrator(
            reputation_store=reputation,
            config=SwarmConfig(),
        )

        # Start
        await orchestrator.start()

        # Register fractals
        orchestrator.register_fractal(fractal1)
        orchestrator.register_fractal(fractal2)

        # Process task
        result = await orchestrator.process_task(task)

        # Stop
        await orchestrator.stop()
    """

    def __init__(
        self,
        reputation_store: ReputationStore | None = None,
        config: SwarmConfig | None = None,
    ) -> None:
        self._reputation = reputation_store or ReputationStore()
        self._config = config or SwarmConfig()
        self._logger = logging.getLogger("gaap.swarm.orchestrator")

        # Auctioneer
        self._auctioneer = TaskAuctioneer(
            reputation_store=self._reputation,
            config=AuctionConfig(
                default_timeout_seconds=self._config.default_auction_timeout,
            ),
        )

        # State
        self._state = OrchestratorState.INITIALIZING

        # Fractals
        self._fractals: dict[str, FractalAgent] = {}
        self._fractal_domains: dict[str, list[str]] = {}

        # Guilds
        self._guilds: dict[str, Guild] = {}

        # Task tracking
        self._pending_tasks: dict[str, Task] = {}
        self._active_executions: dict[str, asyncio.Task] = {}
        self._execution_results: dict[str, GISPResult] = {}

        # Metrics
        self._metrics = SwarmMetrics()

        # Background tasks
        self._monitor_task: asyncio.Task | None = None
        self._guild_task: asyncio.Task | None = None

    @property
    def state(self) -> OrchestratorState:
        return self._state

    @property
    def metrics(self) -> SwarmMetrics:
        self._metrics.active_fractals = len(self._fractals)
        self._metrics.active_guilds = sum(
            1 for g in self._guilds.values() if g.state == GuildState.ACTIVE
        )
        return self._metrics

    async def start(self) -> None:
        """بدء Orchestrator"""
        self._state = OrchestratorState.RUNNING

        # Start background monitoring
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        if self._config.enable_guild_formation:
            self._guild_task = asyncio.create_task(self._guild_formation_loop())

        self._logger.info("Swarm Orchestrator started")

    async def stop(self) -> None:
        """إيقاف Orchestrator"""
        self._state = OrchestratorState.SHUTTING_DOWN

        # Cancel background tasks
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._guild_task:
            self._guild_task.cancel()

        # Wait for active executions
        if self._active_executions:
            self._logger.info(f"Waiting for {len(self._active_executions)} active tasks")
            await asyncio.gather(
                *self._active_executions.values(),
                return_exceptions=True,
            )

        self._state = OrchestratorState.PAUSED
        self._logger.info("Swarm Orchestrator stopped")

    def register_fractal(self, fractal: FractalAgent) -> bool:
        """تسجيل Fractal جديد"""
        if len(self._fractals) >= self._config.max_fractals:
            self._logger.warning(f"Maximum fractals reached: {self._config.max_fractals}")
            return False

        self._fractals[fractal.fractal_id] = fractal
        self._fractal_domains[fractal.fractal_id] = fractal.domains

        self._logger.info(f"Registered fractal {fractal.fractal_id} (domains: {fractal.domains})")

        return True

    def unregister_fractal(self, fractal_id: str) -> bool:
        """إلغاء تسجيل Fractal"""
        if fractal_id not in self._fractals:
            return False

        del self._fractals[fractal_id]
        del self._fractal_domains[fractal_id]

        self._logger.info(f"Unregistered fractal {fractal_id}")
        return True

    async def process_task(
        self,
        task: Task,
        domain: str = "general",
        priority: TaskPriority = TaskPriority.NORMAL,
        complexity: int = 5,
    ) -> GISPResult | None:
        """
        معالجة مهمة عبر المزاد.

        Flow:
        1. Create auction
        2. Broadcast to eligible fractals
        3. Collect bids
        4. Award to winner
        5. Monitor execution
        6. Return result
        """
        if self._state != OrchestratorState.RUNNING:
            self._logger.warning("Orchestrator not running")
            return None

        # Create auction
        auction = TaskAuction(
            task_id=task.id,
            task_description=task.description,
            domain=TaskDomain(domain)
            if domain in [d.value for d in TaskDomain]
            else TaskDomain.GENERAL,
            priority=priority,
            complexity=complexity,
            timeout_seconds=int(self._config.default_auction_timeout),
        )

        # Start auction
        auction_id = await self._auctioneer.start_auction(auction)
        self._metrics.total_auctions += 1

        # Broadcast to fractals
        await self._broadcast_auction(auction)

        # Wait for auction to close
        await asyncio.sleep(auction.timeout_seconds + 0.5)

        # Get result
        result = await self._auctioneer.close_auction(auction_id)

        if result.state != AuctionState.COMPLETED or not result.winner_id:
            self._logger.warning(f"Auction {auction_id} failed: {result.reason}")
            return None

        self._metrics.successful_auctions += 1

        # Create award
        award = self._auctioneer.create_award(result)

        if award is None:
            self._logger.error(f"Failed to create award for auction {auction_id}")
            return None

        # Execute task
        execution_result = await self._execute_task(task, award)

        self._metrics.total_tasks_processed += 1
        if execution_result and execution_result.success:
            self._metrics.successful_tasks += 1
        else:
            self._metrics.failed_tasks += 1

        return execution_result

    async def _broadcast_auction(self, auction: TaskAuction) -> None:
        """بث المزاد للـ Fractals المؤهلين"""
        domain = auction.domain.value

        for fractal_id, fractal in self._fractals.items():
            # Check if fractal can bid
            if not fractal.can_bid_on(auction):
                continue

            # Get estimate
            estimate = fractal.estimate_task(auction)

            if estimate.can_execute:
                bid = fractal.create_bid(auction, estimate)
                await self._auctioneer.receive_bid(bid)

    async def _execute_task(
        self,
        task: Task,
        award: TaskAward,
    ) -> GISPResult | None:
        """تنفيذ المهمة مع الفائز"""
        winner_id = award.winner_id
        fractal = self._fractals.get(winner_id)

        if not fractal:
            self._logger.error(f"Winner {winner_id} not found")
            return None

        start_time = datetime.now()

        try:
            result = await fractal.execute_task(task, award)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            # Update metrics
            n = self._metrics.total_tasks_processed
            self._metrics.avg_task_duration_ms = (
                self._metrics.avg_task_duration_ms * n + duration
            ) / (n + 1)

            return result

        except Exception as e:
            self._logger.error(f"Task execution failed: {e}")
            return GISPResult(
                task_id=task.id,
                fractal_id=winner_id,
                success=False,
                error=str(e),
            )

    async def _monitor_loop(self) -> None:
        """مراقبة صحة النظام"""
        while self._state == OrchestratorState.RUNNING:
            try:
                await asyncio.sleep(self._config.health_check_interval)

                # Check fractal health
                for fractal_id, fractal in list(self._fractals.items()):
                    state = fractal.state
                    if state == FractalState.OFFLINE:
                        self._logger.warning(f"Fractal {fractal_id} is offline")

                # Clear old auctions
                self._auctioneer.clear_completed_auctions()

                # Apply reputation decay
                self._reputation.apply_decay(days=30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Monitor error: {e}")

    async def _guild_formation_loop(self) -> None:
        """تكوين Guilds تلقائي"""
        while self._state == OrchestratorState.RUNNING:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                # Check each domain for guild formation
                domain_fractals: dict[str, list[str]] = {}

                for fractal_id, domains in self._fractal_domains.items():
                    for domain in domains:
                        if domain not in domain_fractals:
                            domain_fractals[domain] = []
                        domain_fractals[domain].append(fractal_id)

                # Check thresholds
                for domain, fractals in domain_fractals.items():
                    if len(fractals) >= self._config.guild_formation_threshold:
                        # Check if guild already exists
                        if any(g.domain == domain for g in self._guilds.values()):
                            continue

                        # Check if can form guild
                        can_form, reason = Guild.can_form_guild(
                            domain=domain,
                            fractals=fractals,
                            reputation_store=self._reputation,
                        )

                        if can_form:
                            await self._form_guild(domain, fractals)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Guild formation error: {e}")

    async def _form_guild(self, domain: str, fractal_ids: list[str]) -> Guild:
        """تكوين Guild جديد"""
        guild_id = f"guild_{domain}_{uuid.uuid4().hex[:6]}"

        guild = Guild(
            guild_id=guild_id,
            domain=domain,
            reputation_store=self._reputation,
        )

        # Add members
        for i, fractal_id in enumerate(fractal_ids):
            role = "founder" if i == 0 else "member"
            guild.add_member(fractal_id, role=role)

        self._guilds[guild_id] = guild

        self._logger.info(
            f"Formed guild {guild_id} for domain {domain} with {len(fractal_ids)} members"
        )

        return guild

    def get_fractal(self, fractal_id: str) -> FractalAgent | None:
        """الحصول على Fractal"""
        return self._fractals.get(fractal_id)

    def get_guild(self, guild_id: str) -> Guild | None:
        """الحصول على Guild"""
        return self._guilds.get(guild_id)

    def get_guild_for_domain(self, domain: str) -> Guild | None:
        """الحصول على Guild لمجال معين"""
        for guild in self._guilds.values():
            if guild.domain == domain and guild.state == GuildState.ACTIVE:
                return guild
        return None

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات شاملة"""
        return {
            "state": self._state.name,
            "metrics": self._metrics.to_dict(),
            "fractals": {fid: f.get_stats() for fid, f in self._fractals.items()},
            "guilds": {gid: g.get_stats() for gid, g in self._guilds.items()},
            "auction_stats": self._auctioneer.get_stats(),
            "reputation_stats": self._reputation.get_stats(),
        }

    async def broadcast_message(
        self,
        message_type: MessageType,
        content: dict[str, Any],
        target_fractals: list[str] | None = None,
    ) -> int:
        """
        بث رسالة للـ Fractals.

        Returns number of fractals that received the message.
        """
        targets = target_fractals or list(self._fractals.keys())

        for fractal_id in targets:
            fractal = self._fractals.get(fractal_id)
            if fractal:
                # Could implement message queue per fractal
                pass

        return len(targets)
