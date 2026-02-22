"""
Task Auction System

Implements the Reputation-Based Task Auction (RBTA) mechanism.

Auction Flow:
1. Orchestrator creates a TaskAuction with requirements
2. Auctioneer broadcasts to eligible fractals
3. Fractals submit TaskBids with utility scores
4. After timeout, auctioneer selects winner
5. Winner receives TaskAward

Smart Features:
- Utility score computation with multiple factors
- Reserve price (min utility threshold)
- Dutch auction option (price decreases over time)
- Backup bidders for fault tolerance
- Auction history for learning
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any
import uuid

from gaap.swarm.reputation import ReputationStore
from gaap.swarm.gisp_protocol import (
    TaskAuction,
    TaskBid,
    TaskAward,
    TaskDomain,
    TaskPriority,
)


class AuctionState(Enum):
    """حالات المزاد"""

    PENDING = auto()  # لم يبدأ بعد
    OPEN = auto()  # مفتوح للعروض
    CLOSED = auto()  # انتهى
    AWARDING = auto()  # يُسند للفائز
    COMPLETED = auto()  # اكتمل
    CANCELLED = auto()  # أُلغي


@dataclass
class UtilityScore:
    """
    تفصيل درجة المنفعة.

    Allows analysis of why a particular bid won.
    """

    total: float
    success_component: float
    reputation_component: float
    cost_component: float
    time_component: float
    load_penalty: float
    confidence_adjustment: float

    def to_dict(self) -> dict[str, float]:
        return {
            "total": round(self.total, 4),
            "success": round(self.success_component, 4),
            "reputation": round(self.reputation_component, 4),
            "cost": round(self.cost_component, 4),
            "time": round(self.time_component, 4),
            "load_penalty": round(self.load_penalty, 4),
            "confidence_adj": round(self.confidence_adjustment, 4),
        }


@dataclass
class AuctionResult:
    """
    نتيجة المزاد.
    """

    auction_id: str
    task_id: str
    state: AuctionState
    winner_id: str | None = None
    winning_bid: TaskBid | None = None
    winning_utility: float = 0.0
    total_bids: int = 0
    runner_ups: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "auction_id": self.auction_id,
            "task_id": self.task_id,
            "state": self.state.name,
            "winner_id": self.winner_id,
            "winning_utility": round(self.winning_utility, 4),
            "total_bids": self.total_bids,
            "runner_ups": self.runner_ups,
            "duration_ms": round(self.duration_ms, 2),
            "reason": self.reason,
        }


@dataclass
class AuctionConfig:
    """
    إعدادات المزاد.
    """

    # Timing
    default_timeout_seconds: float = 30.0
    min_timeout_seconds: float = 5.0
    max_timeout_seconds: float = 120.0

    # Utility weights
    weight_success: float = 0.35
    weight_reputation: float = 0.25
    weight_cost: float = 0.20
    weight_time: float = 0.20

    # Thresholds
    min_utility_threshold: float = 0.3  # Minimum utility to win
    min_bids_required: int = 1  # Minimum bids to award
    max_runner_ups: int = 2  # Backup bidders

    # Dutch auction (optional)
    enable_dutch_auction: bool = False
    dutch_decrement_interval: float = 5.0  # seconds
    dutch_decrement_amount: float = 0.05  # utility reduction

    # Learning
    enable_auction_learning: bool = True
    history_size: int = 100


class TaskAuctioneer:
    """
    نظام المزاد الذكي للمهام.

    Features:
    - Multiple utility computation strategies
    - Dutch auction option for urgent tasks
    - Reserve price enforcement
    - Backup bidder selection
    - Auction history for learning

    Usage:
        auctioneer = TaskAuctioneer(reputation_store)

        # Start auction
        auction = TaskAuction(task_id="task_123", ...)
        auction_id = await auctioneer.start_auction(auction)

        # Receive bids
        await auctioneer.receive_bid(bid)

        # Close and award
        result = await auctioneer.close_auction(auction_id)
        if result.winner_id:
            award = auctioneer.create_award(result)
    """

    def __init__(
        self,
        reputation_store: ReputationStore,
        config: AuctionConfig | None = None,
    ) -> None:
        self._reputation = reputation_store
        self._config = config or AuctionConfig()
        self._logger = logging.getLogger("gaap.swarm.auction")

        # Active auctions
        self._auctions: dict[str, TaskAuction] = {}
        self._bids: dict[str, list[TaskBid]] = {}
        self._auction_states: dict[str, AuctionState] = {}
        self._auction_timers: dict[str, asyncio.Task] = {}

        # Auction history
        self._history: list[AuctionResult] = []

        # Statistics
        self._stats = {
            "total_auctions": 0,
            "successful_awards": 0,
            "cancelled_auctions": 0,
            "avg_bids_per_auction": 0.0,
            "avg_winning_utility": 0.0,
        }

    async def start_auction(
        self,
        auction: TaskAuction,
        auto_close: bool = True,
    ) -> str:
        """
        بدء مزاد جديد.

        Args:
            auction: بيانات المزاد
            auto_close: إغلاق تلقائي بعد المهلة

        Returns:
            auction_id: معرف المزاد
        """
        auction_id = auction.message_id
        if not auction_id:
            auction_id = f"auction_{uuid.uuid4().hex[:8]}"
            auction.message_id = auction_id

        self._auctions[auction_id] = auction
        self._bids[auction_id] = []
        self._auction_states[auction_id] = AuctionState.OPEN

        self._stats["total_auctions"] += 1
        self._logger.info(
            f"Started auction {auction_id} for task {auction.task_id} "
            f"(domain={auction.domain.value}, timeout={auction.timeout_seconds}s)"
        )

        if auto_close:
            self._auction_timers[auction_id] = asyncio.create_task(
                self._auto_close_auction(auction_id, auction.timeout_seconds)
            )

        return auction_id

    async def _auto_close_auction(self, auction_id: str, timeout: float) -> None:
        """إغلاق تلقائي للمزاد"""
        await asyncio.sleep(timeout)

        if self._auction_states.get(auction_id) == AuctionState.OPEN:
            self._logger.info(f"Auto-closing auction {auction_id}")
            await self.close_auction(auction_id)

    async def receive_bid(self, bid: TaskBid) -> bool:
        """
        استقبال عرض.

        Returns:
            True if bid was accepted, False otherwise
        """
        auction_id = bid.correlation_id or ""

        # Find auction by task_id if correlation_id not set
        if not auction_id:
            for aid, auction in self._auctions.items():
                if auction.task_id == bid.task_id:
                    auction_id = aid
                    break

        if not auction_id or auction_id not in self._auctions:
            self._logger.warning(f"Bid received for unknown auction: {bid.task_id}")
            return False

        state = self._auction_states.get(auction_id)
        if state != AuctionState.OPEN:
            self._logger.warning(f"Bid received for closed auction: {auction_id}")
            return False

        auction = self._auctions[auction_id]

        # Check reputation threshold
        reputation = self._reputation.get_domain_reputation(bid.bidder_id, auction.domain.value)

        if reputation < auction.min_reputation:
            self._logger.info(
                f"Bid from {bid.bidder_id} rejected: reputation {reputation:.2f} < {auction.min_reputation}"
            )
            return False

        # Compute utility score
        utility = self.compute_utility(bid, auction, reputation)
        bid.utility_score = utility.total

        # Record participation
        self._reputation.record_auction_participation(bid.bidder_id, participated=True)

        # Store bid
        self._bids[auction_id].append(bid)

        self._logger.info(
            f"Bid from {bid.bidder_id} accepted: utility={utility.total:.4f} "
            f"(success={utility.success_component:.2f}, rep={utility.reputation_component:.2f})"
        )

        return True

    def compute_utility(
        self,
        bid: TaskBid,
        auction: TaskAuction,
        reputation: float,
    ) -> UtilityScore:
        """
        حساب درجة المنفعة الكاملة.

        Formula:
        U = w_s * success + w_r * reputation + w_c * cost_eff + w_t * time_eff
        U = U * (1 - load_penalty) * confidence
        """
        config = self._config

        # Normalize cost (inverse - lower is better)
        max_cost = auction.constraints.get("max_cost", bid.estimated_cost_tokens * 2)
        cost_efficiency = 1.0 - min(bid.estimated_cost_tokens / max(max_cost, 1), 1.0)

        # Normalize time (inverse - lower is better)
        max_time = auction.constraints.get("max_time", bid.estimated_time_seconds * 2)
        time_efficiency = 1.0 - min(bid.estimated_time_seconds / max(max_time, 1), 1.0)

        # Compute components
        success_component = config.weight_success * bid.estimated_success_rate
        reputation_component = config.weight_reputation * reputation
        cost_component = config.weight_cost * cost_efficiency
        time_component = config.weight_time * time_efficiency

        # Base score
        base_score = success_component + reputation_component + cost_component + time_component

        # Adjustments
        load_penalty = bid.current_load * 0.3
        confidence_adjustment = bid.confidence_in_estimate

        # Final score
        adjusted_score = base_score * (1 - load_penalty) * confidence_adjustment
        final_score = max(0.0, min(1.0, adjusted_score))

        return UtilityScore(
            total=final_score,
            success_component=success_component,
            reputation_component=reputation_component,
            cost_component=cost_component,
            time_component=time_component,
            load_penalty=load_penalty,
            confidence_adjustment=confidence_adjustment,
        )

    async def close_auction(self, auction_id: str) -> AuctionResult:
        """
        إغلاق المزاد واختيار الفائز.
        """
        auction = self._auctions.get(auction_id)
        if not auction:
            return AuctionResult(
                auction_id=auction_id,
                task_id="unknown",
                state=AuctionState.CANCELLED,
                reason="Auction not found",
            )

        # Cancel auto-close timer
        if auction_id in self._auction_timers:
            self._auction_timers[auction_id].cancel()
            del self._auction_timers[auction_id]

        bids = self._bids.get(auction_id, [])
        state = self._auction_states.get(auction_id, AuctionState.OPEN)

        if state != AuctionState.OPEN:
            return AuctionResult(
                auction_id=auction_id,
                task_id=auction.task_id,
                state=state,
                reason="Auction already closed",
            )

        start_time = datetime.now()

        # Check minimum bids
        if len(bids) < self._config.min_bids_required:
            self._auction_states[auction_id] = AuctionState.CANCELLED
            self._stats["cancelled_auctions"] += 1

            result = AuctionResult(
                auction_id=auction_id,
                task_id=auction.task_id,
                state=AuctionState.CANCELLED,
                total_bids=len(bids),
                reason=f"Insufficient bids: {len(bids)} < {self._config.min_bids_required}",
            )
            self._add_to_history(result)
            return result

        # Sort bids by utility (descending)
        sorted_bids = sorted(bids, key=lambda b: b.utility_score, reverse=True)

        # Check threshold
        winning_bid = sorted_bids[0]

        if winning_bid.utility_score < self._config.min_utility_threshold:
            self._auction_states[auction_id] = AuctionState.CANCELLED
            self._stats["cancelled_auctions"] += 1

            result = AuctionResult(
                auction_id=auction_id,
                task_id=auction.task_id,
                state=AuctionState.CANCELLED,
                winning_utility=winning_bid.utility_score,
                total_bids=len(bids),
                reason=f"Winning utility below threshold: {winning_bid.utility_score:.2f} < {self._config.min_utility_threshold}",
            )
            self._add_to_history(result)
            return result

        # Award to winner
        self._auction_states[auction_id] = AuctionState.AWARDING

        # Get runner-ups
        runner_ups = [b.bidder_id for b in sorted_bids[1 : self._config.max_runner_ups + 1]]

        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        result = AuctionResult(
            auction_id=auction_id,
            task_id=auction.task_id,
            state=AuctionState.COMPLETED,
            winner_id=winning_bid.bidder_id,
            winning_bid=winning_bid,
            winning_utility=winning_bid.utility_score,
            total_bids=len(bids),
            runner_ups=runner_ups,
            duration_ms=duration_ms,
        )

        self._stats["successful_awards"] += 1
        self._update_stats(result)
        self._add_to_history(result)

        self._logger.info(
            f"Auction {auction_id} completed: winner={winning_bid.bidder_id} "
            f"utility={winning_bid.utility_score:.4f} bids={len(bids)}"
        )

        return result

    def create_award(self, result: AuctionResult) -> TaskAward | None:
        """
        إنشاء رسالة إسناد.
        """
        if not result.winner_id or not result.winning_bid:
            return None

        return TaskAward(
            task_id=result.task_id,
            winner_id=result.winner_id,
            utility_score=result.winning_utility,
            winning_bid=result.winning_bid,
            runner_ups=result.runner_ups,
        )

    def cancel_auction(self, auction_id: str, reason: str = "") -> AuctionResult:
        """
        إلغاء مزاد.
        """
        auction = self._auctions.get(auction_id)

        # Cancel timer
        if auction_id in self._auction_timers:
            self._auction_timers[auction_id].cancel()
            del self._auction_timers[auction_id]

        self._auction_states[auction_id] = AuctionState.CANCELLED
        self._stats["cancelled_auctions"] += 1

        result = AuctionResult(
            auction_id=auction_id,
            task_id=auction.task_id if auction else "unknown",
            state=AuctionState.CANCELLED,
            total_bids=len(self._bids.get(auction_id, [])),
            reason=reason,
        )

        self._add_to_history(result)
        return result

    def get_auction_status(self, auction_id: str) -> AuctionState | None:
        """الحصول على حالة مزاد"""
        return self._auction_states.get(auction_id)

    def get_bids(self, auction_id: str) -> list[TaskBid]:
        """الحصول على عروض مزاد"""
        return self._bids.get(auction_id, [])

    def get_auction(self, auction_id: str) -> TaskAuction | None:
        """الحصول على مزاد"""
        return self._auctions.get(auction_id)

    def _add_to_history(self, result: AuctionResult) -> None:
        """إضافة للسجل"""
        self._history.append(result)

        # Trim history
        if len(self._history) > self._config.history_size:
            self._history = self._history[-self._config.history_size :]

    def _update_stats(self, result: AuctionResult) -> None:
        """تحديث الإحصائيات"""
        n = self._stats["successful_awards"]

        # Running average
        self._stats["avg_bids_per_auction"] = (
            self._stats["avg_bids_per_auction"] * (n - 1) + result.total_bids
        ) / n
        self._stats["avg_winning_utility"] = (
            self._stats["avg_winning_utility"] * (n - 1) + result.winning_utility
        ) / n

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات المزادات"""
        return {
            **self._stats,
            "active_auctions": sum(
                1 for s in self._auction_states.values() if s == AuctionState.OPEN
            ),
            "history_size": len(self._history),
        }

    def get_history(self, limit: int = 10) -> list[AuctionResult]:
        """سجل المزادات"""
        return self._history[-limit:]

    def clear_completed_auctions(self, older_than_seconds: float = 3600) -> int:
        """
        تنظيف المزادات المكتملة.

        Returns:
            Number of auctions cleared
        """
        cutoff = datetime.now() - timedelta(seconds=older_than_seconds)
        cleared = 0

        for auction_id, state in list(self._auction_states.items()):
            if state in (AuctionState.COMPLETED, AuctionState.CANCELLED):
                # Check if auction is old
                auction = self._auctions.get(auction_id)
                if auction and auction.timestamp < cutoff:
                    del self._auctions[auction_id]
                    del self._bids[auction_id]
                    del self._auction_states[auction_id]
                    cleared += 1

        return cleared
