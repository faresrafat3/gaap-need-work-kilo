"""
Budget API - Budget Management Endpoints
========================================

Provides REST API for monitoring and managing budget limits.

Endpoints:
- GET /api/budget - Get current budget status
- GET /api/budget/usage - Get detailed usage breakdown
- GET /api/budget/alerts - Get budget alerts
- PUT /api/budget/limits - Update budget limits
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.core.config import get_config, get_config_manager
from gaap.core.events import EventEmitter, EventType

logger = logging.getLogger("gaap.api.budget")

router = APIRouter(prefix="/api/budget", tags=["budget"])


# =============================================================================
# BudgetTracker - Tracks spending and enforces budget limits
# =============================================================================


@dataclass
class UsageRecord:
    """Single usage record."""

    task_id: str
    timestamp: datetime
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    category: str = "general"


@dataclass
class BudgetAlertInternal:
    """Internal budget alert representation."""

    id: str
    type: str
    threshold: float
    current_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


class BudgetTracker:
    """
    Tracks API spending and enforces budget limits.

    Supports both in-memory and Redis-backed storage.
    Automatically reads limits from configuration.
    """

    def __init__(self, use_redis: bool = True):
        self._lock = threading.RLock()
        self._config = get_config()
        self._alerts: list[BudgetAlertInternal] = []
        self._alert_callbacks: list[Callable[[BudgetAlertInternal], None]] = []

        # In-memory storage
        self._usage_history: list[UsageRecord] = []
        self._daily_spend: float = 0.0
        self._monthly_spend: float = 0.0
        self._last_reset_day: datetime = datetime.now()
        self._last_reset_month: datetime = datetime.now()

        # Redis storage (optional)
        self._redis_client: Any = None
        self._use_redis = False

        if use_redis:
            self._init_redis()

        # Initialize spending from Redis if available
        if self._use_redis:
            self._load_from_redis()

        logger.info(f"BudgetTracker initialized (redis={self._use_redis})")

    def _init_redis(self) -> None:
        """Initialize Redis connection if available."""
        redis_url = os.environ.get("REDIS_URL") or os.environ.get("GAAP_REDIS_URL")
        if not redis_url:
            return

        try:
            import redis

            self._redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            self._use_redis = True
            logger.info("BudgetTracker: Redis storage enabled")
        except Exception as e:
            logger.warning(f"BudgetTracker: Redis unavailable, using memory: {e}")
            self._use_redis = False

    def _get_redis_key(self, key: str) -> str:
        """Generate Redis key with namespace."""
        return f"gaap:budget:{key}"

    def _load_from_redis(self) -> None:
        """Load current spending from Redis."""
        if not self._use_redis or not self._redis_client:
            return

        try:
            daily = self._redis_client.get(self._get_redis_key("daily_spend"))
            monthly = self._redis_client.get(self._get_redis_key("monthly_spend"))

            if daily:
                self._daily_spend = float(daily)
            if monthly:
                self._monthly_spend = float(monthly)

            logger.debug(
                f"Loaded from Redis: daily={self._daily_spend}, monthly={self._monthly_spend}"
            )
        except Exception as e:
            logger.warning(f"Failed to load from Redis: {e}")

    def _save_to_redis(self) -> None:
        """Save current spending to Redis."""
        if not self._use_redis or not self._redis_client:
            return

        try:
            pipe = self._redis_client.pipeline()
            pipe.set(self._get_redis_key("daily_spend"), str(self._daily_spend))
            pipe.set(self._get_redis_key("monthly_spend"), str(self._monthly_spend))

            # Set expiration to end of day/month
            now = datetime.now()
            end_of_day = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end_of_month = (now.replace(day=28) + timedelta(days=4)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )

            pipe.expireat(self._get_redis_key("daily_spend"), int(end_of_day.timestamp()))
            pipe.expireat(self._get_redis_key("monthly_spend"), int(end_of_month.timestamp()))

            pipe.execute()
        except Exception as e:
            logger.warning(f"Failed to save to Redis: {e}")

    def _check_and_reset_periods(self) -> None:
        """Check if daily/monthly periods have passed and reset if needed."""
        now = datetime.now()

        # Check daily reset
        if now.date() != self._last_reset_day.date():
            self._daily_spend = 0.0
            self._last_reset_day = now
            logger.info("Daily budget period reset")

        # Check monthly reset
        if now.month != self._last_reset_month.month or now.year != self._last_reset_month.year:
            self._monthly_spend = 0.0
            self._last_reset_month = now
            logger.info("Monthly budget period reset")

    def record_usage(
        self,
        task_id: str,
        provider: str,
        model: str,
        tokens_used: int,
        cost_usd: float,
        category: str = "general",
    ) -> dict[str, Any]:
        """
        Record API usage and update spending.

        Returns:
            Dict with status and any alerts triggered
        """
        with self._lock:
            self._check_and_reset_periods()

            record = UsageRecord(
                task_id=task_id,
                timestamp=datetime.now(),
                provider=provider,
                model=model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                category=category,
            )

            self._usage_history.append(record)
            self._daily_spend += cost_usd
            self._monthly_spend += cost_usd

            # Trim history if too large (keep last 10000)
            if len(self._usage_history) > 10000:
                self._usage_history = self._usage_history[-10000:]

            # Save to Redis if enabled
            if self._use_redis:
                self._save_to_redis()
                self._add_usage_to_redis(record)

            # Check limits and emit alerts
            alerts_triggered = self._check_limits()

            # Emit budget update event
            try:
                emitter = EventEmitter.get_instance()
                emitter.emit(
                    EventType.BUDGET_UPDATE,
                    {
                        "daily_spend": self._daily_spend,
                        "monthly_spend": self._monthly_spend,
                        "task_id": task_id,
                        "cost": cost_usd,
                    },
                    source="budget_tracker",
                )
            except Exception as e:
                logger.debug(f"Failed to emit budget update event: {e}")

            return {
                "success": True,
                "daily_spend": self._daily_spend,
                "monthly_spend": self._monthly_spend,
                "alerts_triggered": alerts_triggered,
            }

    def _add_usage_to_redis(self, record: UsageRecord) -> None:
        """Add usage record to Redis list."""
        if not self._use_redis or not self._redis_client:
            return

        try:
            key = self._get_redis_key("usage_history")
            data = {
                "task_id": record.task_id,
                "timestamp": record.timestamp.isoformat(),
                "provider": record.provider,
                "model": record.model,
                "tokens": record.tokens_used,
                "cost": record.cost_usd,
                "category": record.category,
            }
            self._redis_client.lpush(key, str(data))
            self._redis_client.ltrim(key, 0, 9999)  # Keep last 10000

            # Set expiration
            now = datetime.now()
            end_of_month = (now.replace(day=28) + timedelta(days=4)).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            self._redis_client.expireat(key, int(end_of_month.timestamp()))
        except Exception as e:
            logger.warning(f"Failed to add usage to Redis: {e}")

    def _check_limits(self) -> list[dict[str, Any]]:
        """Check budget limits and generate alerts."""
        alerts = []
        config = self._config.budget

        monthly_pct = (
            (self._monthly_spend / config.monthly_limit * 100) if config.monthly_limit > 0 else 0
        )
        daily_pct = (self._daily_spend / config.daily_limit * 100) if config.daily_limit > 0 else 0

        # Check thresholds
        for threshold in config.alert_thresholds:
            pct = threshold * 100
            if monthly_pct >= pct:
                alert = BudgetAlertInternal(
                    id=f"threshold-{int(pct)}-{datetime.now().strftime('%Y%m%d')}",
                    type="threshold_exceeded",
                    threshold=pct,
                    current_value=monthly_pct,
                    message=f"Monthly budget exceeded {pct:.0f}% ({self._monthly_spend:.2f}/{config.monthly_limit:.2f})",
                    timestamp=datetime.now(),
                )
                self._add_alert(alert)
                alerts.append({"type": "threshold", "percentage": pct})

        # Check throttling threshold
        if monthly_pct >= config.auto_throttle_at * 100:
            alert = BudgetAlertInternal(
                id=f"throttle-{datetime.now().strftime('%Y%m%d')}",
                type="throttling_active",
                threshold=config.auto_throttle_at * 100,
                current_value=monthly_pct,
                message=f"Budget throttling activated at {monthly_pct:.1f}%",
                timestamp=datetime.now(),
            )
            self._add_alert(alert)
            alerts.append({"type": "throttle"})

        # Check hard stop
        if monthly_pct >= config.hard_stop_at * 100:
            alert = BudgetAlertInternal(
                id=f"hardstop-{datetime.now().strftime('%Y%m%d')}",
                type="hard_stop",
                threshold=config.hard_stop_at * 100,
                current_value=monthly_pct,
                message=f"Budget hard stop reached at {monthly_pct:.1f}%",
                timestamp=datetime.now(),
            )
            self._add_alert(alert)
            alerts.append({"type": "hard_stop"})

        return alerts

    def _add_alert(self, alert: BudgetAlertInternal) -> None:
        """Add alert if not already exists for today."""
        # Check if alert already exists
        for existing in self._alerts:
            if existing.id == alert.id and not existing.acknowledged:
                return

        self._alerts.append(alert)

        # Emit alert event
        try:
            emitter = EventEmitter.get_instance()
            emitter.emit(
                EventType.BUDGET_ALERT,
                {
                    "alert_id": alert.id,
                    "type": alert.type,
                    "message": alert.message,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                },
                source="budget_tracker",
            )
        except Exception as e:
            logger.debug(f"Failed to emit budget alert event: {e}")

        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.warning(f"Alert callback failed: {e}")

    def check_limits(self, estimated_cost: float = 0.0) -> dict[str, Any]:
        """
        Check if a request is within budget limits.

        Returns:
            Dict with allowed status and reason if blocked
        """
        with self._lock:
            self._check_and_reset_periods()
            config = self._config.budget

            # Check per-task limit
            if estimated_cost > config.per_task_limit:
                return {
                    "allowed": False,
                    "reason": f"Estimated cost ${estimated_cost:.2f} exceeds per-task limit ${config.per_task_limit:.2f}",
                    "throttling": False,
                    "hard_stop": False,
                }

            # Check daily limit
            if self._daily_spend + estimated_cost > config.daily_limit:
                return {
                    "allowed": False,
                    "reason": f"Daily budget exceeded (${self._daily_spend:.2f}/${config.daily_limit:.2f})",
                    "throttling": False,
                    "hard_stop": True,
                }

            # Check monthly hard stop
            monthly_pct = (
                (self._monthly_spend / config.monthly_limit * 100)
                if config.monthly_limit > 0
                else 0
            )
            hard_stop = monthly_pct >= config.hard_stop_at * 100

            if hard_stop:
                return {
                    "allowed": False,
                    "reason": f"Monthly budget hard stop reached ({monthly_pct:.1f}%)",
                    "throttling": False,
                    "hard_stop": True,
                }

            # Check throttling
            throttling = monthly_pct >= config.auto_throttle_at * 100

            return {
                "allowed": True,
                "reason": None,
                "throttling": throttling,
                "hard_stop": False,
                "daily_percentage": (self._daily_spend / config.daily_limit * 100)
                if config.daily_limit > 0
                else 0,
                "monthly_percentage": monthly_pct,
            }

    def get_daily_spend(self) -> float:
        """Get current daily spending."""
        with self._lock:
            self._check_and_reset_periods()
            return self._daily_spend

    def get_monthly_spend(self) -> float:
        """Get current monthly spending."""
        with self._lock:
            self._check_and_reset_periods()
            return self._monthly_spend

    def get_usage_history(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get usage history for a date range."""
        with self._lock:
            # Try Redis first if available
            if self._use_redis:
                return self._get_usage_from_redis(start_date, end_date, limit)

            # Use in-memory storage
            results = []
            for record in reversed(self._usage_history):
                if start_date <= record.timestamp <= end_date:
                    results.append(
                        {
                            "task_id": record.task_id,
                            "timestamp": record.timestamp.isoformat(),
                            "provider": record.provider,
                            "model": record.model,
                            "tokens": record.tokens_used,
                            "cost": record.cost_usd,
                            "category": record.category,
                        }
                    )
                    if len(results) >= limit:
                        break
            return results

    def _get_usage_from_redis(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get usage history from Redis."""
        if not self._use_redis or not self._redis_client:
            return []

        try:
            key = self._get_redis_key("usage_history")
            items = self._redis_client.lrange(key, 0, limit * 2)  # Get extra for filtering

            results = []
            for item in items:
                try:
                    data = eval(item)  # Safe since we control the data format
                    ts = datetime.fromisoformat(data["timestamp"])
                    if start_date <= ts <= end_date:
                        results.append(data)
                        if len(results) >= limit:
                            break
                except Exception:
                    continue
            return results
        except Exception as e:
            logger.warning(f"Failed to get usage from Redis: {e}")
            return []

    def get_usage_summary(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict[str, Any]]:
        """Get usage summary grouped by provider/model."""
        history = self.get_usage_history(start_date, end_date, limit=10000)

        # Group by provider/model
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        total_cost = 0.0

        for item in history:
            key = (item["provider"], item["model"])
            if key not in groups:
                groups[key] = {
                    "provider": item["provider"],
                    "model": item["model"],
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                }
            groups[key]["requests"] += 1
            groups[key]["tokens"] += item["tokens"]
            groups[key]["cost"] += item["cost"]
            total_cost += item["cost"]

        # Calculate percentages
        results = []
        for group in groups.values():
            group["percentage"] = (group["cost"] / total_cost * 100) if total_cost > 0 else 0
            results.append(group)

        # Sort by cost descending
        results.sort(key=lambda x: x["cost"], reverse=True)
        return results

    def get_alerts(self, include_acknowledged: bool = False) -> list[dict[str, Any]]:
        """Get current alerts."""
        with self._lock:
            results = []
            for alert in self._alerts:
                if include_acknowledged or not alert.acknowledged:
                    results.append(
                        {
                            "id": alert.id,
                            "type": alert.type,
                            "threshold": alert.threshold,
                            "current_value": alert.current_value,
                            "message": alert.message,
                            "timestamp": alert.timestamp.isoformat(),
                            "acknowledged": alert.acknowledged,
                        }
                    )
            return results

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged")
                    return True
            return False

    def on_alert(self, callback: Callable[[BudgetAlertInternal], None]) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def reset_spending(self) -> None:
        """Reset all spending (for testing)."""
        with self._lock:
            self._daily_spend = 0.0
            self._monthly_spend = 0.0
            self._usage_history.clear()
            if self._use_redis:
                try:
                    self._redis_client.delete(self._get_redis_key("daily_spend"))
                    self._redis_client.delete(self._get_redis_key("monthly_spend"))
                    self._redis_client.delete(self._get_redis_key("usage_history"))
                except Exception as e:
                    logger.warning(f"Failed to reset Redis: {e}")
            logger.info("Budget spending reset")


# =============================================================================
# Global Budget Tracker Instance
# =============================================================================

_budget_tracker: BudgetTracker | None = None
_budget_tracker_lock = threading.Lock()


def _init_budget_tracker() -> BudgetTracker:
    """Initialize the global budget tracker instance."""
    global _budget_tracker
    with _budget_tracker_lock:
        if _budget_tracker is None:
            _budget_tracker = BudgetTracker(use_redis=True)
    return _budget_tracker


class BudgetStatus(BaseModel):
    """Current budget status."""

    monthly_limit: float
    daily_limit: float
    per_task_limit: float
    monthly_spent: float = 0.0
    daily_spent: float = 0.0
    monthly_remaining: float
    daily_remaining: float
    monthly_percentage: float
    daily_percentage: float
    throttling: bool = False
    hard_stop: bool = False


class UsageItem(BaseModel):
    """Single usage item."""

    task_id: str
    timestamp: str
    provider: str
    model: str
    tokens_used: int
    cost_usd: float
    category: str = "general"


class UsageSummary(BaseModel):
    """Usage summary by provider/model."""

    provider: str
    model: str
    total_requests: int
    total_tokens: int
    total_cost: float
    percentage: float


class UsageResponse(BaseModel):
    """Detailed usage response."""

    period: str
    start_date: str
    end_date: str
    total_cost: float
    items: list[UsageItem] = []
    summary: list[UsageSummary] = []


class BudgetAlert(BaseModel):
    """Budget alert."""

    id: str
    type: str
    threshold: float
    current_value: float
    message: str
    timestamp: str
    acknowledged: bool = False


class BudgetAlertsResponse(BaseModel):
    """Budget alerts response."""

    alerts: list[BudgetAlert]
    total: int


class BudgetLimitsRequest(BaseModel):
    """Request to update budget limits."""

    monthly_limit: float | None = Field(None, gt=0)
    daily_limit: float | None = Field(None, gt=0)
    per_task_limit: float | None = Field(None, gt=0)
    auto_throttle_at: float | None = Field(None, ge=0, le=1)
    hard_stop_at: float | None = Field(None, ge=0, le=1)
    cost_optimization_mode: str | None = None


class BudgetLimitsResponse(BaseModel):
    """Budget limits update response."""

    success: bool
    limits: dict[str, Any] | None = None
    error: str | None = None


def get_budget_tracker() -> BudgetTracker:
    """Get the budget tracker instance."""
    global _budget_tracker
    if _budget_tracker is None:
        return _init_budget_tracker()
    return _budget_tracker


def set_budget_tracker(tracker: Any) -> None:
    """Set the budget tracker instance."""
    global _budget_tracker
    _budget_tracker = tracker


def _get_mock_usage() -> tuple[float, float]:
    """Get mock usage data if no tracker available."""
    return (0.0, 0.0)


@router.get("", response_model=BudgetStatus)
async def get_budget_status() -> BudgetStatus:
    """Get current budget status."""
    try:
        config = get_config()
        tracker = get_budget_tracker()

        monthly_spent = 0.0
        daily_spent = 0.0

        if tracker:
            if hasattr(tracker, "get_monthly_spend"):
                monthly_spent = tracker.get_monthly_spend()
            if hasattr(tracker, "get_daily_spend"):
                daily_spent = tracker.get_daily_spend()
        else:
            monthly_spent, daily_spent = _get_mock_usage()

        monthly_remaining = max(0, config.budget.monthly_limit - monthly_spent)
        daily_remaining = max(0, config.budget.daily_limit - daily_spent)

        monthly_percentage = (
            (monthly_spent / config.budget.monthly_limit * 100)
            if config.budget.monthly_limit > 0
            else 0
        )
        daily_percentage = (
            (daily_spent / config.budget.daily_limit * 100) if config.budget.daily_limit > 0 else 0
        )

        throttling = monthly_percentage >= (config.budget.auto_throttle_at * 100)
        hard_stop = monthly_percentage >= (config.budget.hard_stop_at * 100)

        return BudgetStatus(
            monthly_limit=config.budget.monthly_limit,
            daily_limit=config.budget.daily_limit,
            per_task_limit=config.budget.per_task_limit,
            monthly_spent=monthly_spent,
            daily_spent=daily_spent,
            monthly_remaining=monthly_remaining,
            daily_remaining=daily_remaining,
            monthly_percentage=round(monthly_percentage, 2),
            daily_percentage=round(daily_percentage, 2),
            throttling=throttling,
            hard_stop=hard_stop,
        )
    except Exception as e:
        logger.error(f"Failed to get budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/usage", response_model=UsageResponse)
async def get_budget_usage(
    period: str = "daily",
    limit: int = 100,
) -> UsageResponse:
    """Get detailed usage breakdown."""
    try:
        config = get_config()
        tracker = get_budget_tracker()

        now = datetime.now()
        if period == "monthly":
            start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_date = now
        else:
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = now

        items = []
        summary = []
        total_cost = 0.0

        if tracker:
            if hasattr(tracker, "get_usage_history"):
                history = tracker.get_usage_history(start_date, end_date, limit=limit)
                for item in history:
                    items.append(
                        UsageItem(
                            task_id=item.get("task_id", ""),
                            timestamp=item.get("timestamp", ""),
                            provider=item.get("provider", ""),
                            model=item.get("model", ""),
                            tokens_used=item.get("tokens", 0),
                            cost_usd=item.get("cost", 0.0),
                            category=item.get("category", "general"),
                        )
                    )
                    total_cost += item.get("cost", 0.0)

            if hasattr(tracker, "get_usage_summary"):
                summary_data = tracker.get_usage_summary(start_date, end_date)
                for s in summary_data:
                    summary.append(
                        UsageSummary(
                            provider=s.get("provider", ""),
                            model=s.get("model", ""),
                            total_requests=s.get("requests", 0),
                            total_tokens=s.get("tokens", 0),
                            total_cost=s.get("cost", 0.0),
                            percentage=s.get("percentage", 0.0),
                        )
                    )

        return UsageResponse(
            period=period,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            total_cost=total_cost,
            items=items,
            summary=summary,
        )
    except Exception as e:
        logger.error(f"Failed to get budget usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=BudgetAlertsResponse)
async def get_budget_alerts() -> BudgetAlertsResponse:
    """Get budget alerts."""
    try:
        config = get_config()
        tracker = get_budget_tracker()
        alerts = []

        status = await get_budget_status()

        for threshold in config.budget.alert_thresholds:
            percentage = threshold * 100
            if status.monthly_percentage >= percentage:
                alerts.append(
                    BudgetAlert(
                        id=f"alert-{int(threshold * 100)}",
                        type="threshold_exceeded",
                        threshold=percentage,
                        current_value=status.monthly_percentage,
                        message=f"Budget usage has exceeded {percentage:.0f}% of monthly limit",
                        timestamp=datetime.now().isoformat(),
                        acknowledged=False,
                    )
                )

        if status.throttling:
            alerts.append(
                BudgetAlert(
                    id="alert-throttle",
                    type="throttling_active",
                    threshold=config.budget.auto_throttle_at * 100,
                    current_value=status.monthly_percentage,
                    message="Budget throttling is active - requests may be delayed or downgraded",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                )
            )

        if status.hard_stop:
            alerts.append(
                BudgetAlert(
                    id="alert-hardstop",
                    type="hard_stop",
                    threshold=config.budget.hard_stop_at * 100,
                    current_value=status.monthly_percentage,
                    message="Budget hard stop reached - non-critical requests are blocked",
                    timestamp=datetime.now().isoformat(),
                    acknowledged=False,
                )
            )

        if tracker and hasattr(tracker, "get_alerts"):
            tracker_alerts = tracker.get_alerts()
            for alert in tracker_alerts:
                alerts.append(
                    BudgetAlert(
                        id=alert.get("id", ""),
                        type=alert.get("type", ""),
                        threshold=alert.get("threshold", 0),
                        current_value=alert.get("current_value", 0),
                        message=alert.get("message", ""),
                        timestamp=alert.get("timestamp", ""),
                        acknowledged=alert.get("acknowledged", False),
                    )
                )

        return BudgetAlertsResponse(
            alerts=alerts,
            total=len(alerts),
        )
    except Exception as e:
        logger.error(f"Failed to get budget alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/limits", response_model=BudgetLimitsResponse)
async def update_budget_limits(request: BudgetLimitsRequest) -> BudgetLimitsResponse:
    """Update budget limits."""
    try:
        manager = get_config_manager()
        config = manager.config

        if request.monthly_limit is not None:
            config.budget.monthly_limit = request.monthly_limit
        if request.daily_limit is not None:
            config.budget.daily_limit = request.daily_limit
        if request.per_task_limit is not None:
            config.budget.per_task_limit = request.per_task_limit
        if request.auto_throttle_at is not None:
            config.budget.auto_throttle_at = request.auto_throttle_at
        if request.hard_stop_at is not None:
            config.budget.hard_stop_at = request.hard_stop_at
        if request.cost_optimization_mode is not None:
            valid_modes = ["aggressive", "balanced", "quality_first"]
            if request.cost_optimization_mode not in valid_modes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid cost_optimization_mode. Must be one of: {valid_modes}",
                )
            config.budget.cost_optimization_mode = request.cost_optimization_mode

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.CONFIG_CHANGED,
            {"module": "budget", "changes": request.model_dump(exclude_none=True)},
            source="budget_api",
        )

        logger.info("Budget limits updated")

        return BudgetLimitsResponse(
            success=True,
            limits={
                "monthly_limit": config.budget.monthly_limit,
                "daily_limit": config.budget.daily_limit,
                "per_task_limit": config.budget.per_task_limit,
                "auto_throttle_at": config.budget.auto_throttle_at,
                "hard_stop_at": config.budget.hard_stop_at,
                "cost_optimization_mode": config.budget.cost_optimization_mode,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update budget limits: {e}")
        return BudgetLimitsResponse(success=False, error=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str) -> dict[str, Any]:
    """Acknowledge a budget alert."""
    try:
        tracker = get_budget_tracker()

        if tracker and hasattr(tracker, "acknowledge_alert"):
            tracker.acknowledge_alert(alert_id)

        logger.info(f"Alert {alert_id} acknowledged")

        return {"success": True, "message": f"Alert {alert_id} acknowledged"}
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app: Any) -> None:
    """Register budget routes with FastAPI app."""
    app.include_router(router)
