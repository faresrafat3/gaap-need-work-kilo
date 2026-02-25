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
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.core.config import get_config, get_config_manager
from gaap.core.events import EventEmitter, EventType

logger = logging.getLogger("gaap.api.budget")

router = APIRouter(prefix="/api/budget", tags=["budget"])

_budget_tracker: Any = None


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


def get_budget_tracker() -> Any:
    """Get the budget tracker instance."""
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
