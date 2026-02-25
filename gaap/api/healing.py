"""
Healing API - Self-Healing Management Endpoints
================================================

Provides REST API for managing self-healing configuration and monitoring.

Endpoints:
- GET /api/healing/config - Get healing configuration
- PUT /api/healing/config - Update healing configuration
- GET /api/healing/history - Get healing history
- GET /api/healing/patterns - Get detected failure patterns
- POST /api/healing/reset - Reset healing statistics
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.core.events import EventEmitter, EventType
from gaap.healing.healing_config import HealingConfig, create_healing_config

logger = logging.getLogger("gaap.api.healing")

router = APIRouter(prefix="/api/healing", tags=["healing"])

_healing_instance: Any = None


class HealingConfigRequest(BaseModel):
    """Request to update healing configuration."""

    max_healing_level: int | None = None
    max_retries_per_level: int | None = None
    base_delay_seconds: float | None = None
    max_delay_seconds: float | None = None
    exponential_backoff: bool | None = None
    jitter: bool | None = None
    enable_learning: bool | None = None
    enable_observability: bool | None = None
    preset: str | None = None


class HealingConfigResponse(BaseModel):
    """Healing configuration response."""

    success: bool
    config: dict[str, Any] | None = None
    error: str | None = None


class HealingStatsResponse(BaseModel):
    """Healing statistics response."""

    total_attempts: int = 0
    successful_recoveries: int = 0
    escalations: int = 0
    recovery_rate: float = 0.0
    errors_by_category: dict[str, int] = {}
    healing_by_level: dict[str, dict[str, int]] = {}


class HealingHistoryItem(BaseModel):
    """Single healing history item."""

    task_id: str
    level: str
    action: str
    success: bool
    timestamp: str
    error_category: str
    details: str = ""


class HealingHistoryResponse(BaseModel):
    """Healing history response."""

    items: list[HealingHistoryItem]
    total: int


class PatternInfo(BaseModel):
    """Information about a detected pattern."""

    pattern_id: str
    occurrences: int
    last_occurrence: str
    category: str
    sample_error: str


def get_healing_instance() -> Any:
    """Get the healing system instance."""
    return _healing_instance


def set_healing_instance(instance: Any) -> None:
    """Set the healing system instance."""
    global _healing_instance
    _healing_instance = instance


def _config_to_dict(config: HealingConfig) -> dict[str, Any]:
    """Convert HealingConfig to dictionary."""
    return config.to_dict()


@router.get("/config", response_model=HealingConfigResponse)
async def get_healing_config() -> HealingConfigResponse:
    """Get the current healing configuration."""
    try:
        healing = get_healing_instance()
        if healing and hasattr(healing, "_config"):
            return HealingConfigResponse(
                success=True,
                config=_config_to_dict(healing._config),
            )

        config = HealingConfig()
        return HealingConfigResponse(
            success=True,
            config=_config_to_dict(config),
        )
    except Exception as e:
        logger.error(f"Failed to get healing config: {e}")
        return HealingConfigResponse(success=False, error=str(e))


@router.put("/config", response_model=HealingConfigResponse)
async def update_healing_config(request: HealingConfigRequest) -> HealingConfigResponse:
    """Update healing configuration."""
    try:
        healing = get_healing_instance()

        if request.preset:
            config = create_healing_config(request.preset)
        elif healing and hasattr(healing, "_config"):
            config = healing._config
        else:
            config = HealingConfig()

        if request.max_healing_level is not None:
            config.max_healing_level = request.max_healing_level
        if request.max_retries_per_level is not None:
            config.max_retries_per_level = request.max_retries_per_level
        if request.base_delay_seconds is not None:
            config.base_delay_seconds = request.base_delay_seconds
        if request.max_delay_seconds is not None:
            config.max_delay_seconds = request.max_delay_seconds
        if request.exponential_backoff is not None:
            config.exponential_backoff = request.exponential_backoff
        if request.jitter is not None:
            config.jitter = request.jitter
        if request.enable_learning is not None:
            config.enable_learning = request.enable_learning
        if request.enable_observability is not None:
            config.enable_observability = request.enable_observability

        if healing and hasattr(healing, "_config"):
            healing._config = config

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.CONFIG_CHANGED,
            {"module": "healing", "changes": request.model_dump(exclude_none=True)},
            source="healing_api",
        )

        logger.info("Healing configuration updated")

        return HealingConfigResponse(
            success=True,
            config=_config_to_dict(config),
        )
    except Exception as e:
        logger.error(f"Failed to update healing config: {e}")
        return HealingConfigResponse(success=False, error=str(e))


@router.get("/history", response_model=HealingHistoryResponse)
async def get_healing_history(limit: int = 100) -> HealingHistoryResponse:
    """Get healing history."""
    try:
        healing = get_healing_instance()

        items = []
        total = 0

        if healing:
            if hasattr(healing, "_records"):
                records = healing._records[-limit:]
                total = len(healing._records)

                for record in records:
                    items.append(
                        HealingHistoryItem(
                            task_id=record.task_id,
                            level=record.level.name,
                            action=record.action.name,
                            success=record.success,
                            timestamp=record.timestamp.isoformat(),
                            error_category=record.error_category.name,
                            details=record.details,
                        )
                    )

        return HealingHistoryResponse(items=items, total=total)
    except Exception as e:
        logger.error(f"Failed to get healing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patterns", response_model=list[PatternInfo])
async def get_healing_patterns() -> list[PatternInfo]:
    """Get detected failure patterns."""
    try:
        healing = get_healing_instance()
        patterns = []

        if healing and hasattr(healing, "_pattern_history"):
            for pattern_id, occurrences in healing._pattern_history.items():
                if occurrences:
                    last = occurrences[-1]
                    patterns.append(
                        PatternInfo(
                            pattern_id=pattern_id,
                            occurrences=len(occurrences),
                            last_occurrence=last.get("timestamp", ""),
                            category=last.get("category", "UNKNOWN"),
                            sample_error=last.get("error", "")[:200],
                        )
                    )

        return patterns
    except Exception as e:
        logger.error(f"Failed to get healing patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset")
async def reset_healing_stats() -> dict[str, Any]:
    """Reset healing statistics."""
    try:
        healing = get_healing_instance()

        if healing:
            if hasattr(healing, "_records"):
                healing._records = []
            if hasattr(healing, "_error_history"):
                healing._error_history = {}
            if hasattr(healing, "_pattern_history"):
                healing._pattern_history = {}
            if hasattr(healing, "_total_healing_attempts"):
                healing._total_healing_attempts = 0
            if hasattr(healing, "_successful_recoveries"):
                healing._successful_recoveries = 0
            if hasattr(healing, "_escalations"):
                healing._escalations = 0
            if hasattr(healing, "_patterns_detected"):
                healing._patterns_detected = 0

        logger.info("Healing statistics reset")

        return {"success": True, "message": "Healing statistics reset"}
    except Exception as e:
        logger.error(f"Failed to reset healing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=HealingStatsResponse)
async def get_healing_stats() -> HealingStatsResponse:
    """Get healing statistics."""
    try:
        healing = get_healing_instance()

        if healing and hasattr(healing, "get_stats"):
            stats = healing.get_stats()
            return HealingStatsResponse(
                total_attempts=stats.get("total_attempts", 0),
                successful_recoveries=stats.get("successful_recoveries", 0),
                escalations=stats.get("escalations", 0),
                recovery_rate=stats.get("recovery_rate", 0.0),
                errors_by_category=stats.get("errors_by_category", {}),
                healing_by_level=stats.get("healing_by_level", {}),
            )

        return HealingStatsResponse()
    except Exception as e:
        logger.error(f"Failed to get healing stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app: Any) -> None:
    """Register healing routes with FastAPI app."""
    app.include_router(router)
