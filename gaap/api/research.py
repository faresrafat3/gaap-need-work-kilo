"""
REST API Endpoints for Deep Discovery Engine
============================================

API endpoints for controlling research from Web GUI.

Endpoints:
- POST /api/research/search - Execute research
- GET /api/research/config - Get current config
- PUT /api/research/config - Update config
- GET /api/research/history - Get research history
- GET /api/research/stats - Get statistics
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.research import (
    DDEConfig,
    DeepDiscoveryEngine,
    ResearchResult,
)

logger = logging.getLogger("gaap.api.research")


router = APIRouter(prefix="/api/research", tags=["research"])

_engine: DeepDiscoveryEngine | None = None


def get_engine() -> DeepDiscoveryEngine:
    """Get or create engine instance."""
    global _engine
    if _engine is None:
        _engine = DeepDiscoveryEngine()
    return _engine


class SearchRequest(BaseModel):
    """Request for research search."""

    query: str
    depth: int | None = None
    config_override: dict[str, Any] | None = None
    force_fresh: bool = False


class ConfigUpdateRequest(BaseModel):
    """Request to update config."""

    config: dict[str, Any]


class SearchResponse(BaseModel):
    """Response from research search."""

    success: bool
    query: str
    finding: dict[str, Any] | None = None
    metrics: dict[str, Any]
    total_time_ms: str
    error: str | None = None


@router.post("/search", response_model=SearchResponse)  # type: ignore[untyped-decorator]
async def search(request: SearchRequest) -> SearchResponse:
    """
    Execute research on a query.

    Args:
        request: Search request with query and options

    Returns:
        Research result with sources, hypotheses, and metrics
    """
    engine = get_engine()

    result: ResearchResult = await engine.research(
        query=request.query,
        depth=request.depth,
        config_override=request.config_override,
        force_fresh=request.force_fresh,
    )

    return SearchResponse(
        success=result.success,
        query=result.query,
        finding=result.finding.to_dict() if result.finding else None,
        metrics=result.metrics.to_dict(),
        total_time_ms=f"{result.total_time_ms:.1f}",
        error=result.error,
    )


@router.get("/config")  # type: ignore[untyped-decorator]
async def get_config() -> dict[str, Any]:
    """
    Get current research configuration.

    Returns:
        Current DDEConfig as dict
    """
    engine = get_engine()
    return engine.get_config().to_dict()


@router.put("/config")  # type: ignore[untyped-decorator]
async def update_config(request: ConfigUpdateRequest) -> dict[str, str]:
    """
    Update research configuration.

    Args:
        request: New configuration values

    Returns:
        Success status
    """
    engine = get_engine()

    try:
        new_config = DDEConfig.from_dict(request.config)
        engine.update_config(new_config)
        return {"status": "updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/history")  # type: ignore[untyped-decorator]
async def get_history(limit: int = 20) -> list[dict[str, Any]]:
    """
    Get research history.

    Args:
        limit: Maximum number of results

    Returns:
        List of past research findings
    """
    engine = get_engine()

    findings = await engine._knowledge_integrator.get_by_topic("")

    return [f.to_dict() for f in findings[:limit]]


@router.get("/stats")  # type: ignore[untyped-decorator]
async def get_stats() -> dict[str, Any]:
    """
    Get research statistics.

    Returns:
        Engine and component statistics
    """
    engine = get_engine()
    return engine.get_stats()


@router.get("/quick")  # type: ignore[untyped-decorator]
async def quick_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Quick search without deep analysis.

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of sources
    """
    engine = get_engine()

    sources = await engine.quick_search(query, max_results)

    return [s.to_dict() for s in sources]


def register_routes(app: Any) -> None:
    """Register research routes with FastAPI app."""
    app.include_router(router)
