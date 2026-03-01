"""
Swarm API - Multi-Agent Orchestration Endpoints
================================================

Provides endpoints for:
- Fractal agent management
- Task auction and bidding
- Reputation tracking
- Guild management
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/swarm", tags=["swarm"])


class RegisterFractalRequest(BaseModel):
    """Request for registering a fractal agent."""

    fractal_id: str
    specialization: str
    capabilities: list[str] = Field(default_factory=list)


class TaskRequest(BaseModel):
    """Request for processing a task."""

    task: str = Field(..., min_length=1)
    domain: Optional[str] = None
    priority: int = Field(default=5, ge=1, le=10)


class BidRequest(BaseModel):
    """Request for bidding on a task."""

    task_id: str
    fractal_id: str
    utility_score: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)


class GuildRequest(BaseModel):
    """Request for guild operations."""

    guild_name: str
    fractal_id: str


@router.post("/fractal/register")
async def register_fractal(request: RegisterFractalRequest) -> dict:
    """Register a new fractal agent."""
    try:
        # In production, would create actual fractal
        return {
            "status": "registered",
            "fractal_id": request.fractal_id,
            "specialization": request.specialization,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/task")
async def process_task(request: TaskRequest) -> dict:
    """Process a task through the swarm auction."""
    try:
        # In production, would use actual orchestrator
        return {
            "task": request.task,
            "status": "processed",
            "fractal_id": "fractal_01",
            "result": f"Processed: {request.task[:50]}...",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reputation")
async def get_reputation() -> dict:
    """Get reputation scores for all fractals."""
    try:
        # In production, would fetch from actual store
        return {
            "fractals": [
                {"id": "fractal_01", "score": 0.94, "domain": "python"},
                {"id": "fractal_02", "score": 0.87, "domain": "security"},
                {"id": "fractal_03", "score": 0.82, "domain": "sql"},
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/guilds")
async def get_guilds() -> dict:
    """Get all active guilds."""
    try:
        return {
            "guilds": [
                {"name": "Python Guild", "members": 5, "avg_score": 0.91},
                {"name": "Security Guild", "members": 3, "avg_score": 0.88},
                {"name": "SQL Guild", "members": 2, "avg_score": 0.85},
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_swarm_metrics() -> dict:
    """Get swarm performance metrics."""
    try:
        return {
            "total_fractals": 10,
            "active_tasks": 3,
            "completed_tasks": 156,
            "avg_resolution_time_ms": 2450,
            "success_rate": 0.94,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
