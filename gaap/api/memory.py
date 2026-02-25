"""
Memory API - Hierarchical Memory Management Endpoints
=====================================================

Provides REST API for managing GAAP's 4-tier hierarchical memory system.

Endpoints:
- GET /api/memory/stats - Get memory statistics
- GET /api/memory/tiers - Get tier details
- POST /api/memory/consolidate - Trigger memory consolidation
- POST /api/memory/clear/{tier} - Clear a specific memory tier
- GET /api/memory/search - Search memory contents
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from gaap.core.events import EventEmitter, EventType
from gaap.memory.hierarchical import (
    HierarchicalMemory,
    MemoryTier,
    MemoryPriority,
)

logger = logging.getLogger("gaap.api.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])

_memory_instance: HierarchicalMemory | None = None


class MemoryStatsResponse(BaseModel):
    """Memory statistics response."""

    working: dict[str, Any] = {}
    episodic: dict[str, Any] = {}
    semantic: dict[str, Any] = {}
    procedural: dict[str, Any] = {}


class TierInfo(BaseModel):
    """Information about a memory tier."""

    name: str
    level: int
    size: int
    max_size: int | None = None
    description: str


class TierListResponse(BaseModel):
    """List of memory tiers."""

    tiers: list[TierInfo]


class ConsolidateRequest(BaseModel):
    """Request to consolidate memory."""

    source_tier: str = "working"
    target_tier: str = "episodic"
    min_access_count: int = 3


class ConsolidateResponse(BaseModel):
    """Consolidation result."""

    success: bool
    items_consolidated: int = 0
    message: str = ""


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    tier: str
    content_preview: str
    relevance: float
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    """Search results response."""

    query: str
    results: list[SearchResult]
    total: int


class ClearResponse(BaseModel):
    """Clear tier response."""

    success: bool
    tier: str
    items_cleared: int = 0
    message: str = ""


def get_memory_instance() -> HierarchicalMemory:
    """Get or create the memory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = HierarchicalMemory()
    return _memory_instance


def set_memory_instance(instance: HierarchicalMemory) -> None:
    """Set the memory instance."""
    global _memory_instance
    _memory_instance = instance


def _get_tier_description(tier: MemoryTier) -> str:
    """Get description for a memory tier."""
    descriptions = {
        MemoryTier.WORKING: "Fast, limited capacity memory for current context",
        MemoryTier.EPISODIC: "Event history memory for learning from experience",
        MemoryTier.SEMANTIC: "Patterns and rules extracted from episodes",
        MemoryTier.PROCEDURAL: "Acquired skills, templates, and procedures",
    }
    return descriptions.get(tier, "")


@router.get("/stats", response_model=MemoryStatsResponse)
async def get_memory_stats() -> MemoryStatsResponse:
    """Get memory statistics for all tiers."""
    try:
        memory = get_memory_instance()
        stats = memory.get_stats()

        return MemoryStatsResponse(
            working=stats.get("working", {}),
            episodic=stats.get("episodic", {}),
            semantic=stats.get("semantic", {}),
            procedural=stats.get("procedural", {}),
        )
    except Exception as e:
        logger.error(f"Failed to get memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiers", response_model=TierListResponse)
async def get_memory_tiers() -> TierListResponse:
    """Get details about all memory tiers."""
    try:
        memory = get_memory_instance()
        stats = memory.get_stats()

        tiers = []
        for tier in MemoryTier:
            tier_name = tier.name.lower()
            tier_stats = stats.get(tier_name, {})

            max_size = None
            if tier == MemoryTier.WORKING:
                max_size = memory.working.max_size

            tiers.append(
                TierInfo(
                    name=tier_name,
                    level=tier.value,
                    size=tier_stats.get(
                        "size",
                        tier_stats.get(
                            "total_episodes",
                            tier_stats.get("total_rules", tier_stats.get("total_procedures", 0)),
                        ),
                    ),
                    max_size=max_size,
                    description=_get_tier_description(tier),
                )
            )

        return TierListResponse(tiers=tiers)
    except Exception as e:
        logger.error(f"Failed to get memory tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate_memory(request: ConsolidateRequest) -> ConsolidateResponse:
    """Trigger memory consolidation between tiers."""
    try:
        memory = get_memory_instance()
        items_consolidated = 0

        source_tier = request.source_tier.lower()
        target_tier = request.target_tier.lower()

        if source_tier == "working" and target_tier == "episodic":
            working_size_before = memory.working.get_size()

            emitter = EventEmitter.get_instance()
            emitter.emit(
                EventType.SYSTEM_HEALTH,
                {"action": "consolidate", "source": source_tier, "target": target_tier},
                source="memory_api",
            )

            items_consolidated = max(0, working_size_before - memory.working.get_size())

        elif source_tier == "episodic" and target_tier == "semantic":
            lessons = memory.episodic.get_recent_lessons(limit=20)
            items_consolidated = len(lessons)

            for lesson in lessons:
                parts = lesson.split(":", 1)
                if len(parts) == 2:
                    memory.semantic.add_rule(
                        condition=parts[0].strip(),
                        action=parts[1].strip(),
                        confidence=0.6,
                    )

        logger.info(f"Consolidated {items_consolidated} items from {source_tier} to {target_tier}")

        return ConsolidateResponse(
            success=True,
            items_consolidated=items_consolidated,
            message=f"Consolidated {items_consolidated} items from {source_tier} to {target_tier}",
        )
    except Exception as e:
        logger.error(f"Failed to consolidate memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear/{tier}", response_model=ClearResponse)
async def clear_memory_tier(tier: str) -> ClearResponse:
    """Clear a specific memory tier."""
    try:
        memory = get_memory_instance()
        tier_lower = tier.lower()
        items_cleared = 0

        if tier_lower == "working":
            items_cleared = memory.working.get_size()
            memory.working.clear()
        elif tier_lower == "episodic":
            items_cleared = len(memory.episodic._episodes)
            memory.episodic._episodes = []
            memory.episodic._task_index = {}
        elif tier_lower == "semantic":
            items_cleared = len(memory.semantic._rules)
            memory.semantic._rules = {}
            memory.semantic._pattern_index = {}
        elif tier_lower == "procedural":
            items_cleared = len(memory.procedural._procedures)
            memory.procedural._procedures = {}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown tier: {tier}")

        emitter = EventEmitter.get_instance()
        emitter.emit(
            EventType.SYSTEM_HEALTH,
            {"action": "clear_tier", "tier": tier_lower, "items_cleared": items_cleared},
            source="memory_api",
        )

        logger.info(f"Cleared {items_cleared} items from {tier_lower} tier")

        return ClearResponse(
            success=True,
            tier=tier_lower,
            items_cleared=items_cleared,
            message=f"Cleared {items_cleared} items from {tier_lower} tier",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear memory tier: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search", response_model=SearchResponse)
async def search_memory(query: str, limit: int = 10) -> SearchResponse:
    """Search memory contents across all tiers."""
    try:
        memory = get_memory_instance()
        results = []

        semantic_rules = memory.semantic.find_rules(query, min_confidence=0.3)
        for rule in semantic_rules[:limit]:
            results.append(
                SearchResult(
                    id=rule.id,
                    tier="semantic",
                    content_preview=f"{rule.condition} -> {rule.action}",
                    relevance=rule.confidence,
                    metadata={
                        "support_count": rule.support_count,
                        "created_at": rule.created_at.isoformat() if rule.created_at else None,
                    },
                )
            )

        episodes = memory.episodic.get_episodes(limit=limit)
        query_lower = query.lower()
        for episode in episodes:
            if query_lower in episode.action.lower() or query_lower in episode.result.lower():
                results.append(
                    SearchResult(
                        id=episode.task_id,
                        tier="episodic",
                        content_preview=episode.action[:200],
                        relevance=0.7 if episode.success else 0.5,
                        metadata={
                            "success": episode.success,
                            "model": episode.model,
                            "provider": episode.provider,
                            "timestamp": episode.timestamp.isoformat()
                            if episode.timestamp
                            else None,
                        },
                    )
                )

        lessons = memory.episodic.get_recent_lessons(limit=limit * 2)
        for lesson in lessons:
            if query_lower in lesson.lower():
                results.append(
                    SearchResult(
                        id="",
                        tier="episodic",
                        content_preview=lesson[:200],
                        relevance=0.6,
                        metadata={},
                    )
                )

        results = results[:limit]

        return SearchResponse(
            query=query,
            results=results,
            total=len(results),
        )
    except Exception as e:
        logger.error(f"Failed to search memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
async def save_memory() -> dict[str, Any]:
    """Save all memory tiers to disk."""
    try:
        memory = get_memory_instance()
        results = memory.save()

        logger.info(f"Memory save results: {results}")

        return {
            "success": all(results.values()),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def load_memory() -> dict[str, Any]:
    """Load all memory tiers from disk."""
    try:
        memory = get_memory_instance()
        results = memory.load()

        logger.info(f"Memory load results: {results}")

        return {
            "success": all(results.values()),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def register_routes(app: Any) -> None:
    """Register memory routes with FastAPI app."""
    app.include_router(router)
