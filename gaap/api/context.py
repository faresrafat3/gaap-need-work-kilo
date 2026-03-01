"""
Context API - Semantic Context Management Endpoints
=================================================

Provides endpoints for:
- Smart Chunking (semantic code splitting)
- Call Graph (dependency analysis)
- Semantic Index (vector search)
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from gaap.context import (
    CallGraph,
    SemanticIndex,
    SmartChunker,
)

router = APIRouter(prefix="/api/context", tags=["context"])


class ChunkRequest(BaseModel):
    """Request for code chunking."""

    code: str = Field(..., min_length=1)
    file_path: Optional[str] = None
    language: str = Field(default="python")


class ChunkResponse(BaseModel):
    """Response for chunking."""

    chunks: list[dict]
    total_chunks: int


class CallGraphRequest(BaseModel):
    """Request for call graph."""

    project_path: str = Field(..., min_length=1)


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5)


@router.post("/chunk", response_model=ChunkResponse)
async def chunk_code(request: ChunkRequest) -> ChunkResponse:
    """Split code into semantic chunks."""
    try:
        chunker = SmartChunker()
        chunks = chunker.chunk(request.code, request.file_path or "<string>")

        return ChunkResponse(
            chunks=[
                {
                    "content": chunk.content,
                    "type": (
                        chunk.chunk_type.value
                        if hasattr(chunk.chunk_type, "value")
                        else str(chunk.chunk_type)
                    ),
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "dependencies": chunk.dependencies,
                }
                for chunk in chunks
            ],
            total_chunks=len(chunks),
        )
    except Exception as e:
        error_id = str(uuid.uuid4())
        logger.error(f"Context API error [{error_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")


@router.post("/call-graph")
async def build_call_graph(request: CallGraphRequest) -> dict:
    """Build call graph for a project."""
    try:
        graph = CallGraph()
        graph.build(request.project_path)

        # Get summary stats
        nodes = graph.get_nodes()
        edges = graph.get_edges()

        return {
            "nodes": len(nodes),
            "edges": len(edges),
            "files": list(set(n.split("#")[0] for n in nodes)),
        }
    except Exception as e:
        error_id = str(uuid.uuid4())
        logger.error(f"Context API error [{error_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")


@router.post("/search")
async def semantic_search(request: SemanticSearchRequest) -> dict:
    """Search code semantically."""
    try:
        index = SemanticIndex()
        results = index.search(request.query, top_k=request.top_k)

        return {
            "query": request.query,
            "results": [
                {
                    "content": r.content,
                    "file": r.file_path,
                    "score": r.score,
                }
                for r in results
            ],
        }
    except Exception as e:
        error_id = str(uuid.uuid4())
        logger.error(f"Context API error [{error_id}]: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error. Error ID: {error_id}")
