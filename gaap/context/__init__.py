"""
GAAP Context Module
===================

Implements: docs/evolution_plan_2026/40_CONTEXT_AUDIT_SPEC.md

Semantic code understanding and retrieval:

Smart Chunking:
    - Semantic chunking with context
    - Skeleton pattern for class context
    - Cross-file dependency awareness

Call Graph:
    - NetworkX-based call graph
    - Upstream/downstream dependencies
    - 1-hop retrieval

Semantic Index:
    - Vector store integration
    - Embedding generation
    - Similarity search

Usage:
    from gaap.context import SmartChunker, CallGraph, SemanticIndex

    chunker = SmartChunker()
    chunks = chunker.chunk(code, file_path)

    graph = CallGraph()
    graph.build(project_path)
"""

from .call_graph import (
    CallGraph,
    CallGraphConfig,
    CallGraphEdge,
    CallGraphNode,
    create_call_graph,
)
from .semantic_index import (
    IndexConfig,
    IndexEntry,
    SemanticIndex,
    create_semantic_index,
)
from .smart_chunking import (
    ChunkingConfig,
    ChunkType,
    CodeChunk,
    SmartChunker,
    create_chunker,
)

__all__ = [
    "SmartChunker",
    "ChunkingConfig",
    "CodeChunk",
    "ChunkType",
    "create_chunker",
    "CallGraph",
    "CallGraphNode",
    "CallGraphEdge",
    "CallGraphConfig",
    "create_call_graph",
    "SemanticIndex",
    "IndexConfig",
    "IndexEntry",
    "create_semantic_index",
]
