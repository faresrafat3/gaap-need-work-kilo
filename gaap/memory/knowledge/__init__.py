"""
Memory Knowledge Module
=======================

Knowledge Graph components for memory system.

Components:
- KnowledgeGraphBuilder: Build graph from memories
- RelationExtractor: Extract relations between concepts
- DomainDetector: Detect domain from content

Reference: docs/evolution_plan_2026/25_MEMORY_AUDIT_SPEC.md
"""

from .graph_builder import KnowledgeGraphBuilder, MemoryNode, MemoryEdge
from .relation_extractor import RelationExtractor, RelationType

__all__ = [
    "KnowledgeGraphBuilder",
    "MemoryNode",
    "MemoryEdge",
    "RelationExtractor",
    "RelationType",
]
