"""
MemoRAG - Hybrid Memory System (Vector + Knowledge Graph)
Implements: docs/evolution_plan_2026/01_MEMORY_AND_DREAMING.md

Hybrid retrieval combining:
- Vector search for semantic similarity
- Knowledge graph for relationship-aware retrieval
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("gaap.memory.memorag")


@dataclass
class KnowledgeNode:
    """عقدة في الرسم المعرفي"""

    id: str
    content: str
    node_type: str
    importance: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """حافة في الرسم المعرفي"""

    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0
    evidence: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """نتيجة استرجاع هجينة"""

    content: str
    source: str
    score: float
    node_id: str | None = None
    related_nodes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class KnowledgeGraph:
    """
    رسم معرفي للعلاقات بين المفاهيم
    """

    def __init__(self, max_nodes: int = 10000, max_edges: int = 50000) -> None:
        self._max_nodes = max_nodes
        self._max_edges = max_edges
        self.nodes: OrderedDict[str, KnowledgeNode] = OrderedDict()
        self.edges: list[KnowledgeEdge] = []
        self._adjacency: dict[str, set[str]] = {}
        self._reverse_adjacency: dict[str, set[str]] = {}
        self._logger = logging.getLogger("gaap.memory.kg")

    def add_node(
        self,
        node_id: str,
        content: str,
        node_type: str = "concept",
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> KnowledgeNode:
        """إضافة عقدة"""
        # Remove oldest node if at capacity (LRU cleanup)
        if len(self.nodes) >= self._max_nodes:
            self._remove_oldest_node()

        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            importance=importance,
            metadata=metadata or {},
        )
        self.nodes[node_id] = node
        self.nodes.move_to_end(node_id)
        if node_id not in self._adjacency:
            self._adjacency[node_id] = set()
        if node_id not in self._reverse_adjacency:
            self._reverse_adjacency[node_id] = set()
        return node

    def _remove_oldest_node(self) -> None:
        """Remove oldest node and clean up related edges (LRU eviction)."""
        if not self.nodes:
            return
        oldest_key = next(iter(self.nodes))
        del self.nodes[oldest_key]

        # Clean up related edges
        self.edges = [
            e for e in self.edges if e.source_id != oldest_key and e.target_id != oldest_key
        ]

        # Clean up adjacency data
        self._adjacency.pop(oldest_key, None)
        self._reverse_adjacency.pop(oldest_key, None)

        # Clean up references from other adjacency sets
        for adj_set in self._adjacency.values():
            adj_set.discard(oldest_key)
        for rev_set in self._reverse_adjacency.values():
            rev_set.discard(oldest_key)

    def retrieve_hybrid(self, query: str, n_results: int = 5) -> list[RetrievalResult]:
        """
        استرجاع هجين يجمع بين البحث الدلالي والرسم المعرفي.
        """
        self._logger.info(f"MemoRAG: Performing hybrid retrieval for: {query}")

        # 1. استرجاع العقد الدلالية (Semantic Retrieval)
        # (محاكاة الاسترجاع من Vector DB)
        results: list[RetrievalResult] = []

        # 2. توسيع النتائج عبر الرسم المعرفي (KG Expansion)
        # لكل نتيجة دلالية، نسحب العقد المرتبطة بها لزيادة السياق
        expanded_results = []
        for res in results:
            node_id = res.node_id
            if node_id in self.nodes:
                # سحب الجيران المباشرين
                neighbors = self._adjacency.get(node_id, set())
                for neighbor_id in neighbors:
                    neighbor = self.nodes[neighbor_id]
                    expanded_results.append(
                        RetrievalResult(
                            content=neighbor.content,
                            source="knowledge_graph_neighbor",
                            score=res.score * 0.8,  # وزن أقل للجيران
                            node_id=neighbor_id,
                            metadata=neighbor.metadata,
                        )
                    )

        return results + expanded_results[:n_results]

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        evidence: list[str] | None = None,
    ) -> KnowledgeEdge | None:
        """إضافة حافة"""
        if source_id not in self.nodes or target_id not in self.nodes:
            self._logger.warning("Cannot add edge: node not found")
            return None

        # Enforce max edges limit
        if len(self.edges) >= self._max_edges:
            self.edges.pop(0)

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            evidence=evidence or [],
        )
        self.edges.append(edge)

        if source_id in self._adjacency:
            self._adjacency[source_id].add(target_id)
        if target_id in self._reverse_adjacency:
            self._reverse_adjacency[target_id].add(source_id)

        return edge

    def get_neighbors(self, node_id: str, depth: int = 1) -> set[str]:
        """الحصول على الجيران"""
        if node_id not in self.nodes:
            return set()

        visited = {node_id}
        frontier = {node_id}

        for _ in range(depth):
            new_frontier = set()
            for n in frontier:
                new_frontier.update(self._adjacency.get(n, set()))
                new_frontier.update(self._reverse_adjacency.get(n, set()))
            frontier = new_frontier - visited
            visited.update(frontier)

        return visited - {node_id}

    def get_related(self, node_id: str, relation: str | None = None) -> list[str]:
        """الحصول على العقد المرتبطة"""
        related = []
        for edge in self.edges:
            if edge.source_id == node_id and (relation is None or edge.relation == relation):
                related.append(edge.target_id)
            elif edge.target_id == node_id and (relation is None or edge.relation == relation):
                related.append(edge.source_id)
        return related

    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[str]:
        """البحث عن مسار بين عقدتين"""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []

        from collections import deque, OrderedDict

        queue = deque([(source_id, [source_id])])
        visited = {source_id}

        while queue:
            current, path = queue.popleft()
            if current == target_id:
                return path
            if len(path) > max_depth:
                continue

            for neighbor in self._adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def get_node_importance(self, node_id: str) -> float:
        """حساب أهمية العقدة (PageRank-like)"""
        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]
        in_degree = len(self._reverse_adjacency.get(node_id, set()))
        out_degree = len(self._adjacency.get(node_id, set()))

        connectivity_score = (in_degree * 2 + out_degree) / 10.0
        return node.importance * (1 + connectivity_score)


class MemoRAG:
    """
    Hybrid Memory System combining Vector Store and Knowledge Graph

    Features:
    - Semantic search via vector embeddings
    - Relationship-aware retrieval via knowledge graph
    - Cross-modal retrieval (vector + graph)
    - Lesson extraction and storage
    """

    def __init__(
        self,
        persist_dir: str = ".gaap/memory/memorag",
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ) -> None:
        self.persist_dir = persist_dir
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

        self._vector_store: Any = None
        self._knowledge_graph = KnowledgeGraph()
        self._node_to_vector: dict[str, str] = {}
        self._vector_to_node: dict[str, str] = {}

        self._lessons: list[dict[str, Any]] = []
        self._logger = logging.getLogger("gaap.memory.memorag")

        os.makedirs(persist_dir, exist_ok=True)
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        """تهيئة مخزن المتجهات"""
        try:
            from gaap.memory.vector_store import get_vector_store

            self._vector_store = get_vector_store()
            self._logger.info("MemoRAG initialized with VectorStore")
        except Exception as e:
            self._logger.warning(f"VectorStore unavailable: {e}")
            self._vector_store = None

    def store(
        self,
        content: str,
        node_type: str = "memory",
        relations: list[tuple[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        تخزين محتوى مع علاقات

        Args:
            content: المحتوى
            node_type: نوع العقدة
            relations: قائمة العلاقات [(relation_type, target_content), ...]
            metadata: بيانات إضافية

        Returns:
            معرف العقدة
        """
        import hashlib

        node_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        self._knowledge_graph.add_node(
            node_id=node_id,
            content=content,
            node_type=node_type,
            metadata=metadata,
        )

        if self._vector_store:
            vector_id = self._vector_store.add(
                content=content,
                metadata={"node_id": node_id, "type": node_type, **(metadata or {})},
            )
            if vector_id:
                self._node_to_vector[node_id] = vector_id
                self._vector_to_node[vector_id] = node_id

        if relations:
            for relation, target_content in relations:
                target_id = hashlib.sha256(target_content.encode()).hexdigest()[:16]

                if target_id not in self._knowledge_graph.nodes:
                    self._knowledge_graph.add_node(
                        node_id=target_id,
                        content=target_content,
                        node_type="related",
                    )

                self._knowledge_graph.add_edge(
                    source_id=node_id,
                    target_id=target_id,
                    relation=relation,
                )

        return node_id

    def store_lesson(
        self,
        lesson: str,
        context: str,
        category: str,
        success: bool,
        task_type: str | None = None,
    ) -> str:
        """تخزين درس مستفاد"""
        lesson_entry = {
            "lesson": lesson,
            "context": context,
            "category": category,
            "success": success,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
        }
        self._lessons.append(lesson_entry)

        node_type = "lesson_success" if success else "lesson_failure"
        return self.store(
            content=lesson,
            node_type=node_type,
            metadata={
                "context": context,
                "category": category,
                "task_type": task_type,
            },
        )

    def retrieve(
        self,
        query: str,
        k: int = 5,
        include_related: bool = True,
        filter_type: str | None = None,
    ) -> list[RetrievalResult]:
        """
        استرجاع هجين (Vector + Graph)

        Args:
            query: الاستعلام
            k: عدد النتائج
            include_related: تضمين العقد المرتبطة
            filter_type: فلترة حسب النوع

        Returns:
            قائمة نتائج الاسترجاع
        """
        results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        if self._vector_store:
            filter_meta = {"type": filter_type} if filter_type else None
            vector_results = self._vector_store.search(
                query=query, n_results=k * 2, filter_meta=filter_meta
            )

            for vr in vector_results:
                node_id = vr.metadata.get("node_id")
                score = 1.0 / (vector_results.index(vr) + 1)

                if node_id and node_id in self._knowledge_graph.nodes:
                    node = self._knowledge_graph.nodes[node_id]
                    graph_boost = self._knowledge_graph.get_node_importance(node_id)
                    combined_score = (
                        self.vector_weight * score + self.graph_weight * graph_boost * 0.1
                    )

                    related = []
                    if include_related:
                        related = list(self._knowledge_graph.get_neighbors(node_id, depth=1))[:3]

                    results.append(
                        RetrievalResult(
                            content=vr.content,
                            source="hybrid",
                            score=combined_score,
                            node_id=node_id,
                            related_nodes=related,
                            metadata=vr.metadata,
                        )
                    )
                else:
                    results.append(
                        RetrievalResult(
                            content=vr.content,
                            source="vector",
                            score=score,
                            metadata=vr.metadata,
                        )
                    )

                if node_id:
                    seen_ids.add(node_id)

        for node_id, node in self._knowledge_graph.nodes.items():
            if node_id in seen_ids:
                continue

            if filter_type and node.node_type != filter_type:
                continue

            query_lower = query.lower()
            if query_lower in node.content.lower():
                importance = self._knowledge_graph.get_node_importance(node_id)
                related = []
                if include_related:
                    related = list(self._knowledge_graph.get_neighbors(node_id, depth=1))[:3]

                results.append(
                    RetrievalResult(
                        content=node.content,
                        source="graph",
                        score=importance * 0.5,
                        node_id=node_id,
                        related_nodes=related,
                        metadata=node.metadata,
                    )
                )
                seen_ids.add(node_id)

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]

    def retrieve_lessons(
        self,
        query: str,
        k: int = 5,
        success_only: bool = False,
    ) -> list[dict[str, Any]]:
        """استرجاع الدروس المستفادة"""
        filter_type = "lesson_success" if success_only else None
        results = self.retrieve(query=query, k=k, filter_type=filter_type)

        lessons = []
        for r in results:
            if "lesson" in r.metadata.get("type", "") or "lesson" in r.metadata.get(
                "node_type", ""
            ):
                lessons.append(
                    {
                        "lesson": r.content,
                        "context": r.metadata.get("context", ""),
                        "category": r.metadata.get("category", ""),
                        "score": r.score,
                    }
                )

        if not lessons:
            for lesson_entry in self._lessons:
                if success_only and not lesson_entry.get("success"):
                    continue
                if query.lower() in lesson_entry.get("lesson", "").lower():
                    lessons.append(lesson_entry)

        return lessons[:k]

    def get_related_concepts(self, concept: str, depth: int = 2) -> list[str]:
        """الحصول على المفاهيم المرتبطة"""
        import hashlib

        node_id = hashlib.sha256(concept.encode()).hexdigest()[:16]

        if node_id not in self._knowledge_graph.nodes:
            return []

        neighbors = self._knowledge_graph.get_neighbors(node_id, depth=depth)
        return [
            self._knowledge_graph.nodes[n].content
            for n in neighbors
            if n in self._knowledge_graph.nodes
        ]

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات MemoRAG"""
        return {
            "nodes": len(self._knowledge_graph.nodes),
            "edges": len(self._knowledge_graph.edges),
            "lessons": len(self._lessons),
            "vector_available": self._vector_store is not None
            and getattr(self._vector_store, "available", False),
        }


_memorag_instance: MemoRAG | None = None


def get_memorag() -> MemoRAG:
    """Get singleton MemoRAG instance"""
    global _memorag_instance
    if _memorag_instance is None:
        _memorag_instance = MemoRAG()
    return _memorag_instance
