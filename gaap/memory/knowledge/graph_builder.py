"""
Knowledge Graph Builder Module
==============================

Builds and manages knowledge graph from memories.

Features:
- Node creation with types (Episode, Concept, Error, Solution, Pattern)
- Edge creation with relation types
- Graph traversal and querying
- Importance scoring (PageRank-like)
"""

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

from gaap.storage.atomic import atomic_write

logger = logging.getLogger("gaap.memory.knowledge.graph")


class NodeType(Enum):
    """أنواع العقد"""

    EPISODE = auto()  # حدث/تجربة
    CONCEPT = auto()  # مفهوم
    ERROR = auto()  # خطأ
    SOLUTION = auto()  # حل
    PATTERN = auto()  # نمط
    TASK = auto()  # مهمة
    TOOL = auto()  # أداة
    DOMAIN = auto()  # مجال


class RelationType(Enum):
    """أنواع العلاقات"""

    CAUSED = "caused"  # سبب
    FIXED = "fixed"  # أصلح
    RELATED_TO = "related_to"  # مرتبط بـ
    IS_A = "is_a"  # نوع من
    DEPENDS_ON = "depends_on"  # يعتمد على
    CONTRADICTS = "contradicts"  # يتعارض مع
    SIMILAR_TO = "similar_to"  # مشابه لـ
    FOLLOWS = "follows"  # يتبع
    PRECEDES = "precedes"  # يسبق
    IMPLEMENTS = "implements"  # يطبق
    USES = "uses"  # يستخدم


@dataclass
class MemoryNode:
    """
    عقدة في الرسم المعرفي

    Attributes:
        id: معرف فريد
        content: المحتوى
        node_type: نوع العقدة
        importance: الأهمية (0-1)
        domain: التخصص
        created_at: تاريخ الإنشاء
        access_count: عدد مرات الوصول
        last_accessed: آخر وصول
        metadata: بيانات إضافية
    """

    id: str
    content: str
    node_type: NodeType = NodeType.CONCEPT
    importance: float = 1.0
    domain: str = "general"
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:500],
            "node_type": self.node_type.name,
            "importance": self.importance,
            "domain": self.domain,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        return cls(
            id=data["id"],
            content=data["content"],
            node_type=NodeType[data.get("node_type", "CONCEPT")],
            importance=data.get("importance", 1.0),
            domain=data.get("domain", "general"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None
            ),
            metadata=data.get("metadata", {}),
        )

    def touch(self) -> None:
        """تسجيل وصول للعقدة"""
        self.access_count += 1
        self.last_accessed = datetime.now()


@dataclass
class MemoryEdge:
    """
    حافة في الرسم المعرفي

    Attributes:
        source_id: معرف العقدة المصدر
        target_id: معرف العقدة الهدف
        relation: نوع العلاقة
        weight: وزن العلاقة (0-1)
        confidence: ثقة في العلاقة (0-1)
        evidence: أدلة على العلاقة
        created_at: تاريخ الإنشاء
    """

    source_id: str
    target_id: str
    relation: RelationType = RelationType.RELATED_TO
    weight: float = 1.0
    confidence: float = 1.0
    evidence: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEdge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=RelationType(data.get("relation", "related_to")),
            weight=data.get("weight", 1.0),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", []),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
        )


class KnowledgeGraphBuilder:
    """
    بناء وإدارة الرسم المعرفي

    Features:
    - إضافة عقد وحواف
    - البحث عن المسارات
    - حساب الأهمية
    - استخراج العلاقات
    - حفظ واسترجاع
    """

    def __init__(
        self,
        storage_path: str | None = None,
        max_nodes: int = 10000,
        decay_factor: float = 0.95,
    ):
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_nodes = max_nodes
        self.decay_factor = decay_factor

        self._nodes: dict[str, MemoryNode] = {}
        self._edges: list[MemoryEdge] = []
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._edge_index: dict[tuple[str, str, str], MemoryEdge] = {}

        self._logger = logger

        if self.storage_path:
            self._load()

    def add_node(
        self,
        content: str,
        node_type: NodeType = NodeType.CONCEPT,
        domain: str = "general",
        importance: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryNode:
        """
        إضافة عقدة جديدة

        Args:
            content: المحتوى
            node_type: نوع العقدة
            domain: التخصص
            importance: الأهمية
            metadata: بيانات إضافية

        Returns:
            العقدة المضافة
        """
        node_id = self._generate_id(content)

        if node_id in self._nodes:
            existing = self._nodes[node_id]
            existing.touch()
            existing.importance = min(1.0, existing.importance + 0.1)
            return existing

        node = MemoryNode(
            id=node_id,
            content=content,
            node_type=node_type,
            domain=domain,
            importance=importance,
            metadata=metadata or {},
        )

        self._nodes[node_id] = node
        self._logger.debug(f"Added node {node_id} of type {node_type.name}")

        if len(self._nodes) > self.max_nodes:
            self._prune_nodes()

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType = RelationType.RELATED_TO,
        weight: float = 1.0,
        confidence: float = 1.0,
        evidence: list[str] | None = None,
    ) -> MemoryEdge | None:
        """
        إضافة حافة (علاقة)

        Args:
            source_id: معرف العقدة المصدر
            target_id: معرف العقدة الهدف
            relation: نوع العلاقة
            weight: وزن العلاقة
            confidence: ثقة في العلاقة
            evidence: أدلة

        Returns:
            الحافة المضافة أو None إذا فشل
        """
        if source_id not in self._nodes:
            self._logger.warning(f"Source node {source_id} not found")
            return None

        if target_id not in self._nodes:
            self._logger.warning(f"Target node {target_id} not found")
            return None

        edge_key = (source_id, target_id, relation.value)

        if edge_key in self._edge_index:
            existing = self._edge_index[edge_key]
            existing.weight = min(1.0, existing.weight + 0.1)
            existing.confidence = min(1.0, existing.confidence + 0.05)
            if evidence:
                existing.evidence.extend(evidence)
            return existing

        edge = MemoryEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            confidence=confidence,
            evidence=evidence or [],
        )

        self._edges.append(edge)
        self._outgoing[source_id].add(target_id)
        self._incoming[target_id].add(source_id)
        self._edge_index[edge_key] = edge

        self._logger.debug(f"Added edge {source_id} -> {target_id} ({relation.value})")

        return edge

    def get_node(self, node_id: str) -> MemoryNode | None:
        """الحصول على عقدة"""
        node = self._nodes.get(node_id)
        if node:
            node.touch()
        return node

    def get_neighbors(
        self,
        node_id: str,
        depth: int = 1,
        relation: RelationType | None = None,
    ) -> set[str]:
        """
        الحصول على الجيران

        Args:
            node_id: معرف العقدة
            depth: عمق البحث
            relation: نوع العلاقة (اختياري)

        Returns:
            مجموعة معرفات الجيران
        """
        if node_id not in self._nodes:
            return set()

        visited = {node_id}
        frontier = {node_id}

        for _ in range(depth):
            new_frontier = set()

            for n in frontier:
                outgoing = self._outgoing.get(n, set())
                incoming = self._incoming.get(n, set())

                for neighbor in outgoing | incoming:
                    if neighbor not in visited:
                        if relation is None or self._has_relation(n, neighbor, relation):
                            new_frontier.add(neighbor)

            visited.update(new_frontier)
            frontier = new_frontier

            if not frontier:
                break

        return visited - {node_id}

    def _has_relation(self, source_id: str, target_id: str, relation: RelationType) -> bool:
        """فحص وجود علاقة معينة"""
        edge_key = (source_id, target_id, relation.value)
        reverse_key = (target_id, source_id, relation.value)
        return edge_key in self._edge_index or reverse_key in self._edge_index

    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 3,
    ) -> list[str]:
        """
        البحث عن مسار بين عقدتين

        Args:
            source_id: معرف العقدة المصدر
            target_id: معرف العقدة الهدف
            max_depth: أقصى عمق

        Returns:
            قائمة معرفات العقد في المسار
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            return []

        from collections import deque

        queue = deque([(source_id, [source_id])])
        visited = {source_id}

        while queue:
            current, path = queue.popleft()

            if current == target_id:
                return path

            if len(path) > max_depth:
                continue

            neighbors = self._outgoing.get(current, set()) | self._incoming.get(current, set())

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []

    def get_node_importance(self, node_id: str) -> float:
        """
        حساب أهمية العقدة (PageRank-like)

        Factors:
        - درجة الاتصال (in-degree + out-degree)
        - الأهمية الأساسية
        - عدد مرات الوصول
        """
        if node_id not in self._nodes:
            return 0.0

        node = self._nodes[node_id]

        in_degree = len(self._incoming.get(node_id, set()))
        out_degree = len(self._outgoing.get(node_id, set()))

        connectivity_score = (in_degree * 2 + out_degree) / 20.0
        access_score = min(1.0, node.access_count / 10.0)

        total = node.importance * 0.4 + connectivity_score * 0.3 + access_score * 0.3

        return min(1.0, total)

    def get_related_by_relation(
        self,
        node_id: str,
        relation: RelationType,
    ) -> list[str]:
        """الحصول على العقد المرتبطة بعلاقة معينة"""
        related = []

        for edge in self._edges:
            if edge.relation == relation:
                if edge.source_id == node_id:
                    related.append(edge.target_id)
                elif edge.target_id == node_id:
                    related.append(edge.source_id)

        return related

    def get_subgraph(
        self,
        node_ids: list[str],
        include_edges: bool = True,
    ) -> tuple[dict[str, MemoryNode], list[MemoryEdge]]:
        """
        استخراج رسم فرعي

        Args:
            node_ids: قائمة معرفات العقد
            include_edges: تضمين الحواف

        Returns:
            (nodes, edges)
        """
        nodes = {nid: self._nodes[nid] for nid in node_ids if nid in self._nodes}

        edges = []
        if include_edges:
            node_set = set(node_ids)
            for edge in self._edges:
                if edge.source_id in node_set and edge.target_id in node_set:
                    edges.append(edge)

        return nodes, edges

    def decay_importance(self) -> None:
        """تقليل أهمية العقد غير المستخدمة"""
        for node in self._nodes.values():
            if node.last_accessed:
                days_since_access = (datetime.now() - node.last_accessed).days
                node.importance *= self.decay_factor**days_since_access

    def _prune_nodes(self) -> None:
        """حذف العقد الأقل أهمية"""
        if len(self._nodes) <= self.max_nodes:
            return

        sorted_nodes = sorted(
            self._nodes.items(),
            key=lambda x: x[1].importance,
        )

        to_remove = len(self._nodes) - self.max_nodes + 100

        for node_id, _ in sorted_nodes[:to_remove]:
            self._remove_node(node_id)

        self._logger.info(f"Pruned {to_remove} nodes")

    def _remove_node(self, node_id: str) -> None:
        """حذف عقدة وجميع حوافها"""
        if node_id not in self._nodes:
            return

        del self._nodes[node_id]

        for target_id in list(self._outgoing.get(node_id, set())):
            self._incoming[target_id].discard(node_id)

        for source_id in list(self._incoming.get(node_id, set())):
            self._outgoing[source_id].discard(node_id)

        del self._outgoing[node_id]
        del self._incoming[node_id]

        self._edges = [e for e in self._edges if e.source_id != node_id and e.target_id != node_id]

        self._edge_index = {
            k: v for k, v in self._edge_index.items() if k[0] != node_id and k[1] != node_id
        }

    def _generate_id(self, content: str) -> str:
        """توليد معرف فريد"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save(self) -> bool:
        """حفظ الرسم"""
        if not self.storage_path:
            return False

        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)

            nodes_file = self.storage_path / "nodes.json"
            edges_file = self.storage_path / "edges.json"

            nodes_data = {nid: n.to_dict() for nid, n in self._nodes.items()}
            edges_data = [e.to_dict() for e in self._edges]

            atomic_write(nodes_file, json.dumps(nodes_data, indent=2))
            atomic_write(edges_file, json.dumps(edges_data, indent=2))

            self._logger.info(f"Saved {len(self._nodes)} nodes and {len(self._edges)} edges")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save knowledge graph: {e}")
            return False

    def _load(self) -> bool:
        """تحميل الرسم"""
        if not self.storage_path:
            return False

        try:
            nodes_file = self.storage_path / "nodes.json"
            edges_file = self.storage_path / "edges.json"

            if nodes_file.exists():
                with open(nodes_file) as f:
                    nodes_data = json.load(f)

                for nid, ndata in nodes_data.items():
                    self._nodes[nid] = MemoryNode.from_dict(ndata)

            if edges_file.exists():
                with open(edges_file) as f:
                    edges_data = json.load(f)

                for edata in edges_data:
                    edge = MemoryEdge.from_dict(edata)
                    self._edges.append(edge)
                    self._outgoing[edge.source_id].add(edge.target_id)
                    self._incoming[edge.target_id].add(edge.source_id)
                    self._edge_index[(edge.source_id, edge.target_id, edge.relation.value)] = edge

            self._logger.info(f"Loaded {len(self._nodes)} nodes and {len(self._edges)} edges")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load knowledge graph: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الرسم"""
        node_types: dict[str, int] = defaultdict(int)
        for node in self._nodes.values():
            node_types[node.node_type.name] += 1

        relation_types: dict[str, int] = defaultdict(int)
        for edge in self._edges:
            relation_types[edge.relation.value] += 1

        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": dict(node_types),
            "relation_types": dict(relation_types),
            "avg_connections": len(self._edges) * 2 / len(self._nodes) if self._nodes else 0,
        }
