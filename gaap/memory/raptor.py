"""
RAPTOR (Recursive Abstractive Retrieval) Implementation

Implements a tree-structured approach to document retrieval that builds
a hierarchical summary tree from documents, enabling multi-level retrieval.

Key Components:
    - SummaryTreeNode: Node with text, summary, children, and level
    - SummaryTree: Tree structure for hierarchical document organization
    - CollapsedTreeRetrieval: Search across tree layers

Usage:
    from gaap.memory.raptor import SummaryTree, CollapsedTreeRetrieval

    tree = SummaryTree()
    tree.build_from_documents(documents)

    retrieval = CollapsedTreeRetrieval(tree)
    results = retrieval.retrieve("query about topic", k=5)
"""

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.raptor")


class NodeType(Enum):
    LEAF = 0
    INTERNAL = 1
    ROOT = 2


@dataclass
class SummaryTreeNode:
    """
    Node in the RAPTOR summary tree.

    Each node contains:
    - Original text (for leaf nodes) or aggregated text (for internal nodes)
    - Generated summary
    - Children references
    - Level in the tree (0 = leaf level)

    Attributes:
        id: Unique node identifier
        text: Original or aggregated text
        summary: Generated summary of the text
        children: List of child node IDs
        parent: Parent node ID (None for root)
        level: Level in the tree (0 = leaf)
        node_type: Type of node (LEAF, INTERNAL, ROOT)
        metadata: Additional metadata
        embedding: Optional embedding vector
        created_at: Creation timestamp
        importance: Estimated importance score
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    summary: str = ""
    children: list[str] = field(default_factory=list)
    parent: str | None = None
    level: int = 0
    node_type: NodeType = NodeType.LEAF
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: datetime = field(default_factory=datetime.now)
    importance: float = 0.5

    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF

    def is_root(self) -> bool:
        return self.node_type == NodeType.ROOT

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text[:500] + "..." if len(self.text) > 500 else self.text,
            "summary": self.summary[:500] + "..." if len(self.summary) > 500 else self.summary,
            "children": self.children,
            "parent": self.parent,
            "level": self.level,
            "node_type": self.node_type.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "importance": self.importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryTreeNode":
        node_type = NodeType[data.get("node_type", "LEAF")]
        created_at = data.get("created_at")
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at)
            except Exception:
                created_at = datetime.now()
        else:
            created_at = datetime.now()

        return cls(
            id=data.get("id", str(uuid.uuid4())),
            text=data.get("text", ""),
            summary=data.get("summary", ""),
            children=data.get("children", []),
            parent=data.get("parent"),
            level=data.get("level", 0),
            node_type=node_type,
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=created_at,
            importance=data.get("importance", 0.5),
        )


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class SimpleHashEmbedding:
    """Simple hash-based embedding for fallback."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            embedding = [int(text_hash[i : i + 8], 16) / 0xFFFFFFFF for i in range(0, 64, 8)]
            embedding = [e * 2 - 1 for e in embedding]
            while len(embedding) < 384:
                embedding.extend(embedding[: min(384 - len(embedding), len(embedding))])
            embeddings.append(embedding[:384])
        return embeddings


@runtime_checkable
class SummarizerProtocol(Protocol):
    """Protocol for summarization functions."""

    async def summarize(self, texts: list[str]) -> str: ...


class SimpleSummarizer:
    """Simple summarizer that concatenates and truncates."""

    async def summarize(self, texts: list[str]) -> str:
        combined = " ".join(texts)
        if len(combined) > 500:
            return combined[:500] + "..."
        return combined


@dataclass
class Document:
    """Input document for tree building."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
        }


class SummaryTree:
    """
    RAPTOR Summary Tree for hierarchical document organization.

    Builds a tree structure where:
    - Leaf nodes contain original document text
    - Internal nodes contain summaries of their children
    - Root node contains a project-level summary

    The tree enables:
    - Multi-granularity retrieval
    - Context-aware search
    - Hierarchical browsing

    Attributes:
        nodes: Dictionary of all nodes by ID
        root_id: ID of the root node
        max_children: Maximum children per internal node
        embedding_fn: Function to generate embeddings
        summarizer: Function to generate summaries
        _level_indices: Index of nodes by level

    Usage:
        tree = SummaryTree(max_children=5)
        tree.build_from_documents([doc1, doc2, doc3])
        root = tree.get_root()
        children = tree.get_children(root.id)
    """

    def __init__(
        self,
        max_children: int = 5,
        embedding_fn: EmbeddingFunction | None = None,
        summarizer: SummarizerProtocol | None = None,
    ) -> None:
        self.nodes: dict[str, SummaryTreeNode] = {}
        self.root_id: str | None = None
        self.max_children = max_children
        self.embedding_fn = embedding_fn or SimpleHashEmbedding()
        self.summarizer = summarizer or SimpleSummarizer()
        self._level_indices: dict[int, list[str]] = {}
        self._leaf_nodes: list[str] = []
        self._logger = logger

    def build_from_documents(
        self,
        documents: list[Document],
        batch_size: int = 10,
    ) -> str:
        """
        Build tree from a list of documents.

        Args:
            documents: List of Document objects
            batch_size: Batch size for embedding generation

        Returns:
            ID of the root node

        Example:
            docs = [Document(text="..."), Document(text="...")]
            root_id = tree.build_from_documents(docs)
        """
        self.nodes.clear()
        self._level_indices.clear()
        self._leaf_nodes.clear()

        for doc in documents:
            self.add_leaf(doc.text, doc.metadata)

        return self.build_tree()

    def add_leaf(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> SummaryTreeNode:
        """
        Add a leaf node to the tree.

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Created leaf node

        Example:
            node = tree.add_leaf("Document content", {"source": "web"})
        """
        node = SummaryTreeNode(
            text=text,
            summary=text[:200] + "..." if len(text) > 200 else text,
            level=0,
            node_type=NodeType.LEAF,
            metadata=metadata or {},
        )

        embedding = self._get_embedding(text)
        node.embedding = embedding
        node.importance = self._estimate_importance(text)

        self.nodes[node.id] = node
        self._leaf_nodes.append(node.id)
        self._level_indices.setdefault(0, []).append(node.id)

        return node

    def summarize_children(
        self,
        parent_id: str,
    ) -> str:
        """
        Create summary from children nodes.

        Args:
            parent_id: ID of parent node

        Returns:
            Generated summary

        Raises:
            ValueError: If parent node not found or has no children
        """
        parent = self.nodes.get(parent_id)
        if not parent:
            raise ValueError(f"Parent node {parent_id} not found")

        if not parent.children:
            raise ValueError(f"Parent node {parent_id} has no children")

        child_texts = []
        for child_id in parent.children:
            child = self.nodes.get(child_id)
            if child:
                child_texts.append(child.summary or child.text)

        summary = asyncio.run(self.summarizer.summarize(child_texts))
        parent.summary = summary

        combined_text = " ".join(child_texts)
        parent.embedding = self._get_embedding(combined_text)
        parent.importance = sum(
            self.nodes[cid].importance for cid in parent.children if cid in self.nodes
        ) / len(parent.children)

        return summary

    def build_tree(self) -> str:
        """
        Build complete tree from leaf nodes.

        Groups leaf nodes, creates parent summaries, and builds
        up the tree until a single root is reached.

        Returns:
            ID of the root node

        Example:
            root_id = tree.build_tree()
        """
        if not self._leaf_nodes:
            self._logger.warning("No leaf nodes to build tree")
            return ""

        current_level = 0
        current_nodes = self._leaf_nodes.copy()

        while len(current_nodes) > 1:
            next_level_nodes: list[str] = []

            for i in range(0, len(current_nodes), self.max_children):
                batch = current_nodes[i : i + self.max_children]

                parent = SummaryTreeNode(
                    level=current_level + 1,
                    node_type=NodeType.INTERNAL,
                    children=batch,
                )

                self.nodes[parent.id] = parent

                for child_id in batch:
                    child = self.nodes.get(child_id)
                    if child:
                        child.parent = parent.id
                        child_texts = [
                            self.nodes[cid].summary or self.nodes[cid].text
                            for cid in batch
                            if cid in self.nodes
                        ]
                        if child_texts:
                            parent.text = " ".join(child_texts)

                summary = self.summarize_children(parent.id)

                next_level_nodes.append(parent.id)
                self._level_indices.setdefault(current_level + 1, []).append(parent.id)

            current_level += 1
            current_nodes = next_level_nodes

        if current_nodes:
            root_id = current_nodes[0]
            root = self.nodes[root_id]
            root.node_type = NodeType.ROOT
            self.root_id = root_id
            self._logger.info(f"Built tree with root {root_id}, {len(self.nodes)} total nodes")
            return root_id

        return ""

    def get_node(self, node_id: str) -> SummaryTreeNode | None:
        """
        Get node by ID.

        Args:
            node_id: Node identifier

        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[SummaryTreeNode]:
        """
        Get children of a node.

        Args:
            node_id: Parent node identifier

        Returns:
            List of child nodes
        """
        node = self.nodes.get(node_id)
        if not node:
            return []

        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_path_to_root(self, node_id: str) -> list[SummaryTreeNode]:
        """
        Get path from node to root.

        Args:
            node_id: Starting node identifier

        Returns:
            List of nodes from start to root (inclusive)
        """
        path = []
        current_id = node_id

        while current_id:
            node = self.nodes.get(current_id)
            if not node:
                break
            path.append(node)
            current_id = node.parent

        return path

    def get_root(self) -> SummaryTreeNode | None:
        """Get the root node."""
        if self.root_id:
            return self.nodes.get(self.root_id)
        return None

    def get_nodes_at_level(self, level: int) -> list[SummaryTreeNode]:
        """
        Get all nodes at a specific level.

        Args:
            level: Level number (0 = leaf)

        Returns:
            List of nodes at that level
        """
        node_ids = self._level_indices.get(level, [])
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_leaf_nodes(self) -> list[SummaryTreeNode]:
        """Get all leaf nodes."""
        return [self.nodes[nid] for nid in self._leaf_nodes if nid in self.nodes]

    def get_max_level(self) -> int:
        """Get the maximum level in the tree."""
        return max(self._level_indices.keys()) if self._level_indices else 0

    def search_similar(
        self,
        query: str,
        k: int = 5,
        level: int | None = None,
    ) -> list[tuple[SummaryTreeNode, float]]:
        """
        Search for similar nodes.

        Args:
            query: Query text
            k: Number of results
            level: Optional level to search (None = all levels)

        Returns:
            List of (node, similarity) tuples
        """
        query_embedding = self._get_embedding(query)

        candidates = []
        if level is not None:
            candidates = self.get_nodes_at_level(level)
        else:
            candidates = list(self.nodes.values())

        similarities: list[tuple[SummaryTreeNode, float]] = []
        for node in candidates:
            if node.embedding:
                sim = self._cosine_similarity(query_embedding, node.embedding)
                similarities.append((node, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        try:
            embeddings = self.embedding_fn([text])
            return embeddings[0] if embeddings else [0.0] * 384
        except Exception as e:
            self._logger.warning(f"Embedding generation failed: {e}")
            return [0.0] * 384

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _estimate_importance(self, text: str) -> float:
        """Estimate importance of text based on heuristics."""
        score = 0.5

        length_factor = min(len(text) / 1000, 1.0)
        score += length_factor * 0.2

        keywords = ["important", "critical", "key", "essential", "main", "primary"]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        score += min(keyword_count * 0.05, 0.2)

        return min(score, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize tree to dictionary."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "root_id": self.root_id,
            "max_children": self.max_children,
            "level_indices": self._level_indices,
            "leaf_nodes": self._leaf_nodes,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        embedding_fn: EmbeddingFunction | None = None,
        summarizer: SummarizerProtocol | None = None,
    ) -> "SummaryTree":
        """Deserialize tree from dictionary."""
        tree = cls(
            max_children=data.get("max_children", 5),
            embedding_fn=embedding_fn,
            summarizer=summarizer,
        )

        for nid, node_data in data.get("nodes", {}).items():
            tree.nodes[nid] = SummaryTreeNode.from_dict(node_data)

        tree.root_id = data.get("root_id")
        tree._level_indices = data.get("level_indices", {})
        tree._leaf_nodes = data.get("leaf_nodes", [])

        return tree

    def save(self, path: str) -> bool:
        """Save tree to file."""
        import json
        from pathlib import Path

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self._logger.error(f"Failed to save tree: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load tree from file."""
        import json
        from pathlib import Path

        try:
            filepath = Path(path)
            if not filepath.exists():
                return False

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            loaded = SummaryTree.from_dict(
                data,
                embedding_fn=self.embedding_fn,
                summarizer=self.summarizer,
            )
            self.nodes = loaded.nodes
            self.root_id = loaded.root_id
            self._level_indices = loaded._level_indices
            self._leaf_nodes = loaded._leaf_nodes

            return True
        except Exception as e:
            self._logger.error(f"Failed to load tree: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get tree statistics."""
        return {
            "total_nodes": len(self.nodes),
            "leaf_nodes": len(self._leaf_nodes),
            "max_level": self.get_max_level(),
            "nodes_per_level": {level: len(nodes) for level, nodes in self._level_indices.items()},
            "root_id": self.root_id,
        }


class QueryLevel(Enum):
    """Query abstraction level for routing."""

    DETAIL = 0
    SPECIFIC = 1
    GENERAL = 2
    BROAD = 3


@dataclass
class RetrievalResult:
    """Result from tree retrieval."""

    node: SummaryTreeNode
    score: float
    level: int
    path: list[str]
    matched_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node.id,
            "score": self.score,
            "level": self.level,
            "path": self.path,
            "matched_text": self.matched_text[:200] + "..."
            if len(self.matched_text) > 200
            else self.matched_text,
            "summary": self.node.summary[:200] + "..."
            if len(self.node.summary) > 200
            else self.node.summary,
        }


class CollapsedTreeRetrieval:
    """
    Retrieval system that searches across tree layers.

    Implements "collapsed tree" retrieval where all nodes
    from all levels are considered as candidates, with
    query-level routing for optimization.

    Attributes:
        tree: SummaryTree to search
        default_k: Default number of results
        level_weights: Weights for different levels

    Usage:
        retrieval = CollapsedTreeRetrieval(tree)
        results = retrieval.retrieve("query about topic", k=5)
    """

    def __init__(
        self,
        tree: SummaryTree,
        default_k: int = 5,
        level_weights: dict[int, float] | None = None,
    ) -> None:
        self.tree = tree
        self.default_k = default_k
        self.level_weights = level_weights or {0: 1.0, 1: 0.9, 2: 0.8, 3: 0.7}
        self._logger = logger

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        expand_children: bool = True,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant nodes from the tree.

        Args:
            query: Query string
            k: Number of results (default: self.default_k)
            expand_children: Whether to expand results to children

        Returns:
            List of RetrievalResult objects

        Example:
            results = retrieval.retrieve("machine learning algorithms", k=5)
            for result in results:
                print(f"Level {result.level}: {result.matched_text[:100]}")
        """
        k = k or self.default_k

        query_level = self.determine_query_level(query)

        candidates = self._get_candidates(query_level)

        scored = self._score_candidates(query, candidates)

        top_results = scored[:k]

        results = []
        for node, score in top_results:
            path = self.tree.get_path_to_root(node.id)
            path_ids = [n.id for n in path]

            matched_text = node.text if node.is_leaf() else node.summary

            result = RetrievalResult(
                node=node,
                score=score,
                level=node.level,
                path=path_ids,
                matched_text=matched_text,
            )
            results.append(result)

        if expand_children:
            results = self._expand_results(results, k)

        return results

    def determine_query_level(self, query: str) -> QueryLevel:
        """
        Determine which tree level to search based on query.

        Uses heuristics to route queries:
        - Short, specific queries → lower levels (detail)
        - Long, general queries → higher levels (broad)
        - Question words affect routing

        Args:
            query: Query string

        Returns:
            QueryLevel enum value

        Example:
            level = retrieval.determine_query_level("What is the capital?")
            # Returns QueryLevel.GENERAL
        """
        words = query.split()
        word_count = len(words)

        specific_indicators = ["exactly", "specific", "detail", "quote", "line", "section"]
        general_indicators = ["overview", "summary", "main", "general", "all", "list"]
        broad_indicators = ["explain", "describe", "tell me about", "what is"]

        query_lower = query.lower()

        specific_count = sum(1 for ind in specific_indicators if ind in query_lower)
        general_count = sum(1 for ind in general_indicators if ind in query_lower)
        broad_count = sum(1 for ind in broad_indicators if ind in query_lower)

        if specific_count > general_count and specific_count > broad_count:
            return QueryLevel.DETAIL

        if broad_count > general_count:
            return QueryLevel.BROAD

        if general_count > 0:
            return QueryLevel.GENERAL

        if word_count <= 3:
            return QueryLevel.SPECIFIC
        elif word_count <= 7:
            return QueryLevel.GENERAL
        else:
            return QueryLevel.BROAD

    def search_layer(
        self,
        query: str,
        level: int,
        k: int = 5,
    ) -> list[tuple[SummaryTreeNode, float]]:
        """
        Search a specific layer of the tree.

        Args:
            query: Query string
            level: Level to search (0 = leaf)
            k: Number of results

        Returns:
            List of (node, similarity) tuples

        Example:
            results = retrieval.search_layer("query", level=0, k=5)
        """
        return self.tree.search_similar(query, k=k, level=level)

    def hybrid_retrieve(
        self,
        query: str,
        k: int = 5,
        levels: list[int] | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval across multiple levels.

        Combines results from multiple levels with weighted scoring.

        Args:
            query: Query string
            k: Number of results
            levels: Levels to search (None = all)

        Returns:
            List of RetrievalResult objects
        """
        levels = levels or list(range(self.tree.get_max_level() + 1))

        all_results: list[tuple[SummaryTreeNode, float]] = []

        for level in levels:
            level_results = self.search_layer(query, level, k=k)
            weight = self.level_weights.get(level, 0.7)
            weighted = [(node, score * weight) for node, score in level_results]
            all_results.extend(weighted)

        all_results.sort(key=lambda x: x[1], reverse=True)

        seen_ids = set()
        unique_results = []
        for node, score in all_results:
            if node.id not in seen_ids:
                seen_ids.add(node.id)
                unique_results.append((node, score))

        results = []
        for node, score in unique_results[:k]:
            path = self.tree.get_path_to_root(node.id)
            matched_text = node.text if node.is_leaf() else node.summary

            results.append(
                RetrievalResult(
                    node=node,
                    score=score,
                    level=node.level,
                    path=[n.id for n in path],
                    matched_text=matched_text,
                )
            )

        return results

    def contextual_retrieve(
        self,
        query: str,
        context_nodes: list[str],
        k: int = 5,
        context_weight: float = 0.3,
    ) -> list[RetrievalResult]:
        """
        Context-aware retrieval.

        Boosts scores for nodes near provided context nodes.

        Args:
            query: Query string
            context_nodes: Node IDs providing context
            k: Number of results
            context_weight: Weight for context proximity

        Returns:
            List of RetrievalResult objects
        """
        base_results = self.retrieve(query, k=k * 2, expand_children=False)

        context_set = set(context_nodes)
        for ancestor in context_nodes:
            node = self.tree.get_node(ancestor)
            if node:
                for child_id in node.children:
                    context_set.add(child_id)
                path = self.tree.get_path_to_root(ancestor)
                for path_node in path:
                    context_set.add(path_node.id)

        boosted = []
        for result in base_results:
            score = result.score
            if result.node.id in context_set:
                score += context_weight

            is_neighbor = False
            for ctx_id in context_nodes:
                ctx_node = self.tree.get_node(ctx_id)
                if ctx_node:
                    if result.node.parent == ctx_node.parent:
                        is_neighbor = True
                        break
            if is_neighbor:
                score += context_weight * 0.5

            boosted.append((result, score))

        boosted.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in boosted[:k]]

    def _get_candidates(self, query_level: QueryLevel) -> list[SummaryTreeNode]:
        """Get candidate nodes based on query level."""
        max_level = self.tree.get_max_level()

        level_mapping = {
            QueryLevel.DETAIL: [0],
            QueryLevel.SPECIFIC: [0, 1],
            QueryLevel.GENERAL: [1, 2],
            QueryLevel.BROAD: list(range(max(0, max_level - 1), max_level + 1)),
        }

        levels = level_mapping.get(query_level, [0])

        candidates = []
        for level in levels:
            candidates.extend(self.tree.get_nodes_at_level(level))

        return candidates

    def _score_candidates(
        self,
        query: str,
        candidates: list[SummaryTreeNode],
    ) -> list[tuple[SummaryTreeNode, float]]:
        """Score candidates against query."""
        query_embedding = self.tree._get_embedding(query)

        scored = []
        for node in candidates:
            if not node.embedding:
                continue

            similarity = self.tree._cosine_similarity(query_embedding, node.embedding)

            level_weight = self.level_weights.get(node.level, 0.7)
            importance_weight = 1.0 + (node.importance - 0.5) * 0.4

            final_score = similarity * level_weight * importance_weight

            scored.append((node, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _expand_results(
        self,
        results: list[RetrievalResult],
        k: int,
    ) -> list[RetrievalResult]:
        """Expand results to include relevant children."""
        expanded = []

        for result in results:
            expanded.append(result)

            if not result.node.is_leaf():
                children = self.tree.get_children(result.node.id)
                for child in children[:2]:
                    if child.id not in [r.node.id for r in expanded]:
                        child_result = RetrievalResult(
                            node=child,
                            score=result.score * 0.8,
                            level=child.level,
                            path=[n.id for n in self.tree.get_path_to_root(child.id)],
                            matched_text=child.text if child.is_leaf() else child.summary,
                        )
                        expanded.append(child_result)

        return expanded[:k]


def build_raptor_tree(
    documents: list[dict[str, Any]],
    max_children: int = 5,
    embedding_fn: EmbeddingFunction | None = None,
    summarizer: SummarizerProtocol | None = None,
) -> SummaryTree:
    """
    Build a RAPTOR tree from documents.

    Convenience function for quick tree building.

    Args:
        documents: List of document dicts with 'text' and optional 'metadata'
        max_children: Maximum children per internal node
        embedding_fn: Optional embedding function
        summarizer: Optional summarizer

    Returns:
        Built SummaryTree

    Example:
        docs = [{"text": "doc1"}, {"text": "doc2"}]
        tree = build_raptor_tree(docs)
    """
    tree = SummaryTree(
        max_children=max_children,
        embedding_fn=embedding_fn,
        summarizer=summarizer,
    )

    docs = [Document(text=d.get("text", ""), metadata=d.get("metadata", {})) for d in documents]

    tree.build_from_documents(docs)
    return tree
