# Memory Package
import logging
from typing import Any, Callable

logger = logging.getLogger("gaap.memory")

VECTOR_MEMORY_AVAILABLE = False
_get_vector_store_fn: Callable[[], Any] | None = None

try:
    from gaap.memory.vector_store import get_vector_store as _get_vector_store_fn

    VECTOR_MEMORY_AVAILABLE = True
except ImportError:
    logger.debug(
        "Vector memory (chromadb) not installed. Using in-memory fallback. Install with: pip install chromadb"
    )


class LessonStore:
    """Wrapper to store 'Lessons' specifically using VectorStore if available."""

    def __init__(self) -> None:
        self._store: Any = None
        if VECTOR_MEMORY_AVAILABLE and _get_vector_store_fn:
            self._store = _get_vector_store_fn()

    def store_lesson(
        self,
        lesson: str,
        context: str = "",
        category: str = "general",
        success: bool = True,
        task_type: str | None = None,
    ) -> str | None:
        """Store a lesson learned."""
        if self._store is None:
            return None

        metadata = {
            "context": context,
            "category": category,
            "success": success,
            "task_type": task_type or "general",
        }
        result: str | None = self._store.add(content=lesson, metadata=metadata)
        return result

    def search(self, query: str, n: int = 5) -> list[Any]:
        """Search lessons - compatibility method."""
        return self.retrieve_lessons(query, k=n)

    def retrieve_lessons(
        self, query: str, k: int = 5, filter_category: str | None = None
    ) -> list[Any]:
        """Retrieve relevant lessons."""
        if self._store is None:
            return []

        filter_meta = {"category": filter_category} if filter_category else None
        result: list[Any] = self._store.search(query=query, n_results=k, filter_meta=filter_meta)
        return result

    add_lesson = store_lesson


# --- New Memory Agents & Rerankers ---

from gaap.memory.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    LLMReranker,
    RerankResult,
)
from gaap.memory.rerankers.base import RerankRequest

from gaap.memory.agents import (
    RetrievalAgent,
    RetrievalContext,
    RetrievalResult as AgentRetrievalResult,
    SpecialistAgent,
    DomainDecision,
)

from gaap.memory.knowledge import (
    KnowledgeGraphBuilder,
    MemoryNode,
    MemoryEdge,
    RelationExtractor,
    RelationType,
)
from gaap.memory.knowledge.graph_builder import NodeType

from gaap.memory.evolution import (
    REAPEngine,
    REAPResult,
    ClarificationSystem,
    ClarificationRequest,
)

from gaap.memory.raptor import (
    SummaryTreeNode,
    SummaryTree,
    CollapsedTreeRetrieval,
    Document,
    NodeType as RaptorNodeType,
    QueryLevel,
    RetrievalResult as RaptorRetrievalResult,
    build_raptor_tree,
)

from gaap.memory.vector_backends import (
    VectorBackend,
    InMemoryBackend,
    SearchResult,
    VectorRecord,
    get_backend,
    get_available_backends,
)

from gaap.memory.summary_builder import (
    SummaryBuilder,
    SummaryResult,
    KeyConcept,
    HierarchicalSummarizer,
    create_summary_builder,
)

# Few-Shot Retriever (Medprompt-inspired)
from gaap.memory.fewshot_retriever import (
    FewShotRetriever,
    RetrievalResult,
    SuccessLevel,
    SuccessMetrics,
    TaskCategory,
    Trajectory,
    TrajectoryStep,
    create_fewshot_retriever,
)


class MemorySystem:
    """
    Main interface for unified memory system with agents.

    Combines all memory components into a unified system with:
    - Vector store for semantic search
    - Knowledge graph for relationships
    - Retrieval agent for intelligent retrieval
    """

    def __init__(
        self,
        storage_path: str = ".gaap/memory",
        enable_knowledge_graph: bool = True,
        enable_evolution: bool = True,
    ):
        self.storage_path = storage_path
        self._logger = logger

        # Initialize components
        self._vector_store = _get_vector_store_fn() if _get_vector_store_fn else None
        self._knowledge_graph = (
            KnowledgeGraphBuilder(storage_path=f"{storage_path}/knowledge")
            if enable_knowledge_graph
            else None
        )
        self._retrieval_agent: RetrievalAgent | None = None

    def get_retrieval_agent(
        self,
        llm_provider: Any = None,
    ) -> RetrievalAgent:
        """Get or create retrieval agent."""
        if self._retrieval_agent is None:
            self._retrieval_agent = RetrievalAgent(
                vector_store=self._vector_store,
                reranker=CrossEncoderReranker(),
                knowledge_graph=self._knowledge_graph,
                llm_provider=llm_provider,
            )
        return self._retrieval_agent

    def store(self, content: str, **metadata: Any) -> str:
        """Store content in memory."""
        if self._vector_store:
            result: str = self._vector_store.add(content=content, metadata=metadata)
            return result

        # Add to knowledge graph if enabled
        if self._knowledge_graph:
            self._knowledge_graph.add_node(
                content=content,
                node_type=NodeType.CONCEPT,
                domain=metadata.get("domain", "general"),
                metadata=metadata,
            )

        return ""

    def retrieve(self, query: str, k: int = 5, **kwargs: Any) -> list[Any]:
        """Retrieve relevant memories."""
        if self._vector_store:
            results: list[Any] = self._vector_store.search(query=query, n_results=k)
            return results

        return []

    async def smart_retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        llm_provider: Any = None,
    ) -> AgentRetrievalResult:
        """Intelligent retrieval using agents."""
        agent = self.get_retrieval_agent(llm_provider)
        ctx = RetrievalContext(**context) if context else RetrievalContext()
        return await agent.retrieve(query, ctx)


# --- Legacy Exports ---

from gaap.memory.hierarchical import (
    HierarchicalMemory,
    MemoryTier,
    WorkingMemory,
    EpisodicMemory,
    EpisodicMemoryStore,
    SemanticMemoryStore,
    SemanticRule,
    ProceduralMemoryStore,
)

try:
    from gaap.memory.dream_processor import DreamProcessor
except ImportError:
    DreamProcessor = None  # type: ignore

try:
    from gaap.memory.memorag import (
        MemoRAG,
        KnowledgeGraph as _KnowledgeGraph,
        RetrievalResult,
        get_memorag as _get_memorag,
    )
except ImportError:
    MemoRAG = None  # type: ignore
    _KnowledgeGraph = None  # type: ignore
    RetrievalResult = None  # type: ignore
    _get_memorag = None  # type: ignore


def _get_vector_store_class() -> type[Any] | None:
    """Lazy import of VectorStore class."""
    if VECTOR_MEMORY_AVAILABLE:
        from gaap.memory.vector_store import VectorStore as VS

        return VS
    return None


VectorStore = _get_vector_store_class()

__all__ = [
    "HierarchicalMemory",
    "MemoryTier",
    "WorkingMemory",
    "EpisodicMemory",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "SemanticRule",
    "ProceduralMemoryStore",
    "DreamProcessor",
    "VECTOR_MEMORY_AVAILABLE",
    "LessonStore",
    "VectorStore",
    "MemoRAG",
    "KnowledgeGraph",
    "RetrievalResult",
    "get_memorag",
    "MemorySystem",
    "BaseReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "RerankRequest",
    "RerankResult",
    "RetrievalAgent",
    "RetrievalContext",
    "AgentRetrievalResult",
    "SpecialistAgent",
    "DomainDecision",
    "KnowledgeGraphBuilder",
    "MemoryNode",
    "MemoryEdge",
    "RelationExtractor",
    "RelationType",
    "REAPEngine",
    "REAPResult",
    "ClarificationSystem",
    "ClarificationRequest",
    "SummaryTreeNode",
    "SummaryTree",
    "CollapsedTreeRetrieval",
    "Document",
    "RaptorNodeType",
    "QueryLevel",
    "RaptorRetrievalResult",
    "build_raptor_tree",
    "VectorBackend",
    "InMemoryBackend",
    "SearchResult",
    "VectorRecord",
    "get_backend",
    "get_available_backends",
    "SummaryBuilder",
    "SummaryResult",
    "KeyConcept",
    "HierarchicalSummarizer",
    "create_summary_builder",
    # Few-Shot Retriever
    "FewShotRetriever",
    "RetrievalResult",
    "SuccessLevel",
    "SuccessMetrics",
    "TaskCategory",
    "Trajectory",
    "TrajectoryStep",
    "create_fewshot_retriever",
]


def get_memorag() -> Any:
    """Get MemoRAG instance."""
    if _get_memorag is not None:
        return _get_memorag()
    return None


# Type alias for backward compatibility
KnowledgeGraph = _KnowledgeGraph
