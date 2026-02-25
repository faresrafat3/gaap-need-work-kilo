"""
Tests for Memory Evolution Components
====================================

Tests for new memory agents, rerankers, and evolution engine.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.memory.rerankers import BaseReranker, RerankResult, RerankRequest, CrossEncoderReranker
from gaap.memory.agents import (
    RetrievalAgent,
    RetrievalContext,
    RetrievalResult,
    SpecialistAgent,
    DomainDecision,
)
from gaap.memory.agents.specialist_agent import ScopeType, Domain
from gaap.memory.knowledge import KnowledgeGraphBuilder, MemoryNode, MemoryEdge, RelationType
from gaap.memory.knowledge.graph_builder import NodeType
from gaap.memory.evolution import REAPEngine, REAPResult, ClarificationSystem, ClarificationRequest

pytest_plugins = ("pytest_asyncio",)


class TestRerankRequest:
    """Tests for RerankRequest dataclass"""

    def test_create_request(self):
        request = RerankRequest(
            query="test query",
            candidates=["a", "b", "c"],
            top_k=5,
        )
        assert request.query == "test query"
        assert len(request.candidates) == 3
        assert request.top_k == 5

    def test_request_defaults(self):
        request = RerankRequest(query="test", candidates=[])
        assert request.top_k == 5
        assert request.context == {}
        assert request.metadata_list == []


class TestRerankResult:
    """Tests for RerankResult dataclass"""

    def test_create_result(self):
        result = RerankResult(
            content="test content",
            score=0.95,
            original_score=0.8,
            rank=1,
        )
        assert result.content == "test content"
        assert result.score == 0.95
        assert result.rank == 1

    def test_result_to_dict(self):
        result = RerankResult(
            content="test content",
            score=0.9,
            original_score=0.7,
            rank=2,
            source="test",
        )
        d = result.to_dict()
        assert d["score"] == 0.9
        assert d["rank"] == 2
        assert "content" in d


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker"""

    @pytest.mark.asyncio
    async def test_rerank_basic(self):
        reranker = CrossEncoderReranker()
        request = RerankRequest(
            query="python error",
            candidates=["Python traceback error", "Java exception", "Python syntax error"],
            top_k=2,
        )
        results = await reranker.rerank(request)
        assert len(results) <= 2
        assert all(isinstance(r, RerankResult) for r in results)

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        request = RerankRequest(query="test", candidates=[], top_k=5)
        results = await reranker.rerank(request)
        assert results == []

    def test_reranker_stats(self):
        reranker = CrossEncoderReranker()
        stats = reranker.get_stats()
        assert "name" in stats
        assert "total_reranks" in stats


class TestRetrievalContext:
    """Tests for RetrievalContext"""

    def test_context_creation(self):
        ctx = RetrievalContext(
            task_type="code_generation",
            active_domain="python",
            conversation_history=[{"role": "user", "content": "msg1"}],
        )
        assert ctx.task_type == "code_generation"
        assert ctx.active_domain == "python"
        assert len(ctx.conversation_history) == 1

    def test_context_defaults(self):
        ctx = RetrievalContext()
        assert ctx.task_type is None
        assert ctx.active_domain is None
        assert ctx.conversation_history == []


class TestRetrievalAgent:
    """Tests for RetrievalAgent"""

    def test_agent_initialization(self):
        agent = RetrievalAgent()
        assert agent._vector_store is None
        assert agent._reranker is None

    def test_agent_with_reranker(self):
        reranker = CrossEncoderReranker()
        agent = RetrievalAgent(reranker=reranker)
        assert agent._reranker is reranker

    @pytest.mark.asyncio
    async def test_retrieve_empty(self):
        agent = RetrievalAgent()
        ctx = RetrievalContext()
        result = await agent.retrieve("test query", ctx)
        assert isinstance(result, RetrievalResult)


class TestSpecialistAgent:
    """Tests for SpecialistAgent"""

    def test_agent_initialization(self):
        agent = SpecialistAgent()
        assert len(SpecialistAgent.DOMAIN_PROFILES) > 0

    @pytest.mark.asyncio
    async def test_detect_domain_python(self):
        agent = SpecialistAgent()
        decision = await agent.determine_domain("fix the python import error in my flask app", {})
        assert decision.domain in ["python", "api", "general"]
        assert decision.confidence >= 0

    @pytest.mark.asyncio
    async def test_detect_domain_database(self):
        agent = SpecialistAgent()
        decision = await agent.determine_domain("postgres connection timeout in production", {})
        assert decision.domain in ["database", "devops", "general"]
        assert decision.confidence >= 0

    @pytest.mark.asyncio
    async def test_detect_domain_frontend(self):
        agent = SpecialistAgent()
        decision = await agent.determine_domain("react component not rendering correctly", {})
        assert decision.domain in ["frontend", "general"]
        assert decision.confidence >= 0

    @pytest.mark.asyncio
    async def test_detect_domain_devops(self):
        agent = SpecialistAgent()
        decision = await agent.determine_domain("configure nginx and docker for deployment", {})
        assert decision.domain in ["devops", "general"]


class TestDomainDecision:
    """Tests for DomainDecision"""

    def test_decision_creation(self):
        decision = DomainDecision(
            domain="python",
            confidence=0.85,
            scope=ScopeType.NARROW,
            needs_confirmation=False,
        )
        assert decision.domain == "python"
        assert decision.confidence == 0.85
        assert decision.scope == ScopeType.NARROW

    def test_decision_to_dict(self):
        decision = DomainDecision(
            domain="database",
            confidence=0.9,
            scope=ScopeType.MODERATE,
            reasoning="Found database keywords",
        )
        d = decision.to_dict()
        assert d["domain"] == "database"
        assert d["confidence"] == 0.9
        assert "scope" in d


class TestKnowledgeGraphBuilder:
    """Tests for KnowledgeGraphBuilder"""

    def test_builder_initialization(self):
        builder = KnowledgeGraphBuilder()
        assert len(builder._nodes) == 0
        assert len(builder._edges) == 0

    def test_add_node(self):
        builder = KnowledgeGraphBuilder()
        node = builder.add_node(
            content="test concept",
            node_type=NodeType.CONCEPT,
            domain="test",
        )
        assert node is not None
        assert node.id in builder._nodes

    def test_add_edge(self):
        builder = KnowledgeGraphBuilder()
        node1 = builder.add_node("node1", NodeType.CONCEPT, "test")
        node2 = builder.add_node("node2", NodeType.CONCEPT, "test")
        edge = builder.add_edge(node1.id, node2.id, RelationType.RELATED_TO, 0.8)
        assert edge is not None
        assert len(builder._edges) == 1

    def test_get_neighbors(self):
        builder = KnowledgeGraphBuilder()
        node1 = builder.add_node("python error", NodeType.ERROR, "python")
        node2 = builder.add_node("fix python error", NodeType.SOLUTION, "python")
        builder.add_edge(node1.id, node2.id, RelationType.FIXED, 0.9)

        neighbors = builder.get_neighbors(node1.id)
        assert isinstance(neighbors, set)


class TestMemoryNode:
    """Tests for MemoryNode"""

    def test_node_creation(self):
        node = MemoryNode(
            id="test-node",
            content="test content",
            node_type=NodeType.CONCEPT,
            domain="test",
        )
        assert node.id == "test-node"
        assert node.node_type == NodeType.CONCEPT


class TestRelationType:
    """Tests for RelationType enum"""

    def test_relation_types_exist(self):
        assert RelationType.CAUSED is not None
        assert RelationType.FIXED is not None
        assert RelationType.RELATED_TO is not None
        assert RelationType.IS_A is not None


class TestREAPEngine:
    """Tests for REAP Engine"""

    def test_engine_initialization(self):
        engine = REAPEngine()
        assert engine is not None

    @pytest.mark.asyncio
    async def test_run_cycle_empty(self):
        engine = REAPEngine()
        result = await engine.run_cycle([])
        assert isinstance(result, REAPResult)


class TestClarificationSystem:
    """Tests for ClarificationSystem"""

    def test_system_initialization(self):
        system = ClarificationSystem()
        assert system is not None

    @pytest.mark.asyncio
    async def test_clarify(self):
        system = ClarificationSystem()
        request = ClarificationRequest(
            query="fix the error",
            context={"task_type": "code"},
            ambiguity_score=0.8,
            detected_domain="python",
        )
        response = await system.clarify(request)
        assert response is not None
        assert response.question is not None


class TestClarificationRequest:
    """Tests for ClarificationRequest"""

    def test_request_creation(self):
        request = ClarificationRequest(
            query="test query",
            context={"key": "value"},
            ambiguity_score=0.7,
        )
        assert request.query == "test query"
        assert request.ambiguity_score == 0.7


class TestIntegration:
    """Integration tests"""

    @pytest.mark.asyncio
    async def test_full_retrieval_flow(self):
        agent = RetrievalAgent()
        ctx = RetrievalContext(task_type="debug")
        result = await agent.retrieve("fix python import error", ctx)
        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_specialist_then_retrieval(self):
        specialist = SpecialistAgent()
        decision = await specialist.determine_domain("fix django migration error", {})

        agent = RetrievalAgent()
        ctx = RetrievalContext(active_domain=decision.domain)
        result = await agent.retrieve("migration error", ctx)

        assert isinstance(result, RetrievalResult)

    @pytest.mark.asyncio
    async def test_knowledge_graph_building(self):
        builder = KnowledgeGraphBuilder()

        error_node = builder.add_node(
            content="ImportError: No module named 'xyz'",
            node_type=NodeType.ERROR,
            domain="python",
        )

        solution_node = builder.add_node(
            content="pip install xyz",
            node_type=NodeType.SOLUTION,
            domain="python",
        )

        edge = builder.add_edge(error_node.id, solution_node.id, RelationType.FIXED, 0.95)

        assert len(builder._nodes) == 2
        assert edge is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
