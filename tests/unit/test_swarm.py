"""
Unit Tests for GAAP Swarm Intelligence Module (v2.0)
Tests: ReputationStore, TaskAuctioneer, FractalAgent, Guild, GISP Protocol
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock

import pytest

from gaap.core.types import Task, TaskType, TaskPriority, TaskComplexity
from gaap.swarm.reputation import (
    ReputationStore,
    ReputationEntry,
    DomainExpertise,
    ReputationScore,
)
from gaap.swarm.gisp_protocol import (
    TaskAuction,
    TaskBid,
    TaskAward,
    TaskDomain,
    TaskPriority as GISPPriority,
    MessageType,
    ConsensusVote,
    GuildForm,
)
from gaap.swarm.auction import (
    TaskAuctioneer,
    AuctionConfig,
    AuctionState,
    AuctionResult,
)
from gaap.swarm.fractal import (
    FractalAgent,
    FractalState,
    FractalCapability,
    TaskEstimate,
)
from gaap.swarm.guild import (
    Guild,
    GuildState,
    GuildMembership,
)
from gaap.swarm.orchestrator import (
    SwarmOrchestrator,
    SwarmConfig,
    SwarmMetrics,
    OrchestratorState,
)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def reputation_store(temp_dir):
    return ReputationStore(storage_path=str(Path(temp_dir) / "reputation.json"))


@pytest.fixture
def auctioneer(reputation_store):
    return TaskAuctioneer(
        reputation_store=reputation_store,
        config=AuctionConfig(default_timeout_seconds=5.0),
    )


@pytest.fixture
def mock_provider():
    provider = Mock()
    provider.default_model = "test-model"
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Test output"
    provider.chat_completion = AsyncMock(return_value=response)
    return provider


@pytest.fixture
def mock_memory():
    memory = Mock()
    memory.retrieve = Mock(return_value=[])
    memory.record_episode = AsyncMock()
    return memory


@pytest.fixture
def sample_fractal(reputation_store, mock_provider, mock_memory):
    for _ in range(5):
        reputation_store.record_success("coder_01", "python")

    return FractalAgent(
        fractal_id="coder_01",
        domains=["python", "testing"],
        provider=mock_provider,
        memory=mock_memory,
        reputation_store=reputation_store,
    )


class TestReputationStore:
    """Tests for ReputationStore"""

    def test_record_success(self, reputation_store):
        reputation_store.record_success("coder_01", "python")

        score = reputation_store.get_domain_reputation("coder_01", "python")
        assert score > 0.5

    def test_record_failure(self, reputation_store):
        reputation_store.record_success("coder_01", "python")
        score_before = reputation_store.get_domain_reputation("coder_01", "python")

        reputation_store.record_failure("coder_01", "python", predicted=False)
        score_after = reputation_store.get_domain_reputation("coder_01", "python")

        assert score_after < score_before

    def test_predicted_failure_reduced_penalty(self, reputation_store):
        reputation_store.record_success("coder_01", "python")
        reputation_store.record_success("coder_02", "python")

        reputation_store.record_failure("coder_01", "python", predicted=True)
        reputation_store.record_failure("coder_02", "python", predicted=False)

        score_01 = reputation_store.get_domain_reputation("coder_01", "python")
        score_02 = reputation_store.get_domain_reputation("coder_02", "python")

        assert score_01 > score_02

    def test_get_top_fractals(self, reputation_store):
        for _ in range(5):
            reputation_store.record_success("coder_01", "python")
        for _ in range(2):
            reputation_store.record_success("coder_02", "python")

        top = reputation_store.get_top_fractals("python", limit=2)

        assert len(top) == 2
        assert top[0][0] == "coder_01"

    def test_persistence(self, reputation_store, temp_dir):
        reputation_store.record_success("coder_01", "python")
        reputation_store.save()

        new_store = ReputationStore(storage_path=str(Path(temp_dir) / "reputation.json"))

        score = new_store.get_domain_reputation("coder_01", "python")
        assert score > 0.5


class TestGISPProtocol:
    """Tests for GISP Protocol"""

    def test_task_auction_creation(self):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Implement authentication",
            domain=TaskDomain.SECURITY,
            complexity=7,
            priority=GISPPriority.HIGH,
        )

        assert auction.task_id == "task_123"
        assert auction.domain == TaskDomain.SECURITY

    def test_task_bid_utility(self):
        bid = TaskBid(
            task_id="task_123",
            bidder_id="coder_01",
            estimated_success_rate=0.9,
            estimated_cost_tokens=100,
            estimated_time_seconds=60,
            confidence_in_estimate=0.8,
            current_load=0.2,
        )

        utility = bid.compute_utility_score(reputation=0.85)

        assert 0 < utility < 1
        assert bid.utility_score == utility

    def test_message_serialization(self):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Test task",
        )

        data = auction.to_dict()

        assert data["task_id"] == "task_123"
        assert data["message_type"] == "TASK_AUCTION"

    def test_consensus_vote(self):
        vote = ConsensusVote(
            proposal_id="prop_001",
            proposal_type="SOP",
            voter_id="coder_01",
            vote="APPROVE",
            confidence=0.9,
        )

        assert vote.vote == "APPROVE"
        assert vote.confidence == 0.9


class TestTaskAuctioneer:
    """Tests for TaskAuctioneer"""

    @pytest.mark.asyncio
    async def test_start_auction(self, auctioneer):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Test task",
            domain=TaskDomain.PYTHON,
            timeout_seconds=5,
        )

        auction_id = await auctioneer.start_auction(auction, auto_close=False)

        assert auction_id is not None
        assert auctioneer.get_auction_status(auction_id) == AuctionState.OPEN

    @pytest.mark.asyncio
    async def test_receive_bid(self, auctioneer):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Test task",
            domain=TaskDomain.PYTHON,
            timeout_seconds=5,
        )

        await auctioneer.start_auction(auction, auto_close=False)

        auctioneer._reputation.record_success("coder_01", "python")

        bid = TaskBid(
            task_id="task_123",
            bidder_id="coder_01",
            estimated_success_rate=0.9,
            confidence_in_estimate=0.8,
        )
        bid.compute_utility_score(
            reputation=auctioneer._reputation.get_domain_reputation("coder_01", "python")
        )

        accepted = await auctioneer.receive_bid(bid)

        assert accepted is True

    @pytest.mark.asyncio
    async def test_close_auction(self, auctioneer):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Test task",
            domain=TaskDomain.PYTHON,
            timeout_seconds=5,
        )

        await auctioneer.start_auction(auction, auto_close=False)

        for _ in range(5):
            auctioneer._reputation.record_success("coder_01", "python")

        bid = TaskBid(
            task_id="task_123",
            bidder_id="coder_01",
            estimated_success_rate=0.9,
            confidence_in_estimate=0.9,
        )
        bid.compute_utility_score(
            reputation=auctioneer._reputation.get_domain_reputation("coder_01", "python")
        )

        await auctioneer.receive_bid(bid)

        result = await auctioneer.close_auction(auction.message_id)

        assert result.state == AuctionState.COMPLETED
        assert result.winner_id == "coder_01"


class TestFractalAgent:
    """Tests for FractalAgent"""

    def test_initial_state(self, sample_fractal):
        assert sample_fractal.state == FractalState.IDLE
        assert sample_fractal.current_load == 0.0

    def test_estimate_task(self, sample_fractal):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Write a Python function",
            domain=TaskDomain.PYTHON,
            complexity=5,
        )

        estimate = sample_fractal.estimate_task(auction)

        assert estimate.can_execute is True
        assert 0 <= estimate.estimated_success <= 1

    def test_create_bid(self, sample_fractal):
        auction = TaskAuction(
            task_id="task_123",
            task_description="Write a Python function",
            domain=TaskDomain.PYTHON,
            complexity=5,
        )

        estimate = sample_fractal.estimate_task(auction)
        bid = sample_fractal.create_bid(auction, estimate)

        assert bid.bidder_id == "coder_01"
        assert bid.utility_score > 0

    @pytest.mark.asyncio
    async def test_execute_task(self, sample_fractal):
        task = Mock()
        task.id = "task_123"
        task.description = "Write a function"

        award = TaskAward(
            task_id="task_123",
            winner_id="coder_01",
            utility_score=0.9,
        )

        result = await sample_fractal.execute_task(task, award)

        assert result is not None
        assert result.fractal_id == "coder_01"


class TestGuild:
    """Tests for Guild"""

    @pytest.fixture
    def guild(self, reputation_store):
        return Guild(
            guild_id="python_guild",
            domain="python",
            reputation_store=reputation_store,
        )

    def test_add_member(self, guild, reputation_store):
        for _ in range(10):
            reputation_store.record_success("coder_01", "python")

        membership = guild.add_member("coder_01", role="founder")

        assert membership is not None
        assert guild.member_count == 1

    def test_guild_activation(self, guild, reputation_store):
        for i in range(1, 5):
            fractal_id = f"coder_{i:02d}"
            for _ in range(10):
                reputation_store.record_success(fractal_id, "python")
            role = "founder" if i == 1 else "member"
            guild.add_member(fractal_id, role=role)

        assert guild.state == GuildState.ACTIVE
        assert guild.member_count >= 3

    def test_proposal_voting(self, guild, reputation_store):
        for i in range(1, 4):
            fractal_id = f"coder_{i:02d}"
            for _ in range(10):
                reputation_store.record_success(fractal_id, "python")
            guild.add_member(fractal_id, role="founder" if i == 1 else "member")

        proposal = guild.create_proposal(
            proposal_type="SOP",
            content="Always use type hints",
            proposer_id="coder_01",
        )

        assert proposal is not None

        vote1 = ConsensusVote(
            proposal_id=proposal.proposal_id,
            voter_id="coder_02",
            vote="APPROVE",
            confidence=0.9,
        )
        vote2 = ConsensusVote(
            proposal_id=proposal.proposal_id,
            voter_id="coder_03",
            vote="APPROVE",
            confidence=0.8,
        )

        guild.vote(proposal.proposal_id, vote1)
        guild.vote(proposal.proposal_id, vote2)

        assert len(guild.get_sops()) > 0


class TestSwarmOrchestrator:
    """Tests for SwarmOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        return SwarmOrchestrator(
            config=SwarmConfig(
                default_auction_timeout=2.0,
                guild_formation_threshold=3,
            )
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, orchestrator):
        await orchestrator.start()
        assert orchestrator.state == OrchestratorState.RUNNING

        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.PAUSED

    def test_register_fractal(self, orchestrator, sample_fractal):
        success = orchestrator.register_fractal(sample_fractal)

        assert success is True
        assert len(orchestrator._fractals) == 1

    @pytest.mark.asyncio
    async def test_process_task(self, orchestrator, sample_fractal):
        orchestrator.register_fractal(sample_fractal)

        for _ in range(5):
            orchestrator._reputation.record_success("coder_01", "python")

        await orchestrator.start()

        task = Mock()
        task.id = "task_123"
        task.description = "Write a function"

        result = await orchestrator.process_task(
            task=task,
            domain="python",
        )

        await orchestrator.stop()

        assert orchestrator.metrics.total_tasks_processed >= 1


class TestIntegration:
    """Integration tests"""

    @pytest.fixture
    def full_setup(self, temp_dir):
        store = ReputationStore(storage_path=str(Path(temp_dir) / "reputation.json"))

        orchestrator = SwarmOrchestrator(
            reputation_store=store,
            config=SwarmConfig(
                default_auction_timeout=2.0,
                enable_guild_formation=True,
            ),
        )

        fractals = []
        for i in range(5):
            provider = Mock()
            provider.default_model = "test-model"

            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message.content = f"Output from fractal {i}"
            provider.chat_completion = AsyncMock(return_value=response)

            memory = Mock()
            memory.retrieve = Mock(return_value=[])
            memory.record_episode = AsyncMock()

            fractal = FractalAgent(
                fractal_id=f"coder_{i:02d}",
                domains=["python", "testing"],
                provider=provider,
                memory=memory,
                reputation_store=store,
            )

            for _ in range(i * 2 + 1):
                store.record_success(f"coder_{i:02d}", "python")

            fractals.append(fractal)

        return orchestrator, fractals, store

    @pytest.mark.asyncio
    async def test_full_workflow(self, full_setup):
        orchestrator, fractals, store = full_setup

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        task = Mock()
        task.id = "integration_task"
        task.description = "Implement feature X"

        result = await orchestrator.process_task(
            task=task,
            domain="python",
            complexity=5,
        )

        await orchestrator.stop()

        stats = orchestrator.get_stats()

        assert stats["metrics"]["total_tasks"] >= 1
        assert len(stats["fractals"]) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
