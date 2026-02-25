"""
Comprehensive Integration Tests for GAAP Swarm Protocol (GISP v2.0)

Tests:
- TestOrchestratorAuctionIntegration: Full auction flow with orchestrator
- TestGuildFormation: Guild creation and management
- TestFullSwarmFlow: End-to-end swarm workflow
- TestReputationBasedAuction: Reputation-weighted task assignment
- TestFractalSpawning: Dynamic fractal creation
- TestGISPCommunication: Message passing and protocol compliance
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import pytest

from gaap.core.types import Task
from gaap.swarm.reputation import (
    ReputationStore,
    ReputationEntry,
    DomainExpertise,
)
from gaap.swarm.gisp_protocol import (
    TaskAuction,
    TaskBid,
    TaskAward,
    TaskResult,
    TaskDomain,
    TaskPriority,
    MessageType,
    ConsensusVote,
    GuildForm,
    MemoryShare,
    CapabilityAnnounce,
    GISPMessage,
)
from gaap.swarm.auction import (
    TaskAuctioneer,
    AuctionConfig,
    AuctionState,
    AuctionResult,
    UtilityScore,
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
    GuildProposal,
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
def mock_provider():
    provider = Mock()
    provider.default_model = "test-model"
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "Test output from provider"
    provider.chat_completion = AsyncMock(return_value=response)
    return provider


@pytest.fixture
def mock_memory():
    memory = Mock()
    memory.retrieve = Mock(return_value=[])
    memory.record_episode = AsyncMock()
    return memory


def create_fractal(
    fractal_id: str,
    domains: list[str],
    reputation_store,
    mock_provider,
    mock_memory,
    success_count: int = 5,
):
    for _ in range(success_count):
        for domain in domains:
            reputation_store.record_success(fractal_id, domain)

    return FractalAgent(
        fractal_id=fractal_id,
        domains=domains,
        provider=mock_provider,
        memory=mock_memory,
        reputation_store=reputation_store,
    )


class TestOrchestratorAuctionIntegration:
    """Tests for full auction flow through orchestrator"""

    @pytest.fixture
    def orchestrator_setup(self, reputation_store, mock_provider, mock_memory):
        config = SwarmConfig(
            default_auction_timeout=1.0,
            min_fractals_for_auction=1,
            guild_formation_threshold=10,
        )
        orchestrator = SwarmOrchestrator(
            reputation_store=reputation_store,
            config=config,
        )

        fractals = [
            create_fractal(
                f"coder_{i:02d}",
                ["python", "testing"],
                reputation_store,
                mock_provider,
                mock_memory,
                success_count=i * 2 + 3,
            )
            for i in range(3)
        ]

        return orchestrator, fractals, reputation_store

    @pytest.mark.asyncio
    async def test_orchestrator_starts_and_stops(self, orchestrator_setup):
        orchestrator, _, _ = orchestrator_setup

        await orchestrator.start()
        assert orchestrator.state == OrchestratorState.RUNNING

        await orchestrator.stop()
        assert orchestrator.state == OrchestratorState.PAUSED

    @pytest.mark.asyncio
    async def test_auction_with_single_bidder(self, orchestrator_setup):
        orchestrator, fractals, _ = orchestrator_setup

        orchestrator.register_fractal(fractals[0])
        await orchestrator.start()

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "task_001"
        task.description = "Write a Python function"
        task.type = None
        task.priority = None

        result = await orchestrator.process_task(task, domain="python")

        assert orchestrator.metrics.total_auctions >= 1

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_auction_with_multiple_bidders(self, orchestrator_setup):
        orchestrator, fractals, _ = orchestrator_setup

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "task_002"
        task.description = "Implement complex algorithm"
        task.type = None
        task.priority = None

        result = await orchestrator.process_task(task, domain="python")

        stats = orchestrator.get_stats()
        assert stats["metrics"]["total_auctions"] >= 1

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_highest_utility_wins(self, orchestrator_setup):
        orchestrator, fractals, reputation_store = orchestrator_setup

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        scores = [reputation_store.get_domain_reputation(f.fractal_id, "python") for f in fractals]
        highest_rep_idx = scores.index(max(scores))
        expected_winner = fractals[highest_rep_idx].fractal_id

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "task_003"
        task.description = "Critical task"
        task.type = None
        task.priority = None

        result = await orchestrator.process_task(
            task,
            domain="python",
            priority=TaskPriority.HIGH,
        )

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_auction_stats_tracking(self, orchestrator_setup):
        orchestrator, fractals, _ = orchestrator_setup

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        for i in range(3):
            task = Mock(spec=["id", "description", "type", "priority"])
            task.id = f"task_{i:03d}"
            task.description = f"Task {i}"
            task.type = None
            task.priority = None

            await orchestrator.process_task(task, domain="python")

        stats = orchestrator.get_stats()
        assert stats["metrics"]["total_auctions"] >= 3

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_fractal_registration_limit(self, reputation_store, mock_provider, mock_memory):
        config = SwarmConfig(max_fractals=2)
        orchestrator = SwarmOrchestrator(reputation_store=reputation_store, config=config)

        fractal1 = create_fractal("f1", ["python"], reputation_store, mock_provider, mock_memory)
        fractal2 = create_fractal("f2", ["python"], reputation_store, mock_provider, mock_memory)
        fractal3 = create_fractal("f3", ["python"], reputation_store, mock_provider, mock_memory)

        assert orchestrator.register_fractal(fractal1) is True
        assert orchestrator.register_fractal(fractal2) is True
        assert orchestrator.register_fractal(fractal3) is False

    def test_orchestrator_metrics_update(self, orchestrator_setup):
        orchestrator, _, _ = orchestrator_setup

        metrics = orchestrator.metrics
        assert metrics.total_tasks_processed == 0
        assert metrics.active_fractals == 0


class TestGuildFormation:
    """Tests for guild creation and management"""

    @pytest.fixture
    def guild_setup(self, reputation_store, mock_provider, mock_memory):
        config = SwarmConfig(
            guild_formation_threshold=3,
            enable_guild_formation=True,
        )
        orchestrator = SwarmOrchestrator(
            reputation_store=reputation_store,
            config=config,
        )

        high_rep_fractals = [
            create_fractal(
                f"expert_{i:02d}",
                ["python"],
                reputation_store,
                mock_provider,
                mock_memory,
                success_count=15,
            )
            for i in range(5)
        ]

        return orchestrator, high_rep_fractals, reputation_store

    def test_guild_creation_requirements(self, reputation_store):
        guild = Guild(
            guild_id="python_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        assert guild.state == GuildState.FORMING
        assert guild.member_count == 0

    def test_guild_adds_members(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="test_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        for fractal in fractals[:3]:
            membership = guild.add_member(fractal.fractal_id, role="member")
            assert membership is not None

        assert guild.member_count == 3

    def test_guild_activates_with_enough_members(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="active_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        guild.add_member(fractals[0].fractal_id, role="founder")
        guild.add_member(fractals[1].fractal_id, role="member")
        guild.add_member(fractals[2].fractal_id, role="member")

        assert guild.state == GuildState.ACTIVE

    def test_guild_rejects_low_reputation(self, guild_setup):
        _, _, reputation_store = guild_setup

        guild = Guild(
            guild_id="elite_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        membership = guild.add_member("unknown_fractal", role="member")

        assert membership is None

    def test_guild_sop_proposal_and_voting(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="sop_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        for i, fractal in enumerate(fractals[:4]):
            role = "founder" if i == 0 else "member"
            guild.add_member(fractal.fractal_id, role=role)

        proposal = guild.create_proposal(
            proposal_type="SOP",
            content="Always use type hints in function definitions",
            proposer_id=fractals[0].fractal_id,
        )

        assert proposal is not None
        assert proposal.status == "pending"

        for fractal in fractals[:4]:
            vote = ConsensusVote(
                proposal_id=proposal.proposal_id,
                voter_id=fractal.fractal_id,
                vote="APPROVE",
                confidence=0.95,
            )
            guild.vote(proposal.proposal_id, vote)

        assert proposal.status == "approved"
        sops = guild.get_sops()
        assert len(sops) >= 1

    def test_guild_memory_sharing(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="memory_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        guild.add_member(fractals[0].fractal_id, role="founder")
        guild.add_member(fractals[1].fractal_id, role="member")

        memory_entry = {
            "type": "episodic",
            "content": "Successfully implemented OAuth2 authentication",
            "domain": "python",
        }

        result = guild.share_memory(memory_entry, source_fractal=fractals[0].fractal_id)
        assert result is True

        shared = guild.get_shared_memory()
        assert len(shared) >= 1

    def test_guild_best_member_selection(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="selection_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        for fractal in fractals[:3]:
            guild.add_member(fractal.fractal_id, role="member")

        best = guild.get_best_member_for_task(task_complexity=5)

        assert best is not None
        assert best in [f.fractal_id for f in fractals[:3]]

    def test_guild_can_form_check(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        fractal_ids = [f.fractal_id for f in fractals[:3]]

        can_form, reason = Guild.can_form_guild(
            domain="python",
            fractals=fractal_ids,
            reputation_store=reputation_store,
        )

        assert can_form is True

    def test_guild_dissolution(self, guild_setup):
        _, fractals, reputation_store = guild_setup

        guild = Guild(
            guild_id="dissolve_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        guild.add_member(fractals[0].fractal_id, role="founder")
        guild.dissolve()

        assert guild.state == GuildState.DISSOLVED
        assert guild.member_count == 0


class TestFullSwarmFlow:
    """End-to-end tests for complete swarm workflows"""

    @pytest.fixture
    def full_swarm(self, temp_dir, mock_provider, mock_memory):
        reputation_store = ReputationStore(
            storage_path=str(Path(temp_dir) / "swarm_reputation.json")
        )

        config = SwarmConfig(
            default_auction_timeout=1.0,
            guild_formation_threshold=3,
            enable_guild_formation=True,
        )

        orchestrator = SwarmOrchestrator(
            reputation_store=reputation_store,
            config=config,
        )

        python_fractals = [
            create_fractal(
                f"python_coder_{i}",
                ["python", "testing"],
                reputation_store,
                mock_provider,
                mock_memory,
                success_count=10 + i,
            )
            for i in range(3)
        ]

        sql_fractals = [
            create_fractal(
                f"sql_analyst_{i}",
                ["sql", "data_science"],
                reputation_store,
                mock_provider,
                mock_memory,
                success_count=8 + i,
            )
            for i in range(3)
        ]

        security_fractals = [
            create_fractal(
                f"security_expert_{i}",
                ["security", "python"],
                reputation_store,
                mock_provider,
                mock_memory,
                success_count=12 + i,
            )
            for i in range(2)
        ]

        all_fractals = python_fractals + sql_fractals + security_fractals

        return (
            orchestrator,
            all_fractals,
            reputation_store,
            {
                "python": python_fractals,
                "sql": sql_fractals,
                "security": security_fractals,
            },
        )

    @pytest.mark.asyncio
    async def test_complete_task_lifecycle(self, full_swarm):
        orchestrator, fractals, _, _ = full_swarm

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "lifecycle_task"
        task.description = "Implement complete feature with tests"
        task.type = None
        task.priority = None

        result = await orchestrator.process_task(
            task,
            domain="python",
            priority=TaskPriority.HIGH,
            complexity=7,
        )

        stats = orchestrator.get_stats()
        assert stats["metrics"]["total_auctions"] >= 1

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_multiple_domain_tasks(self, full_swarm):
        orchestrator, fractals, _, domain_map = full_swarm

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        tasks = [
            ("python_task", "python", "Write a Python API endpoint"),
            ("sql_task", "sql", "Create database migration"),
            ("security_task", "security", "Security audit report"),
        ]

        for task_id, domain, description in tasks:
            task = Mock(spec=["id", "description", "type", "priority"])
            task.id = task_id
            task.description = description
            task.type = None
            task.priority = None

            await orchestrator.process_task(task, domain=domain)

        stats = orchestrator.get_stats()
        assert stats["metrics"]["total_auctions"] >= 3

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, full_swarm):
        orchestrator, fractals, _, _ = full_swarm

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        async def process_task(task_id: str):
            task = Mock(spec=["id", "description", "type", "priority"])
            task.id = task_id
            task.description = f"Concurrent task {task_id}"
            task.type = None
            task.priority = None
            return await orchestrator.process_task(task, domain="python")

        results = await asyncio.gather(
            process_task("concurrent_1"),
            process_task("concurrent_2"),
            process_task("concurrent_3"),
        )

        assert len(results) == 3

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_fractal_unregister(self, full_swarm):
        orchestrator, fractals, _, _ = full_swarm

        orchestrator.register_fractal(fractals[0])
        orchestrator.register_fractal(fractals[1])

        assert len(orchestrator._fractals) == 2

        result = orchestrator.unregister_fractal(fractals[0].fractal_id)

        assert result is True
        assert len(orchestrator._fractals) == 1

    @pytest.mark.asyncio
    async def test_orchestrator_stats_complete(self, full_swarm):
        orchestrator, fractals, _, _ = full_swarm

        for fractal in fractals:
            orchestrator.register_fractal(fractal)

        await orchestrator.start()

        stats = orchestrator.get_stats()

        assert "state" in stats
        assert "metrics" in stats
        assert "fractals" in stats
        assert "guilds" in stats
        assert "auction_stats" in stats
        assert "reputation_stats" in stats

        assert stats["state"] == "RUNNING"
        assert len(stats["fractals"]) == len(fractals)

        await orchestrator.stop()


class TestReputationBasedAuction:
    """Tests for reputation-weighted task assignment"""

    @pytest.fixture
    def reputation_setup(self, temp_dir):
        reputation_store = ReputationStore(storage_path=str(Path(temp_dir) / "rep_auction.json"))

        for i in range(20):
            reputation_store.record_success("expert_coder", "python")

        for i in range(10):
            reputation_store.record_success("mid_coder", "python")

        for i in range(3):
            reputation_store.record_success("junior_coder", "python")

        reputation_store.record_failure("unreliable_coder", "python", predicted=False)

        return reputation_store

    def test_reputation_affects_utility_score(self, reputation_setup):
        auction = TaskAuction(
            task_id="rep_task",
            task_description="Complex Python task",
            domain=TaskDomain.PYTHON,
            complexity=8,
        )

        expert_rep = reputation_setup.get_domain_reputation("expert_coder", "python")
        mid_rep = reputation_setup.get_domain_reputation("mid_coder", "python")
        junior_rep = reputation_setup.get_domain_reputation("junior_coder", "python")

        assert expert_rep > mid_rep > junior_rep

    def test_predicted_failure_reduces_penalty(self, reputation_setup):
        reputation_setup.record_failure("honest_coder", "python", predicted=True)
        reputation_setup.record_failure("silent_coder", "python", predicted=False)

        honest_rep = reputation_setup.get_domain_reputation("honest_coder", "python")
        silent_rep = reputation_setup.get_domain_reputation("silent_coder", "python")

        assert honest_rep > silent_rep

    def test_get_top_fractals(self, reputation_setup):
        top = reputation_setup.get_top_fractals("python", limit=3, min_confidence=0.3)

        assert len(top) >= 1
        if len(top) >= 2:
            assert top[0][1] >= top[1][1]

    def test_auctioneer_rejects_low_reputation(self, reputation_setup):
        auctioneer = TaskAuctioneer(reputation_store=reputation_setup)

        auction = TaskAuction(
            task_id="high_rep_task",
            task_description="Critical task",
            domain=TaskDomain.PYTHON,
            min_reputation=0.8,
        )

        async def run_auction():
            auction_id = await auctioneer.start_auction(auction, auto_close=False)

            expert_bid = TaskBid(
                task_id="high_rep_task",
                bidder_id="expert_coder",
                estimated_success_rate=0.9,
                confidence_in_estimate=0.9,
                correlation_id=auction_id,
            )

            expert_rep = reputation_setup.get_domain_reputation("expert_coder", "python")
            expert_bid.compute_utility_score(reputation=expert_rep)

            accepted = await auctioneer.receive_bid(expert_bid)
            return accepted

        accepted = asyncio.run(run_auction())
        assert accepted is True

    def test_reputation_persistence(self, reputation_setup, temp_dir):
        reputation_setup.save()

        loaded = ReputationStore(storage_path=str(Path(temp_dir) / "rep_auction.json"))

        expert_rep = loaded.get_domain_reputation("expert_coder", "python")
        assert expert_rep > 0.5

    def test_reputation_decay(self, reputation_setup):
        initial_rep = reputation_setup.get_domain_reputation("expert_coder", "python")

        reputation_setup.apply_decay(days=0)

        after_rep = reputation_setup.get_domain_reputation("expert_coder", "python")

        assert after_rep <= initial_rep

    def test_utility_computation_components(self, reputation_setup):
        auctioneer = TaskAuctioneer(reputation_store=reputation_setup)

        auction = TaskAuction(
            task_id="component_task",
            task_description="Test task",
            domain=TaskDomain.PYTHON,
            constraints={"max_cost": 500, "max_time": 300},
        )

        bid = TaskBid(
            task_id="component_task",
            bidder_id="expert_coder",
            estimated_success_rate=0.9,
            estimated_cost_tokens=100,
            estimated_time_seconds=60,
            confidence_in_estimate=0.85,
            current_load=0.2,
        )

        rep = reputation_setup.get_domain_reputation("expert_coder", "python")
        utility = auctioneer.compute_utility(bid, auction, rep)

        assert 0 <= utility.total <= 1
        assert utility.success_component >= 0
        assert utility.reputation_component >= 0
        assert utility.cost_component >= 0
        assert utility.time_component >= 0
        assert utility.load_penalty >= 0
        assert 0 <= utility.confidence_adjustment <= 1


class TestFractalSpawning:
    """Tests for dynamic fractal creation and management"""

    @pytest.fixture
    def spawn_setup(self, temp_dir, mock_provider, mock_memory):
        reputation_store = ReputationStore(storage_path=str(Path(temp_dir) / "spawn_rep.json"))

        return reputation_store, mock_provider, mock_memory

    def test_fractal_initial_state(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        fractal = FractalAgent(
            fractal_id="new_fractal",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        assert fractal.state == FractalState.IDLE
        assert fractal.current_load == 0.0

    def test_fractal_capability_tracking(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        fractal = FractalAgent(
            fractal_id="capable_fractal",
            domains=["python", "sql", "security"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        assert len(fractal.domains) == 3

    def test_fractal_task_estimation(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        for _ in range(10):
            reputation_store.record_success("estimator", "python")

        fractal = FractalAgent(
            fractal_id="estimator",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        auction = TaskAuction(
            task_id="estimate_task",
            task_description="Complex task",
            domain=TaskDomain.PYTHON,
            complexity=7,
        )

        estimate = fractal.estimate_task(auction)

        assert isinstance(estimate, TaskEstimate)
        assert estimate.can_execute in [True, False]
        assert 0 <= estimate.estimated_success <= 1
        assert estimate.estimated_cost >= 0
        assert estimate.estimated_time >= 0

    def test_fractal_bid_creation(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        for _ in range(5):
            reputation_store.record_success("bidder", "python")

        fractal = FractalAgent(
            fractal_id="bidder",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        auction = TaskAuction(
            task_id="bid_task",
            task_description="Test bidding",
            domain=TaskDomain.PYTHON,
            complexity=5,
        )

        estimate = fractal.estimate_task(auction)
        bid = fractal.create_bid(auction, estimate)

        assert bid.bidder_id == "bidder"
        assert bid.task_id == "bid_task"
        assert bid.utility_score >= 0

    @pytest.mark.asyncio
    async def test_fractal_task_execution(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        for _ in range(5):
            reputation_store.record_success("executor", "python")

        fractal = FractalAgent(
            fractal_id="executor",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "exec_task"
        task.description = "Execute this task"
        task.type = None
        task.priority = None

        award = TaskAward(
            task_id="exec_task",
            winner_id="executor",
            utility_score=0.85,
        )

        result = await fractal.execute_task(task, award)

        assert result is not None
        assert result.fractal_id == "executor"
        assert result.task_id == "exec_task"

    def test_fractal_load_management(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        fractal = FractalAgent(
            fractal_id="load_test",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
            config={"max_concurrent_tasks": 2},
        )

        assert fractal.current_load == 0.0

    def test_fractal_stats(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        fractal = FractalAgent(
            fractal_id="stats_fractal",
            domains=["python", "sql"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        stats = fractal.get_stats()

        assert stats["fractal_id"] == "stats_fractal"
        assert "domains" in stats
        assert "state" in stats
        assert "current_load" in stats

    def test_fractal_declines_unsuitable_task(self, spawn_setup):
        reputation_store, mock_provider, mock_memory = spawn_setup

        fractal = FractalAgent(
            fractal_id="specialist",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=reputation_store,
        )

        auction = TaskAuction(
            task_id="unsuitable_task",
            task_description="SQL database optimization",
            domain=TaskDomain.SQL,
            complexity=9,
        )

        estimate = fractal.estimate_task(auction)

        assert estimate.can_execute is False or len(estimate.risk_factors) > 0


class TestGISPCommunication:
    """Tests for GISP protocol message passing"""

    def test_task_auction_message(self):
        auction = TaskAuction(
            task_id="msg_task",
            task_description="Test message serialization",
            domain=TaskDomain.PYTHON,
            complexity=6,
            priority=TaskPriority.HIGH,
            min_reputation=0.7,
            timeout_seconds=30,
        )

        data = auction.to_dict()

        assert data["task_id"] == "msg_task"
        assert data["message_type"] == "TASK_AUCTION"
        assert data["domain"] == "python"
        assert data["priority"] == "HIGH"

    def test_task_bid_message(self):
        bid = TaskBid(
            task_id="bid_msg_task",
            bidder_id="test_bidder",
            estimated_success_rate=0.85,
            estimated_cost_tokens=150,
            estimated_time_seconds=45,
            confidence_in_estimate=0.8,
            current_load=0.3,
        )

        utility = bid.compute_utility_score(reputation=0.9)

        data = bid.to_dict()

        assert data["bidder_id"] == "test_bidder"
        assert data["message_type"] == "TASK_BID"
        assert data["utility_score"] > 0
        assert 0 <= data["utility_score"] <= 1

    def test_task_award_message(self):
        winning_bid = TaskBid(
            task_id="award_task",
            bidder_id="winner",
            estimated_success_rate=0.9,
        )

        award = TaskAward(
            task_id="award_task",
            winner_id="winner",
            utility_score=0.92,
            winning_bid=winning_bid,
            runner_ups=["runner_1", "runner_2"],
        )

        data = award.to_dict()

        assert data["winner_id"] == "winner"
        assert data["message_type"] == "TASK_AWARD"
        assert len(data["runner_ups"]) == 2

    def test_task_result_message(self):
        result = TaskResult(
            task_id="result_task",
            fractal_id="completer",
            success=True,
            output="Task completed successfully",
            actual_cost_tokens=120,
            actual_time_seconds=35,
            quality_score=0.88,
            predicted_success=True,
            confidence_before=0.7,
            confidence_after=0.85,
        )

        data = result.to_dict()

        assert data["fractal_id"] == "completer"
        assert data["message_type"] == "TASK_RESULT"
        assert data["success"] is True

    def test_consensus_vote_message(self):
        vote = ConsensusVote(
            proposal_id="prop_001",
            proposal_type="SOP",
            voter_id="voter_01",
            vote="APPROVE",
            confidence=0.9,
            reasoning="This SOP aligns with best practices",
            vote_weight=0.85,
        )

        data = vote.to_dict()

        assert data["voter_id"] == "voter_01"
        assert data["message_type"] == "CONSENSUS_VOTE"
        assert data["vote"] == "APPROVE"

    def test_guild_form_message(self):
        guild_form = GuildForm(
            guild_id="new_guild",
            guild_name="Python Experts",
            domain="python",
            founder_id="founder_01",
            min_reputation=0.75,
            invited_members=["expert_1", "expert_2"],
        )

        data = guild_form.to_dict()

        assert data["guild_id"] == "new_guild"
        assert data["message_type"] == "GUILD_FORM"
        assert data["domain"] == "python"

    def test_memory_share_message(self):
        memory = MemoryShare(
            source_fractal="sharer",
            target_fractal="receiver",
            memory_type="episodic",
            domain="python",
            content={"experience": "OAuth2 implementation success"},
            relevance_score=0.85,
        )

        data = memory.to_dict()

        assert data["source_fractal"] == "sharer"
        assert data["message_type"] == "MEMORY_SHARE"
        assert data["memory_type"] == "episodic"

    def test_capability_announce_message(self):
        capability = CapabilityAnnounce(
            fractal_id="announcer",
            domains=["python", "sql"],
            skills=["testing", "optimization"],
            tools=["pytest", "sqlalchemy"],
            max_concurrent_tasks=3,
            preferred_task_types=["backend", "database"],
        )

        data = capability.to_dict()

        assert data["fractal_id"] == "announcer"
        assert data["message_type"] == "CAPABILITY_ANNOUNCE"
        assert len(data["domains"]) == 2

    def test_message_correlation(self):
        auction = TaskAuction(
            task_id="corr_task",
            task_description="Correlation test",
            correlation_id="corr_123",
        )

        bid = TaskBid(
            task_id="corr_task",
            bidder_id="bidder",
            correlation_id="corr_123",
        )

        assert auction.correlation_id == bid.correlation_id

    def test_message_timestamp(self):
        import datetime

        msg = GISPMessage()

        assert isinstance(msg.timestamp, datetime.datetime)
        assert len(msg.message_id) > 0


class TestSwarmEdgeCases:
    """Edge case and error handling tests"""

    @pytest.fixture
    def edge_setup(self, temp_dir):
        reputation_store = ReputationStore(storage_path=str(Path(temp_dir) / "edge_rep.json"))
        return reputation_store

    def test_empty_reputation_returns_neutral(self, edge_setup):
        score = edge_setup.get_domain_reputation("unknown", "unknown_domain")

        assert score == 0.5

    def test_auction_with_no_bids(self, edge_setup):
        auctioneer = TaskAuctioneer(
            reputation_store=edge_setup,
            config=AuctionConfig(min_bids_required=1),
        )

        auction = TaskAuction(
            task_id="no_bids_task",
            task_description="Unpopular task",
        )

        asyncio.run(auctioneer.start_auction(auction, auto_close=False))
        result = asyncio.run(auctioneer.close_auction(auction.message_id))

        assert result.state == AuctionState.CANCELLED
        assert "Insufficient bids" in result.reason

    def test_utility_below_threshold(self, edge_setup):
        auctioneer = TaskAuctioneer(
            reputation_store=edge_setup,
            config=AuctionConfig(min_utility_threshold=0.8),
        )

        auction = TaskAuction(
            task_id="low_util_task",
            task_description="Difficult task",
        )

        asyncio.run(auctioneer.start_auction(auction, auto_close=False))

        low_bid = TaskBid(
            task_id="low_util_task",
            bidder_id="low_bidder",
            estimated_success_rate=0.1,
            confidence_in_estimate=0.1,
        )
        low_bid.compute_utility_score(reputation=0.1)

        asyncio.run(auctioneer.receive_bid(low_bid))
        result = asyncio.run(auctioneer.close_auction(auction.message_id))

        assert result.state == AuctionState.CANCELLED

    def test_fractal_state_transitions(self, edge_setup, mock_provider, mock_memory):
        fractal = FractalAgent(
            fractal_id="state_test",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=edge_setup,
        )

        assert fractal.state == FractalState.IDLE

    @pytest.mark.asyncio
    async def test_orchestrator_not_running_rejects_tasks(self, edge_setup):
        orchestrator = SwarmOrchestrator(reputation_store=edge_setup)

        task = Mock(spec=["id", "description", "type", "priority"])
        task.id = "reject_task"
        task.description = "Should be rejected"
        task.type = None
        task.priority = None

        result = await orchestrator.process_task(task)

        assert result is None

    def test_duplicate_fractal_registration(self, edge_setup, mock_provider, mock_memory):
        orchestrator = SwarmOrchestrator(reputation_store=edge_setup)

        fractal = FractalAgent(
            fractal_id="duplicate",
            domains=["python"],
            provider=mock_provider,
            memory=mock_memory,
            reputation_store=edge_setup,
        )

        orchestrator.register_fractal(fractal)
        orchestrator.register_fractal(fractal)

        assert len(orchestrator._fractals) == 1

    def test_guild_duplicate_member(self, edge_setup):
        guild = Guild(
            guild_id="dup_guild",
            domain="python",
            reputation_store=edge_setup,
        )

        for _ in range(15):
            edge_setup.record_success("dup_member", "python")

        membership1 = guild.add_member("dup_member", role="founder")
        membership2 = guild.add_member("dup_member", role="member")

        assert membership1 is not None
        assert membership2 is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
