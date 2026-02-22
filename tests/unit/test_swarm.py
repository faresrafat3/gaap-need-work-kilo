"""
Unit Tests for GAAP Swarm Intelligence Module
Tests: ReputationStore, TaskAuction, Fractals, GISP Protocol
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from gaap.core.types import Task, TaskType, TaskPriority, TaskComplexity
from gaap.swarm import (
    ReputationStore,
    ReputationScore,
    FractalProfile,
    TaskBid,
    TaskDomain,
    TaskAuction,
    TaskBroadcast,
    AuctionResult,
    ConsensusOracle,
    Arbitrator,
    GISPMessage,
    GISPHeader,
    MessageType,
    create_reputation_store,
    create_task_auction,
    create_bid_message,
    create_auction_message,
)
from gaap.swarm.fractals import (
    BaseFractal,
    CoderFractal,
    CriticFractal,
    ResearcherFractal,
    FractalResult,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def reputation_store(temp_dir):
    """Create a fresh reputation store"""
    return ReputationStore(persist_path=str(Path(temp_dir) / "reputation"))


@pytest.fixture
def task_auction(reputation_store):
    """Create a task auction instance"""
    return TaskAuction(reputation_store=reputation_store)


@pytest.fixture
def sample_task():
    """Create a sample task"""
    return Task(
        id="test_task_001",
        description="Write a Python function to calculate fibonacci",
        type=TaskType.CODE_GENERATION,
        priority=TaskPriority.NORMAL,
        complexity=TaskComplexity.SIMPLE,
    )


class TestReputationScore:
    """Tests for ReputationScore"""

    def test_initial_score(self):
        """Test initial reputation score"""
        score = ReputationScore(domain=TaskDomain.PYTHON)
        assert score.score == 50.0
        assert score.total_tasks == 0
        assert score.success_rate == 0.0

    def test_update_success(self):
        """Test updating with success"""
        score = ReputationScore(domain=TaskDomain.PYTHON)
        score.update(success=True)
        assert score.total_tasks == 1
        assert score.successful_tasks == 1
        assert score.score == 51.0

    def test_update_failure(self):
        """Test updating with failure"""
        score = ReputationScore(domain=TaskDomain.PYTHON)
        score.update(success=False)
        assert score.total_tasks == 1
        assert score.failed_tasks == 1
        assert score.score == 48.0

    def test_self_predicted_failure_no_penalty(self):
        """Test self-predicted failure saves reputation"""
        score = ReputationScore(domain=TaskDomain.PYTHON)
        score.update(success=False, self_predicted=True)
        assert score.score == 50.0
        assert score.self_predicted_failures == 1

    def test_success_rate(self):
        """Test success rate calculation"""
        score = ReputationScore(domain=TaskDomain.PYTHON)
        score.update(success=True)
        score.update(success=True)
        score.update(success=False)
        assert score.success_rate == pytest.approx(2 / 3, rel=0.01)


class TestFractalProfile:
    """Tests for FractalProfile"""

    def test_profile_creation(self):
        """Test creating a fractal profile"""
        profile = FractalProfile(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        assert profile.fractal_id == "coder_001"
        assert profile.specialization == TaskDomain.PYTHON
        assert profile.guild is None

    def test_get_reputation_creates_if_missing(self):
        """Test getting reputation creates if missing"""
        profile = FractalProfile(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        rep = profile.get_reputation(TaskDomain.JAVASCRIPT)
        assert rep.domain == TaskDomain.JAVASCRIPT
        assert TaskDomain.JAVASCRIPT in profile.reputation_scores

    def test_utility_score_specialization_bonus(self):
        """Test utility score bonus for specialization"""
        profile = FractalProfile(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        utility_python = profile.get_utility_score(TaskDomain.PYTHON)
        utility_js = profile.get_utility_score(TaskDomain.JAVASCRIPT)
        assert utility_python > utility_js


class TestReputationStore:
    """Tests for ReputationStore"""

    def test_register_fractal(self, reputation_store):
        """Test registering a fractal"""
        profile = reputation_store.register_fractal(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        assert profile.fractal_id == "coder_001"
        assert reputation_store.get_profile("coder_001") is not None

    def test_record_task_result_success(self, reputation_store):
        """Test recording successful task"""
        reputation_store.register_fractal(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        reputation_store.record_task_result(
            fractal_id="coder_001",
            domain=TaskDomain.PYTHON,
            success=True,
        )
        profile = reputation_store.get_profile("coder_001")
        assert profile.total_tasks == 1
        rep = profile.get_reputation(TaskDomain.PYTHON)
        assert rep.successful_tasks == 1

    def test_create_bid(self, reputation_store):
        """Test creating a bid"""
        reputation_store.register_fractal(
            fractal_id="coder_001",
            name="TestCoder",
            specialization=TaskDomain.PYTHON,
        )
        bid = reputation_store.create_bid(
            fractal_id="coder_001",
            task_id="task_001",
            domain=TaskDomain.PYTHON,
        )
        assert bid is not None
        assert bid.task_id == "task_001"
        assert bid.bidder_id == "coder_001"

    def test_select_winner(self, reputation_store):
        """Test selecting auction winner"""
        reputation_store.register_fractal("f1", "Coder1", TaskDomain.PYTHON)
        reputation_store.register_fractal("f2", "Coder2", TaskDomain.JAVASCRIPT)
        reputation_store.register_fractal("f3", "Coder3", TaskDomain.PYTHON)

        reputation_store.record_task_result("f1", TaskDomain.PYTHON, True)
        reputation_store.record_task_result("f1", TaskDomain.PYTHON, True)
        reputation_store.record_task_result("f3", TaskDomain.PYTHON, False)

        bid1 = reputation_store.create_bid("f1", "task_001", TaskDomain.PYTHON)
        bid2 = reputation_store.create_bid("f2", "task_001", TaskDomain.PYTHON)
        bid3 = reputation_store.create_bid("f3", "task_001", TaskDomain.PYTHON)

        winner = reputation_store.select_winner([bid1, bid2, bid3])
        assert winner is not None
        assert winner.bidder_id == "f1"

    def test_guild_formation(self, reputation_store):
        """Test guild formation after sufficient tasks"""
        reputation_store.register_fractal(
            fractal_id="expert_coder",
            name="ExpertCoder",
            specialization=TaskDomain.PYTHON,
        )
        for _ in range(10):
            reputation_store.record_task_result("expert_coder", TaskDomain.PYTHON, True)

        profile = reputation_store.get_profile("expert_coder")
        assert profile.guild == "PYTHON_Guild"

    def test_get_stats(self, reputation_store):
        """Test getting statistics"""
        reputation_store.register_fractal("f1", "Coder1", TaskDomain.PYTHON)
        reputation_store.register_fractal("f2", "Coder2", TaskDomain.JAVASCRIPT)

        stats = reputation_store.get_stats()
        assert stats["total_fractals"] == 2


class TestTaskAuction:
    """Tests for TaskAuction"""

    def test_register_fractal(self, task_auction):
        """Test registering fractals"""
        coder = CoderFractal(fractal_id="coder_001")
        task_auction.register_fractal(coder)
        stats = task_auction.get_fractal_stats()
        assert stats["total_fractals"] == 1

    def test_broadcast_task(self, task_auction, sample_task):
        """Test broadcasting a task"""
        broadcast = asyncio.run(task_auction.broadcast_task(sample_task))
        assert broadcast.task_id == sample_task.id
        assert broadcast.domain == TaskDomain.PYTHON

    def test_collect_bids_no_fractals(self, task_auction, sample_task):
        """Test collecting bids with no registered fractals"""
        bids = asyncio.run(task_auction.collect_bids(sample_task))
        assert len(bids) == 0

    def test_collect_bids_with_fractals(self, task_auction, sample_task):
        """Test collecting bids with registered fractals"""
        task_auction.register_fractal(CoderFractal(fractal_id="coder_001"))
        task_auction.register_fractal(ResearcherFractal(fractal_id="researcher_001"))

        bids = asyncio.run(task_auction.collect_bids(sample_task))

        assert len(bids) >= 1
        for bid in bids:
            assert bid.utility > 0

    def test_run_auction(self, task_auction, sample_task):
        """Test running complete auction"""
        task_auction.register_fractal(CoderFractal(fractal_id="coder_001"))
        task_auction.register_fractal(CriticFractal(fractal_id="critic_001"))

        result = asyncio.run(task_auction.run_auction(sample_task))

        assert result.task_id == sample_task.id
        assert result.winner_id is not None
        assert len(result.all_bids) >= 1

    def test_execute_with_winner(self, task_auction, sample_task):
        """Test executing task with auction winner"""
        task_auction.register_fractal(CoderFractal(fractal_id="coder_001"))

        result = asyncio.run(task_auction.execute_with_winner(sample_task, {"test": True}))

        assert result is not None
        assert result.success is not None


class TestFractals:
    """Tests for Fractal agents"""

    def test_coder_fractal_creation(self):
        """Test creating coder fractal"""
        coder = CoderFractal(fractal_id="test_coder")
        assert coder.name == "Coder"
        assert coder.specialization == TaskDomain.PYTHON

    def test_coder_assess_capability(self):
        """Test coder capability assessment"""
        coder = CoderFractal()
        task = Task(
            description="Write a Python function",
            type=TaskType.CODE_GENERATION,
        )
        confidence = coder.assess_capability(task)
        assert 0.0 <= confidence <= 1.0

    def test_coder_execute(self):
        """Test coder execution"""
        coder = CoderFractal()
        task = Task(
            id="test_001",
            description="Write a simple hello world function in Python",
            type=TaskType.CODE_GENERATION,
        )
        result = asyncio.run(coder.execute(task, {}))
        assert result.success
        assert result.output is not None

    def test_critic_fractal_creation(self):
        """Test creating critic fractal"""
        critic = CriticFractal(fractal_id="test_critic")
        assert critic.name == "Critic"
        assert critic.specialization == TaskDomain.SECURITY

    def test_critic_analyze_code(self):
        """Test critic code analysis"""
        critic = CriticFractal()
        code = """
password = "secret123"
eval(user_input)
"""
        task = Task(
            id="review_001",
            description="Review this code",
            type=TaskType.CODE_REVIEW,
        )
        result = asyncio.run(critic.execute(task, {"code": code}))
        assert result.success
        issues = result.output.get("issues", {})
        assert len(issues.get("security", [])) >= 2

    def test_researcher_fractal_creation(self):
        """Test creating researcher fractal"""
        researcher = ResearcherFractal(fractal_id="test_researcher")
        assert researcher.name == "Researcher"
        assert researcher.specialization == TaskDomain.RESEARCH

    def test_researcher_execute(self):
        """Test researcher execution"""
        researcher = ResearcherFractal()
        task = Task(
            id="research_001",
            description="Research best practices for Python async",
            type=TaskType.RESEARCH,
        )
        result = asyncio.run(researcher.execute(task, {}))
        assert result.success
        assert "findings" in result.output

    def test_fractal_memory(self):
        """Test fractal local memory"""
        coder = CoderFractal()
        coder.remember({"task": "test", "success": True})
        similar = coder.recall_similar("test")
        assert len(similar) == 1

    def test_epistemic_doubt(self):
        """Test epistemic doubt mechanism"""
        coder = CoderFractal()
        assert coder.check_epistemic_doubt(0.3) is True
        assert coder.check_epistemic_doubt(0.7) is False


class TestGISPProtocol:
    """Tests for GISP Protocol v2.0"""

    def test_message_creation(self):
        """Test creating GISP message"""
        header = GISPHeader(
            sender_id="test_sender",
            recipient_id="test_recipient",
            trace_id="trace_001",
        )
        message = GISPMessage(
            header=header,
            msg_type=MessageType.TASK_BID,
            payload={"test": "data"},
        )
        assert message.msg_type == MessageType.TASK_BID
        assert message.header.protocol_version == "GISP/2.0"

    def test_create_bid_message(self):
        """Test creating bid message"""
        message = create_bid_message(
            task_id="task_001",
            bidder_id="coder_001",
            utility=85.5,
            success_rate=0.9,
            cost=0.05,
            reputation=75.0,
            rationale="Python expert",
            trace_id="trace_001",
        )
        assert message.msg_type == MessageType.TASK_BID
        assert message.payload["utility"] == 85.5

    def test_create_auction_message(self):
        """Test creating auction message"""
        message = create_auction_message(
            task_id="task_001",
            description="Write Python code",
            domain="PYTHON",
            requirements={"language": "python"},
            trace_id="trace_001",
        )
        assert message.msg_type == MessageType.TASK_AUCTION
        assert message.payload["domain"] == "PYTHON"


class TestConsensusOracle:
    """Tests for ConsensusOracle"""

    def test_cast_vote(self):
        """Test casting votes"""
        oracle = ConsensusOracle()
        message = GISPMessage(
            header=GISPHeader(
                sender_id="voter_001",
                recipient_id="oracle",
                trace_id="debate_001",
            ),
            msg_type=MessageType.CONSENSUS_VOTE,
            payload={"decision": "APPROVE", "rationale": "Looks good"},
        )
        oracle.cast_vote(message)
        assert "debate_001" in oracle.active_debates

    def test_evaluate_consensus_approved(self):
        """Test evaluating consensus - approved"""
        oracle = ConsensusOracle(quorum_threshold=0.66)

        for i in range(3):
            message = GISPMessage(
                header=GISPHeader(
                    sender_id=f"voter_{i}",
                    recipient_id="oracle",
                    trace_id="debate_001",
                ),
                msg_type=MessageType.CONSENSUS_VOTE,
                payload={"decision": "APPROVE"},
            )
            oracle.cast_vote(message)

        result = oracle.evaluate_consensus("debate_001")
        assert result["reached"] is True
        assert result["verdict"] == "APPROVED"

    def test_evaluate_consensus_rejected(self):
        """Test evaluating consensus - rejected"""
        oracle = ConsensusOracle(quorum_threshold=0.66)

        for i in range(3):
            message = GISPMessage(
                header=GISPHeader(
                    sender_id=f"voter_{i}",
                    recipient_id="oracle",
                    trace_id="debate_002",
                ),
                msg_type=MessageType.CONSENSUS_VOTE,
                payload={"decision": "REJECT"},
            )
            oracle.cast_vote(message)

        result = oracle.evaluate_consensus("debate_002")
        assert result["reached"] is True
        assert result["verdict"] == "REJECTED"

    def test_clear_debate(self):
        """Test clearing debate"""
        oracle = ConsensusOracle()
        message = GISPMessage(
            header=GISPHeader(
                sender_id="voter_001",
                recipient_id="oracle",
                trace_id="debate_003",
            ),
            msg_type=MessageType.CONSENSUS_VOTE,
            payload={"decision": "APPROVE"},
        )
        oracle.cast_vote(message)
        oracle.clear_debate("debate_003")
        assert "debate_003" not in oracle.active_debates


class TestTaskDomain:
    """Tests for TaskDomain enum"""

    def test_all_domains_exist(self):
        """Test all expected domains exist"""
        expected = {
            "PYTHON",
            "JAVASCRIPT",
            "SQL",
            "FRONTEND",
            "BACKEND",
            "SECURITY",
            "TESTING",
            "DOCUMENTATION",
            "RESEARCH",
            "GENERAL",
        }
        actual = {d.name for d in TaskDomain}
        assert expected == actual


class TestSwarmOrchestrator:
    """Tests for SwarmOrchestrator"""

    def test_create_orchestrator(self):
        """Test creating orchestrator"""
        from gaap.swarm import SwarmOrchestrator, SwarmConfig

        config = SwarmConfig(max_parallel_tasks=5)
        orchestrator = SwarmOrchestrator(config=config)

        assert orchestrator is not None
        assert len(orchestrator._fractals) >= 3

    def test_register_custom_fractal(self):
        """Test registering custom fractal"""
        from gaap.swarm import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        initial_count = len(orchestrator._fractals)

        custom_coder = CoderFractal(fractal_id="custom_coder_001")
        orchestrator.register_fractal(custom_coder)

        assert len(orchestrator._fractals) == initial_count + 1

    def test_execute_task_simple(self, sample_task):
        """Test simple task execution"""
        from gaap.swarm import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        result = asyncio.run(orchestrator.execute_task(sample_task))

        assert result.task_id == sample_task.id
        assert result.winner_id != ""
        assert result.all_bids >= 1

    def test_execute_parallel(self):
        """Test parallel task execution"""
        from gaap.swarm import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()

        tasks = [Task(id=f"task_{i}", description=f"Write Python function {i}") for i in range(3)]

        results = asyncio.run(orchestrator.execute_parallel(tasks))

        assert len(results) == 3
        for r in results:
            assert r.task_id.startswith("task_")

    def test_get_stats(self):
        """Test getting orchestrator stats"""
        from gaap.swarm import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        stats = orchestrator.get_stats()

        assert "total_fractals" in stats
        assert "total_tasks_executed" in stats
        assert stats["total_fractals"] >= 3

    def test_get_top_performers(self):
        """Test getting top performers"""
        from gaap.swarm import SwarmOrchestrator

        orchestrator = SwarmOrchestrator()
        top = orchestrator.get_top_performers(TaskDomain.PYTHON, limit=3)

        assert isinstance(top, list)
        assert len(top) <= 3


class TestGAAPEngineSwarmIntegration:
    """Tests for GAAPEngine swarm integration"""

    def test_engine_with_swarm(self):
        """Test creating engine with swarm enabled"""
        from gaap.gaap_engine import GAAPEngine

        engine = GAAPEngine(enable_swarm=True, enable_memory=False, enable_security=False)

        assert engine.swarm_orchestrator is not None

    def test_engine_without_swarm(self):
        """Test creating engine without swarm"""
        from gaap.gaap_engine import GAAPEngine

        engine = GAAPEngine(enable_swarm=False, enable_memory=False, enable_security=False)

        assert engine.swarm_orchestrator is None

    def test_engine_swarm_execute(self):
        """Test engine swarm execute method"""
        from gaap.gaap_engine import GAAPEngine

        engine = GAAPEngine(enable_swarm=True, enable_memory=False, enable_security=False)

        task = Task(
            id="engine_task_001",
            description="Write a Python hello world",
        )

        result = asyncio.run(engine.swarm_execute(task))

        assert "success" in result
        assert "winner" in result
        assert "bids_received" in result

    def test_engine_swarm_stats(self):
        """Test engine stats include swarm stats"""
        from gaap.gaap_engine import GAAPEngine

        engine = GAAPEngine(enable_swarm=True, enable_memory=False, enable_security=False)

        stats = engine.get_stats()

        assert "swarm_stats" in stats
