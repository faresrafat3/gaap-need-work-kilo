"""
Tests for Layer 2 Evolution - Intelligent Tactical Planning
=============================================================

Tests:
- Layer2Config
- IntelligentTask schema
- Phase discovery and reassessment
- Semantic dependency resolution
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from gaap.layers.layer2_config import (
    Layer2Config,
    create_layer2_config,
)
from gaap.layers.task_schema import (
    IntelligentTask,
    Phase,
    TaskPhase,
    RiskLevel,
    RiskFactor,
    RiskType,
    SchemaDefinition,
    ToolRecommendation,
    ReassessmentResult,
)
from gaap.layers.phase_planner import (
    PhaseDiscoveryEngine,
    PhaseReassessor,
    PhaseExpander,
    PhaseDiscoveryContext,
    create_phase_planner,
)
from gaap.layers.semantic_dependencies import (
    SemanticDependencyEngine,
    Dependency,
    DependencyGraph,
    ResolutionContext,
    create_dependency_engine,
)
from gaap.layers.task_injector import (
    DynamicTaskInjector,
    InjectionContext,
    InjectionDecision,
    create_task_injector,
)
from gaap.layers.layer2_learner import (
    Layer2Learner,
    ExecutionEpisode,
    LearningPattern,
    create_layer2_learner,
)


class TestLayer2Config:
    """Tests for Layer2Config."""

    def test_default_config(self):
        config = Layer2Config()

        assert config.phase_discovery_mode == "auto"
        assert config.phase_reassessment_mode == "auto"
        assert config.dependency_timing == "auto"
        assert config.dependency_depth == "standard"
        assert config.injection_autonomy == "auto"
        assert config.learning_enabled is True

    def test_high_quality_preset(self):
        config = Layer2Config.high_quality()

        assert config.phase_discovery_mode == "deep"
        assert config.phase_reassessment_mode == "full"
        assert config.dependency_depth == "exhaustive"
        assert config.risk_analysis_mode == "paranoid"

    def test_fast_preset(self):
        config = Layer2Config.fast()

        assert config.phase_discovery_mode == "standard"
        assert config.phase_reassessment_mode == "risk_based"
        assert config.learning_enabled is False

    def test_balanced_preset(self):
        config = Layer2Config.balanced()

        assert config.phase_discovery_mode == "auto"
        assert config.dependency_depth == "deep"
        assert config.learning_enabled is True

    def test_config_validation(self):
        with pytest.raises(ValueError):
            Layer2Config(injection_risk_threshold=1.5)

        with pytest.raises(ValueError):
            Layer2Config(parallel_task_limit=0)

    def test_to_dict_and_from_dict(self):
        config = Layer2Config(
            phase_discovery_mode="deep",
            dependency_depth="exhaustive",
        )

        d = config.to_dict()
        restored = Layer2Config.from_dict(d)

        assert restored.phase_discovery_mode == "deep"
        assert restored.dependency_depth == "exhaustive"

    def test_factory_function(self):
        config = create_layer2_config("high_quality", parallel_task_limit=10)

        assert config.phase_discovery_mode == "deep"
        assert config.parallel_task_limit == 10


class TestIntelligentTask:
    """Tests for IntelligentTask."""

    def test_task_creation(self):
        task = IntelligentTask(
            id="task-1",
            name="Implement User API",
            description="Create user CRUD endpoints",
            category="api",
        )

        assert task.id == "task-1"
        assert task.status == "pending"
        assert task.expansion_status == "expanded"

    def test_task_with_schema(self):
        task = IntelligentTask(
            id="task-1",
            name="Test User Model",
            description="Write tests for user model",
            category="testing",
            input_schema={
                "user_model": SchemaDefinition(
                    name="user_model",
                    type="file",
                    description="Path to user model",
                )
            },
            output_schema={
                "test_file": SchemaDefinition(
                    name="test_file",
                    type="file",
                    description="Generated test file",
                ),
            },
        )

        assert "user_model" in task.input_schema
        assert "test_file" in task.output_schema

    def test_task_with_tools(self):
        task = IntelligentTask(
            id="task-1",
            name="Test API",
            description="Test API endpoints",
            category="testing",
            recommended_tools=[
                ToolRecommendation(
                    tool="pytest",
                    reason="Best for Python testing",
                    priority=1,
                ),
                ToolRecommendation(
                    tool="httpx",
                    reason="For API testing",
                    priority=2,
                ),
            ],
        )

        sorted_tools = task.get_sorted_tools()
        assert sorted_tools[0].tool == "pytest"
        assert sorted_tools[0].priority == 1

    def test_task_with_risks(self):
        task = IntelligentTask(
            id="task-1",
            name="Database Migration",
            description="Migrate user data",
            category="database",
            risk_factors=[
                RiskFactor(
                    type=RiskType.DATA_LOSS,
                    level=RiskLevel.HIGH,
                    description="Potential data loss during migration",
                    mitigation="Backup before migration",
                    probability=0.8,
                    impact=0.8,
                ),
            ],
            overall_risk_level=RiskLevel.HIGH,
        )

        assert task.is_high_risk()
        risk_score = task.get_risk_score()
        assert risk_score > 0.0

    def test_task_to_dict_and_from_dict(self):
        task = IntelligentTask(
            id="task-1",
            name="Test",
            description="Test task",
            category="testing",
            semantic_intent="Verify functionality",
        )

        d = task.to_dict()
        restored = IntelligentTask.from_dict(d)

        assert restored.id == "task-1"
        assert restored.semantic_intent == "Verify functionality"


class TestPhase:
    """Tests for Phase."""

    def test_phase_creation(self):
        phase = Phase(
            id="phase-1",
            name="Core Implementation",
            description="Build core features",
            order=1,
        )

        assert phase.id == "phase-1"
        assert phase.status == TaskPhase.PLACEHOLDER
        assert len(phase.tasks) == 0

    def test_phase_readiness(self):
        phase = Phase(
            id="phase-1",
            name="Test Phase",
            description="Test",
            order=1,
            status=TaskPhase.PLACEHOLDER,
        )

        assert phase.is_ready_to_expand()
        assert not phase.is_ready_to_execute()

        phase.status = TaskPhase.EXPANDED
        phase.tasks = [IntelligentTask(id="t1", name="T1", description="", category="setup")]

        assert phase.is_ready_to_execute()

    def test_phase_progress(self):
        phase = Phase(
            id="phase-1",
            name="Test Phase",
            description="Test",
            order=1,
            status=TaskPhase.EXECUTING,
            tasks=[
                IntelligentTask(
                    id="t1", name="T1", description="", category="setup", status="completed"
                ),
                IntelligentTask(
                    id="t2", name="T2", description="", category="setup", status="completed"
                ),
                IntelligentTask(
                    id="t3", name="T3", description="", category="setup", status="pending"
                ),
            ],
        )

        progress = phase.get_progress()
        assert progress == pytest.approx(2 / 3, rel=0.01)


class TestDependencyGraph:
    """Tests for DependencyGraph."""

    def test_graph_creation(self):
        graph = DependencyGraph()

        assert len(graph.tasks) == 0
        assert len(graph.dependencies) == 0

    def test_add_tasks_and_dependencies(self):
        graph = DependencyGraph()

        t1 = IntelligentTask(id="t1", name="T1", description="", category="setup")
        t2 = IntelligentTask(id="t2", name="T2", description="", category="api")

        graph.add_task(t1)
        graph.add_task(t2)

        dep = Dependency(
            from_task="t2",
            to_task="t1",
            dependency_type="hard",
            reason="T2 depends on T1",
        )
        graph.add_dependency(dep)

        assert len(graph.tasks) == 2
        assert len(graph.dependencies) == 1
        assert graph.get_dependencies("t2") == ["t1"]
        assert graph.get_dependents("t1") == ["t2"]

    def test_get_ready_tasks(self):
        graph = DependencyGraph()

        t1 = IntelligentTask(id="t1", name="T1", description="", category="setup")
        t2 = IntelligentTask(id="t2", name="T2", description="", category="api")
        t3 = IntelligentTask(id="t3", name="T3", description="", category="testing")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency(
            Dependency(from_task="t2", to_task="t1", dependency_type="hard", reason="")
        )
        graph.add_dependency(
            Dependency(from_task="t3", to_task="t2", dependency_type="hard", reason="")
        )

        ready = graph.get_ready_tasks(set())
        assert "t1" in ready

        ready = graph.get_ready_tasks({"t1"})
        assert "t2" in ready

        ready = graph.get_ready_tasks({"t1", "t2"})
        assert "t3" in ready

    def test_cycle_detection(self):
        graph = DependencyGraph()

        t1 = IntelligentTask(id="t1", name="T1", description="", category="setup")
        t2 = IntelligentTask(id="t2", name="T2", description="", category="api")
        t3 = IntelligentTask(id="t3", name="T3", description="", category="setup")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency(
            Dependency(from_task="t2", to_task="t1", dependency_type="hard", reason="")
        )
        graph.add_dependency(
            Dependency(from_task="t3", to_task="t2", dependency_type="hard", reason="")
        )
        graph.add_dependency(
            Dependency(from_task="t1", to_task="t3", dependency_type="hard", reason="")
        )

        cycles = graph.detect_cycles()
        assert len(cycles) >= 1

    def test_topological_sort(self):
        graph = DependencyGraph()

        t1 = IntelligentTask(id="t1", name="T1", description="", category="setup")
        t2 = IntelligentTask(id="t2", name="T2", description="", category="api")
        t3 = IntelligentTask(id="t3", name="T3", description="", category="testing")

        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.add_dependency(
            Dependency(from_task="t2", to_task="t1", dependency_type="hard", reason="")
        )
        graph.add_dependency(
            Dependency(from_task="t3", to_task="t2", dependency_type="hard", reason="")
        )

        order = graph.topological_sort()

        assert order.index("t1") < order.index("t2")
        assert order.index("t2") < order.index("t3")


class TestPhaseDiscoveryEngine:
    """Tests for PhaseDiscoveryEngine."""

    @pytest.fixture
    def discovery_context(self):
        from gaap.layers.layer1_strategic import ArchitectureSpec, ArchitectureParadigm

        spec = ArchitectureSpec(
            spec_id="test-spec",
            timestamp=datetime.now(),
            paradigm=ArchitectureParadigm.MODULAR_MONOLITH,
        )

        return PhaseDiscoveryContext(
            architecture_spec=spec,
            original_request="Build a user management system",
            complexity_score=0.5,
        )

    def test_fallback_discovery(self, discovery_context):
        engine = PhaseDiscoveryEngine()

        phases = engine._fallback_discover_phases(discovery_context)

        assert len(phases) >= 3
        assert all(isinstance(p, Phase) for p in phases)
        assert all(p.status == TaskPhase.PLACEHOLDER for p in phases)

    @pytest.mark.asyncio
    async def test_discover_phases_without_provider(self, discovery_context):
        engine = PhaseDiscoveryEngine()

        phases = await engine.discover_phases(discovery_context)

        assert len(phases) > 0
        assert engine._discovery_count == 1


class TestSemanticDependencyEngine:
    """Tests for SemanticDependencyEngine."""

    @pytest.fixture
    def resolution_context(self):
        tasks = [
            IntelligentTask(
                id="t1",
                name="Implement User Model",
                description="Create user model",
                category="database",
                semantic_scope=["models/user.py"],
            ),
            IntelligentTask(
                id="t2",
                name="Test User Model",
                description="Test user model",
                category="testing",
                semantic_intent="Test the user model implementation",
            ),
            IntelligentTask(
                id="t3",
                name="Create User API",
                description="Build user API endpoints",
                category="api",
                semantic_scope=["api/user.py"],
            ),
        ]

        return ResolutionContext(tasks=tasks)

    def test_tasks_related(self):
        engine = SemanticDependencyEngine()

        t1 = IntelligentTask(
            id="t1",
            name="User Model",
            description="Create user model",
            category="database",
        )
        t2 = IntelligentTask(
            id="t2",
            name="Test User Model",
            description="Test user model",
            category="testing",
        )
        t3 = IntelligentTask(
            id="t3",
            name="Product API",
            description="Create product API",
            category="api",
        )

        assert engine._tasks_related(t1, t2) is True
        assert engine._tasks_related(t1, t3) is False

    @pytest.mark.asyncio
    async def test_resolve_dependencies(self, resolution_context):
        engine = SemanticDependencyEngine()

        graph = await engine.resolve(resolution_context)

        assert len(graph.tasks) == 3
        assert engine._resolutions == 1


class TestReassessmentResult:
    """Tests for ReassessmentResult."""

    def test_reassessment_result_creation(self):
        result = ReassessmentResult(
            replan_needed=True,
            reasoning="High-risk phase requires re-evaluation",
            confidence=0.85,
            affected_files=["models/user.py", "api/user.py"],
        )

        assert result.replan_needed is True
        assert result.confidence == 0.85
        assert len(result.affected_files) == 2

    def test_reassessment_with_new_tasks(self):
        new_task = IntelligentTask(
            id="injected-1",
            name="Security Audit",
            description="Run security audit",
            category="security",
        )

        result = ReassessmentResult(
            replan_needed=True,
            reasoning="Security vulnerability found",
            confidence=0.85,
            new_tasks_to_inject=[new_task],
        )

        assert len(result.new_tasks_to_inject) == 1


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_phase_planner(self):
        discovery, reassessor, expander = create_phase_planner()

        assert isinstance(discovery, PhaseDiscoveryEngine)
        assert isinstance(reassessor, PhaseReassessor)
        assert isinstance(expander, PhaseExpander)

    def test_create_dependency_engine(self):
        engine = create_dependency_engine()

        assert isinstance(engine, SemanticDependencyEngine)

    def test_create_task_injector(self):
        injector = create_task_injector()

        assert isinstance(injector, DynamicTaskInjector)

    def test_create_layer2_learner(self):
        learner = create_layer2_learner()

        assert isinstance(learner, Layer2Learner)


class TestDynamicTaskInjector:
    """Tests for DynamicTaskInjector."""

    @pytest.fixture
    def injection_context(self):
        phase = Phase(
            id="phase-1",
            name="Test Phase",
            description="Test",
            order=1,
            status=TaskPhase.EXECUTING,
        )

        return InjectionContext(
            current_phase=phase,
            completed_phases=[],
            pending_tasks=[],
        )

    def test_injector_creation(self):
        injector = DynamicTaskInjector()

        assert injector._config.injection_autonomy == "auto"
        assert injector._injections == 0

    @pytest.mark.asyncio
    async def test_analyze_no_injection_needed(self, injection_context):
        injector = DynamicTaskInjector()

        decision = await injector.analyze_and_inject(injection_context)

        assert isinstance(decision, InjectionDecision)
        assert injector._injections == 1

    @pytest.mark.asyncio
    async def test_analyze_with_failures(self, injection_context):
        injector = DynamicTaskInjector()

        injection_context.failed_tasks = [{"task_id": "t1", "error": "test"}]

        decision = await injector.analyze_and_inject(injection_context)

        assert decision.should_inject is True
        assert len(decision.tasks_to_inject) > 0

    def test_injector_stats(self):
        injector = DynamicTaskInjector()

        stats = injector.get_stats()

        assert "total_injections" in stats
        assert "approved_injections" in stats
        assert "learned_patterns" in stats


class TestLayer2Learner:
    """Tests for Layer2Learner."""

    def test_learner_creation(self):
        learner = Layer2Learner()

        assert learner._learning_enabled is True
        assert len(learner._task_episodes) == 0

    def test_record_task_execution(self):
        learner = Layer2Learner()

        task = IntelligentTask(
            id="task-1",
            name="Implement API",
            description="Create API",
            category="api",
            estimated_duration_minutes=10,
        )

        learner.record_task_execution(
            task=task,
            actual_duration_minutes=15,
            status="completed",
        )

        assert len(learner._task_episodes) == 1
        episode = learner._task_episodes[0]
        assert episode.estimation_error == pytest.approx(0.5, rel=0.1)

    def test_get_duration_estimate(self):
        learner = Layer2Learner()

        for i in range(5):
            task = IntelligentTask(
                id=f"task-{i}",
                name="Test Task",
                description="Test",
                category="testing",
                estimated_duration_minutes=10,
            )
            learner.record_task_execution(
                task=task,
                actual_duration_minutes=12,
                status="completed",
            )

        estimate, confidence = learner.get_duration_estimate("testing", "testing")

        assert estimate == 12
        assert confidence > 0

    def test_apply_learning_to_task(self):
        learner = Layer2Learner()

        for i in range(5):
            task = IntelligentTask(
                id=f"task-{i}",
                name="Test Task",
                description="Test",
                category="testing",
                estimated_duration_minutes=10,
            )
            learner.record_task_execution(
                task=task,
                actual_duration_minutes=20,
                status="completed",
            )

        new_task = IntelligentTask(
            id="new-task",
            name="Test Task",
            description="Test",
            category="testing",
            estimated_duration_minutes=10,
        )

        improved_task = learner.apply_learning_to_task(new_task)

        assert improved_task.estimated_duration_minutes != 10

    def test_learner_stats(self):
        learner = Layer2Learner()

        task = IntelligentTask(
            id="task-1",
            name="Test",
            description="Test",
            category="setup",
            estimated_duration_minutes=5,
        )
        learner.record_task_execution(task, 6, "completed")

        stats = learner.get_stats()

        assert stats["task_episodes"] == 1
        assert stats["duration_patterns"] >= 0

    def test_save_and_load_episodes(self):
        learner = Layer2Learner()

        task = IntelligentTask(
            id="task-1",
            name="Test",
            description="Test",
            category="setup",
            estimated_duration_minutes=5,
        )
        learner.record_task_execution(task, 6, "completed")

        data = learner.save_episodes()

        assert "task_episodes" in data
        assert len(data["task_episodes"]) == 1

        new_learner = Layer2Learner()
        new_learner.load_episodes(data)

        assert len(new_learner._task_episodes) == 1


class TestExecutionEpisode:
    """Tests for ExecutionEpisode."""

    def test_episode_creation(self):
        episode = ExecutionEpisode(
            task_id="task-1",
            task_name="Test Task",
            category="api",
            phase_id="phase-1",
            estimated_duration_minutes=10,
            actual_duration_minutes=12,
        )

        assert episode.task_id == "task-1"
        assert episode.status == "completed"
        assert episode.retry_count == 0

    def test_episode_hash(self):
        episode = ExecutionEpisode(
            task_id="task-1",
            task_name="Test Task",
            category="api",
            phase_id=None,
            estimated_duration_minutes=10,
            actual_duration_minutes=10,
        )

        hash_id = episode.get_id_hash()

        assert len(hash_id) == 12
        assert isinstance(hash_id, str)


class TestLearningPattern:
    """Tests for LearningPattern."""

    def test_pattern_creation(self):
        pattern = LearningPattern(
            pattern_type="duration",
            pattern_key="api:general",
        )

        assert pattern.sample_count == 0
        assert pattern.confidence == 0.0

    def test_pattern_update(self):
        pattern = LearningPattern(
            pattern_type="duration",
            pattern_key="test",
        )

        pattern.update(10.0)
        assert pattern.sample_count == 1
        assert pattern.average_value == 10.0

        pattern.update(20.0)
        assert pattern.sample_count == 2
        assert pattern.average_value == 15.0

        pattern.update(15.0)
        assert pattern.confidence > 0
