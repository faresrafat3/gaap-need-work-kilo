"""
Unit Tests for GAAP SOTA Research Hub Module (Spec 20)
Tests: Signature, Teleprompter, Artifacts, FewShotRetriever, ProfileEvolver, SOPManager
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from gaap.core.signatures import (
    FieldType,
    Signature,
    SignatureField,
    Teleprompter,
    Example,
    Module,
    ModuleRegistry,
    OptimizationResult,
)
from gaap.core.artifacts import (
    Artifact,
    ArtifactBuilder,
    ArtifactMetadata,
    ArtifactRegistry,
    ArtifactStatus,
    ArtifactType,
    ArtifactLink,
)
from gaap.memory.fewshot_retriever import (
    FewShotRetriever,
    SuccessLevel,
    SuccessMetrics,
    TaskCategory,
    Trajectory,
    TrajectoryStep,
    RetrievalResult,
)
from gaap.swarm.profile_evolver import (
    EvolutionStatus,
    EvolutionTrigger,
    PerformanceSnapshot,
    ProfileEvolution,
    ProfileEvolver,
)
from gaap.layers.sop_manager import (
    ArtifactValidationResult,
    QualityGate,
    QualityGateStatus,
    SOP,
    SOPManager,
    SOPStep,
    StepType,
)


class TestSignature:
    """Tests for Signature class"""

    def test_create(self):
        """Test creating a signature with input and output fields"""
        signature = Signature(
            name="code_review",
            description="Review code for issues",
            inputs=[
                SignatureField(
                    name="code",
                    field_type=FieldType.STRING,
                    description="Code to review",
                    required=True,
                ),
                SignatureField(
                    name="language",
                    field_type=FieldType.STRING,
                    description="Programming language",
                    required=False,
                ),
            ],
            outputs=[
                SignatureField(
                    name="issues",
                    field_type=FieldType.LIST,
                    description="List of issues found",
                    required=True,
                ),
                SignatureField(
                    name="score",
                    field_type=FieldType.FLOAT,
                    description="Quality score 0-10",
                    required=True,
                    constraints={"min_value": 0, "max_value": 10},
                ),
            ],
            instructions="Provide a thorough code review",
        )

        assert signature.name == "code_review"
        assert signature.description == "Review code for issues"
        assert len(signature.inputs) == 2
        assert len(signature.outputs) == 2
        assert signature.instructions == "Provide a thorough code review"

    def test_create_with_defaults(self):
        """Test creating a signature with default values"""
        signature = Signature(name="simple_task")

        assert signature.name == "simple_task"
        assert signature.description == ""
        assert signature.inputs == []
        assert signature.outputs == []
        assert signature.instructions == ""

    def test_create_from_dict(self):
        """Test creating a signature from dictionary"""
        data = {
            "name": "test_sig",
            "description": "Test description",
            "inputs": [
                {"name": "input1", "type": "STRING", "required": True},
                {"name": "input2", "type": "INTEGER", "required": False, "default": 0},
            ],
            "outputs": [
                {"name": "output1", "type": "BOOLEAN"},
            ],
            "instructions": "Test instructions",
        }

        signature = Signature.from_dict(data)

        assert signature.name == "test_sig"
        assert len(signature.inputs) == 2
        assert len(signature.outputs) == 1
        assert signature.inputs[0].field_type == FieldType.STRING
        assert signature.inputs[1].field_type == FieldType.INTEGER
        assert signature.inputs[1].required is False
        assert signature.inputs[1].default == 0

    def test_validate_input(self):
        """Test validating input data against signature"""
        signature = Signature(
            name="test",
            inputs=[
                SignatureField(name="required_str", field_type=FieldType.STRING, required=True),
                SignatureField(name="optional_int", field_type=FieldType.INTEGER, required=False),
                SignatureField(
                    name="constrained_list",
                    field_type=FieldType.LIST,
                    constraints={"min_length": 1, "max_length": 5},
                ),
            ],
        )

        valid_data = {
            "required_str": "hello",
            "constrained_list": [1, 2, 3],
        }
        is_valid, errors = signature.validate_input(valid_data)
        assert is_valid is True
        assert len(errors) == 0

        invalid_data = {
            "required_str": 123,
            "constrained_list": [],
        }
        is_valid, errors = signature.validate_input(invalid_data)
        assert is_valid is False
        assert len(errors) > 0

        missing_data = {"optional_int": 5}
        is_valid, errors = signature.validate_input(missing_data)
        assert is_valid is False
        assert any("required_str" in e for e in errors)

    def test_validate_output(self):
        """Test validating output data against signature"""
        signature = Signature(
            name="test",
            outputs=[
                SignatureField(name="result", field_type=FieldType.STRING, required=True),
                SignatureField(
                    name="confidence",
                    field_type=FieldType.FLOAT,
                    required=True,
                    constraints={"min_value": 0.0, "max_value": 1.0},
                ),
            ],
        )

        valid_output = {"result": "success", "confidence": 0.85}
        is_valid, errors = signature.validate_output(valid_output)
        assert is_valid is True

        invalid_output = {"result": "success", "confidence": 1.5}
        is_valid, errors = signature.validate_output(invalid_output)
        assert is_valid is False

        missing_output = {"result": "success"}
        is_valid, errors = signature.validate_output(missing_output)
        assert is_valid is False

    def test_to_prompt(self):
        """Test converting signature to prompt format"""
        signature = Signature(
            name="summarize",
            description="Summarize a document",
            inputs=[
                SignatureField(
                    name="document", field_type=FieldType.STRING, description="Text to summarize"
                ),
            ],
            outputs=[
                SignatureField(
                    name="summary", field_type=FieldType.STRING, description="Generated summary"
                ),
                SignatureField(
                    name="key_points", field_type=FieldType.LIST, description="Main points"
                ),
            ],
            instructions="Create a concise summary highlighting key points.",
        )

        prompt = signature.to_prompt()

        assert "Task: summarize" in prompt
        assert "Description: Summarize a document" in prompt
        assert "Inputs:" in prompt
        assert "document [string] (required)" in prompt
        assert "Outputs:" in prompt
        assert "summary [string] (required)" in prompt
        assert "Instructions: Create a concise summary" in prompt

    def test_get_input_schema(self):
        """Test getting JSON schema for inputs"""
        signature = Signature(
            name="test",
            inputs=[
                SignatureField(name="text", field_type=FieldType.STRING, description="Input text"),
                SignatureField(
                    name="count", field_type=FieldType.INTEGER, required=False, default=10
                ),
            ],
        )

        schema = signature.get_input_schema()

        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "count" in schema["properties"]
        assert schema["properties"]["text"]["type"] == "string"
        assert "text" in schema["required"]
        assert "count" not in schema["required"]
        assert schema["properties"]["count"]["default"] == 10

    def test_get_output_schema(self):
        """Test getting JSON schema for outputs"""
        signature = Signature(
            name="test",
            outputs=[
                SignatureField(
                    name="score",
                    field_type=FieldType.FLOAT,
                    constraints={"min_value": 0, "max_value": 100},
                ),
            ],
        )

        schema = signature.get_output_schema()

        assert schema["type"] == "object"
        assert schema["properties"]["score"]["type"] == "number"
        assert schema["properties"]["score"]["minimum"] == 0
        assert schema["properties"]["score"]["maximum"] == 100


class TestTeleprompter:
    """Tests for Teleprompter class"""

    @pytest.fixture
    def teleprompter(self):
        """Create a teleprompter instance"""
        return Teleprompter(max_examples=5, min_quality_score=0.7)

    @pytest.fixture
    def sample_signature(self):
        """Create a sample signature"""
        return Signature(
            name="test_task",
            inputs=[SignatureField(name="input", field_type=FieldType.STRING)],
            outputs=[SignatureField(name="output", field_type=FieldType.STRING)],
        )

    def test_optimize(self, teleprompter, sample_signature):
        """Test optimizing a prompt"""
        examples = [
            Example(
                inputs={"input": "hello"},
                outputs={"output": "HELLO"},
                quality_score=0.9,
            ),
            Example(
                inputs={"input": "world"},
                outputs={"output": "WORLD"},
                quality_score=0.85,
            ),
        ]

        result = teleprompter.optimize(sample_signature, examples, context="Convert to uppercase")

        assert isinstance(result, OptimizationResult)
        assert "Convert to uppercase" in result.optimized_prompt
        assert "Task: test_task" in result.optimized_prompt
        assert len(result.examples_used) == 2
        assert result.improvement_score > 0

    def test_optimize_without_examples(self, teleprompter, sample_signature):
        """Test optimizing without examples"""
        result = teleprompter.optimize(sample_signature)

        assert result is not None
        assert "Task: test_task" in result.optimized_prompt
        assert len(result.examples_used) == 0

    def test_get_best_examples(self, teleprompter, sample_signature):
        """Test getting best examples for a signature"""
        teleprompter.index_example(
            sample_signature,
            Example(inputs={"x": "1"}, outputs={"y": "2"}, quality_score=0.95),
        )
        teleprompter.index_example(
            sample_signature,
            Example(inputs={"x": "3"}, outputs={"y": "4"}, quality_score=0.75),
        )
        teleprompter.index_example(
            sample_signature,
            Example(inputs={"x": "5"}, outputs={"y": "6"}, quality_score=0.65),
        )

        best = teleprompter.get_best_examples(sample_signature, k=2)

        assert len(best) == 2
        assert best[0].quality_score == 0.95
        assert best[1].quality_score == 0.75

    def test_get_best_examples_with_filter(self, teleprompter, sample_signature):
        """Test getting best examples with filter function"""
        teleprompter.index_example(
            sample_signature,
            Example(inputs={"x": "1"}, outputs={"y": "2"}, quality_score=0.95, source="valid"),
        )
        teleprompter.index_example(
            sample_signature,
            Example(inputs={"x": "3"}, outputs={"y": "4"}, quality_score=0.85, source="invalid"),
        )

        best = teleprompter.get_best_examples(
            sample_signature,
            k=2,
            filter_fn=lambda e: e.source == "valid",
        )

        assert len(best) == 1
        assert best[0].source == "valid"

    def test_index_example(self, teleprompter, sample_signature):
        """Test indexing examples"""
        example = Example(
            inputs={"input": "test"},
            outputs={"output": "TEST"},
            quality_score=0.8,
        )

        teleprompter.index_example(sample_signature, example)

        examples = teleprompter._example_cache.get("test_task", [])
        assert len(examples) == 1
        assert examples[0] == example

    def test_get_optimization_stats(self, teleprompter, sample_signature):
        """Test getting optimization statistics"""
        teleprompter.optimize(sample_signature)
        teleprompter.optimize(sample_signature)

        stats = teleprompter.get_optimization_stats()

        assert stats["total_optimizations"] == 2
        assert "avg_improvement" in stats
        assert "test_task" in stats["signatures_optimized"]

    def test_example_limit(self, teleprompter, sample_signature):
        """Test that example cache is limited"""
        for i in range(150):
            teleprompter.index_example(
                sample_signature,
                Example(inputs={"x": str(i)}, outputs={"y": str(i)}, quality_score=0.8),
            )

        examples = teleprompter._example_cache.get("test_task", [])
        assert len(examples) == 100


class TestArtifacts:
    """Tests for Artifact system"""

    def test_create(self):
        """Test creating an artifact"""
        artifact = Artifact(
            type=ArtifactType.CODE,
            name="main.py",
            content="def hello():\n    print('Hello')",
            created_by="agent_01",
            metadata=ArtifactMetadata(
                version="1.0.0",
                tags=["python", "core"],
                language="python",
            ),
        )

        assert artifact.type == ArtifactType.CODE
        assert artifact.name == "main.py"
        assert artifact.status == ArtifactStatus.DRAFT
        assert artifact.created_by == "agent_01"
        assert "python" in artifact.metadata.tags

    def test_create_with_builder(self):
        """Test creating an artifact using builder pattern"""
        artifact = (
            ArtifactBuilder()
            .type(ArtifactType.SPEC)
            .name("requirements.md")
            .content({"description": "Test spec", "requirements": ["req1", "req2"]})
            .created_by("architect_01")
            .tag("documentation")
            .tag("requirements")
            .version("2.0.0")
            .language("markdown")
            .build()
        )

        assert artifact.type == ArtifactType.SPEC
        assert artifact.name == "requirements.md"
        assert artifact.metadata.version == "2.0.0"
        assert len(artifact.metadata.tags) == 2

    def test_validate(self):
        """Test validating artifacts"""
        valid_code = Artifact(
            type=ArtifactType.CODE,
            name="utils.py",
            content="def util(): pass",
        )
        is_valid, errors = valid_code.validate()
        assert is_valid is True
        assert len(errors) == 0

        invalid_code = Artifact(
            type=ArtifactType.CODE,
            name="short.py",
            content="x",
        )
        is_valid, errors = invalid_code.validate()
        assert is_valid is False
        assert any("too short" in e.lower() for e in errors)

        valid_spec = Artifact(
            type=ArtifactType.SPEC,
            name="spec",
            content={"description": "A spec", "requirements": ["req1"]},
        )
        is_valid, errors = valid_spec.validate()
        assert is_valid is True

        invalid_spec = Artifact(
            type=ArtifactType.SPEC,
            name="bad_spec",
            content={"description": "Missing requirements"},
        )
        is_valid, errors = invalid_spec.validate()
        assert is_valid is False

        valid_test = Artifact(
            type=ArtifactType.TEST_RESULT,
            name="results",
            content={"passed": 10, "total": 10},
        )
        is_valid, errors = valid_test.validate()
        assert is_valid is True

        valid_pr = Artifact(
            type=ArtifactType.PR,
            name="pr_123",
            content={"title": "Fix bug", "description": "Fixes #123", "changes": ["file1.py"]},
        )
        is_valid, errors = valid_pr.validate()
        assert is_valid is True

    def test_registry(self):
        """Test artifact registry operations"""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(storage_path=tmpdir)

            artifact1 = Artifact(
                type=ArtifactType.CODE,
                name="module.py",
                content="def module(): pass",
                created_by="agent_01",
            )
            artifact2 = Artifact(
                type=ArtifactType.DOCUMENT,
                name="README.md",
                content="# Documentation",
                created_by="agent_02",
            )

            id1 = registry.register(artifact1)
            id2 = registry.register(artifact2)

            assert id1 is not None
            assert id2 is not None

            retrieved = registry.get(id1)
            assert retrieved is not None
            assert retrieved.name == "module.py"

            code_artifacts = registry.get_by_type(ArtifactType.CODE)
            assert len(code_artifacts) == 1
            assert code_artifacts[0].name == "module.py"

            agent_artifacts = registry.get_by_creator("agent_01")
            assert len(agent_artifacts) == 1

            link = registry.link(id1, id2, "documented_by")
            assert link is not None
            assert link.relationship == "documented_by"

            links = registry.get_links(id1)
            assert len(links) == 1

            linked = registry.get_linked_artifacts(id1)
            assert len(linked) == 1
            assert linked[0].name == "README.md"

            registry.update_status(id1, ArtifactStatus.APPROVED)
            updated = registry.get(id1)
            assert updated is not None
            assert updated.status == ArtifactStatus.APPROVED

            draft_artifacts = registry.get_by_status(ArtifactStatus.DRAFT)
            assert len(draft_artifacts) == 1

            stats = registry.get_stats()
            assert stats["total_artifacts"] == 2
            assert stats["total_links"] == 1

            assert registry.delete(id1) is True
            assert registry.get(id1) is None
            assert len(registry.get_links(id2)) == 0

    def test_artifact_hash(self):
        """Test artifact content hashing"""
        artifact = Artifact(
            type=ArtifactType.CODE,
            name="test.py",
            content="def test(): pass",
        )

        hash1 = artifact.compute_hash()
        hash2 = artifact.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 16

        artifact.content = "def modified(): pass"
        hash3 = artifact.compute_hash()
        assert hash3 != hash1

    def test_artifact_to_dict_roundtrip(self):
        """Test artifact serialization roundtrip"""
        original = Artifact(
            type=ArtifactType.CODE,
            name="test.py",
            content="print('hello')",
            metadata=ArtifactMetadata(version="1.0.0", tags=["test"]),
            created_by="agent_01",
            status=ArtifactStatus.APPROVED,
        )

        data = original.to_dict()
        restored = Artifact.from_dict(data)

        assert restored.type == original.type
        assert restored.name == original.name
        assert restored.content == original.content
        assert restored.metadata.version == original.metadata.version
        assert restored.status == original.status


class TestFewShotRetriever:
    """Tests for FewShotRetriever class"""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FewShotRetriever(storage_path=tmpdir, min_success_level=SuccessLevel.SUCCESS)

    def test_index(self, retriever):
        """Test indexing trajectories"""
        trajectory = Trajectory(
            task_type=TaskCategory.CODE_GENERATION,
            task_description="Write a function to sort a list",
            steps=[
                TrajectoryStep(step_id=1, action="analyze", reasoning="Need to sort numbers"),
                TrajectoryStep(step_id=2, action="implement", reasoning="Using quicksort"),
            ],
            result={"code": "def sort_list(l): return sorted(l)"},
            success_metrics=SuccessMetrics(
                completion_rate=1.0,
                accuracy_score=0.95,
                efficiency_score=0.9,
                quality_score=0.9,
            ),
            success_level=SuccessLevel.EXEMPLARY,
            signature_name="code_gen",
        )

        traj_id = retriever.index_trajectory(trajectory)

        assert traj_id == trajectory.id
        assert trajectory.id in retriever._trajectories
        assert trajectory.id in retriever._type_index[TaskCategory.CODE_GENERATION]
        assert trajectory.id in retriever._signature_index.get("code_gen", [])

    def test_index_low_success_filtered(self, retriever):
        """Test that low success trajectories are filtered"""
        trajectory = Trajectory(
            task_type=TaskCategory.CODE_GENERATION,
            task_description="Write a function",
            success_level=SuccessLevel.FAILED,
        )

        retriever.index_trajectory(trajectory)

        assert len(retriever._trajectories) == 0

    def test_retrieve_similar(self, retriever):
        """Test retrieving similar trajectories"""
        retriever.index_trajectory(
            Trajectory(
                id="traj_1",
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Write a function to sort a list in Python",
                success_metrics=SuccessMetrics(quality_score=0.9),
                success_level=SuccessLevel.SUCCESS,
            )
        )
        retriever.index_trajectory(
            Trajectory(
                id="traj_2",
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Create a function to sort an array",
                success_metrics=SuccessMetrics(quality_score=0.8),
                success_level=SuccessLevel.SUCCESS,
            )
        )
        retriever.index_trajectory(
            Trajectory(
                id="traj_3",
                task_type=TaskCategory.DEBUGGING,
                task_description="Debug the sorting function",
                success_metrics=SuccessMetrics(quality_score=0.85),
                success_level=SuccessLevel.SUCCESS,
            )
        )

        result = retriever.retrieve_similar("Write a function to sort numbers", k=2)

        assert isinstance(result, RetrievalResult)
        assert result.query == "Write a function to sort numbers"
        assert len(result.trajectories) <= 2
        assert result.total_candidates >= 2
        assert len(result.similarity_scores) == len(result.trajectories)
        assert result.retrieval_time_ms > 0

    def test_retrieve_similar_with_type_filter(self, retriever):
        """Test retrieving with task type filter"""
        retriever.index_trajectory(
            Trajectory(
                id="traj_1",
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Write code",
                success_metrics=SuccessMetrics(quality_score=0.9),
                success_level=SuccessLevel.SUCCESS,
            )
        )
        retriever.index_trajectory(
            Trajectory(
                id="traj_2",
                task_type=TaskCategory.TESTING,
                task_description="Write tests",
                success_metrics=SuccessMetrics(quality_score=0.85),
                success_level=SuccessLevel.SUCCESS,
            )
        )

        result = retriever.retrieve_similar(
            "Write something",
            k=5,
            task_type=TaskCategory.CODE_GENERATION,
        )

        for traj in result.trajectories:
            assert traj.task_type == TaskCategory.CODE_GENERATION

    def test_build_prompt(self, retriever):
        """Test building few-shot prompt"""
        trajectory = Trajectory(
            task_type=TaskCategory.CODE_GENERATION,
            task_description="Write a function to reverse a string",
            steps=[
                TrajectoryStep(
                    step_id=1,
                    action="analyze",
                    reasoning="Need to reverse string characters",
                ),
                TrajectoryStep(
                    step_id=2,
                    action="implement",
                    reasoning="Using slice notation",
                ),
            ],
            result={"code": "def reverse(s): return s[::-1]"},
            success_metrics=SuccessMetrics(quality_score=0.9),
            success_level=SuccessLevel.SUCCESS,
        )

        prompt = retriever.build_few_shot_prompt(
            "Write a function to reverse a list",
            examples=[trajectory],
        )

        assert "Write a function to reverse a string" in prompt
        assert "Approach:" in prompt
        assert "analyze" in prompt
        assert "Current Task" in prompt
        assert "Write a function to reverse a list" in prompt

    def test_build_prompt_auto_retrieve(self, retriever):
        """Test building prompt with auto-retrieval"""
        retriever.index_trajectory(
            Trajectory(
                id="traj_1",
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Write a sorting function",
                success_metrics=SuccessMetrics(quality_score=0.9),
                success_level=SuccessLevel.SUCCESS,
            )
        )

        prompt = retriever.build_few_shot_prompt("Write a sorting function")

        assert "sorting function" in prompt.lower()

    def test_build_prompt_no_examples(self, retriever):
        """Test building prompt when no examples available"""
        prompt = retriever.build_few_shot_prompt("New task type")

        assert "No similar examples available" in prompt

    def test_get_stats(self, retriever):
        """Test getting retriever statistics"""
        retriever.index_trajectory(
            Trajectory(
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Task 1",
                success_metrics=SuccessMetrics(quality_score=0.9),
                success_level=SuccessLevel.SUCCESS,
            )
        )
        retriever.index_trajectory(
            Trajectory(
                task_type=TaskCategory.CODE_REVIEW,
                task_description="Task 2",
                success_metrics=SuccessMetrics(quality_score=0.8),
                success_level=SuccessLevel.SUCCESS,
            )
        )

        stats = retriever.get_stats()

        assert stats["total_trajectories"] == 2
        assert "CODE_GENERATION" in stats["by_task_type"]
        assert "CODE_REVIEW" in stats["by_task_type"]
        assert "avg_success_score" in stats


class TestProfileEvolver:
    """Tests for ProfileEvolver class"""

    @pytest.fixture
    def evolver(self):
        """Create an evolver instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ProfileEvolver(
                storage_path=tmpdir,
                min_tasks_for_evolution=5,
                evolution_cooldown_hours=1,
            )

    def test_analyze(self, evolver):
        """Test analyzing fractal performance"""
        evolver.register_profile(
            "coder_01",
            specialty="general",
            capabilities={"python": 0.5, "javascript": 0.5},
        )

        for i in range(5):
            snapshot = PerformanceSnapshot(
                total_tasks=10 + i * 5,
                successful_tasks=8 + i * 4,
                failed_tasks=2 + i,
                avg_quality_score=0.75 + i * 0.05,
                domain_breakdown={
                    "sql": {"success_rate": 0.95, "avg_quality": 0.9},
                    "css": {"success_rate": 0.3, "avg_quality": 0.4},
                },
                capability_scores={"python": 0.7, "sql": 0.9},
            )
            evolver.record_performance("coder_01", snapshot)

        analysis = evolver.analyze_performance("coder_01")

        assert analysis["status"] == "analyzed"
        assert "current_performance" in analysis
        assert "trend" in analysis
        assert "strengths" in analysis
        assert "weaknesses" in analysis
        assert "recommendations" in analysis
        assert "sql" in analysis["strengths"]
        assert "css" in analysis["weaknesses"]

    def test_analyze_no_data(self, evolver):
        """Test analyzing with no performance data"""
        analysis = evolver.analyze_performance("unknown_fractal")

        assert analysis["status"] == "no_data"

    def test_suggest_evolution(self, evolver):
        """Test suggesting profile evolution"""
        evolver.register_profile(
            "coder_01",
            specialty="general",
            capabilities={"python": 0.5},
        )

        snapshot1 = PerformanceSnapshot(
            total_tasks=5,
            successful_tasks=2,
            failed_tasks=3,
            avg_quality_score=0.4,
            domain_breakdown={},
            capability_scores={"python": 0.3},
        )
        snapshot2 = PerformanceSnapshot(
            total_tasks=10,
            successful_tasks=9,
            failed_tasks=1,
            avg_quality_score=0.9,
            domain_breakdown={"sql": {"success_rate": 0.95}},
            capability_scores={"python": 0.5, "sql": 0.9},
        )

        evolver.record_performance("coder_01", snapshot1)
        evolver.record_performance("coder_01", snapshot2)

        evolution = evolver.suggest_evolution("coder_01")

        assert evolution is not None
        assert evolution.fractal_id == "coder_01"
        assert evolution.status == EvolutionStatus.PROPOSED
        assert evolution.confidence >= 0.0

    def test_suggest_evolution_cooldown(self, evolver):
        """Test evolution cooldown prevents suggestions"""
        evolver.register_profile("coder_01", "general", {"python": 0.5})

        snapshot = PerformanceSnapshot(total_tasks=10, successful_tasks=9)
        evolver.record_performance("coder_01", snapshot)
        evolver.record_performance("coder_01", snapshot)

        evolution = ProfileEvolution(
            fractal_id="coder_01",
            old_specialty="general",
            new_specialty="sql",
            status=EvolutionStatus.APPLIED,
            applied_at=datetime.now(),
        )
        evolver._last_evolution["coder_01"] = datetime.now()

        result = evolver.suggest_evolution("coder_01")

        assert result is None

    def test_apply(self, evolver):
        """Test applying an evolution"""
        evolver.register_profile("coder_01", "general", {"python": 0.5})

        evolution = ProfileEvolution(
            fractal_id="coder_01",
            old_specialty="general",
            new_specialty="sql",
            old_capabilities={"python": 0.5},
            new_capabilities={"python": 0.5, "sql": 0.9},
            trigger=EvolutionTrigger.TASK_SUCCESS_PATTERN,
            confidence=0.8,
            status=EvolutionStatus.PROPOSED,
        )

        result = evolver.apply_evolution(evolution)

        assert result is True
        assert evolution.status == EvolutionStatus.APPLIED
        assert evolution.applied_at is not None

        profile = evolver.get_current_profile("coder_01")
        assert profile["specialty"] == "sql"
        assert "sql" in profile["capabilities"]

    def test_apply_low_confidence_rejected(self, evolver):
        """Test that low confidence evolution is rejected"""
        evolver.register_profile("coder_01", "general", {})

        evolution = ProfileEvolution(
            fractal_id="coder_01",
            old_specialty="general",
            new_specialty="sql",
            confidence=0.3,
            status=EvolutionStatus.PROPOSED,
        )

        result = evolver.apply_evolution(evolution)

        assert result is False
        assert evolution.status == EvolutionStatus.REJECTED

    def test_apply_wrong_status(self, evolver):
        """Test applying evolution with wrong status"""
        evolution = ProfileEvolution(
            fractal_id="coder_01",
            status=EvolutionStatus.APPLIED,
        )

        result = evolver.apply_evolution(evolution)

        assert result is False

    def test_get_evolution_history(self, evolver):
        """Test getting evolution history"""
        evolver.register_profile("coder_01", "general", {})

        evolution = ProfileEvolution(
            fractal_id="coder_01",
            old_specialty="general",
            new_specialty="sql",
            confidence=0.8,
            status=EvolutionStatus.PROPOSED,
        )
        evolver.apply_evolution(evolution)

        history = evolver.get_evolution_history("coder_01")

        assert len(history) == 1
        assert history[0].new_specialty == "sql"

    def test_rollback_evolution(self, evolver):
        """Test rolling back an evolution"""
        evolver.register_profile("coder_01", "general", {"python": 0.5})

        evolution1 = ProfileEvolution(
            id="evo_1",
            fractal_id="coder_01",
            old_specialty="general",
            new_specialty="python",
            old_capabilities={"python": 0.5},
            new_capabilities={"python": 0.8},
            confidence=0.8,
            status=EvolutionStatus.PROPOSED,
        )
        evolver.apply_evolution(evolution1)

        result = evolver.rollback_evolution("evo_1")

        assert result is True
        assert evolution1.status == EvolutionStatus.ROLLED_BACK
        profile = evolver.get_current_profile("coder_01")
        assert profile["specialty"] == "general"

    def test_get_stats(self, evolver):
        """Test getting evolver statistics"""
        evolver.register_profile("coder_01", "general", {})
        evolver.register_profile("coder_02", "sql", {})

        evolution = ProfileEvolution(
            fractal_id="coder_01",
            confidence=0.8,
            status=EvolutionStatus.PROPOSED,
        )
        evolver.apply_evolution(evolution)

        stats = evolver.get_stats()

        assert stats["total_profiles"] == 2
        assert stats["total_evolutions"] == 1
        assert stats["applied"] == 1


class TestSOPManager:
    """Tests for SOPManager class"""

    @pytest.fixture
    def sop_manager(self):
        """Create an SOP manager instance"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SOPManager(sop_dir=tmpdir)

    def test_get_sop(self, sop_manager):
        """Test getting SOP by role"""
        sop = sop_manager.get_sop_for_role("coder")

        assert sop is not None
        assert sop.role == "coder"
        assert len(sop.steps) > 0
        assert len(sop.artifacts_produced) > 0

    def test_get_sop_unknown_role(self, sop_manager):
        """Test getting SOP for unknown role"""
        sop = sop_manager.get_sop_for_role("unknown_role")

        assert sop is None

    def test_get_sop_by_id(self, sop_manager):
        """Test getting SOP by ID"""
        all_sops = sop_manager.list_sops()
        if all_sops:
            sop = sop_manager.get_sop(all_sops[0])
            assert sop is not None

    def test_validate_artifact(self, sop_manager):
        """Test validating artifact against SOP"""
        sop = sop_manager.get_sop_for_role("coder")
        assert sop is not None

        artifact = Mock()
        artifact.name = "source_code"
        artifact.type = ArtifactType.CODE
        artifact.content = "def hello():\n    print('hello')"

        result = sop_manager.validate_artifact_against_sop(artifact, sop)

        assert isinstance(result, ArtifactValidationResult)
        assert result.artifact_name == "source_code"
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.quality_gate_results, list)

    def test_validate_artifact_none(self, sop_manager):
        """Test validating None artifact"""
        sop = sop_manager.get_sop_for_role("coder")
        assert sop is not None

        result = sop_manager.validate_artifact_against_sop(None, sop)

        assert result.is_valid is False
        assert "None" in result.issues[0]

    def test_validate_artifact_with_step(self, sop_manager):
        """Test validating artifact for specific step"""
        sop = sop_manager.get_sop_for_role("coder")
        assert sop is not None

        artifact = Mock()
        artifact.name = "source_code"
        artifact.type = ArtifactType.CODE
        artifact.content = "print('test')"

        result = sop_manager.validate_artifact_against_sop(artifact, sop, step_id=3)

        assert result is not None
        assert result.score >= 0.0

    def test_register_sop(self, sop_manager):
        """Test registering a new SOP"""
        new_sop = SOP(
            role="custom_role",
            name="Custom SOP",
            description="A custom SOP",
            steps=[
                SOPStep(step_id=1, name="Step 1", step_type=StepType.ACTION),
                SOPStep(step_id=2, name="Step 2", step_type=StepType.ACTION, dependencies=[1]),
            ],
            artifacts_produced=["output"],
        )

        sop_manager.register_sop(new_sop)

        retrieved = sop_manager.get_sop_for_role("custom_role")
        assert retrieved is not None
        assert retrieved.name == "Custom SOP"

    def test_register_invalid_sop(self, sop_manager):
        """Test registering an invalid SOP"""
        invalid_sop = SOP(
            role="invalid",
            name="Invalid SOP",
            steps=[
                SOPStep(step_id=1, name="Step 1", dependencies=[99]),
            ],
        )

        with pytest.raises(ValueError):
            sop_manager.register_sop(invalid_sop)

    def test_get_next_step(self, sop_manager):
        """Test getting next step in SOP"""
        sop = sop_manager.get_sop_for_role("coder")
        assert sop is not None
        assert len(sop.steps) >= 2

        next_step = sop_manager.get_next_step(sop, sop.steps[0].step_id)

        assert next_step == sop.steps[1].step_id

    def test_get_executable_steps(self, sop_manager):
        """Test getting executable steps"""
        sop = sop_manager.get_sop_for_role("coder")
        assert sop is not None

        executable = sop_manager.get_executable_steps(sop, completed_steps=set())

        assert 1 in executable

        executable = sop_manager.get_executable_steps(sop, completed_steps={1, 2})

        for step in sop.steps:
            if all(dep in {1, 2} for dep in step.dependencies):
                if step.step_id not in {1, 2}:
                    assert step.step_id in executable

    def test_list_sops(self, sop_manager):
        """Test listing all SOPs"""
        sops = sop_manager.list_sops()

        assert len(sops) >= 3

    def test_list_roles(self, sop_manager):
        """Test listing all roles"""
        roles = sop_manager.list_roles()

        assert "coder" in roles
        assert "reviewer" in roles
        assert "architect" in roles

    def test_get_stats(self, sop_manager):
        """Test getting SOP manager statistics"""
        stats = sop_manager.get_stats()

        assert stats["total_sops"] >= 3
        assert stats["total_roles"] >= 3
        assert stats["total_steps"] > 0
        assert "sop_directory" in stats

    def test_quality_gate_check(self, sop_manager):
        """Test quality gate checking"""
        gate = QualityGate(
            name="test_gate",
            description="Test gate",
            required_artifacts=["test_artifact"],
        )

        artifact = Mock()
        artifact.name = "test_artifact"
        artifact.content = "test content"
        artifact.type = ArtifactType.DOCUMENT

        status = sop_manager._check_quality_gate(artifact, gate)

        assert status in [
            QualityGateStatus.PASSED,
            QualityGateStatus.FAILED,
            QualityGateStatus.SKIPPED,
        ]

    def test_sop_step_order_validation(self):
        """Test SOP step order validation"""
        valid_sop = SOP(
            role="test",
            name="Valid SOP",
            steps=[
                SOPStep(step_id=1, name="First"),
                SOPStep(step_id=2, name="Second", dependencies=[1]),
                SOPStep(step_id=3, name="Third", dependencies=[2]),
            ],
        )

        is_valid, errors = valid_sop.validate_step_order()
        assert is_valid is True

        invalid_sop = SOP(
            role="test",
            name="Invalid SOP",
            steps=[
                SOPStep(step_id=1, name="First", dependencies=[5]),
            ],
        )

        is_valid, errors = invalid_sop.validate_step_order()
        assert is_valid is False
        assert len(errors) > 0


class TestIntegration:
    """Integration tests for SOTA Research Hub components"""

    def test_signature_teleprompter_workflow(self):
        """Test Signature + Teleprompter workflow"""
        signature = Signature(
            name="translate",
            inputs=[SignatureField(name="text", field_type=FieldType.STRING)],
            outputs=[SignatureField(name="translation", field_type=FieldType.STRING)],
        )

        teleprompter = Teleprompter()
        teleprompter.index_example(
            signature,
            Example(
                inputs={"text": "Hello"},
                outputs={"translation": "Hola"},
                quality_score=0.95,
            ),
        )
        teleprompter.index_example(
            signature,
            Example(
                inputs={"text": "Goodbye"},
                outputs={"translation": "Adios"},
                quality_score=0.9,
            ),
        )

        result = teleprompter.optimize(signature)

        assert result is not None
        assert len(result.examples_used) == 2
        assert result.improvement_score > 0

    def test_artifact_fewshot_workflow(self):
        """Test Artifact + FewShotRetriever workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ArtifactRegistry(storage_path=tmpdir)
            retriever = FewShotRetriever(storage_path=tmpdir)

            trajectory = Trajectory(
                task_type=TaskCategory.CODE_GENERATION,
                task_description="Create utility module",
                steps=[TrajectoryStep(step_id=1, action="write code")],
                result={"code": "def util(): pass"},
                success_metrics=SuccessMetrics(quality_score=0.9),
                success_level=SuccessLevel.SUCCESS,
            )
            retriever.index_trajectory(trajectory)

            artifact = Artifact(
                type=ArtifactType.CODE,
                name="utils.py",
                content=trajectory.result["code"],
            )
            registry.register(artifact)

            assert len(retriever._trajectories) == 1
            assert len(registry._artifacts) == 1

    def test_profile_evolution_retriever_workflow(self):
        """Test ProfileEvolver + FewShotRetriever workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            evolver = ProfileEvolver(storage_path=tmpdir)
            retriever = FewShotRetriever(
                storage_path=tmpdir, min_success_level=SuccessLevel.EXEMPLARY
            )

            evolver.register_profile("agent_01", "general", {"code": 0.5})

            trajectory = Trajectory(
                task_type=TaskCategory.CODE_GENERATION,
                task_description="SQL optimization task",
                success_metrics=SuccessMetrics(quality_score=0.95),
                success_level=SuccessLevel.EXEMPLARY,
                signature_name="sql_task",
            )
            retriever.index_trajectory(trajectory)

            snapshot = PerformanceSnapshot(
                total_tasks=10,
                successful_tasks=9,
                domain_breakdown={"sql": {"success_rate": 0.95}},
                capability_scores={"sql": 0.95},
            )
            evolver.record_performance("agent_01", snapshot)
            evolver.record_performance("agent_01", snapshot)

            examples = retriever.get_examples_for_signature("sql_task")
            assert len(examples) >= 1 or len(retriever._trajectories) == 1

    def test_sop_artifact_validation_workflow(self):
        """Test SOPManager + Artifact validation workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            sop_manager = SOPManager(sop_dir=tmpdir)
            registry = ArtifactRegistry(storage_path=tmpdir)

            sop = sop_manager.get_sop_for_role("coder")
            assert sop is not None

            artifact = Artifact(
                type=ArtifactType.CODE,
                name="source_code",
                content="def hello(): return 'world'",
            )

            is_valid, errors = artifact.validate()
            assert is_valid is True

            result = sop_manager.validate_artifact_against_sop(artifact, sop)
            assert result is not None

            artifact_id = registry.register(artifact)
            assert artifact_id is not None
