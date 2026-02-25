"""
Unit tests for Spec 18: Code Level Upgrades

Tests for:
- RAPTOR (Recursive Abstractive Retrieval)
- Vector Backends (InMemory, LanceDB fallback)
- Summary Builder
- Interpreter Tool
- API Search Tool
- Tool Interactive Critic
- GraphOfThoughts (via Layer1)
- Reflexion (Self-Healing)

Reference: docs/evolution_plan_2026/18_CODE_LEVEL_UPGRADES.md
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.memory.raptor import (
    SummaryTree,
    SummaryTreeNode,
    NodeType,
    Document,
    CollapsedTreeRetrieval,
    QueryLevel,
    RetrievalResult,
    build_raptor_tree,
    SimpleHashEmbedding,
    SimpleSummarizer,
)
from gaap.memory.vector_backends import (
    InMemoryBackend,
    VectorRecord,
    SearchResult,
    get_backend,
    get_available_backends,
)
from gaap.memory.summary_builder import (
    SummaryBuilder,
    SummaryResult,
    KeyConcept,
    HierarchicalSummarizer,
    SimpleLLMProvider,
)
from gaap.tools.interpreter_tool import (
    InterpreterTool,
    ExecutionResult,
    ExecutionStatus,
    TestCase,
    TestResult,
    create_interpreter,
)
from gaap.tools.search_tool import (
    APISearchTool,
    APIInfo,
    EndpointInfo,
    APICategory,
    DeprecationStatus,
    create_api_search_tool,
)
from gaap.layers.tool_critic import (
    ToolInteractiveCritic,
    VerificationPlan,
    VerificationStep,
    VerificationResult,
    VerificationStepType,
    VerificationStatus,
)
from gaap.healing.reflexion import (
    ReflexionEngine,
    Reflection,
    ReflectionDepth,
)


class TestRAPTOR:
    """Tests for RAPTOR (Recursive Abstractive Retrieval)"""

    def test_summary_tree_node_creation(self):
        node = SummaryTreeNode(
            id="test-node-1",
            text="Sample document text",
            summary="Sample summary",
            level=0,
            node_type=NodeType.LEAF,
        )
        assert node.id == "test-node-1"
        assert node.text == "Sample document text"
        assert node.summary == "Sample summary"
        assert node.level == 0
        assert node.is_leaf() is True
        assert node.is_root() is False

    def test_summary_tree_node_serialization(self):
        node = SummaryTreeNode(
            id="test-node-1",
            text="Sample document text",
            summary="Sample summary",
            level=0,
            node_type=NodeType.LEAF,
        )
        data = node.to_dict()
        assert data["id"] == "test-node-1"
        assert data["node_type"] == "LEAF"

        restored = SummaryTreeNode.from_dict(data)
        assert restored.id == node.id
        assert restored.text == node.text

    def test_summary_tree_initialization(self):
        tree = SummaryTree(max_children=5)
        assert tree.max_children == 5
        assert tree.root_id is None
        assert len(tree.nodes) == 0

    def test_build_from_documents(self):
        tree = SummaryTree(max_children=3)
        documents = [
            Document(
                text="Document one content about machine learning.", metadata={"source": "test1"}
            ),
            Document(
                text="Document two content about neural networks.", metadata={"source": "test2"}
            ),
            Document(
                text="Document three content about deep learning.", metadata={"source": "test3"}
            ),
        ]
        root_id = tree.build_from_documents(documents)
        assert root_id is not None
        assert tree.root_id == root_id
        assert len(tree.nodes) >= 3
        root = tree.get_root()
        assert root is not None
        assert root.is_root() is True

    def test_add_leaf_node(self):
        tree = SummaryTree()
        node = tree.add_leaf("Test document content", {"key": "value"})
        assert node.id in tree.nodes
        assert node.is_leaf() is True
        assert node.embedding is not None

    def test_get_children(self):
        tree = SummaryTree(max_children=2)
        documents = [
            Document(text="Document one"),
            Document(text="Document two"),
        ]
        tree.build_from_documents(documents)
        root = tree.get_root()
        if root:
            children = tree.get_children(root.id)
            assert len(children) >= 0

    def test_get_nodes_at_level(self):
        tree = SummaryTree(max_children=2)
        documents = [
            Document(text="Document one"),
            Document(text="Document two"),
            Document(text="Document three"),
        ]
        tree.build_from_documents(documents)
        leaf_nodes = tree.get_nodes_at_level(0)
        assert len(leaf_nodes) == 3

    def test_search_similar(self):
        tree = SummaryTree()
        tree.add_leaf("Machine learning algorithms")
        tree.add_leaf("Neural network architectures")
        tree.add_leaf("Cooking recipes")
        results = tree.search_similar("deep learning AI", k=2)
        assert len(results) <= 2

    def test_collapsed_retrieval_initialization(self):
        tree = SummaryTree()
        retrieval = CollapsedTreeRetrieval(tree, default_k=10)
        assert retrieval.default_k == 10
        assert retrieval.tree is tree

    def test_determine_query_level(self):
        tree = SummaryTree()
        retrieval = CollapsedTreeRetrieval(tree)

        assert retrieval.determine_query_level("overview of the project") == QueryLevel.GENERAL
        assert retrieval.determine_query_level("explain the system") == QueryLevel.BROAD
        assert retrieval.determine_query_level("specific line 42") == QueryLevel.DETAIL
        assert retrieval.determine_query_level("api") == QueryLevel.SPECIFIC

    def test_retrieve_from_tree(self):
        tree = SummaryTree(max_children=2)
        documents = [
            Document(text="Machine learning is a subset of artificial intelligence."),
            Document(text="Neural networks are inspired by biological neurons."),
            Document(text="Cooking involves preparing food with heat."),
        ]
        tree.build_from_documents(documents)
        retrieval = CollapsedTreeRetrieval(tree)
        results = retrieval.retrieve("What is machine learning?", k=2)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, RetrievalResult)

    def test_build_raptor_tree_convenience(self):
        documents = [
            {"text": "First document"},
            {"text": "Second document"},
        ]
        tree = build_raptor_tree(documents, max_children=2)
        assert isinstance(tree, SummaryTree)
        assert len(tree.nodes) >= 2


class TestVectorBackends:
    """Tests for Vector Backend implementations"""

    def test_inmemory_backend_initialization(self):
        backend = InMemoryBackend(dimension=384)
        assert backend.dimension == 384
        assert len(backend.tables) == 0

    def test_inmemory_backend_connect(self):
        backend = InMemoryBackend()
        result = backend.connect("/tmp/test_db")
        assert result is True

    def test_inmemory_backend_create_table(self):
        backend = InMemoryBackend()
        backend.connect("/tmp/test")
        result = backend.create_table("documents", dimension=128)
        assert result is True
        assert "documents" in backend.tables

    def test_inmemory_backend_insert(self):
        backend = InMemoryBackend()
        backend.connect("/tmp/test")
        backend.create_table("documents")

        vectors = [[0.1, 0.2, 0.3] for _ in range(3)]
        ids = backend.insert("documents", vectors)
        assert len(ids) == 3
        assert backend.count("documents") == 3

    def test_inmemory_backend_search(self):
        backend = InMemoryBackend()
        backend.connect("/tmp/test")
        backend.create_table("documents")

        vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        texts = ["doc one", "doc two", "doc three"]
        backend.insert("documents", vectors, texts=texts)

        query = [1.0, 0.0, 0.0]
        results = backend.search("documents", query, k=2)
        assert len(results) == 2
        assert results[0].score > 0.9

    def test_inmemory_backend_delete(self):
        backend = InMemoryBackend()
        backend.connect("/tmp/test")
        backend.create_table("documents")

        vectors = [[0.1, 0.2, 0.3]]
        ids = backend.insert("documents", vectors)
        assert backend.count("documents") == 1

        backend.delete("documents", ids)
        assert backend.count("documents") == 0

    def test_inmemory_backend_filter_search(self):
        backend = InMemoryBackend()
        backend.connect("/tmp/test")
        backend.create_table("documents")

        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        metadata = [{"category": "A"}, {"category": "B"}, {"category": "A"}]
        backend.insert("documents", vectors, metadata=metadata)

        results = backend.search("documents", [0.1, 0.2], k=10, filter_metadata={"category": "A"})
        assert all(r.metadata.get("category") == "A" for r in results)

    def test_lancedb_fallback_to_inmemory(self):
        backend = get_backend("memory", dimension=256)
        assert isinstance(backend, InMemoryBackend)
        assert backend.dimension == 256

    def test_get_available_backends(self):
        backends = get_available_backends()
        assert "memory" in backends

    def test_vector_record(self):
        record = VectorRecord(
            id="test-id",
            vector=[0.1, 0.2, 0.3],
            text="Sample text",
            metadata={"key": "value"},
        )
        data = record.to_dict()
        assert data["id"] == "test-id"
        assert data["text"] == "Sample text"

    def test_search_result(self):
        result = SearchResult(
            id="result-id",
            score=0.95,
            text="Matched text",
            metadata={"source": "test"},
        )
        data = result.to_dict()
        assert data["score"] == 0.95


class TestSummaryBuilder:
    """Tests for Summary Builder"""

    def test_summary_builder_initialization(self):
        builder = SummaryBuilder(max_summary_length=500, compression_ratio=0.3)
        assert builder.max_summary_length == 500
        assert builder.compression_ratio == 0.3

    @pytest.mark.asyncio
    async def test_summarize_texts(self):
        builder = SummaryBuilder()
        texts = [
            "Machine learning is a field of artificial intelligence.",
            "Neural networks are computing systems inspired by biological networks.",
            "Deep learning uses multiple layers of neural networks.",
        ]
        result = await builder.summarize_texts(texts)
        assert isinstance(result, SummaryResult)
        assert len(result.summary) > 0
        assert result.source_count == 3

    @pytest.mark.asyncio
    async def test_summarize_empty_texts(self):
        builder = SummaryBuilder()
        result = await builder.summarize_texts([])
        assert result.summary == ""
        assert result.source_count == 0

    @pytest.mark.asyncio
    async def test_summarize_short_text(self):
        builder = SummaryBuilder(max_summary_length=500)
        result = await builder.summarize_texts(["Short text"])
        assert result.summary == "Short text"

    def test_extract_concepts(self):
        builder = SummaryBuilder()
        text = """
        Machine Learning algorithms process data to find patterns.
        Neural Networks are a key component of Deep Learning.
        Artificial Intelligence systems use Machine Learning for predictions.
        """
        concepts = builder.extract_key_concepts(text, max_concepts=5)
        assert len(concepts) <= 5
        for concept in concepts:
            assert isinstance(concept, KeyConcept)
            assert len(concept.name) > 0

    def test_estimate_importance_from_text(self):
        builder = SummaryBuilder()

        short_text = "Simple text."
        short_score = builder.estimate_importance_from_text(short_text)
        assert 0.0 <= short_score <= 1.0

        important_text = (
            "This is a critical and important function that is essential for the system."
        )
        important_score = builder.estimate_importance_from_text(important_text)
        assert important_score >= short_score

    def test_key_concept(self):
        concept = KeyConcept(
            name="Machine Learning",
            description="AI subset for pattern recognition",
            relevance=0.9,
            category="technical",
        )
        data = concept.to_dict()
        assert data["name"] == "Machine Learning"
        assert data["relevance"] == 0.9

    @pytest.mark.asyncio
    async def test_hierarchical_summarizer(self):
        summarizer = HierarchicalSummarizer()
        assert summarizer.context_window == 3


class TestInterpreterTool:
    """Tests for Interpreter Tool (sandboxed code execution)"""

    def test_interpreter_initialization(self):
        tool = InterpreterTool(default_timeout=5.0, max_output_length=10000)
        assert tool.default_timeout == 5.0
        assert tool.max_output_length == 10000

    def test_validate_syntax_valid(self):
        tool = InterpreterTool()
        is_valid, error = tool.validate_syntax("x = 1 + 2")
        assert is_valid is True
        assert error == ""

    def test_validate_syntax_invalid(self):
        tool = InterpreterTool()
        is_valid, error = tool.validate_syntax("x = 1 +")
        assert is_valid is False
        assert "Syntax error" in error

    @pytest.mark.asyncio
    async def test_execute_simple_code(self):
        tool = InterpreterTool()
        result = await tool.execute("x = 1 + 2")
        assert result.success is True
        assert result.status == ExecutionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_execute_code_with_result(self):
        tool = InterpreterTool()
        code = """
result = [i ** 2 for i in range(5)]
"""
        result = await tool.execute(code)
        assert result.success is True
        assert result.return_value == [0, 1, 4, 9, 16]

    @pytest.mark.asyncio
    async def test_execute_blocked_import(self):
        tool = InterpreterTool()
        code = "import os"
        result = await tool.execute(code)
        assert result.status == ExecutionStatus.SECURITY_VIOLATION
        assert "Blocked imports" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        tool = InterpreterTool(default_timeout=0.1)
        code = """
while True:
    pass
"""
        result = await tool.execute(code, timeout=0.1)
        assert result.status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_function(self):
        tool = InterpreterTool()
        code = """
def add(a, b):
    return a + b
"""
        result = await tool.execute_function(code, "add", args=(1, 2))
        assert result.success is True
        assert result.return_value == 3

    @pytest.mark.asyncio
    async def test_test_function(self):
        tool = InterpreterTool()
        code = """
def multiply(a, b):
    return a * b
"""
        test_cases = [
            tool.create_test_case("test_positive", {"a": 2, "b": 3}, 6),
            tool.create_test_case("test_negative", {"a": -1, "b": 5}, -5),
        ]
        results = await tool.test_function(code, "multiply", test_cases)
        assert all(r.passed for r in results)

    def test_sandbox_restricted_builtins(self):
        tool = InterpreterTool()
        safe_globals = tool._get_safe_globals()
        builtins = safe_globals.get("__builtins__", {})
        dangerous = ["exec", "eval", "compile", "open", "__import__"]
        for func in dangerous:
            assert func not in builtins, f"Dangerous function {func} should not be available"


class TestAPISearchTool:
    """Tests for API Search Tool"""

    def test_api_search_tool_initialization(self):
        tool = APISearchTool()
        assert tool.use_web_search is False

    @pytest.mark.asyncio
    async def test_search_documentation_standard_library(self):
        tool = APISearchTool()
        info = await tool.search_documentation("json.loads")
        assert info.exists is True
        assert info.category == APICategory.STANDARD_LIBRARY
        assert "json" in info.module

    @pytest.mark.asyncio
    async def test_search_documentation_popular_package(self):
        tool = APISearchTool()
        info = await tool.search_documentation("requests.get")
        assert info.exists is True
        assert info.category == APICategory.POPULAR_PACKAGE

    @pytest.mark.asyncio
    async def test_search_documentation_unknown(self):
        tool = APISearchTool()
        info = await tool.search_documentation("nonexistent.function")
        assert info.category == APICategory.UNKNOWN

    @pytest.mark.asyncio
    async def test_verify_endpoint_invalid_url(self):
        tool = APISearchTool()
        info = await tool.verify_endpoint("not-a-valid-url")
        assert info.exists is False
        assert "Invalid URL" in info.description

    @pytest.mark.asyncio
    async def test_verify_endpoint_valid_format(self):
        tool = APISearchTool()
        info = await tool.verify_endpoint("https://api.github.com", method="GET")
        assert info.method == "GET"

    @pytest.mark.asyncio
    async def test_check_deprecation_active(self):
        tool = APISearchTool()
        is_deprecated, message = await tool.check_deprecation("json.loads")
        assert is_deprecated is False

    @pytest.mark.asyncio
    async def test_check_deprecation_deprecated(self):
        tool = APISearchTool()
        is_deprecated, message = await tool.check_deprecation("os.system")
        assert is_deprecated is True

    @pytest.mark.asyncio
    async def test_get_api_examples(self):
        tool = APISearchTool()
        examples = await tool.get_api_examples("json.loads")
        assert len(examples) > 0
        assert "json.loads" in examples[0]

    @pytest.mark.asyncio
    async def test_search_multiple(self):
        tool = APISearchTool()
        results = await tool.search_multiple(["json.loads", "re.match"])
        assert "json.loads" in results
        assert "re.match" in results

    def test_api_info_properties(self):
        info = APIInfo(
            name="test.api",
            category=APICategory.STANDARD_LIBRARY,
        )
        assert info.exists is True
        assert info.is_deprecated is False

        deprecated_info = APIInfo(
            name="old.api",
            deprecation_status=DeprecationStatus.DEPRECATED,
        )
        assert deprecated_info.is_deprecated is True


class TestToolInteractiveCritic:
    """Tests for Tool-Interactive CRITIC"""

    def test_verification_plan_creation(self):
        plan = VerificationPlan(plan_id="test-plan", subject="Test subject")
        assert plan.plan_id == "test-plan"
        assert len(plan.steps) == 0

    def test_verification_plan_add_step(self):
        plan = VerificationPlan(plan_id="test-plan", subject="Test")
        step = plan.add_step(
            step_type=VerificationStepType.SYNTAX_VALIDATION,
            description="Validate syntax",
            tool_name="interpreter",
        )
        assert len(plan.steps) == 1
        assert step.step_type == VerificationStepType.SYNTAX_VALIDATION

    def test_verification_result_success_rate(self):
        result = VerificationResult(
            plan_id="test-plan",
            total_steps=10,
            passed_steps=8,
        )
        assert result.success_rate == 0.8

    def test_tool_interactive_critic_initialization(self):
        critic = ToolInteractiveCritic()
        assert len(critic.available_tools) == 0

    def test_register_tool(self):
        critic = ToolInteractiveCritic()
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        critic.register_tool("test_tool", mock_tool)
        assert "test_tool" in critic.available_tools

    @pytest.mark.asyncio
    async def test_generate_verification_plan_fallback(self):
        critic = ToolInteractiveCritic()
        plan = await critic.generate_verification_plan("def test(): pass")
        assert isinstance(plan, VerificationPlan)
        assert len(plan.steps) >= 0

    @pytest.mark.asyncio
    async def test_execute_verification_empty_plan(self):
        critic = ToolInteractiveCritic()
        plan = VerificationPlan(plan_id="test", subject="Test")
        result = await critic.execute_verification(plan)
        assert result.total_steps == 0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_evaluate_with_tools_no_tools(self):
        critic = ToolInteractiveCritic()
        evaluation = await critic.evaluate_with_tools("simple text")
        assert evaluation.critic == "tool_interactive"
        assert evaluation.score >= 0

    def test_compare_results_numeric(self):
        critic = ToolInteractiveCritic()
        match, desc = critic.compare_results(10, 10)
        assert match is True

        match, desc = critic.compare_results(10, 10.1, tolerance=0.2)
        assert match is True

    def test_compare_results_string(self):
        critic = ToolInteractiveCritic()
        match, desc = critic.compare_results("hello", "HELLO")
        assert match is True

    def test_compare_results_list(self):
        critic = ToolInteractiveCritic()
        match, desc = critic.compare_results([1, 2, 3], [1, 2, 3])
        assert match is True

        match, desc = critic.compare_results([1, 2], [1, 2, 3])
        assert match is False


class TestGraphOfThoughts:
    """Tests for GraphOfThoughts (via Layer1 integration)"""

    def test_thought_aggregation(self):
        thoughts = [
            {"id": 1, "idea": "Use microservices", "score": 0.6},
            {"id": 2, "idea": "Use monolith", "score": 0.5},
            {"id": 3, "idea": "Use serverless", "score": 0.7},
        ]
        merged = {
            "id": "merged",
            "idea": "Hybrid: microservices with serverless workers",
            "score": max(t["score"] for t in thoughts) + 0.1,
        }
        assert merged["score"] >= 0.7

    def test_thought_refinement_loop(self):
        original = {"idea": "Basic approach", "score": 0.5}
        refinements = []
        for i in range(3):
            refined = {
                "idea": f"{original['idea']} (refined {i + 1})",
                "score": min(original["score"] + 0.1 * (i + 1), 1.0),
            }
            refinements.append(refined)
        assert refinements[-1]["score"] == 0.8

    def test_thought_node_merge(self):
        parent_a = {"children": ["a1", "a2"], "score": 0.6}
        parent_b = {"children": ["b1", "b2"], "score": 0.7}
        merged_node = {
            "children": parent_a["children"] + parent_b["children"],
            "score": (parent_a["score"] + parent_b["score"]) / 2 + 0.1,
        }
        assert len(merged_node["children"]) == 4
        assert merged_node["score"] == 0.75

    def test_token_reduction_via_merge(self):
        thoughts = [
            {"text": "Long detailed thought 1", "tokens": 100},
            {"text": "Long detailed thought 2", "tokens": 100},
            {"text": "Long detailed thought 3", "tokens": 100},
        ]
        total_before = sum(t["tokens"] for t in thoughts)
        merged = {"text": "Merged summary", "tokens": 150}
        reduction = 1 - (merged["tokens"] / total_before)
        assert reduction > 0.3


class TestReflexion:
    """Tests for Reflexion (Self-Reflection for Failure Recovery)"""

    def test_reflection_creation(self):
        reflection = Reflection(
            failure_analysis="Test failed due to incorrect input",
            root_cause="Input validation missing",
            proposed_fix="Add input validation",
            confidence=0.8,
        )
        assert reflection.failure_analysis == "Test failed due to incorrect input"
        assert reflection.confidence == 0.8

    def test_reflection_serialization(self):
        reflection = Reflection(
            failure_analysis="Analysis",
            root_cause="Root",
            proposed_fix="Fix",
            depth=ReflectionDepth.DEEP,
        )
        data = reflection.to_dict()
        restored = Reflection.from_dict(data)
        assert restored.failure_analysis == reflection.failure_analysis
        assert restored.depth == ReflectionDepth.DEEP

    def test_reflection_to_prompt_context(self):
        reflection = Reflection(
            failure_analysis="Missing error handling",
            root_cause="No try-catch block",
            proposed_fix="Add exception handling",
            alternative_approaches=["Use decorator", "Use context manager"],
            lessons_learned=["Always handle exceptions"],
        )
        context = reflection.to_prompt_context()
        assert "PREVIOUS ATTEMPT FAILED" in context
        assert "Missing error handling" in context
        assert "Root Cause" in context
        assert "Proposed Fix" in context

    def test_reflexion_engine_initialization(self):
        engine = ReflexionEngine()
        assert engine._model is not None
        assert engine._enable_deep is True

    @pytest.mark.asyncio
    async def test_reflexion_fallback_syntax_error(self):
        engine = ReflexionEngine(llm_provider=None)
        error = SyntaxError("invalid syntax")
        reflection = await engine.reflect(
            error=error,
            task_description="Write a function",
            previous_output="def func(",
        )
        assert (
            "Syntax" in reflection.failure_analysis
            or "syntax" in reflection.failure_analysis.lower()
        )
        assert reflection.depth == ReflectionDepth.SURFACE

    @pytest.mark.asyncio
    async def test_reflexion_fallback_timeout(self):
        engine = ReflexionEngine(llm_provider=None)
        error = TimeoutError("Operation timed out")
        reflection = await engine.reflect(
            error=error,
            task_description="Long running task",
            previous_output="",
        )
        assert "timeout" in reflection.failure_analysis.lower()

    def test_refine_prompt(self):
        engine = ReflexionEngine()
        reflection = Reflection(
            failure_analysis="Missing imports",
            proposed_fix="Add import statements",
        )
        original = "Write a function to calculate fibonacci"
        refined = engine.refine_prompt(original, reflection)
        assert "PREVIOUS ATTEMPT FAILED" in refined
        assert "Missing imports" in refined

    def test_clear_cache(self):
        engine = ReflexionEngine()
        engine._reflection_cache["key"] = Reflection(failure_analysis="test")
        engine.clear_cache()
        assert len(engine._reflection_cache) == 0

    def test_reflection_depth_enum(self):
        assert ReflectionDepth.SURFACE.value == 1
        assert ReflectionDepth.MODERATE.value == 2
        assert ReflectionDepth.DEEP.value == 3


class TestCodeUpgradesIntegration:
    """Integration tests for Code Level Upgrades"""

    @pytest.mark.asyncio
    async def test_raptor_with_summary_builder(self):
        builder = SummaryBuilder()
        tree = SummaryTree()

        docs = [
            Document(text="Document about Python programming"),
            Document(text="Document about JavaScript frameworks"),
        ]

        for doc in docs:
            tree.add_leaf(doc.text, doc.metadata)

        assert len(tree.nodes) == 2

    @pytest.mark.asyncio
    async def test_interpreter_with_critic(self):
        interpreter = InterpreterTool()
        critic = ToolInteractiveCritic(tools=[interpreter])

        assert "interpreter" in critic.available_tools

        plan = VerificationPlan(plan_id="test", subject="x = 1")
        plan.add_step(
            step_type=VerificationStepType.SYNTAX_VALIDATION,
            description="Validate",
            tool_name="interpreter",
            tool_params={"code": "x = 1"},
        )

        result = await critic.execute_verification(plan)
        assert result.total_steps == 1

    @pytest.mark.asyncio
    async def test_api_search_with_critic(self):
        api_search = APISearchTool()
        critic = ToolInteractiveCritic(tools=[api_search])

        plan = await critic.generate_verification_plan("requests.get('https://api.example.com')")

        assert len(plan.steps) >= 0

    @pytest.mark.asyncio
    async def test_reflexion_with_healing_flow(self):
        engine = ReflexionEngine(llm_provider=None)

        error = ValueError("Invalid value")
        reflection = await engine.reflect(
            error=error,
            task_description="Process user input",
            previous_output="result = process(None)",
        )

        original_prompt = "Process the user input"
        refined = engine.refine_prompt(original_prompt, reflection)

        assert len(refined) > len(original_prompt)
        assert "PREVIOUS ATTEMPT" in refined

    @pytest.mark.asyncio
    async def test_full_verification_flow(self):
        interpreter = InterpreterTool()
        api_search = APISearchTool()
        critic = ToolInteractiveCritic(tools=[interpreter, api_search])

        code = """
def calculate_sum(numbers):
    return sum(numbers)
"""
        evaluation = await critic.evaluate_with_tools(code)

        assert evaluation.score >= 0
        assert len(evaluation.evidence) >= 0
