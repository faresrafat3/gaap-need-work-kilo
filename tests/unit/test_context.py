"""
Tests for GAAP Context Module
"""

import pytest

from gaap.context.call_graph import (
    CallGraph,
    CallGraphConfig,
    CallGraphEdge,
    CallGraphNode,
)


class TestCallGraphConfig:
    def test_defaults(self):
        config = CallGraphConfig()
        assert config.include_imports is True
        assert config.include_inheritance is True
        assert config.max_depth == 3
        assert config.exclude_private is True

    def test_default_classmethod(self):
        config = CallGraphConfig.default()
        assert config.max_depth == 3

    def test_deep_classmethod(self):
        config = CallGraphConfig.deep()
        assert config.max_depth == 5

    def test_shallow_classmethod(self):
        config = CallGraphConfig.shallow()
        assert config.max_depth == 1
        assert config.include_nested_calls is False


class TestCallGraphNode:
    def test_defaults(self):
        node = CallGraphNode(
            node_id="test_id",
            name="test_func",
            node_type="function",
            file_path="/path/to/file.py",
            line=10,
        )
        assert node.node_id == "test_id"
        assert node.name == "test_func"
        assert node.node_type == "function"
        assert node.line == 10

    def test_to_dict(self):
        node = CallGraphNode(
            node_id="test_id",
            name="test_func",
            node_type="function",
            file_path="/path/to/file.py",
            line=10,
        )
        d = node.to_dict()
        assert d["id"] == "test_id"
        assert d["name"] == "test_func"
        assert d["type"] == "function"
        assert d["file"] == "/path/to/file.py"
        assert d["line"] == 10


class TestCallGraphEdge:
    def test_defaults(self):
        edge = CallGraphEdge(source_id="caller", target_id="callee", edge_type="calls")
        assert edge.source_id == "caller"
        assert edge.target_id == "callee"
        assert edge.edge_type == "calls"

    def test_to_dict(self):
        edge = CallGraphEdge(source_id="caller", target_id="callee", edge_type="calls", line=5)
        d = edge.to_dict()
        assert d["source"] == "caller"
        assert d["target"] == "callee"
        assert d["type"] == "calls"
        assert d["line"] == 5


class TestCallGraph:
    @pytest.fixture(autouse=True)
    def check_networkx(self):
        pytest.importorskip("networkx")

    def test_init_with_default_config(self):
        graph = CallGraph()
        assert graph.config is not None
        assert isinstance(graph.config, CallGraphConfig)

    def test_build_from_code(self):
        graph = CallGraph()
        code = """
def foo():
    return 1

def bar():
    return foo()
"""
        result = graph.build_from_code(code)
        assert result == 1
        assert len(graph._nodes) >= 2

    def test_build_from_code_with_class(self):
        graph = CallGraph()
        code = """
class MyClass:
    def method(self):
        return 42
"""
        result = graph.build_from_code(code)
        assert result == 1

    def test_build_from_code_syntax_error(self):
        graph = CallGraph()
        code = "def invalid syntax here"
        result = graph.build_from_code(code)
        assert result == 0

    def test_get_node(self):
        graph = CallGraph()
        code = "def foo(): pass"
        graph.build_from_code(code)

        nodes = list(graph._nodes.values())
        assert len(nodes) > 0

    def test_add_node(self):
        graph = CallGraph()
        node = CallGraphNode(
            node_id="test_node",
            name="test",
            node_type="function",
            file_path="/test.py",
            line=1,
        )
        graph._add_node(node)
        assert "test_node" in graph._nodes
