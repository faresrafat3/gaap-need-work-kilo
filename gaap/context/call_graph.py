"""
Call Graph - Cross-File Dependency Analysis
Implements: docs/evolution_plan_2026/40_CONTEXT_AUDIT_SPEC.md

Features:
- NetworkX-based call graph construction
- Upstream callers detection
- Downstream dependencies detection
- 1-hop retrieval
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger("gaap.context.call_graph")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None
    logger.warning("NetworkX not installed. Install with: pip install networkx")


@dataclass
class CallGraphNode:
    node_id: str
    name: str
    node_type: str  # function, method, class, module
    file_path: str
    line: int
    signature: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "name": self.name,
            "type": self.node_type,
            "file": self.file_path,
            "line": self.line,
            "signature": self.signature,
        }


@dataclass
class CallGraphEdge:
    source_id: str
    target_id: str
    edge_type: str  # calls, imports, inherits, contains
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type,
            "line": self.line,
        }


@dataclass
class CallGraphConfig:
    include_imports: bool = True
    include_inheritance: bool = True
    include_nested_calls: bool = True
    max_depth: int = 3
    exclude_private: bool = True
    exclude_tests: bool = True

    @classmethod
    def default(cls) -> CallGraphConfig:
        return cls()

    @classmethod
    def deep(cls) -> CallGraphConfig:
        return cls(max_depth=5, include_nested_calls=True)

    @classmethod
    def shallow(cls) -> CallGraphConfig:
        return cls(max_depth=1, include_nested_calls=False)


class CallGraph:
    """
    Cross-file call graph builder and analyzer.

    Features:
    - Build call graph from project
    - Find upstream callers
    - Find downstream dependencies
    - 1-hop retrieval

    Usage:
        graph = CallGraph()
        graph.build_from_project("src/")

        callers = graph.get_callers("my_function")
        callees = graph.get_callees("my_function")
    """

    def __init__(self, config: CallGraphConfig | None = None) -> None:
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for CallGraph. Install with: pip install networkx"
            )

        self.config = config or CallGraphConfig.default()
        self._graph: Any = nx.DiGraph() if nx else None
        self._nodes: dict[str, CallGraphNode] = {}
        self._logger = logger

    @property
    def graph(self) -> Any:
        return self._graph

    def build_from_project(self, project_path: str | Path) -> int:
        path = Path(project_path)
        if not path.exists():
            self._logger.error(f"Project path not found: {project_path}")
            return 0

        files_processed = 0

        for py_file in path.rglob("*.py"):
            if self.config.exclude_tests and "test" in py_file.name.lower():
                continue

            try:
                self._process_file(py_file)
                files_processed += 1
            except Exception as e:
                self._logger.warning(f"Error processing {py_file}: {e}")

        self._logger.info(
            f"Built call graph from {files_processed} files, {len(self._nodes)} nodes"
        )
        return files_processed

    def build_from_files(self, files: list[str | Path]) -> int:
        files_processed = 0

        for file_path in files:
            path = Path(file_path)
            if path.exists() and path.suffix == ".py":
                try:
                    self._process_file(path)
                    files_processed += 1
                except Exception as e:
                    self._logger.warning(f"Error processing {path}: {e}")

        return files_processed

    def build_from_code(self, code: str, file_path: str = "<string>") -> int:
        try:
            tree = ast.parse(code)
            self._process_tree(tree, file_path, code)
            return 1
        except SyntaxError as e:
            self._logger.warning(f"Syntax error in {file_path}: {e}")
            return 0

    def _process_file(self, file_path: Path) -> None:
        code = file_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(code)
            self._process_tree(tree, str(file_path), code)
        except SyntaxError as e:
            self._logger.warning(f"Syntax error in {file_path}: {e}")

    def _process_tree(self, tree: ast.AST, file_path: str, code: str) -> None:
        current_class = ""
        current_function = ""

        module_id = f"module:{file_path}"
        self._add_node(
            CallGraphNode(
                node_id=module_id,
                name=Path(file_path).stem,
                node_type="module",
                file_path=file_path,
                line=1,
            )
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                current_class = node.name
                class_id = f"class:{file_path}:{node.name}"

                self._add_node(
                    CallGraphNode(
                        node_id=class_id,
                        name=node.name,
                        node_type="class",
                        file_path=file_path,
                        line=node.lineno,
                    )
                )

                self._graph.add_edge(module_id, class_id, type="contains")

                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_id = f"class:{base.id}"
                        self._graph.add_edge(class_id, base_id, type="inherits")

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if current_class:
                    func_id = f"method:{file_path}:{current_class}.{node.name}"
                    func_type = "method"
                    parent_id = f"class:{file_path}:{current_class}"
                else:
                    func_id = f"function:{file_path}:{node.name}"
                    func_type = "function"
                    parent_id = module_id

                self._add_node(
                    CallGraphNode(
                        node_id=func_id,
                        name=node.name,
                        node_type=func_type,
                        file_path=file_path,
                        line=node.lineno,
                    )
                )

                self._graph.add_edge(parent_id, func_id, type="contains")

                self._extract_calls(node, func_id, file_path)

        if self.config.include_imports:
            self._extract_imports(tree, module_id, file_path)

    def _extract_calls(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        caller_id: str,
        file_path: str,
    ) -> None:
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                callee_name = self._get_call_name(node)

                if callee_name:
                    if "." in callee_name:
                        callee_id = f"method:{callee_name}"
                    else:
                        callee_id = f"function:{file_path}:{callee_name}"

                    if not self._graph.has_node(callee_id):
                        self._add_node(
                            CallGraphNode(
                                node_id=callee_id,
                                name=callee_name,
                                node_type="unknown",
                                file_path="",
                                line=0,
                            )
                        )

                    self._graph.add_edge(
                        caller_id,
                        callee_id,
                        type="calls",
                        line=node.lineno,
                    )

    def _extract_imports(
        self,
        tree: ast.AST,
        module_id: str,
        file_path: str,
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_id = f"module:{alias.name}"
                    self._graph.add_edge(module_id, import_id, type="imports")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    import_id = f"module:{node.module}"
                    self._graph.add_edge(module_id, import_id, type="imports")

    def _get_call_name(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current: ast.expr = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return ""

    def _add_node(self, node: CallGraphNode) -> None:
        self._nodes[node.node_id] = node
        self._graph.add_node(node.node_id, **node.to_dict())

    def get_callers(self, name: str, depth: int = 1) -> list[CallGraphNode]:
        node_id = self._find_node_id(name)
        if not node_id:
            return []

        callers: list[CallGraphNode] = []
        visited = {node_id}

        def _find_callers(current_id: str, current_depth: int) -> None:
            if current_depth > depth:
                return

            for predecessor in self._graph.predecessors(current_id):
                if predecessor not in visited:
                    visited.add(predecessor)
                    edge_data = self._graph.get_edge_data(predecessor, current_id)
                    if edge_data and edge_data.get("type") == "calls":
                        if predecessor in self._nodes:
                            callers.append(self._nodes[predecessor])
                    _find_callers(predecessor, current_depth + 1)

        _find_callers(node_id, 1)
        return callers

    def get_callees(self, name: str, depth: int = 1) -> list[CallGraphNode]:
        node_id = self._find_node_id(name)
        if not node_id:
            return []

        callees: list[CallGraphNode] = []
        visited = {node_id}

        def _find_callees(current_id: str, current_depth: int) -> None:
            if current_depth > depth:
                return

            for successor in self._graph.successors(current_id):
                if successor not in visited:
                    visited.add(successor)
                    edge_data = self._graph.get_edge_data(current_id, successor)
                    if edge_data and edge_data.get("type") == "calls":
                        if successor in self._nodes:
                            callees.append(self._nodes[successor])
                    _find_callees(successor, current_depth + 1)

        _find_callees(node_id, 1)
        return callees

    def _find_node_id(self, name: str) -> str | None:
        for node_id, node in self._nodes.items():
            if node.name == name:
                return node_id
        return None

    def get_node(self, name: str) -> CallGraphNode | None:
        node_id = self._find_node_id(name)
        if node_id:
            return self._nodes.get(node_id)
        return None

    def get_all_nodes(self) -> list[CallGraphNode]:
        return list(self._nodes.values())

    def get_stats(self) -> dict[str, Any]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "functions": sum(1 for n in self._nodes.values() if n.node_type == "function"),
            "methods": sum(1 for n in self._nodes.values() if n.node_type == "method"),
            "classes": sum(1 for n in self._nodes.values() if n.node_type == "class"),
        }

    def export_dot(self) -> str:
        if nx:
            return nx.nx_pydot.to_pydot(self._graph).to_string()  # type: ignore[no-any-return]
        return ""

    def export_json(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **self._graph.get_edge_data(u, v),
                }
                for u, v in self._graph.edges()
            ],
        }


def create_call_graph(
    include_imports: bool = True,
    max_depth: int = 3,
) -> CallGraph:
    config = CallGraphConfig(
        include_imports=include_imports,
        max_depth=max_depth,
    )
    return CallGraph(config)
