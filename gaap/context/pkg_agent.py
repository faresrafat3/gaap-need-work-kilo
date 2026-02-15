# Package Agent
import ast
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================

class NodeType(Enum):
    """أنواع العقد"""
    PROJECT = "project"
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    VARIABLE = "variable"
    IMPORT = "import"
    FILE = "file"


class EdgeType(Enum):
    """أنواع الحواف"""
    CONTAINS = "contains"
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GraphNode:
    """عقدة في الرسم البياني"""
    id: str
    name: str
    node_type: NodeType
    file_path: str = ""
    line_number: int = 0
    signature: str = ""
    docstring: str = ""
    summary: str = ""
    importance: float = 0.0
    connections: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.node_type.value,
            "file": self.file_path,
            "line": self.line_number,
            "signature": self.signature,
            "summary": self.summary,
            "importance": self.importance,
        }


@dataclass
class GraphEdge:
    """حافة في الرسم البياني"""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Knowledge Graph
# =============================================================================

class KnowledgeGraph:
    """الرسم البياني المعرفي"""

    def __init__(self):
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        self._node_index: dict[str, set[str]] = {}  # type -> node_ids
        self._adjacency: dict[str, set[str]] = {}   # node_id -> connected_ids

    def add_node(self, node: GraphNode) -> None:
        """إضافة عقدة"""
        self.nodes[node.id] = node

        # فهرسة بالنوع
        if node.node_type.value not in self._node_index:
            self._node_index[node.node_type.value] = set()
        self._node_index[node.node_type.value].add(node.id)

        # تهيئة قائمة الجوار
        if node.id not in self._adjacency:
            self._adjacency[node.id] = set()

    def add_edge(self, edge: GraphEdge) -> None:
        """إضافة حافة"""
        self.edges.append(edge)

        # تحديث قائمة الجوار
        if edge.source_id in self._adjacency:
            self._adjacency[edge.source_id].add(edge.target_id)

        # تحديث عدد الاتصالات
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].connections += 1
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].connections += 1

    def get_node(self, node_id: str) -> GraphNode | None:
        """الحصول على عقدة"""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """الحصول على عقد بالنوع"""
        ids = self._node_index.get(node_type.value, set())
        return [self.nodes[nid] for nid in ids if nid in self.nodes]

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[GraphNode]:
        """الحصول على الجيران"""
        if depth <= 0:
            return []

        neighbors = []
        connected_ids = self._adjacency.get(node_id, set())

        for nid in connected_ids:
            if nid in self.nodes:
                neighbors.append(self.nodes[nid])
                # استدعاء متكرر للعمق الأكبر
                if depth > 1:
                    neighbors.extend(self.get_neighbors(nid, depth - 1))

        return neighbors

    def get_most_important(self, limit: int = 20) -> list[GraphNode]:
        """الحصول على أهم العقد"""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (n.importance, n.connections),
            reverse=True
        )
        return sorted_nodes[:limit]

    def search(self, query: str, limit: int = 10) -> list[GraphNode]:
        """بحث في العقد"""
        query_lower = query.lower()
        results = []

        for node in self.nodes.values():
            score = 0

            # تطابق الاسم
            if query_lower in node.name.lower():
                score += 10

            # تطابق التوقيع
            if query_lower in node.signature.lower():
                score += 5

            # تطابق الملخص
            if query_lower in node.summary.lower():
                score += 3

            # تطابق الدوكستريمج
            if query_lower in node.docstring.lower():
                score += 2

            if score > 0:
                results.append((node, score * node.importance))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:limit]]

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الرسم البياني"""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": {
                t: len(ids) for t, ids in self._node_index.items()
            },
            "avg_connections": sum(n.connections for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
        }


# =============================================================================
# PKG Agent
# =============================================================================

class PKGAgent:
    """
    وكيل رسم الخرائط المعرفية للمشروع
    
    يحول المشروع إلى رسم بياني:
    - يقلل 1B+ tokens إلى ~50k nodes
    - يحتفظ بالعلاقات والتبعيات
    - يدعم البحث والاستكشاف
    """

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.graph = KnowledgeGraph()
        self._logger = logging.getLogger("gaap.context.pkg")
        self._is_built = False

    async def build_graph(self, force: bool = False) -> KnowledgeGraph:
        """بناء الرسم البياني"""
        if self._is_built and not force:
            return self.graph

        self._logger.info(f"Building knowledge graph for: {self.project_path}")

        # إضافة عقدة المشروع
        project_node = GraphNode(
            id="project_root",
            name=os.path.basename(self.project_path),
            node_type=NodeType.PROJECT,
            importance=1.0
        )
        self.graph.add_node(project_node)

        # تحليل الملفات
        for root, dirs, files in os.walk(self.project_path):
            # تجاهل المجلدات المخفية
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]

            for file in files:
                if file.startswith('.'):
                    continue

                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                # تحليل حسب نوع الملف
                if ext == '.py':
                    await self._analyze_python_file(file_path)
                elif ext in ('.js', '.ts', '.jsx', '.tsx'):
                    await self._analyze_js_file(file_path)
                else:
                    await self._analyze_generic_file(file_path)

        # حساب الأهمية
        self._calculate_importance()

        self._is_built = True

        self._logger.info(
            f"Graph built: {len(self.graph.nodes)} nodes, "
            f"{len(self.graph.edges)} edges"
        )

        return self.graph

    async def _analyze_python_file(self, file_path: str) -> None:
        """تحليل ملف Python"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # إضافة عقدة الملف
            file_node = GraphNode(
                id=f"file:{file_path}",
                name=os.path.basename(file_path),
                node_type=NodeType.FILE,
                file_path=file_path,
                summary=self._generate_file_summary(content)
            )
            self.graph.add_node(file_node)

            self.graph.add_edge(GraphEdge(
                source_id="project_root",
                target_id=file_node.id,
                edge_type=EdgeType.CONTAINS
            ))

            # تحليل AST
            try:
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        await self._add_class_node(node, file_path)
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        await self._add_function_node(node, file_path)
                    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                        await self._add_import_node(node, file_path)

            except SyntaxError:
                self._logger.debug(f"Syntax error in {file_path}")

        except Exception as e:
            self._logger.debug(f"Could not analyze {file_path}: {e}")

    async def _add_class_node(self, node: ast.ClassDef, file_path: str) -> None:
        """إضافة عقدة Class"""
        class_id = f"class:{file_path}:{node.name}"

        # استخراج التوثيق
        docstring = ast.get_docstring(node) or ""

        # استخراج التوقيع
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        signature = f"class {node.name}({', '.join(methods[:5])}{'...' if len(methods) > 5 else ''})"

        class_node = GraphNode(
            id=class_id,
            name=node.name,
            node_type=NodeType.CLASS,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            docstring=docstring,
            summary=self._generate_summary(docstring, node.name),
            metadata={"methods": methods}
        )

        self.graph.add_node(class_node)

        self.graph.add_edge(GraphEdge(
            source_id=f"file:{file_path}",
            target_id=class_id,
            edge_type=EdgeType.CONTAINS
        ))

        # الوراثة
        for base in node.bases:
            if isinstance(base, ast.Name):
                self.graph.add_edge(GraphEdge(
                    source_id=class_id,
                    target_id=f"class:{base.id}",
                    edge_type=EdgeType.INHERITS,
                    weight=0.5
                ))

    async def _add_function_node(self, node, file_path: str) -> None:
        """إضافة عقدة Function"""
        func_id = f"func:{file_path}:{node.name}:{node.lineno}"

        # استخراج التوثيق
        docstring = ast.get_docstring(node) or ""

        # استخراج التوقيع
        args = [a.arg for a in node.args.args]
        signature = f"def {node.name}({', '.join(args)})"

        func_node = GraphNode(
            id=func_id,
            name=node.name,
            node_type=NodeType.FUNCTION,
            file_path=file_path,
            line_number=node.lineno,
            signature=signature,
            docstring=docstring,
            summary=self._generate_summary(docstring, node.name),
            metadata={"args": args}
        )

        self.graph.add_node(func_node)

        self.graph.add_edge(GraphEdge(
            source_id=f"file:{file_path}",
            target_id=func_id,
            edge_type=EdgeType.CONTAINS
        ))

    async def _add_import_node(self, node, file_path: str) -> None:
        """إضافة عقدة Import"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_id = f"import:{alias.name}"
                import_node = GraphNode(
                    id=import_id,
                    name=alias.name,
                    node_type=NodeType.IMPORT,
                    file_path=file_path,
                    line_number=node.lineno
                )
                self.graph.add_node(import_node)

                self.graph.add_edge(GraphEdge(
                    source_id=f"file:{file_path}",
                    target_id=import_id,
                    edge_type=EdgeType.IMPORTS
                ))

    async def _analyze_js_file(self, file_path: str) -> None:
        """تحليل ملف JavaScript/TypeScript"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # إضافة عقدة الملف
            file_node = GraphNode(
                id=f"file:{file_path}",
                name=os.path.basename(file_path),
                node_type=NodeType.FILE,
                file_path=file_path,
                summary=self._generate_file_summary(content)
            )
            self.graph.add_node(file_node)

            # استخراج الدوال بـ Regex (مبسط)
            func_pattern = r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()'
            for match in re.finditer(func_pattern, content):
                func_name = match.group(1) or match.group(2)
                if func_name:
                    func_node = GraphNode(
                        id=f"func:{file_path}:{func_name}",
                        name=func_name,
                        node_type=NodeType.FUNCTION,
                        file_path=file_path,
                        line_number=content[:match.start()].count('\n') + 1
                    )
                    self.graph.add_node(func_node)

                    self.graph.add_edge(GraphEdge(
                        source_id=f"file:{file_path}",
                        target_id=func_node.id,
                        edge_type=EdgeType.CONTAINS
                    ))

        except Exception as e:
            self._logger.debug(f"Could not analyze {file_path}: {e}")

    async def _analyze_generic_file(self, file_path: str) -> None:
        """تحليل ملف عام"""
        # فقط إضافة كعقدة ملف
        file_node = GraphNode(
            id=f"file:{file_path}",
            name=os.path.basename(file_path),
            node_type=NodeType.FILE,
            file_path=file_path
        )
        self.graph.add_node(file_node)

    def _generate_file_summary(self, content: str) -> str:
        """توليد ملخص الملف"""
        lines = content.split('\n')

        # البحث عن docstring أو تعليق أول
        for line in lines[:20]:
            line = line.strip()
            if line.startswith('"""') or line.startswith("'''"):
                # استخراج docstring
                end_idx = content.find(line[:3], 3)
                if end_idx > 0:
                    return content[3:end_idx].strip()[:200]
            elif line.startswith('#') and len(line) > 2:
                return line[2:].strip()[:200]

        return ""

    def _generate_summary(self, docstring: str, name: str) -> str:
        """توليد ملخص"""
        if docstring:
            # أول جملة
            sentences = docstring.split('.')
            return sentences[0][:100] + ('.' if len(sentences[0]) < 100 else '...')

        return name

    def _calculate_importance(self) -> None:
        """حساب أهمية العقد"""
        # PageRank-like algorithm مبسط

        # العقد ذات اتصالات أكثر أهم
        max_connections = max(
            (n.connections for n in self.graph.nodes.values()),
            default=1
        )

        for node in self.graph.nodes.values():
            # أهمية أساسية من نوع العقدة
            type_importance = {
                NodeType.PROJECT: 1.0,
                NodeType.CLASS: 0.8,
                NodeType.FUNCTION: 0.6,
                NodeType.FILE: 0.5,
                NodeType.MODULE: 0.7,
                NodeType.IMPORT: 0.3,
                NodeType.VARIABLE: 0.2,
            }

            base = type_importance.get(node.node_type, 0.5)
            connection_bonus = node.connections / max_connections if max_connections > 0 else 0

            node.importance = base * 0.6 + connection_bonus * 0.4

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def find_relevant_nodes(
        self,
        query: str,
        limit: int = 20
    ) -> list[GraphNode]:
        """العثور على عقد ذات صلة"""
        if not self._is_built:
            await self.build_graph()

        return self.graph.search(query, limit)

    async def get_context_for_node(
        self,
        node_id: str,
        depth: int = 2
    ) -> list[GraphNode]:
        """الحصول على سياق لعقدة"""
        if not self._is_built:
            await self.build_graph()

        node = self.graph.get_node(node_id)
        if not node:
            return []

        # الحصول على الجيران
        neighbors = self.graph.get_neighbors(node_id, depth)

        # إضافة العقدة نفسها
        result = [node] + neighbors

        return result

    async def get_architecture_overview(self) -> str:
        """الحصول على نظرة معمارية"""
        if not self._is_built:
            await self.build_graph()

        # أهم المكونات
        important = self.graph.get_most_important(50)

        # تصنيف حسب النوع
        classes = [n for n in important if n.node_type == NodeType.CLASS]
        functions = [n for n in important if n.node_type == NodeType.FUNCTION]
        files = [n for n in important if n.node_type == NodeType.FILE]

        overview = "# Project Architecture\n\n"
        overview += f"Total nodes: {len(self.graph.nodes)}\n"
        overview += f"Total edges: {len(self.graph.edges)}\n\n"

        overview += f"## Key Classes ({len(classes)})\n"
        for c in classes[:10]:
            overview += f"- {c.name}: {c.summary}\n"

        overview += f"\n## Key Functions ({len(functions)})\n"
        for f in functions[:10]:
            overview += f"- {f.signature}\n"

        return overview

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return self.graph.get_stats()
