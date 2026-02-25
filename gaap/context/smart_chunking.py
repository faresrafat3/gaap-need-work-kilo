"""
Smart Chunking - Semantic Code Chunking with Context
Implements: docs/evolution_plan_2026/40_CONTEXT_AUDIT_SPEC.md

Features:
- Semantic chunking preserving context
- Skeleton pattern for class context
- Function/class/method chunk types
- Context-aware boundaries
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


logger = logging.getLogger("gaap.context.smart_chunking")


class ChunkType(Enum):
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    ASYNC_FUNCTION = auto()
    ASYNC_METHOD = auto()
    IMPORT_BLOCK = auto()
    CONSTANT_BLOCK = auto()
    COMMENT_BLOCK = auto()
    UNKNOWN = auto()


@dataclass
class CodeChunk:
    chunk_id: str
    chunk_type: ChunkType
    content: str
    file_path: str
    start_line: int
    end_line: int
    name: str = ""
    parent_class: str = ""
    parent_module: str = ""
    signature: str = ""
    docstring: str = ""
    imports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    skeleton_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chunk_type": self.chunk_type.name,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "signature": self.signature,
            "docstring": self.docstring,
            "parent_class": self.parent_class,
            "imports": self.imports,
            "dependencies": self.dependencies,
            "skeleton_context": self.skeleton_context,
        }

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


@dataclass
class ChunkingConfig:
    max_chunk_lines: int = 100
    min_chunk_lines: int = 5
    include_skeleton: bool = True
    include_imports: bool = True
    include_docstrings: bool = True
    split_methods: bool = True
    language: str = "python"

    @classmethod
    def default(cls) -> ChunkingConfig:
        return cls()

    @classmethod
    def fine_grained(cls) -> ChunkingConfig:
        return cls(max_chunk_lines=50, min_chunk_lines=3)

    @classmethod
    def coarse(cls) -> ChunkingConfig:
        return cls(max_chunk_lines=200, min_chunk_lines=10, split_methods=False)

    @classmethod
    def for_context(cls) -> ChunkingConfig:
        return cls(
            max_chunk_lines=80,
            include_skeleton=True,
            include_imports=True,
        )


class SmartChunker:
    """
    Semantic code chunker with context preservation.

    Features:
    - AST-based chunking
    - Skeleton pattern for class context
    - Import tracking
    - Dependency extraction

    Usage:
        chunker = SmartChunker()
        chunks = chunker.chunk(code, "path/to/file.py")
        for chunk in chunks:
            print(f"{chunk.name}: lines {chunk.start_line}-{chunk.end_line}")
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig.default()
        self._logger = logger
        self._chunk_counter = 0

    def chunk(
        self,
        code: str,
        file_path: str,
        include_skeleton: bool | None = None,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            self._logger.warning(f"Syntax error in {file_path}: {e}")
            return chunks

        module_name = Path(file_path).stem
        imports = self._extract_imports(tree)

        if self.config.include_imports and imports:
            import_chunk = self._create_import_chunk(code, imports, file_path, module_name)
            if import_chunk:
                chunks.append(import_chunk)

        class_contexts: dict[str, ast.ClassDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_contexts[node.name] = node

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_chunks = self._chunk_class(
                    node, code, file_path, module_name, imports, class_contexts
                )
                chunks.extend(class_chunks)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                chunk = self._create_function_chunk(node, code, file_path, module_name, "", imports)
                if chunk:
                    chunks.append(chunk)

        if include_skeleton is None:
            include_skeleton = self.config.include_skeleton

        if include_skeleton:
            for chunk in chunks:
                if chunk.parent_class:
                    chunk.skeleton_context = self._create_skeleton(
                        class_contexts.get(chunk.parent_class),
                        chunk.name,
                        code,
                    )

        return chunks

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        imports: list[str] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)

        return imports

    def _create_import_chunk(
        self,
        code: str,
        imports: list[str],
        file_path: str,
        module_name: str,
    ) -> CodeChunk | None:
        lines = code.splitlines()
        import_lines = []
        start_line: int = 0
        end_line: int = 0

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_lines.append(line)
                if start_line == 0:
                    start_line = i
                else:
                    start_line = min(start_line, i)
                end_line = max(end_line, i)
            elif import_lines and stripped and not stripped.startswith("#"):
                break

        if not import_lines:
            return None

        self._chunk_counter += 1

        return CodeChunk(
            chunk_id=f"chunk_{self._chunk_counter}",
            chunk_type=ChunkType.IMPORT_BLOCK,
            content="\n".join(import_lines),
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            name="imports",
            parent_module=module_name,
            imports=imports,
        )

    def _chunk_class(
        self,
        node: ast.ClassDef,
        code: str,
        file_path: str,
        module_name: str,
        imports: list[str],
        class_contexts: dict[str, ast.ClassDef],
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []

        lines = code.splitlines()

        class_start = node.lineno
        class_end = node.end_lineno or class_start
        class_content = "\n".join(lines[class_start - 1 : class_end])

        docstring = ast.get_docstring(node) or ""

        self._chunk_counter += 1
        class_chunk = CodeChunk(
            chunk_id=f"chunk_{self._chunk_counter}",
            chunk_type=ChunkType.CLASS,
            content=class_content,
            file_path=file_path,
            start_line=class_start,
            end_line=class_end,
            name=node.name,
            parent_module=module_name,
            signature=f"class {node.name}",
            docstring=docstring,
            imports=imports,
        )
        chunks.append(class_chunk)

        if self.config.split_methods:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = self._create_function_chunk(
                        item, code, file_path, module_name, node.name, imports
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

        return chunks

    def _create_function_chunk(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        code: str,
        file_path: str,
        module_name: str,
        parent_class: str,
        imports: list[str],
    ) -> CodeChunk | None:
        lines = code.splitlines()

        start_line = node.lineno
        end_line = node.end_lineno or start_line

        if self.config.max_chunk_lines > 0:
            chunk_lines = end_line - start_line + 1
            if chunk_lines > self.config.max_chunk_lines:
                self._logger.debug(
                    f"Chunk {node.name} exceeds max lines ({chunk_lines} > {self.config.max_chunk_lines})"
                )

        content = "\n".join(lines[start_line - 1 : end_line])

        is_async = isinstance(node, ast.AsyncFunctionDef)
        is_method = bool(parent_class)

        if is_method:
            chunk_type = ChunkType.ASYNC_METHOD if is_async else ChunkType.METHOD
        else:
            chunk_type = ChunkType.ASYNC_FUNCTION if is_async else ChunkType.FUNCTION

        signature = self._get_signature(node, is_async)
        docstring = ast.get_docstring(node) or ""
        dependencies = self._extract_dependencies(node)

        self._chunk_counter += 1

        return CodeChunk(
            chunk_id=f"chunk_{self._chunk_counter}",
            chunk_type=chunk_type,
            content=content,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            name=node.name,
            parent_class=parent_class,
            parent_module=module_name,
            signature=signature,
            docstring=docstring,
            imports=imports,
            dependencies=dependencies,
        )

    def _get_signature(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        is_async: bool,
    ) -> str:
        args = []

        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_string(arg.annotation)}"
            args.append(arg_str)

        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        for arg in node.args.kwonlyargs:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_annotation_string(arg.annotation)}"
            args.append(arg_str)

        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        prefix = "async def" if is_async else "def"
        returns = ""
        if node.returns:
            returns = f" -> {self._get_annotation_string(node.returns)}"

        return f"{prefix} {node.name}({', '.join(args)}){returns}"

    def _get_annotation_string(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            value = self._get_annotation_string(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_str = self._get_annotation_string(node.slice)
            return f"{value}[{slice_str}]"
        elif isinstance(node, ast.Tuple):
            elements = [self._get_annotation_string(e) for e in node.elts]
            return ", ".join(elements)
        return ""

    def _extract_dependencies(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> list[str]:
        dependencies: set[str] = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name):
                    dependencies.add(f"{child.value.id}.{child.attr}")

        builtins = {
            "print",
            "len",
            "range",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "tuple",
            "bool",
            "None",
            "True",
            "False",
            "if",
            "else",
            "for",
            "while",
            "return",
            "raise",
            "try",
            "except",
            "with",
        }

        return [d for d in dependencies if d not in builtins and not d.startswith("_")]

    def _create_skeleton(
        self,
        class_node: ast.ClassDef | None,
        highlight_method: str,
        code: str,
    ) -> str:
        if not class_node:
            return ""

        lines = [f"class {class_node.name}:"]

        for item in class_node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name == highlight_method:
                    lines.append(f"    {self._get_signature(item, False)}:")
                    lines.append("        # ... method body ...")
                else:
                    lines.append(f"    # {item.name}(...)")
            elif isinstance(item, ast.AsyncFunctionDef):
                if item.name == highlight_method:
                    lines.append(f"    {self._get_signature(item, True)}:")
                    lines.append("        # ... method body ...")
                else:
                    lines.append(f"    # async {item.name}(...)")
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        lines.append(f"    {target.id} = ...")

        return "\n".join(lines)

    def chunk_file(self, file_path: str | Path) -> list[CodeChunk]:
        path = Path(file_path)
        if not path.exists():
            self._logger.error(f"File not found: {file_path}")
            return []

        code = path.read_text(encoding="utf-8")
        return self.chunk(code, str(path))

    def get_stats(self) -> dict[str, Any]:
        return {
            "config": {
                "max_chunk_lines": self.config.max_chunk_lines,
                "min_chunk_lines": self.config.min_chunk_lines,
                "include_skeleton": self.config.include_skeleton,
            },
            "chunks_created": self._chunk_counter,
        }


def create_chunker(
    max_lines: int = 100,
    include_skeleton: bool = True,
) -> SmartChunker:
    config = ChunkingConfig(
        max_chunk_lines=max_lines,
        include_skeleton=include_skeleton,
    )
    return SmartChunker(config)
