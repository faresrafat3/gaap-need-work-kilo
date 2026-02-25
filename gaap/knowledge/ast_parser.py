"""
AST Parser - Source Code Analysis
==================================

Parse source files using Tree-Sitter (if available) or fallback to Python ast.

Extracts:
- Classes & Inheritance
- Functions & Signatures
- Docstrings
- Imports

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md
"""

import ast
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.knowledge.knowledge_config import KnowledgeConfig

logger = logging.getLogger("gaap.knowledge.parser")

_tree_sitter_available = False
_languages = None

try:
    import tree_sitter_python

    _tree_sitter_available = True
except ImportError:
    pass

try:
    import tree_sitter_javascript

    _tree_sitter_available = True
except ImportError:
    pass

try:
    import tree_sitter_typescript

    _tree_sitter_available = True
except ImportError:
    pass

try:
    from tree_sitter import Language, Parser
except ImportError:
    _tree_sitter_available = False


@dataclass
class Parameter:
    """Function/method parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_varargs: bool = False
    is_kwargs: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type_annotation": self.type_annotation,
            "default_value": self.default_value,
            "is_varargs": self.is_varargs,
            "is_kwargs": self.is_kwargs,
        }


@dataclass
class FunctionInfo:
    """Function or method information."""

    name: str
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    decorators: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": [p.to_dict() for p in self.parameters],
            "return_type": self.return_type,
            "docstring": self.docstring,
            "is_async": self.is_async,
            "is_method": self.is_method,
            "is_classmethod": self.is_classmethod,
            "is_staticmethod": self.is_staticmethod,
            "decorators": self.decorators,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "file_path": self.file_path,
        }

    @property
    def signature(self) -> str:
        params = ", ".join(
            f"{p.name}" + (f": {p.type_annotation}" if p.type_annotation else "")
            for p in self.parameters
        )
        ret = f" -> {self.return_type}" if self.return_type else ""
        return f"{self.name}({params}){ret}"


@dataclass
class ClassInfo:
    """Class information."""

    name: str
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None
    methods: list[FunctionInfo] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0
    file_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bases": self.bases,
            "docstring": self.docstring,
            "methods": [m.to_dict() for m in self.methods],
            "attributes": self.attributes,
            "decorators": self.decorators,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "file_path": self.file_path,
        }

    @property
    def public_methods(self) -> list[FunctionInfo]:
        return [m for m in self.methods if not m.name.startswith("_")]


@dataclass
class ImportInfo:
    """Import statement information."""

    module: str
    names: list[str] = field(default_factory=list)
    alias: str | None = None
    is_from: bool = False
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "names": self.names,
            "alias": self.alias,
            "is_from": self.is_from,
            "line": self.line,
        }


@dataclass
class ParsedFile:
    """Result of parsing a single file."""

    file_path: str
    language: str
    classes: list[ClassInfo] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    module_docstring: str | None = None
    total_lines: int = 0
    parse_time_ms: float = 0.0
    error: str | None = None
    parsed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "language": self.language,
            "classes": [c.to_dict() for c in self.classes],
            "functions": [f.to_dict() for f in self.functions],
            "imports": [i.to_dict() for i in self.imports],
            "module_docstring": self.module_docstring,
            "total_lines": self.total_lines,
            "parse_time_ms": self.parse_time_ms,
            "error": self.error,
            "parsed_at": self.parsed_at.isoformat(),
        }


class ASTParser:
    """
    Parse source files to extract structure.

    Uses Tree-Sitter if available, falls back to Python ast module.
    """

    def __init__(self, config: KnowledgeConfig | None = None):
        self._config = config or KnowledgeConfig()
        self._logger = logger

        self._parsers: dict[str, Any] = {}
        self._tree_sitter_enabled = _tree_sitter_available and self._config.use_tree_sitter

        if self._tree_sitter_enabled:
            self._init_tree_sitter()

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def parse_file(self, file_path: Path) -> ParsedFile:
        """
        Parse a single source file.

        Args:
            file_path: Path to the source file

        Returns:
            ParsedFile with extracted structure
        """
        import time

        start_time = time.time()

        language = self._detect_language(file_path)

        result = ParsedFile(
            file_path=str(file_path),
            language=language,
        )

        if language not in self._config.supported_languages:
            result.error = f"Unsupported language: {language}"
            return result

        if not file_path.exists():
            result.error = "File does not exist"
            return result

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self._config.max_file_size_mb:
            result.error = f"File too large: {file_size_mb:.1f}MB"
            return result

        try:
            content = file_path.read_text(encoding="utf-8")
            result.total_lines = len(content.splitlines())

            if language == "python":
                self._parse_python(file_path, content, result)
            elif language in ("javascript", "typescript"):
                self._parse_js_ts(file_path, content, result, language)

        except Exception as e:
            result.error = str(e)
            self._logger.warning(f"Error parsing {file_path}: {e}")

        result.parse_time_ms = (time.time() - start_time) * 1000
        return result

    def parse_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[ParsedFile]:
        """
        Parse all supported files in a directory.

        Args:
            directory: Directory to parse
            recursive: Whether to parse subdirectories

        Returns:
            List of ParsedFile results
        """
        results: list[ParsedFile] = []
        file_count = 0
        total_size = 0.0

        patterns = self._config.include_patterns

        for pattern in patterns:
            for file_path in directory.glob(pattern if recursive else pattern):
                if not file_path.is_file():
                    continue

                if self._should_exclude(file_path):
                    continue

                if file_count >= self._config.max_files_per_repo:
                    self._logger.warning(f"Max files reached: {file_count}")
                    break

                size_mb = file_path.stat().st_size / (1024 * 1024)
                if total_size + size_mb > self._config.max_total_size_mb:
                    self._logger.warning(f"Max total size reached: {total_size:.1f}MB")
                    break

                result = self.parse_file(file_path)
                results.append(result)
                file_count += 1
                total_size += size_mb

        return results

    def _parse_python(
        self,
        file_path: Path,
        content: str,
        result: ParsedFile,
    ) -> None:
        """Parse Python file using ast module."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            result.error = f"Syntax error: {e}"
            return

        result.module_docstring = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(
                        ImportInfo(
                            module=alias.name,
                            alias=alias.asname,
                            line=node.lineno,
                        )
                    )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=names,
                        is_from=True,
                        line=node.lineno,
                    )
                )

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                result.classes.append(self._extract_class(node, str(file_path)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result.functions.append(self._extract_function(node, str(file_path)))

    def _extract_class(self, node: ast.ClassDef, file_path: str) -> ClassInfo:
        """Extract class information from AST node."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(self._get_attribute_name(base))

        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))

        methods: list[FunctionInfo] = []
        attributes: list[str] = []

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func = self._extract_function(item, file_path, is_method=True)
                methods.append(func)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)

        return ClassInfo(
            name=node.name,
            bases=bases,
            docstring=ast.get_docstring(node),
            methods=methods,
            attributes=attributes,
            decorators=decorators,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            file_path=file_path,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: str,
        is_method: bool = False,
    ) -> FunctionInfo:
        """Extract function/method information from AST node."""
        parameters: list[Parameter] = []

        args = node.args

        if args.posonlyargs:
            for arg in args.posonlyargs:
                parameters.append(
                    Parameter(
                        name=arg.arg,
                        type_annotation=self._get_annotation(arg.annotation),
                    )
                )

        if is_method and args.args:
            args.args = args.args[1:]

        for arg in args.args:
            parameters.append(
                Parameter(
                    name=arg.arg,
                    type_annotation=self._get_annotation(arg.annotation),
                )
            )

        if args.vararg:
            parameters.append(
                Parameter(
                    name=args.vararg.arg,
                    type_annotation=self._get_annotation(args.vararg.annotation),
                    is_varargs=True,
                )
            )

        for arg in args.kwonlyargs:
            parameters.append(
                Parameter(
                    name=arg.arg,
                    type_annotation=self._get_annotation(arg.annotation),
                )
            )

        if args.kwarg:
            parameters.append(
                Parameter(
                    name=args.kwarg.arg,
                    type_annotation=self._get_annotation(args.kwarg.annotation),
                    is_kwargs=True,
                )
            )

        defaults = args.defaults
        if defaults:
            for i, default in enumerate(defaults):
                idx = len(parameters) - len(defaults) + i
                if idx < len(parameters):
                    parameters[idx].default_value = self._get_value(default)

        decorators = []
        is_classmethod = False
        is_staticmethod = False

        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
                if dec.id == "classmethod":
                    is_classmethod = True
                elif dec.id == "staticmethod":
                    is_staticmethod = True
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))

        return FunctionInfo(
            name=node.name,
            parameters=parameters,
            return_type=self._get_annotation(node.returns),
            docstring=ast.get_docstring(node),
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            decorators=decorators,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            file_path=file_path,
        )

    def _get_annotation(self, node: ast.AST | None) -> str | None:
        """Extract type annotation string."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Subscript):
            base = self._get_annotation(node.value)
            slice_val = self._get_annotation(node.slice)
            return f"{base}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._get_annotation(el) or "Any" for el in node.elts)
            return f"({elements})"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            left = self._get_annotation(node.left)
            right = self._get_annotation(node.right)
            return f"{left} | {right}"

        return None

    def _get_value(self, node: ast.AST) -> str:
        """Get string representation of a value."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.List):
            return "[]"
        elif isinstance(node, ast.Dict):
            return "{}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}()"
        return "..."

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get full attribute name (e.g., 'typing.Optional')."""
        parts = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def _parse_js_ts(
        self,
        file_path: Path,
        content: str,
        result: ParsedFile,
        language: str,
    ) -> None:
        """Parse JavaScript/TypeScript file."""
        if not self._tree_sitter_enabled:
            result.error = f"Tree-sitter required for {language}"
            return

        parser = self._parsers.get(language)
        if parser is None:
            result.error = f"No parser for {language}"
            return

        try:
            tree = parser.parse(bytes(content, "utf-8"))
            self._extract_js_ts_structure(tree.root_node, result)
        except Exception as e:
            result.error = str(e)

    def _extract_js_ts_structure(self, node: Any, result: ParsedFile) -> None:
        """Extract structure from JS/TS tree-sitter node."""
        pass

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".mjs": "javascript",
            ".cjs": "javascript",
        }
        return ext_to_lang.get(file_path.suffix.lower(), "unknown")

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        path_str = str(path)
        for pattern in self._config.exclude_patterns:
            pattern_regex = pattern.replace("**", ".*").replace("*", "[^/]*")
            if pattern.startswith("**") or "/" not in pattern:
                import re

                if re.search(pattern_regex, path_str):
                    return True
        return False

    def _init_tree_sitter(self) -> None:
        """Initialize tree-sitter parsers."""
        pass


def create_parser(config: KnowledgeConfig | None = None) -> ASTParser:
    """Create an ASTParser instance."""
    return ASTParser(config=config)
