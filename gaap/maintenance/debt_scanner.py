"""
Debt Scanner - Technical Debt Detection Engine
==============================================

Scans codebase for various types of technical debt:
- TODO/FIXME/HACK markers
- Cyclomatic complexity
- Duplicate code
- Dead code
- Long functions

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md
"""

import ast
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.maintenance.debt_config import DebtConfig, DebtPriority, DebtType

logger = logging.getLogger("gaap.maintenance.scanner")


@dataclass
class DebtItem:
    """Represents a single technical debt item."""

    id: str
    type: DebtType
    file_path: str
    line_number: int
    end_line: int
    message: str
    priority: DebtPriority = DebtPriority.MEDIUM
    complexity: int | None = None
    function_name: str | None = None
    snippet: str | None = None
    interest_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "end_line": self.end_line,
            "message": self.message,
            "priority": self.priority.name,
            "complexity": self.complexity,
            "function_name": self.function_name,
            "snippet": self.snippet,
            "interest_score": self.interest_score,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DebtItem":
        return cls(
            id=data["id"],
            type=DebtType[data["type"]],
            file_path=data["file_path"],
            line_number=data["line_number"],
            end_line=data.get("end_line", data["line_number"]),
            message=data["message"],
            priority=DebtPriority[data.get("priority", "MEDIUM")],
            complexity=data.get("complexity"),
            function_name=data.get("function_name"),
            snippet=data.get("snippet"),
            interest_score=data.get("interest_score", 0.0),
            metadata=data.get("metadata", {}),
            discovered_at=(
                datetime.fromisoformat(data["discovered_at"])
                if "discovered_at" in data
                else datetime.now()
            ),
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DebtItem):
            return False
        return self.id == other.id


@dataclass
class ScanResult:
    """Result of a debt scan operation."""

    scanned_files: int = 0
    total_debt_items: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_priority: dict[str, int] = field(default_factory=dict)
    items: list[DebtItem] = field(default_factory=list)
    scan_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "total_debt_items": self.total_debt_items,
            "by_type": self.by_type,
            "by_priority": self.by_priority,
            "items": [item.to_dict() for item in self.items],
            "scan_time_ms": self.scan_time_ms,
            "errors": self.errors,
        }


class DebtScanner:
    """
    Scans codebase for technical debt.

    Features:
    - Marker detection (TODO, FIXME, etc.)
    - Cyclomatic complexity analysis
    - Duplicate code detection
    - Dead code detection
    - Long function detection
    """

    def __init__(self, config: DebtConfig | None = None):
        self._config = config or DebtConfig()
        self._logger = logger

        self._marker_pattern = re.compile(
            r"#\s*(" + "|".join(self._config.markers) + r")\s*(:?\s*(.+))?", re.IGNORECASE
        )

    @property
    def config(self) -> DebtConfig:
        return self._config

    def scan_directory(
        self,
        path: Path | str,
        recursive: bool = True,
    ) -> ScanResult:
        """
        Scan a directory for technical debt.

        Args:
            path: Directory path to scan
            recursive: Whether to scan subdirectories

        Returns:
            ScanResult with all debt items found
        """
        import time

        start_time = time.time()
        result = ScanResult()
        root_path = Path(path)

        if not root_path.exists():
            result.errors.append(f"Path does not exist: {path}")
            return result

        patterns = ["**/*.py"] if recursive else ["*.py"]

        for pattern in patterns:
            for file_path in root_path.glob(pattern):
                if self._should_exclude(file_path):
                    continue

                try:
                    items = self.scan_file(file_path)
                    result.items.extend(items)
                    result.scanned_files += 1
                except Exception as e:
                    result.errors.append(f"Error scanning {file_path}: {str(e)}")

        for item in result.items:
            type_name = item.type.name
            priority_name = item.priority.name
            result.by_type[type_name] = result.by_type.get(type_name, 0) + 1
            result.by_priority[priority_name] = result.by_priority.get(priority_name, 0) + 1

        result.total_debt_items = len(result.items)
        result.scan_time_ms = (time.time() - start_time) * 1000

        self._logger.info(
            f"Scanned {result.scanned_files} files, found {result.total_debt_items} debt items"
        )

        return result

    def scan_file(self, file_path: Path | str) -> list[DebtItem]:
        """
        Scan a single file for technical debt.

        Args:
            file_path: File path to scan

        Returns:
            List of DebtItem found in the file
        """
        items: list[DebtItem] = []
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            return items

        try:
            content = path.read_text(encoding="utf-8")
            lines = content.splitlines()
        except Exception:
            return items

        items.extend(self._scan_markers(path, content))
        items.extend(self._scan_complexity(path, content))
        items.extend(self._scan_long_functions(path, content))
        items.extend(self._scan_dead_code(path, content, lines))

        return items

    def scan_duplicates(
        self,
        files: list[Path],
        min_lines: int | None = None,
    ) -> list[DebtItem]:
        """
        Detect duplicate code blocks across files.

        Args:
            files: List of files to compare
            min_lines: Minimum lines for duplicate detection

        Returns:
            List of DebtItem for duplicate code
        """
        min_lines = min_lines or self._config.duplicate_min_lines
        items: list[DebtItem] = []

        code_blocks: dict[str, tuple[Path, int, int, str]] = {}

        for file_path in files:
            if self._should_exclude(file_path):
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                lines = content.splitlines()

                for i in range(len(lines) - min_lines + 1):
                    block = "\n".join(lines[i : i + min_lines])
                    normalized = self._normalize_code(block)
                    block_hash = hashlib.md5(normalized.encode()).hexdigest()

                    if block_hash in code_blocks:
                        original = code_blocks[block_hash]
                        items.append(
                            DebtItem(
                                id=f"dup-{block_hash[:8]}-{file_path.stem}-{i}",
                                type=DebtType.DUPLICATE,
                                file_path=str(file_path),
                                line_number=i + 1,
                                end_line=i + min_lines,
                                message=f"Duplicate of {original[0]}:{original[1]}",
                                priority=DebtPriority.MEDIUM,
                                snippet=block[:200],
                            )
                        )
                    else:
                        code_blocks[block_hash] = (file_path, i + 1, min_lines, block)
            except Exception as e:
                self._logger.debug(f"Error scanning duplicates in {file_path}: {e}")

        return items

    def _scan_markers(self, file_path: Path, content: str) -> list[DebtItem]:
        """Scan for TODO/FIXME/etc markers."""
        items: list[DebtItem] = []
        lines = content.splitlines()

        for i, line in enumerate(lines):
            match = self._marker_pattern.search(line)
            if match:
                marker = match.group(1).upper()
                message = match.group(3) if match.group(3) else ""
                message = message.strip() if message else f"{marker} found"

                priority = self._config.marker_priorities.get(marker, DebtPriority.MEDIUM)

                items.append(
                    DebtItem(
                        id=self._generate_id(
                            file_path,
                            i + 1,
                            DebtType[marker] if marker in DebtType.__members__ else DebtType.TODO,
                        ),
                        type=DebtType[marker] if marker in DebtType.__members__ else DebtType.TODO,
                        file_path=str(file_path),
                        line_number=i + 1,
                        end_line=i + 1,
                        message=message,
                        priority=priority,
                        snippet=line.strip()[:100],
                    )
                )

        return items

    def _scan_complexity(self, file_path: Path, content: str) -> list[DebtItem]:
        """Scan for high cyclomatic complexity."""
        items: list[DebtItem] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return items

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                complexity = self._calculate_complexity(node)

                if complexity >= self._config.complexity_warning:
                    priority = (
                        DebtPriority.CRITICAL
                        if complexity >= self._config.complexity_critical
                        else DebtPriority.HIGH
                    )

                    items.append(
                        DebtItem(
                            id=self._generate_id(file_path, node.lineno, DebtType.COMPLEXITY),
                            type=DebtType.COMPLEXITY,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            message=f"Function '{node.name}' has complexity {complexity}",
                            priority=priority,
                            complexity=complexity,
                            function_name=node.name,
                        )
                    )

        return items

    def _scan_long_functions(self, file_path: Path, content: str) -> list[DebtItem]:
        """Scan for excessively long functions."""
        items: list[DebtItem] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return items

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno is None:
                    continue

                lines = node.end_lineno - node.lineno + 1

                if lines >= self._config.long_function_lines:
                    priority = (
                        DebtPriority.HIGH
                        if lines >= self._config.long_function_warning
                        else DebtPriority.MEDIUM
                    )

                    items.append(
                        DebtItem(
                            id=self._generate_id(file_path, node.lineno, DebtType.LONG_FUNCTION),
                            type=DebtType.LONG_FUNCTION,
                            file_path=str(file_path),
                            line_number=node.lineno,
                            end_line=node.end_lineno,
                            message=f"Function '{node.name}' is {lines} lines long",
                            priority=priority,
                            function_name=node.name,
                            metadata={"lines": lines},
                        )
                    )

        return items

    def _scan_dead_code(
        self,
        file_path: Path,
        content: str,
        lines: list[str],
    ) -> list[DebtItem]:
        """Scan for potentially dead code."""
        items: list[DebtItem] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return items

        defined_functions: set[str] = set()
        used_names: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined_functions.add(node.name)
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        unused = defined_functions - used_names

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in unused:
                if node.name.startswith("_"):
                    continue

                items.append(
                    DebtItem(
                        id=self._generate_id(file_path, node.lineno, DebtType.DEAD_CODE),
                        type=DebtType.DEAD_CODE,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        message=f"Function '{node.name}' appears to be unused",
                        priority=DebtPriority.LOW,
                        function_name=node.name,
                    )
                )

        return items

    def _calculate_complexity(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
                if child.ifs:
                    complexity += len(child.ifs)

        return complexity

    def _normalize_code(self, code: str) -> str:
        """Normalize code for duplicate detection."""
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "", code, flags=re.DOTALL)
        code = re.sub(r"\s+", " ", code)
        return code.strip()

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded from scanning."""
        path_str = str(path)
        for pattern in self._config.exclude_patterns:
            pattern_regex = pattern.replace("**", ".*").replace("*", "[^/]*")
            if re.search(pattern_regex, path_str):
                return True
        return False

    def _generate_id(self, file_path: Path, line: int, debt_type: DebtType) -> str:
        """Generate unique ID for a debt item."""
        content = f"{file_path}:{line}:{debt_type.name}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{debt_type.name.lower()}-{file_path.stem}-{line}-{hash_part}"


def create_scanner(config: DebtConfig | None = None) -> DebtScanner:
    """Create a DebtScanner instance."""
    return DebtScanner(config=config)
