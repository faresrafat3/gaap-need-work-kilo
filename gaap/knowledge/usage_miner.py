"""
Usage Miner - Extract Usage Patterns from Tests and Examples
=============================================================

Mines usage patterns from test files and example code to understand
how a library is supposed to be used.

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.knowledge.ast_parser import ASTParser
from gaap.knowledge.knowledge_config import KnowledgeConfig

logger = logging.getLogger("gaap.knowledge.miner")


@dataclass
class UsageExample:
    """A usage example extracted from tests or examples."""

    id: str
    code: str
    intent: str
    source_file: str
    source_line: int
    function_name: str | None = None
    class_name: str | None = None
    imports: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.5
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "intent": self.intent,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "imports": self.imports,
            "tags": self.tags,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageExample":
        return cls(
            id=data["id"],
            code=data["code"],
            intent=data["intent"],
            source_file=data["source_file"],
            source_line=data["source_line"],
            function_name=data.get("function_name"),
            class_name=data.get("class_name"),
            imports=data.get("imports", []),
            tags=data.get("tags", []),
            confidence=data.get("confidence", 0.5),
            extracted_at=(
                datetime.fromisoformat(data["extracted_at"])
                if "extracted_at" in data
                else datetime.now()
            ),
        )


@dataclass
class UsagePattern:
    """A common usage pattern identified across multiple examples."""

    pattern_id: str
    description: str
    examples: list[UsageExample] = field(default_factory=list)
    frequency: int = 0
    avg_confidence: float = 0.0
    function_calls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "examples": [e.to_dict() for e in self.examples],
            "frequency": self.frequency,
            "avg_confidence": self.avg_confidence,
            "function_calls": self.function_calls,
        }


@dataclass
class MiningResult:
    """Result of usage mining operation."""

    total_files: int = 0
    total_examples: int = 0
    examples: list[UsageExample] = field(default_factory=list)
    patterns: list[UsagePattern] = field(default_factory=list)
    by_function: dict[str, int] = field(default_factory=dict)
    by_class: dict[str, int] = field(default_factory=dict)
    mine_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_examples": self.total_examples,
            "examples": [e.to_dict() for e in self.examples],
            "patterns": [p.to_dict() for p in self.patterns],
            "by_function": self.by_function,
            "by_class": self.by_class,
            "mine_time_ms": self.mine_time_ms,
            "errors": self.errors,
        }


class UsageMiner:
    """
    Mines usage patterns from test files and examples.

    Features:
    - Extract code snippets from tests
    - Infer intent from test names
    - Identify common patterns
    - Build usage documentation
    """

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        parser: ASTParser | None = None,
    ):
        self._config = config or KnowledgeConfig()
        self._parser = parser or ASTParser(self._config)
        self._logger = logger

        self._intent_patterns = [
            (r"test_(\w+)", "Testing {0}"),
            (r"it_(\w+)", "Should {0}"),
            (r"describe_(\w+)", "Describes {0}"),
            (r"example_(\w+)", "Example: {0}"),
            (r"demo_(\w+)", "Demo: {0}"),
        ]

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def mine_tests(self, repo_path: Path) -> MiningResult:
        """
        Mine usage examples from test directories.

        Args:
            repo_path: Root path of the repository

        Returns:
            MiningResult with extracted examples
        """
        import time

        start_time = time.time()
        result = MiningResult()

        test_dirs: list[Path] = []
        for test_dir in self._config.test_directories:
            test_path = repo_path / test_dir
            if test_path.exists() and test_path.is_dir():
                test_dirs.append(test_path)

        if not test_dirs:
            self._logger.info(f"No test directories found in {repo_path}")
            result.mine_time_ms = (time.time() - start_time) * 1000
            return result

        for test_dir in test_dirs:  # type: ignore[assignment]
            self._mine_directory(test_dir, result, is_test=True)  # type: ignore[arg-type]

        self._identify_patterns(result)
        result.mine_time_ms = (time.time() - start_time) * 1000

        return result

    def mine_examples(self, repo_path: Path) -> MiningResult:
        """
        Mine usage examples from example directories.

        Args:
            repo_path: Root path of the repository

        Returns:
            MiningResult with extracted examples
        """
        import time

        start_time = time.time()
        result = MiningResult()

        example_dirs: list[Path] = []
        for example_dir in self._config.example_directories:
            example_path = repo_path / example_dir
            if example_path.exists() and example_path.is_dir():
                example_dirs.append(example_path)

        if not example_dirs:
            self._logger.info(f"No example directories found in {repo_path}")
            result.mine_time_ms = (time.time() - start_time) * 1000
            return result

        for example_dir in example_dirs:  # type: ignore[assignment]
            self._mine_directory(example_dir, result, is_test=False)  # type: ignore[arg-type]

        self._identify_patterns(result)
        result.mine_time_ms = (time.time() - start_time) * 1000

        return result

    def mine_all(self, repo_path: Path) -> MiningResult:
        """
        Mine from both tests and examples.

        Args:
            repo_path: Root path of the repository

        Returns:
            Combined MiningResult
        """
        import time

        start_time = time.time()

        test_result = self.mine_tests(repo_path)
        example_result = self.mine_examples(repo_path)

        combined = MiningResult()
        combined.total_files = test_result.total_files + example_result.total_files
        combined.total_examples = test_result.total_examples + example_result.total_examples
        combined.examples = test_result.examples + example_result.examples

        for key, value in test_result.by_function.items():
            combined.by_function[key] = combined.by_function.get(key, 0) + value
        for key, value in example_result.by_function.items():
            combined.by_function[key] = combined.by_function.get(key, 0) + value

        combined.by_class = {**test_result.by_class, **example_result.by_class}
        combined.errors = test_result.errors + example_result.errors

        self._identify_patterns(combined)
        combined.mine_time_ms = (time.time() - start_time) * 1000

        return combined

    def _mine_directory(
        self,
        directory: Path,
        result: MiningResult,
        is_test: bool = True,
    ) -> None:
        """Mine a single directory for examples."""
        for file_path in directory.rglob("*.py"):
            if self._should_skip(file_path):
                continue

            try:
                examples = self._extract_from_file(file_path, is_test)
                result.examples.extend(examples)
                result.total_files += 1
                result.total_examples += len(examples)

                for ex in examples:
                    if ex.function_name:
                        result.by_function[ex.function_name] = (
                            result.by_function.get(ex.function_name, 0) + 1
                        )
                    if ex.class_name:
                        result.by_class[ex.class_name] = result.by_class.get(ex.class_name, 0) + 1

            except Exception as e:
                result.errors.append(f"Error mining {file_path}: {e}")

    def _extract_from_file(
        self,
        file_path: Path,
        is_test: bool,
    ) -> list[UsageExample]:
        """Extract usage examples from a single file."""
        examples: list[UsageExample] = []

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return examples

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return examples

        file_imports = self._extract_imports(tree)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                example = self._extract_from_function(
                    node, file_path, content, file_imports, is_test
                )
                if example:
                    examples.append(example)

        return examples

    def _extract_from_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        file_path: Path,
        content: str,
        imports: list[str],
        is_test: bool,
    ) -> UsageExample | None:
        """Extract usage example from a function."""
        name = node.name

        if name.startswith("_") and not name.startswith("__"):
            return None

        intent = self._infer_intent(name)

        start_line = node.lineno
        end_line = node.end_lineno or node.lineno

        lines = content.splitlines()
        code_lines = lines[start_line - 1 : end_line]
        code = "\n".join(code_lines)

        if len(code) < 20:
            return None

        confidence = 0.9 if is_test else 0.7

        tags = []
        if is_test:
            tags.append("test")
        else:
            tags.append("example")

        if "async" in code:
            tags.append("async")
        if "await" in code:
            tags.append("await")

        return UsageExample(
            id=f"usage-{file_path.stem}-{start_line}",
            code=code[:1000],
            intent=intent,
            source_file=str(file_path),
            source_line=start_line,
            function_name=name,
            imports=imports,
            tags=tags,
            confidence=confidence,
        )

    def _infer_intent(self, name: str) -> str:
        """Infer intent from function/test name."""
        name_lower = name.lower()

        for pattern, template in self._intent_patterns:
            match = re.match(pattern, name_lower)
            if match:
                intent_part = match.group(1).replace("_", " ")
                return template.format(intent_part)

        name_formatted = name.replace("_", " ").lower()
        return f"Usage: {name_formatted}"

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extract import statements from AST."""
        imports: list[str] = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)

        return imports

    def _identify_patterns(self, result: MiningResult) -> None:
        """Identify common patterns across examples."""
        function_examples: dict[str, list[UsageExample]] = {}

        for example in result.examples:
            calls = self._extract_function_calls(example.code)
            for call in calls:
                if call not in function_examples:
                    function_examples[call] = []
                function_examples[call].append(example)

        for func_name, examples in sorted(function_examples.items(), key=lambda x: -len(x[1])):
            if len(examples) >= 2:
                avg_conf = sum(e.confidence for e in examples) / len(examples)

                pattern = UsagePattern(
                    pattern_id=f"pattern-{func_name}",
                    description=f"Common usage of {func_name}",
                    examples=examples[:3],
                    frequency=len(examples),
                    avg_confidence=avg_conf,
                    function_calls=[func_name],
                )
                result.patterns.append(pattern)

    def _extract_function_calls(self, code: str) -> list[str]:
        """Extract function calls from code string."""
        pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        calls = re.findall(pattern, code)

        excluded = {
            "print",
            "len",
            "str",
            "int",
            "float",
            "list",
            "dict",
            "set",
            "range",
            "open",
            "type",
            "isinstance",
            "hasattr",
            "getattr",
            "assert",
            "raise",
            "return",
            "if",
            "for",
            "while",
            "with",
        }

        return [c for c in calls if c not in excluded]

    def _should_skip(self, path: Path) -> bool:
        """Check if file should be skipped."""
        path_str = str(path)

        skip_patterns = ["__pycache__", ".pyc", "conftest.py"]
        for pattern in skip_patterns:
            if pattern in path_str:
                return True

        return False


def create_usage_miner(
    config: KnowledgeConfig | None = None,
    parser: ASTParser | None = None,
) -> UsageMiner:
    """Create a UsageMiner instance."""
    return UsageMiner(config=config, parser=parser)
