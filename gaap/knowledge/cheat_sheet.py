"""
Cheat Sheet Generator - Reference Card Generation
==================================================

Generates reference cards (cheat sheets) from parsed library code.

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import Counter

from gaap.knowledge.knowledge_config import KnowledgeConfig
from gaap.knowledge.ast_parser import ParsedFile, ClassInfo, FunctionInfo
from gaap.knowledge.usage_miner import UsageExample, UsagePattern, MiningResult

logger = logging.getLogger("gaap.knowledge.cheatsheet")


@dataclass
class FunctionSummary:
    """Summary of a frequently used function."""

    name: str
    signature: str
    docstring: str | None = None
    class_name: str | None = None
    usage_count: int = 0
    example: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signature": self.signature,
            "docstring": self.docstring,
            "class_name": self.class_name,
            "usage_count": self.usage_count,
            "example": self.example,
        }


@dataclass
class PatternExample:
    """A common usage pattern example."""

    description: str
    code: str
    frequency: int = 1
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "code": self.code,
            "frequency": self.frequency,
            "tags": self.tags,
        }


@dataclass
class BreakingChange:
    """Detected breaking change."""

    description: str
    old_api: str | None = None
    new_api: str | None = None
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "old_api": self.old_api,
            "new_api": self.new_api,
            "severity": self.severity,
        }


@dataclass
class ReferenceCard:
    """Complete reference card for a library."""

    library_name: str
    version: str | None = None
    description: str | None = None
    top_functions: list[FunctionSummary] = field(default_factory=list)
    top_classes: list[dict[str, Any]] = field(default_factory=list)
    common_patterns: list[PatternExample] = field(default_factory=list)
    breaking_changes: list[BreakingChange] = field(default_factory=list)
    imports_used: list[str] = field(default_factory=list)
    total_files_analyzed: int = 0
    total_functions_found: int = 0
    total_classes_found: int = 0
    generated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_name": self.library_name,
            "version": self.version,
            "description": self.description,
            "top_functions": [f.to_dict() for f in self.top_functions],
            "top_classes": self.top_classes,
            "common_patterns": [p.to_dict() for p in self.common_patterns],
            "breaking_changes": [b.to_dict() for b in self.breaking_changes],
            "imports_used": self.imports_used,
            "total_files_analyzed": self.total_files_analyzed,
            "total_functions_found": self.total_functions_found,
            "total_classes_found": self.total_classes_found,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Generate markdown representation."""
        lines = [
            f"# {self.library_name} Reference Card",
            "",
        ]

        if self.description:
            lines.extend(
                [
                    f"*{self.description}*",
                    "",
                ]
            )

        lines.extend(
            [
                f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M')}",
                f"Files analyzed: {self.total_files_analyzed}",
                "",
            ]
        )

        if self.top_functions:
            lines.extend(
                [
                    "## Top Functions",
                    "",
                ]
            )
            for func in self.top_functions[: self._top_count]:
                lines.append(f"### `{func.signature}`")
                if func.docstring:
                    lines.extend(
                        [
                            "",
                            func.docstring.split("\n")[0],
                        ]
                    )
                if func.example:
                    lines.extend(
                        [
                            "",
                            "```python",
                            func.example[:200],
                            "```",
                        ]
                    )
                lines.append("")

        if self.top_classes:
            lines.extend(
                [
                    "## Key Classes",
                    "",
                ]
            )
            for cls in self.top_classes[: self._top_count]:
                lines.append(f"### `{cls['name']}`")
                if cls.get("bases"):
                    lines.append(f"Inherits from: {', '.join(cls['bases'])}")
                if cls.get("docstring"):
                    lines.extend(["", cls["docstring"].split("\n")[0]])
                lines.append("")

        if self.common_patterns:
            lines.extend(
                [
                    "## Common Patterns",
                    "",
                ]
            )
            for pattern in self.common_patterns:
                lines.append(f"### {pattern.description}")
                lines.extend(
                    [
                        "",
                        "```python",
                        pattern.code[:500],
                        "```",
                        "",
                    ]
                )

        if self.breaking_changes:
            lines.extend(
                [
                    "## ⚠️ Breaking Changes",
                    "",
                ]
            )
            for change in self.breaking_changes:
                lines.append(f"- **{change.severity.upper()}**: {change.description}")
                if change.old_api:
                    lines.append(f"  - Old: `{change.old_api}`")
                if change.new_api:
                    lines.append(f"  - New: `{change.new_api}`")
                lines.append("")

        return "\n".join(lines)

    @property
    def _top_count(self) -> int:
        return 10


class CheatSheetGenerator:
    """
    Generates reference cards from parsed library code.

    Features:
    - Extract top functions by usage
    - Identify common patterns
    - Detect breaking changes
    - Generate markdown/JSON output
    """

    def __init__(self, config: KnowledgeConfig | None = None):
        self._config = config or KnowledgeConfig()
        self._logger = logger

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    def generate(
        self,
        parsed_files: list[ParsedFile],
        mining_result: MiningResult | None = None,
        library_name: str = "unknown",
        version: str | None = None,
        description: str | None = None,
    ) -> ReferenceCard:
        """
        Generate a reference card from parsed files.

        Args:
            parsed_files: List of parsed source files
            mining_result: Optional mining result for usage patterns
            library_name: Name of the library
            version: Version string
            description: Library description

        Returns:
            Generated ReferenceCard
        """
        card = ReferenceCard(
            library_name=library_name,
            version=version,
            description=description,
        )

        card.total_files_analyzed = len(parsed_files)

        all_functions: list[FunctionInfo] = []
        all_classes: list[ClassInfo] = []
        all_imports: list[str] = []

        for pf in parsed_files:
            all_functions.extend(pf.functions)
            all_classes.extend(pf.classes)
            for imp in pf.imports:
                if imp.module:
                    all_imports.append(imp.module)

        card.total_functions_found = len(all_functions)
        card.total_classes_found = len(all_classes)

        usage_counts: dict[str, int] = {}
        if mining_result:
            usage_counts = mining_result.by_function.copy()

        card.top_functions = self._extract_top_functions(all_functions, usage_counts, mining_result)

        card.top_classes = self._extract_top_classes(all_classes)

        if mining_result:
            card.common_patterns = self._extract_patterns(mining_result)

        card.imports_used = self._get_common_imports(all_imports)

        return card

    def _extract_top_functions(
        self,
        functions: list[FunctionInfo],
        usage_counts: dict[str, int],
        mining_result: MiningResult | None,
    ) -> list[FunctionSummary]:
        """Extract top functions by usage and importance."""
        summaries: list[FunctionSummary] = []

        for func in functions:
            if func.name.startswith("_"):
                continue

            usage_count = usage_counts.get(func.name, 0)

            example = None
            if mining_result:
                for ex in mining_result.examples:
                    if ex.function_name == func.name or func.name in ex.code:
                        example = ex.code
                        break

            summaries.append(
                FunctionSummary(
                    name=func.name,
                    signature=func.signature,
                    docstring=func.docstring,
                    class_name=func.file_path.split("/")[-1].replace(".py", ""),
                    usage_count=usage_count,
                    example=example,
                )
            )

        summaries.sort(key=lambda x: (-x.usage_count, x.name))

        return summaries[: self._config.top_functions_count]

    def _extract_top_classes(
        self,
        classes: list[ClassInfo],
    ) -> list[dict[str, Any]]:
        """Extract top classes by method count and complexity."""
        class_data = []

        for cls in classes:
            if cls.name.startswith("_"):
                continue

            class_data.append(
                {
                    "name": cls.name,
                    "bases": cls.bases,
                    "docstring": cls.docstring,
                    "public_methods": len(cls.public_methods),
                    "total_methods": len(cls.methods),
                }
            )

        class_data.sort(key=lambda x: (-(x["public_methods"] if isinstance(x["public_methods"], int) else int(x["public_methods"]) if isinstance(x["public_methods"], str) else 0), x["name"]))

        return class_data[: self._config.top_functions_count]

    def _extract_patterns(
        self,
        mining_result: MiningResult,
    ) -> list[PatternExample]:
        """Extract common patterns from mining result."""
        patterns: list[PatternExample] = []

        for pattern in mining_result.patterns[: self._config.common_patterns_count]:
            if pattern.examples:
                best_example = max(pattern.examples, key=lambda x: x.confidence)
                patterns.append(
                    PatternExample(
                        description=pattern.description,
                        code=best_example.code,
                        frequency=pattern.frequency,
                        tags=best_example.tags,
                    )
                )

        return patterns

    def _get_common_imports(
        self,
        imports: list[str],
        top_n: int = 10,
    ) -> list[str]:
        """Get most commonly used imports."""
        counter = Counter(imports)
        return [imp for imp, _ in counter.most_common(top_n)]

    def save_card(
        self,
        card: ReferenceCard,
        output_path: Path | str,
        format: str = "json",
    ) -> Path:
        """
        Save reference card to file.

        Args:
            card: ReferenceCard to save
            output_path: Output file path
            format: 'json' or 'markdown'

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "markdown":
            content = card.to_markdown()
            output_path = output_path.with_suffix(".md")
        else:
            content = card.to_json()
            output_path = output_path.with_suffix(".json")

        output_path.write_text(content, encoding="utf-8")

        self._logger.info(f"Saved reference card to {output_path}")
        return output_path


def create_cheat_sheet_generator(
    config: KnowledgeConfig | None = None,
) -> CheatSheetGenerator:
    """Create a CheatSheetGenerator instance."""
    return CheatSheetGenerator(config=config)
