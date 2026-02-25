"""
Knowledge Ingestion - Repository Learning Engine
================================================

Main orchestrator for ingesting and learning from repositories.

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md

Usage:
    from gaap.knowledge import KnowledgeIngestion

    ingestion = KnowledgeIngestion()
    result = await ingestion.ingest_repo("https://github.com/pydantic/pydantic")

    # Later, load learned knowledge
    knowledge = ingestion.load_library("pydantic")
"""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from gaap.knowledge.knowledge_config import KnowledgeConfig, create_knowledge_config
from gaap.knowledge.ast_parser import ASTParser, ParsedFile, create_parser
from gaap.knowledge.usage_miner import UsageMiner, MiningResult, create_usage_miner
from gaap.knowledge.cheat_sheet import (
    CheatSheetGenerator,
    ReferenceCard,
    create_cheat_sheet_generator,
)

logger = logging.getLogger("gaap.knowledge.ingestion")


@dataclass
class IngestionResult:
    """Result of repository ingestion."""

    library_name: str
    source: str
    success: bool = True

    files_parsed: int = 0
    functions_found: int = 0
    classes_found: int = 0
    examples_mined: int = 0
    patterns_identified: int = 0

    parse_time_ms: float = 0.0
    mine_time_ms: float = 0.0
    total_time_ms: float = 0.0

    output_path: str | None = None
    reference_card: ReferenceCard | None = None

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    ingested_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_name": self.library_name,
            "source": self.source,
            "success": self.success,
            "files_parsed": self.files_parsed,
            "functions_found": self.functions_found,
            "classes_found": self.classes_found,
            "examples_mined": self.examples_mined,
            "patterns_identified": self.patterns_identified,
            "parse_time_ms": self.parse_time_ms,
            "mine_time_ms": self.mine_time_ms,
            "total_time_ms": self.total_time_ms,
            "output_path": self.output_path,
            "reference_card": self.reference_card.to_dict() if self.reference_card else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "ingested_at": self.ingested_at.isoformat(),
        }


@dataclass
class LibraryKnowledge:
    """Loaded knowledge about a library."""

    library_name: str
    version: str | None = None
    description: str | None = None
    reference_card: ReferenceCard | None = None
    parsed_files: list[ParsedFile] = field(default_factory=list)
    usage_examples: list[dict[str, Any]] = field(default_factory=list)
    ingested_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "library_name": self.library_name,
            "version": self.version,
            "description": self.description,
            "reference_card": self.reference_card.to_dict() if self.reference_card else None,
            "parsed_files": [pf.to_dict() for pf in self.parsed_files],
            "usage_examples": self.usage_examples,
            "ingested_at": self.ingested_at.isoformat(),
        }

    def get_context_for_prompt(self, max_tokens: int = 4000) -> str:
        """Generate context string for LLM prompts."""
        lines = [
            f"## Library: {self.library_name}",
            "",
        ]

        if self.reference_card:
            if self.reference_card.description:
                lines.append(f"Description: {self.reference_card.description}")
                lines.append("")

            if self.reference_card.top_functions:
                lines.append("### Key Functions:")
                for func in self.reference_card.top_functions[:5]:
                    lines.append(f"- `{func.signature}`")
                    if func.docstring:
                        lines.append(f"  {func.docstring.split(chr(10))[0][:100]}")
                lines.append("")

            if self.reference_card.top_classes:
                lines.append("### Key Classes:")
                for cls in self.reference_card.top_classes[:5]:
                    bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
                    lines.append(f"- {cls['name']}{bases}")
                lines.append("")

            if self.reference_card.common_patterns:
                lines.append("### Common Patterns:")
                for pattern in self.reference_card.common_patterns[:3]:
                    lines.append(f"- {pattern.description}")
                    lines.append(f"  ```python")
                    lines.append(f"  {pattern.code[:200]}")
                    lines.append(f"  ```")
                lines.append("")

        context = "\n".join(lines)

        if len(context) > max_tokens * 4:
            context = context[: max_tokens * 4]

        return context


class KnowledgeIngestion:
    """
    Main orchestrator for repository ingestion.

    Features:
    - Clone remote repositories
    - Parse source files (Python, JS, TS)
    - Mine usage examples from tests
    - Generate reference cards
    - Store knowledge for later use

    Usage:
        ingestion = KnowledgeIngestion()
        result = await ingestion.ingest_repo("https://github.com/user/repo")
        knowledge = ingestion.load_library("repo")
    """

    def __init__(
        self,
        config: KnowledgeConfig | None = None,
        project_root: Path | str | None = None,
    ):
        self._config = config or KnowledgeConfig()
        self._project_root = Path(project_root) if project_root else Path.cwd()
        self._logger = logger

        self._parser = create_parser(self._config)
        self._miner = create_usage_miner(self._config, self._parser)
        self._generator = create_cheat_sheet_generator(self._config)

        self._knowledge_path = self._project_root / self._config.storage_path
        self._knowledge_path.mkdir(parents=True, exist_ok=True)

    @property
    def config(self) -> KnowledgeConfig:
        return self._config

    async def ingest_repo(
        self,
        source: str | Path,
        library_name: str | None = None,
        version: str | None = None,
        description: str | None = None,
    ) -> IngestionResult:
        """
        Ingest a repository and learn its patterns.

        Args:
            source: URL or local path to repository
            library_name: Optional name override
            version: Optional version string
            description: Optional description

        Returns:
            IngestionResult with parsing and mining results
        """
        import time

        start_time = time.time()

        if isinstance(source, Path):
            source_str = str(source)
        else:
            source_str = source

        if library_name is None:
            library_name = self._extract_library_name(source_str)

        result = IngestionResult(
            library_name=library_name,
            source=source_str,
        )

        repo_path: Path | None = None
        is_temp = False

        try:
            if source_str.startswith(("http://", "https://", "git@")):
                repo_path = await self._clone_repo(source_str)
                is_temp = True
            else:
                repo_path = Path(source_str)
                if not repo_path.exists():
                    result.errors.append(f"Path does not exist: {source_str}")
                    result.success = False
                    return result

            parsed_files = self._parser.parse_directory(repo_path)

            result.files_parsed = len(parsed_files)
            result.functions_found = sum(len(pf.functions) for pf in parsed_files)
            result.classes_found = sum(len(pf.classes) for pf in parsed_files)
            result.parse_time_ms = sum(pf.parse_time_ms for pf in parsed_files)

            mining_result = self._miner.mine_all(repo_path)

            result.examples_mined = mining_result.total_examples
            result.patterns_identified = len(mining_result.patterns)
            result.mine_time_ms = mining_result.mine_time_ms

            reference_card = self._generator.generate(
                parsed_files=parsed_files,
                mining_result=mining_result,
                library_name=library_name,
                version=version,
                description=description,
            )
            result.reference_card = reference_card

            output_path = await self._save_knowledge(
                library_name=library_name,
                parsed_files=parsed_files,
                mining_result=mining_result,
                reference_card=reference_card,
            )
            result.output_path = str(output_path)

            result.warnings.extend(mining_result.errors)

        except Exception as e:
            result.errors.append(str(e))
            result.success = False
            self._logger.error(f"Ingestion failed: {e}")

        finally:
            if is_temp and repo_path:
                shutil.rmtree(repo_path, ignore_errors=True)

        result.total_time_ms = (time.time() - start_time) * 1000
        return result

    def load_library(self, library_name: str) -> LibraryKnowledge | None:
        """
        Load previously ingested library knowledge.

        Args:
            library_name: Name of the library

        Returns:
            LibraryKnowledge if found, None otherwise
        """
        knowledge_file = self._knowledge_path / f"{library_name}.json"

        if not knowledge_file.exists():
            self._logger.warning(f"Library not found: {library_name}")
            return None

        try:
            with open(knowledge_file) as f:
                data = json.load(f)

            reference_card = None
            if data.get("reference_card"):
                card_data = data["reference_card"]
                from gaap.knowledge.cheat_sheet import (
                    ReferenceCard,
                    FunctionSummary,
                    PatternExample,
                )

                top_functions = [FunctionSummary(**f) for f in card_data.get("top_functions", [])]
                common_patterns = [
                    PatternExample(**p) for p in card_data.get("common_patterns", [])
                ]

                reference_card = ReferenceCard(
                    library_name=card_data.get("library_name", library_name),
                    version=card_data.get("version"),
                    description=card_data.get("description"),
                    top_functions=top_functions,
                    top_classes=card_data.get("top_classes", []),
                    common_patterns=common_patterns,
                    imports_used=card_data.get("imports_used", []),
                )

            return LibraryKnowledge(
                library_name=data.get("library_name", library_name),
                version=data.get("version"),
                description=data.get("description"),
                reference_card=reference_card,
                usage_examples=data.get("usage_examples", []),
                ingested_at=datetime.fromisoformat(data["ingested_at"])
                if "ingested_at" in data
                else datetime.now(),
            )

        except Exception as e:
            self._logger.error(f"Failed to load library {library_name}: {e}")
            return None

    def list_libraries(self) -> list[str]:
        """List all ingested libraries."""
        libraries = []

        for file_path in self._knowledge_path.glob("*.json"):
            libraries.append(file_path.stem)

        return sorted(libraries)

    def delete_library(self, library_name: str) -> bool:
        """Delete an ingested library."""
        knowledge_file = self._knowledge_path / f"{library_name}.json"

        if knowledge_file.exists():
            knowledge_file.unlink()
            self._logger.info(f"Deleted library: {library_name}")
            return True

        return False

    async def _clone_repo(self, repo_url: str) -> Path:
        """Clone a remote repository."""
        temp_dir = Path(tempfile.mkdtemp(prefix="gaap_repo_"))

        self._logger.info(f"Cloning {repo_url}...")

        try:
            result = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                str(self._config.clone_depth),
                repo_url,
                str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                result.communicate(),
                timeout=self._config.clone_timeout_seconds,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Git clone failed: {stderr.decode()}")

            self._logger.info(f"Cloned to {temp_dir}")
            return temp_dir

        except asyncio.TimeoutError:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(f"Clone timed out after {self._config.clone_timeout_seconds}s")

    async def _save_knowledge(
        self,
        library_name: str,
        parsed_files: list[ParsedFile],
        mining_result: MiningResult,
        reference_card: ReferenceCard,
    ) -> Path:
        """Save knowledge to disk."""
        output_path = self._knowledge_path / f"{library_name}.json"

        data = {
            "library_name": library_name,
            "version": reference_card.version,
            "description": reference_card.description,
            "reference_card": reference_card.to_dict(),
            "parsed_files": [pf.to_dict() for pf in parsed_files[:100]],
            "usage_examples": [ex.to_dict() for ex in mining_result.examples[:50]],
            "ingested_at": datetime.now().isoformat(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self._logger.info(f"Saved knowledge to {output_path}")

        card_path = self._knowledge_path / f"{library_name}_card.md"
        self._generator.save_card(reference_card, card_path, format="markdown")

        return output_path

    def _extract_library_name(self, source: str) -> str:
        """Extract library name from URL or path."""
        if source.startswith(("http://", "https://", "git@")):
            parsed = urlparse(source.replace("git@", "https://"))
            path = parsed.path.rstrip("/")
            name = path.split("/")[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name.lower()
        else:
            return Path(source).name.lower()


def create_knowledge_ingestion(
    config: KnowledgeConfig | None = None,
    project_root: Path | str | None = None,
) -> KnowledgeIngestion:
    """Create a KnowledgeIngestion instance."""
    return KnowledgeIngestion(config=config, project_root=project_root)
