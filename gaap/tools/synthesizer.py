"""
Tool Synthesizer (Just-in-Time Tooling)
=======================================

New Module - Evolution 2026
Implements: docs/evolution_plan_2026/14_JUST_IN_TIME_TOOLING.md

Full-featured tool synthesizer that:
- Discovers relevant libraries
- Generates code using LLM
- Validates and hot-loads tools
- Caches synthesized skills

Usage:
    from gaap.tools.synthesizer import ToolSynthesizer, SynthesizedTool

    synthesizer = ToolSynthesizer()
    tool = await synthesizer.synthesize_with_discovery("parse JSON files")
    if tool:
        result = tool.module.run(data='{"key": "value"}')
"""

from __future__ import annotations

import ast
import importlib.util
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gaap.core.axioms import AxiomValidator
from gaap.core.events import EventEmitter, EventType
from gaap.core.logging import get_standard_logger
from gaap.security.dlp import DLPScanner
from gaap.tools.code_synthesizer import CodeSynthesizer, SynthesisRequest, SynthesisResult
from gaap.tools.library_discoverer import LibraryDiscoverer
from gaap.tools.skill_cache import SkillCache, SkillMetadata

logger = get_standard_logger("gaap.tools.synthesizer")


@dataclass
class SynthesizedTool:
    """A synthesized and validated tool ready for use."""

    id: str
    name: str
    code: str
    description: str
    file_path: Path
    is_safe: bool
    module: Any = None
    category: str = "other"
    dependencies: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    tests: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "code": self.code,
            "description": self.description,
            "file_path": str(self.file_path),
            "is_safe": self.is_safe,
            "category": self.category,
            "dependencies": self.dependencies,
            "quality_score": self.quality_score,
            "tests": self.tests,
            "metadata": self.metadata,
        }


@dataclass
class SynthesisStats:
    """Statistics for the tool synthesizer."""

    tools_synthesized: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    failures: int = 0
    most_common_categories: list[tuple[str, int]] = field(default_factory=list)


class ToolSynthesizer:
    """
    Generates, validates, and hot-loads new Python tools on the fly.

    Full synthesis flow:
    1. Check cache first (maybe skill already exists)
    2. Search for relevant libraries using LibraryDiscoverer
    3. Generate code using CodeSynthesizer
    4. Validate the code
    5. Store in SkillCache
    6. Return SynthesizedTool
    """

    def __init__(
        self,
        workspace_path: Path | str = ".gaap/custom_tools",
        cache_path: Path | str = ".gaap/skills",
        github_token: str | None = None,
        llm_provider: Any = None,
    ):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        self.axiom_validator = AxiomValidator()
        self.dlp = DLPScanner()

        self.library_discoverer = LibraryDiscoverer(github_token=github_token)
        self.code_synthesizer = CodeSynthesizer(provider=llm_provider)
        self.skill_cache = SkillCache(cache_path=cache_path)

        self.event_emitter = EventEmitter.get_instance()

        self._stats = SynthesisStats()
        self._category_counter: Counter = Counter()

    def __repr__(self) -> str:
        return (
            f"ToolSynthesizer("
            f"workspace={self.workspace_path}, "
            f"cache_entries={len(self.skill_cache.list_all())})"
        )

    async def synthesize(
        self,
        intent: str,
        context: dict[str, Any] | None = None,
    ) -> SynthesizedTool | None:
        """
        Synthesize a tool for the given intent.

        Args:
            intent: Description of what the tool should do
            context: Optional context for code generation

        Returns:
            SynthesizedTool or None if synthesis failed
        """
        self.event_emitter.emit(
            EventType.TOOL_SYNTHESIS_STARTED,
            {"intent": intent, "context": context},
            source="ToolSynthesizer",
        )

        logger.info(f"Synthesizing tool for intent: {intent}")

        try:
            request = SynthesisRequest(
                intent=intent,
                context=context or {},
            )

            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_PROGRESS,
                {"stage": "code_generation", "intent": intent},
                source="ToolSynthesizer",
            )

            result = await self.code_synthesizer.synthesize(request)

            if not result.success:
                self._stats.failures += 1
                self.event_emitter.emit(
                    EventType.TOOL_SYNTHESIS_FAILED,
                    {"intent": intent, "error": result.error, "attempts": result.attempts},
                    source="ToolSynthesizer",
                )
                logger.error(f"Code synthesis failed: {result.error}")
                return None

            return await self._finalize_tool(intent, result, context)

        except Exception as e:
            self._stats.failures += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_FAILED,
                {"intent": intent, "error": str(e)},
                source="ToolSynthesizer",
            )
            logger.exception(f"Tool synthesis failed: {e}")
            return None

    async def synthesize_with_discovery(self, intent: str) -> SynthesizedTool | None:
        """
        Synthesize a tool with automatic library discovery.

        Full synthesis flow:
        1. Check cache first
        2. Discover relevant libraries
        3. Generate code
        4. Validate
        5. Cache and return

        Args:
            intent: Description of what the tool should do

        Returns:
            SynthesizedTool or None if synthesis failed
        """
        self.event_emitter.emit(
            EventType.TOOL_SYNTHESIS_STARTED,
            {"intent": intent, "discovery": True},
            source="ToolSynthesizer",
        )

        logger.info(f"Synthesizing tool with discovery for: {intent}")

        cached = self._check_cache_for_intent(intent)
        if cached:
            self._stats.cache_hits += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_COMPLETE,
                {"intent": intent, "tool_id": cached.id, "source": "cache"},
                source="ToolSynthesizer",
            )
            return cached

        self._stats.cache_misses += 1

        try:
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_PROGRESS,
                {"stage": "library_discovery", "intent": intent},
                source="ToolSynthesizer",
            )

            libraries = await self.library_discoverer.recommend_for_task(intent, limit=3)
            library_names = [lib.name for lib in libraries]

            logger.info(f"Discovered libraries: {library_names}")

            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_PROGRESS,
                {"stage": "code_generation", "intent": intent, "libraries": library_names},
                source="ToolSynthesizer",
            )

            request = SynthesisRequest(
                intent=intent,
                libraries=library_names,
                context={"discovered_libraries": [lib.to_dict() for lib in libraries]},
            )

            result = await self.code_synthesizer.synthesize(request)

            if not result.success:
                self._stats.failures += 1
                self.event_emitter.emit(
                    EventType.TOOL_SYNTHESIS_FAILED,
                    {"intent": intent, "error": result.error},
                    source="ToolSynthesizer",
                )
                logger.error(f"Code synthesis failed: {result.error}")
                return None

            return await self._finalize_tool(intent, result, {"libraries": libraries})

        except Exception as e:
            self._stats.failures += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_FAILED,
                {"intent": intent, "error": str(e)},
                source="ToolSynthesizer",
            )
            logger.exception(f"Tool synthesis with discovery failed: {e}")
            return None

    async def _finalize_tool(
        self,
        intent: str,
        result: SynthesisResult,
        context: dict[str, Any] | None = None,
    ) -> SynthesizedTool | None:
        """Validate, save, and load a synthesized tool."""
        self.event_emitter.emit(
            EventType.TOOL_SYNTHESIS_PROGRESS,
            {"stage": "validation", "intent": intent},
            source="ToolSynthesizer",
        )

        axiom_results = self.axiom_validator.validate(result.code)
        failed_axioms = [r for r in axiom_results if not r.passed]

        if failed_axioms:
            logger.error(f"Tool rejected by Constitution: {[r.axiom_name for r in failed_axioms]}")
            self._stats.failures += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_FAILED,
                {
                    "intent": intent,
                    "error": "Axiom validation failed",
                    "failed_axioms": [r.axiom_name for r in failed_axioms],
                },
                source="ToolSynthesizer",
            )
            return None

        try:
            ast.parse(result.code)
        except SyntaxError as e:
            logger.error(f"Generated tool has syntax error: {e}")
            self._stats.failures += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_FAILED,
                {"intent": intent, "error": f"Syntax error: {e}"},
                source="ToolSynthesizer",
            )
            return None

        tool_id = str(uuid.uuid4())[:8]
        tool_name = f"tool_{tool_id}"
        file_path = self.workspace_path / f"{tool_name}.py"

        with open(file_path, "w") as f:
            f.write(result.code)

        self.event_emitter.emit(
            EventType.TOOL_SYNTHESIS_PROGRESS,
            {"stage": "hot_loading", "intent": intent, "tool_id": tool_id},
            source="ToolSynthesizer",
        )

        try:
            spec = importlib.util.spec_from_file_location(tool_name, file_path)
            if not spec or not spec.loader:
                raise ImportError("Failed to create module spec")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if not hasattr(module, "run") and not hasattr(module, "execute"):
                logger.warning(f"Tool {tool_name} has no entry point (run/execute).")

            category = self._determine_category(intent, context)
            self._category_counter[category] += 1

            tool = SynthesizedTool(
                id=tool_id,
                name=tool_name,
                code=result.code,
                description=intent,
                file_path=file_path,
                is_safe=True,
                module=module,
                category=category,
                dependencies=result.dependencies,
                quality_score=result.quality_score,
                tests=result.tests,
                metadata={
                    **result.metadata,
                    "context": context or {},
                },
            )

            self._store_in_cache(tool, context)

            self._stats.tools_synthesized += 1
            self._stats.most_common_categories = self._category_counter.most_common(5)

            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_COMPLETE,
                {
                    "intent": intent,
                    "tool_id": tool_id,
                    "tool_name": tool_name,
                    "category": category,
                    "quality_score": result.quality_score,
                    "dependencies": result.dependencies,
                },
                source="ToolSynthesizer",
            )

            logger.info(f"Successfully synthesized tool {tool_name}")
            return tool

        except Exception as e:
            logger.error(f"Failed to hot-load tool: {e}")
            self._stats.failures += 1
            self.event_emitter.emit(
                EventType.TOOL_SYNTHESIS_FAILED,
                {"intent": intent, "error": f"Hot-load failed: {e}"},
                source="ToolSynthesizer",
            )
            return None

    def _check_cache_for_intent(self, intent: str) -> SynthesizedTool | None:
        """Check if a similar tool already exists in cache."""
        all_skills = self.skill_cache.list_all()

        intent_lower = intent.lower()
        intent_words = set(intent_lower.split())

        for skill in all_skills:
            desc_lower = skill.description.lower()
            desc_words = set(desc_lower.split())

            overlap = len(intent_words & desc_words)
            if overlap >= min(3, len(intent_words)):
                logger.info(f"Found cached skill matching intent: {skill.id}")
                return self.skill_cache.retrieve(skill.id)

            for tag in skill.tags:
                if tag.lower() in intent_lower:
                    logger.info(f"Found cached skill by tag: {skill.id}")
                    return self.skill_cache.retrieve(skill.id)

        return None

    def _determine_category(
        self,
        intent: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Determine the category for a tool."""
        intent_lower = intent.lower()

        categories = {
            "coding": ["code", "function", "class", "module", "generate", "refactor"],
            "research": ["search", "find", "lookup", "research", "discover", "query"],
            "analysis": ["analyze", "parse", "extract", "process", "transform"],
            "automation": ["automate", "schedule", "batch", "pipeline", "workflow"],
            "utility": ["helper", "util", "format", "convert", "validate"],
        }

        for category, keywords in categories.items():
            if any(kw in intent_lower for kw in keywords):
                return category

        if context and "libraries" in context:
            for lib in context["libraries"]:
                if isinstance(lib, dict):
                    lib_name = lib.get("name", "").lower()
                else:
                    lib_name = getattr(lib, "name", "").lower()

                if any(db in lib_name for db in ["sql", "mongo", "redis"]):
                    return "analysis"
                if any(web in lib_name for web in ["flask", "django", "fastapi", "requests"]):
                    return "automation"

        return "other"

    def _store_in_cache(
        self,
        tool: SynthesizedTool,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Store a tool in the skill cache."""
        tags = []
        if context and "libraries" in context:
            for lib in context["libraries"]:
                if isinstance(lib, dict):
                    tags.append(lib.get("name", ""))
                else:
                    tags.append(getattr(lib, "name", ""))

        intent_words = tool.description.lower().split()
        tags.extend([w for w in intent_words if len(w) > 3][:5])

        self.skill_cache.store(
            tool,
            {
                "description": tool.description,
                "tags": tags,
                "dependencies": tool.dependencies,
            },
        )

    def load_skill(self, skill_id: str) -> SynthesizedTool | None:
        """
        Load a skill from the cache by ID.

        Args:
            skill_id: The skill ID to load

        Returns:
            SynthesizedTool or None if not found
        """
        tool = self.skill_cache.retrieve(skill_id)
        if tool:
            self.skill_cache.update_usage(skill_id)
            logger.info(f"Loaded skill {skill_id}")
        return tool

    def list_skills(self) -> list[SkillMetadata]:
        """
        List all cached skills.

        Returns:
            List of SkillMetadata for all skills
        """
        return self.skill_cache.list_all()

    def cleanup_old_skills(self, days: int = 30) -> int:
        """
        Remove skills not used in the specified number of days.

        Args:
            days: Number of days of inactivity before cleanup

        Returns:
            Count of deleted skills
        """
        deleted = self.skill_cache.cleanup_unused(days=days)
        logger.info(f"Cleaned up {deleted} old skills")
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """
        Get synthesis statistics.

        Returns:
            Dict with statistics about synthesis operations
        """
        cache_stats = self.skill_cache.get_stats()

        return {
            "tools_synthesized": self._stats.tools_synthesized,
            "cache_hits": self._stats.cache_hits,
            "cache_misses": self._stats.cache_misses,
            "failures": self._stats.failures,
            "most_common_categories": self._stats.most_common_categories,
            "cache_stats": cache_stats.to_dict(),
        }

    def set_llm_provider(self, provider: Any) -> None:
        """Set the LLM provider for code generation."""
        self.code_synthesizer.set_provider(provider)

    async def close(self) -> None:
        """Clean up resources."""
        await self.library_discoverer.close()

    def cleanup(self) -> None:
        """Removes temporary tools from workspace."""
        try:
            for py_file in self.workspace_path.glob("tool_*.py"):
                py_file.unlink()
                logger.debug(f"Removed temporary tool: {py_file}")
        except Exception as e:
            logger.error(f"Failed to cleanup temporary tools: {e}")


_synthesizer_instance: ToolSynthesizer | None = None


def get_tool_synthesizer(**kwargs: Any) -> ToolSynthesizer:
    """Get singleton ToolSynthesizer instance."""
    global _synthesizer_instance

    if _synthesizer_instance is None:
        _synthesizer_instance = ToolSynthesizer(**kwargs)
    return _synthesizer_instance
