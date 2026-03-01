"""
Semantic Dependency Engine - Intelligent Dependency Resolution
================================================================

Evolution 2026: LLM-driven semantic dependency analysis.

Key Features:
- Understands task relationships semantically (not regex)
- Detects hidden dependencies from context
- Supports multiple depth levels
- Learns from past dependency patterns

Depth Levels:
- standard: Obvious dependencies (test → implementation)
- deep: Semantic analysis of task descriptions
- exhaustive: Full codebase analysis with LLM
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.layer2_config import Layer2Config
from gaap.layers.task_schema import IntelligentTask

logger = get_logger("gaap.layer2.dependencies")


@dataclass
class Dependency:
    """Represents a dependency between two tasks"""

    from_task: str
    to_task: str
    dependency_type: Literal["hard", "soft", "conditional"]
    reason: str
    confidence: float = 0.8
    detected_by: str = "unknown"  # rule, semantic, llm, pattern

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_task": self.from_task,
            "to_task": self.to_task,
            "dependency_type": self.dependency_type,
            "reason": self.reason,
            "confidence": self.confidence,
            "detected_by": self.detected_by,
        }


@dataclass
class DependencyGraph:
    """Dependency graph for tasks"""

    tasks: dict[str, IntelligentTask] = field(default_factory=dict)
    dependencies: list[Dependency] = field(default_factory=list)

    # Cached structures
    _adjacency: dict[str, list[str]] = field(default_factory=dict)
    _reverse_adjacency: dict[str, list[str]] = field(default_factory=dict)

    def add_task(self, task: IntelligentTask) -> None:
        self.tasks[task.id] = task

    def add_dependency(self, dep: Dependency) -> None:
        """
        Add Dependency.

        dep.from_task depends on dep.to_task
        means: dep.to_task must complete before dep.from_task

        _adjacency[task] = tasks that depend on task (my dependents)
        _reverse_adjacency[task] = tasks that task depends on (my dependencies)
        """
        self.dependencies.append(dep)

        # from_task depends on to_task, so to_task has from_task as a dependent
        if dep.to_task not in self._adjacency:
            self._adjacency[dep.to_task] = []
        self._adjacency[dep.to_task].append(dep.from_task)

        # from_task has to_task as a dependency
        if dep.from_task not in self._reverse_adjacency:
            self._reverse_adjacency[dep.from_task] = []
        self._reverse_adjacency[dep.from_task].append(dep.to_task)

    def get_dependencies(self, task_id: str) -> list[str]:
        """Get tasks that this task depends on"""
        return self._reverse_adjacency.get(task_id, [])

    def get_dependents(self, task_id: str) -> list[str]:
        """Get tasks that depend on this task"""
        return self._adjacency.get(task_id, [])

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """Get tasks ready to execute (all dependencies satisfied)"""
        ready = []
        for task_id, task in self.tasks.items():
            if task_id in completed:
                continue
            deps = self.get_dependencies(task_id)
            if all(d in completed for d in deps):
                ready.append(task_id)
        return ready

    def detect_cycles(self) -> list[list[str]]:
        """Detect cycles in dependency graph"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {task_id: WHITE for task_id in self.tasks}
        cycles = []

        def dfs(task_id: str, path: list[str]) -> None:
            color[task_id] = GRAY
            path.append(task_id)

            for dep_id in self._adjacency.get(task_id, []):
                if dep_id not in self.tasks:
                    continue

                if color[dep_id] == GRAY:
                    cycle_start = path.index(dep_id)
                    cycles.append(path[cycle_start:] + [dep_id])
                elif color[dep_id] == WHITE:
                    dfs(dep_id, path.copy())

            color[task_id] = BLACK

        for task_id in self.tasks:
            if color[task_id] == WHITE:
                dfs(task_id, [])

        return cycles

    def topological_sort(self) -> list[str]:
        """Get topological order of tasks"""
        in_degree = {task_id: len(self.get_dependencies(task_id)) for task_id in self.tasks}
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for dependent in self.get_dependents(current):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result if len(result) == len(self.tasks) else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks": list(self.tasks.keys()),
            "dependencies": [d.to_dict() for d in self.dependencies],
        }


@dataclass
class ResolutionContext:
    """Context for dependency resolution"""

    tasks: list[IntelligentTask]
    codebase_context: dict[str, str] = field(default_factory=dict)
    previous_executions: list[Any] = field(default_factory=list)
    architecture_context: dict[str, Any] = field(default_factory=dict)


class SemanticDependencyEngine:
    """
    LLM-driven semantic dependency analysis.

    Unlike regex-based or simple rule-based systems, this understands
    task relationships through semantic analysis.
    """

    def __init__(
        self,
        provider: Any = None,
        config: Layer2Config | None = None,
    ):
        self._provider = provider
        self._config = config or Layer2Config()
        self._logger = logger

        self._resolutions = 0
        self._llm_resolutions = 0
        self._rule_resolutions = 0

    async def resolve(self, context: ResolutionContext) -> DependencyGraph:
        """
        Resolve dependencies between tasks.

        Uses configured depth level for analysis.
        """
        start_time = time.time()
        self._resolutions += 1

        graph = DependencyGraph()
        for task in context.tasks:
            graph.add_task(task)

        timing: Literal["auto", "pre", "jit", "continuous", "hybrid"] = (
            self._config.dependency_timing
        )
        if timing == "auto":
            inferred = self._infer_timing(context)
            timing = inferred  # type: ignore[assignment]

        depth = self._config.dependency_depth

        rule_deps = await self._apply_rule_based_detection(context)
        for dep in rule_deps:
            graph.add_dependency(dep)

        if depth in ("deep", "exhaustive") and self._provider:
            semantic_deps = await self._semantic_analysis(context, depth)
            for dep in semantic_deps:
                if not self._dependency_exists(graph, dep):
                    graph.add_dependency(dep)
            self._llm_resolutions += 1
        else:
            self._rule_resolutions += 1

        for task in context.tasks:
            for dep_id in task.dependencies:
                if dep_id in graph.tasks and not self._dependency_exists(
                    graph,
                    Dependency(
                        from_task=task.id,
                        to_task=dep_id,
                        dependency_type="hard",
                        reason="Explicitly declared",
                        detected_by="explicit",
                    ),
                ):
                    graph.add_dependency(
                        Dependency(
                            from_task=task.id,
                            to_task=dep_id,
                            dependency_type="hard",
                            reason="Explicitly declared",
                            detected_by="explicit",
                        )
                    )

        cycles = graph.detect_cycles()
        if cycles:
            self._logger.warning(f"Cycles detected: {cycles}")
            graph = self._break_cycles(graph, cycles)

        elapsed = time.time() - start_time
        self._logger.info(
            f"Dependency resolution complete: {len(graph.dependencies)} deps, "
            f"{elapsed:.2f}s, depth={depth}"
        )

        return graph

    def _infer_timing(self, context: ResolutionContext) -> str:
        """Infer best timing strategy"""
        if len(context.tasks) > 20:
            return "jit"
        return "hybrid"

    async def _apply_rule_based_detection(self, context: ResolutionContext) -> list[Dependency]:
        """Apply rule-based dependency detection"""
        dependencies = []
        tasks = context.tasks

        test_tasks = {}
        impl_tasks = {}

        for task in tasks:
            name_lower = task.name.lower()
            desc_lower = task.description.lower()

            if "test" in name_lower or "test" in desc_lower:
                test_tasks[task.id] = task
            else:
                impl_tasks[task.id] = task

        for test_id, test_task in test_tasks.items():
            for impl_id, impl_task in impl_tasks.items():
                if self._tasks_related(test_task, impl_task):
                    dependencies.append(
                        Dependency(
                            from_task=test_id,
                            to_task=impl_id,
                            dependency_type="hard",
                            reason="Testing depends on implementation",
                            confidence=0.9,
                            detected_by="rule",
                        )
                    )

        file_modifications: dict[str, list[IntelligentTask]] = {}
        for task in tasks:
            for file_path in task.semantic_scope:
                if file_path not in file_modifications:
                    file_modifications[file_path] = []
                file_modifications[file_path].append(task)

        for file_path, related_tasks in file_modifications.items():
            if len(related_tasks) > 1:
                for i, task1 in enumerate(related_tasks):
                    for task2 in related_tasks[i + 1 :]:
                        if (
                            "write" in task1.semantic_intent.lower()
                            and "read" in task2.semantic_intent.lower()
                        ):
                            dependencies.append(
                                Dependency(
                                    from_task=task2.id,
                                    to_task=task1.id,
                                    dependency_type="soft",
                                    reason=f"Both modify {file_path}",
                                    confidence=0.6,
                                    detected_by="rule",
                                )
                            )

        return dependencies

    def _tasks_related(self, task1: IntelligentTask, task2: IntelligentTask) -> bool:
        """Check if two tasks are semantically related"""
        name1_words = set(task1.name.lower().replace("_", " ").split())
        name2_words = set(task2.name.lower().replace("_", " ").split())

        name1_words.discard("test")
        name2_words.discard("test")

        overlap = name1_words & name2_words
        return len(overlap) >= 1

    def _dependency_exists(self, graph: DependencyGraph, dep: Dependency) -> bool:
        """Check if dependency already exists"""
        for existing in graph.dependencies:
            if existing.from_task == dep.from_task and existing.to_task == dep.to_task:
                return True
        return False

    async def _semantic_analysis(
        self,
        context: ResolutionContext,
        depth: str,
    ) -> list[Dependency]:
        """Use LLM for semantic dependency analysis"""

        tasks_desc = "\n".join(
            [f"- [{t.id}] {t.name}: {t.description[:100]}" for t in context.tasks[:20]]
        )

        depth_instruction = {
            "deep": "Analyze task descriptions to find semantic dependencies. Look for: data flow, API calls, shared state.",
            "exhaustive": "Perform exhaustive analysis including: code paths, data transformations, error handling chains, resource dependencies.",
        }.get(depth, "Analyze dependencies between tasks.")

        prompt = f"""You are an expert software architect. Analyze dependencies between these tasks.

TASKS:
{tasks_desc}

INSTRUCTIONS:
{depth_instruction}

Identify dependencies where Task A MUST complete before Task B can start.

Output ONLY valid JSON array:
[
  {{
    "from": "task_id_B",
    "to": "task_id_A",
    "type": "hard|soft",
    "reason": "Why A must complete before B",
    "confidence": 0.0-1.0
  }}
]

IMPORTANT: Only output dependencies that truly exist, not all possible combinations.
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=self._config.llm_temperature,
                max_tokens=2048,
            )

            if not response.choices:
                return []

            content = response.choices[0].message.content
            return self._parse_dependency_response(content)

        except Exception as e:
            self._logger.warning(f"LLM dependency analysis failed: {e}")
            return []

    def _parse_dependency_response(self, content: str) -> list[Dependency]:
        """Parse LLM dependency response"""
        dependencies: list[Dependency] = []

        json_match = re.search(r"\[[\s\S]*\]", content)
        if not json_match:
            return dependencies

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return dependencies

        for item in data[:30]:
            try:
                dep = Dependency(
                    from_task=item.get("from", ""),
                    to_task=item.get("to", ""),
                    dependency_type=item.get("type", "soft"),
                    reason=item.get("reason", ""),
                    confidence=item.get("confidence", 0.5),
                    detected_by="llm",
                )
                if dep.from_task and dep.to_task:
                    dependencies.append(dep)
            except Exception:
                continue

        return dependencies

    def _break_cycles(
        self,
        graph: DependencyGraph,
        cycles: list[list[str]],
    ) -> DependencyGraph:
        """Break cycles by removing weakest dependencies"""
        for cycle in cycles:
            if len(cycle) < 2:
                continue

            weakest_dep = None
            weakest_conf = 1.0

            for i in range(len(cycle) - 1):
                from_task = cycle[i]
                to_task = cycle[i + 1]

                for dep in graph.dependencies:
                    if dep.from_task == from_task and dep.to_task == to_task:
                        if dep.confidence < weakest_conf:
                            weakest_conf = dep.confidence
                            weakest_dep = dep

            if weakest_dep:
                graph.dependencies.remove(weakest_dep)
                if weakest_dep.from_task in graph._adjacency:
                    graph._adjacency[weakest_dep.from_task].remove(weakest_dep.to_task)
                if weakest_dep.to_task in graph._reverse_adjacency:
                    graph._reverse_adjacency[weakest_dep.to_task].remove(weakest_dep.from_task)

                self._logger.info(
                    f"Broke cycle by removing: {weakest_dep.from_task} -> {weakest_dep.to_task}"
                )

        return graph

    async def detect_hidden_dependencies(
        self,
        tasks: list[IntelligentTask],
        previous_results: list[Any],
    ) -> list[Dependency]:
        """
        Detect hidden dependencies from previous execution results.

        Uses LLM to analyze patterns in previous executions.
        """
        if not self._provider or not previous_results:
            return []

        failure_patterns = []
        for result in previous_results:
            if hasattr(result, "error") and result.error:
                failure_patterns.append(
                    {
                        "task": result.task_id if hasattr(result, "task_id") else "unknown",
                        "error": str(result.error)[:200],
                    }
                )

        if not failure_patterns:
            return []

        tasks_desc = "\n".join([f"- [{t.id}] {t.name}" for t in tasks[:15]])

        prompt = f"""Analyze these task execution failures to find hidden dependencies.

TASKS:
{tasks_desc}

FAILURE PATTERNS:
{json.dumps(failure_patterns[:10], indent=2)}

Identify dependencies that might not be obvious from task descriptions but are revealed by failure patterns.

Output JSON:
[
  {{
    "from": "task_id",
    "to": "task_id",
    "reason": "Hidden dependency revealed by failure",
    "confidence": 0.0-1.0
  }}
]
"""

        try:
            response = await self._provider.chat_completion(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                model=getattr(self._provider, "default_model", "llama-3.3-70b-versatile"),
                temperature=0.3,
                max_tokens=1024,
            )

            if response.choices:
                content = response.choices[0].message.content
                return self._parse_dependency_response(content)

        except Exception as e:
            self._logger.warning(f"Hidden dependency detection failed: {e}")

        return []

    def get_stats(self) -> dict[str, int]:
        """Get resolution statistics"""
        return {
            "total_resolutions": self._resolutions,
            "llm_resolutions": self._llm_resolutions,
            "rule_resolutions": self._rule_resolutions,
        }


def create_dependency_engine(
    provider: Any = None,
    config: Layer2Config | None = None,
) -> SemanticDependencyEngine:
    """Factory function to create SemanticDependencyEngine"""
    return SemanticDependencyEngine(provider=provider, config=config)
