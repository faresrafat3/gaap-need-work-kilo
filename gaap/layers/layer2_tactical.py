# mypy: ignore-errors
# Layer 2: Tactical Layer
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from gaap.core.base import BaseLayer
from gaap.core.types import (
    ExecutionStatus,
    LayerType,
    Message,
    MessageRole,
    Task,
    TaskComplexity,
    TaskPriority,
    TaskResult,
    TaskType,
)
from gaap.layers.layer1_strategic import ArchitectureSpec

# =============================================================================
# Logger Setup
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Enums
# =============================================================================


class TaskCategory(Enum):
    """تصنيفات المهام"""

    SETUP = auto()
    DATABASE = auto()
    API = auto()
    FRONTEND = auto()
    TESTING = auto()
    DOCUMENTATION = auto()
    INTEGRATION = auto()
    SECURITY = auto()
    INFRASTRUCTURE = auto()


class DependencyType(Enum):
    """أنواع التبعيات"""

    HARD = "hard"  # يجب أن تكتمل قبل البدء
    SOFT = "soft"  # يفضل أن تكتمل
    CONDITIONAL = "conditional"  # بناءً على شرط


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AtomicTask:
    """مهمة ذرية"""

    id: str
    name: str
    description: str
    category: TaskCategory

    # التفاصيل
    type: TaskType = TaskType.CODE_GENERATION
    priority: TaskPriority = TaskPriority.NORMAL
    complexity: TaskComplexity = TaskComplexity.SIMPLE

    # القيود
    constraints: dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: list[str] = field(default_factory=list)

    # التبعيات
    dependencies: list[str] = field(default_factory=list)
    dependency_type: dict[str, DependencyType] = field(default_factory=dict)

    # الموارد
    estimated_tokens: int = 500
    estimated_time_minutes: int = 5
    estimated_cost_usd: float = 0.01

    # الحالة
    status: ExecutionStatus = ExecutionStatus.PENDING
    result: TaskResult | None = None

    # الـ metadata
    assigned_agent: str = ""
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_task(self) -> Task:
        """تحويل لـ Task"""
        return Task(
            id=self.id,
            description=self.description,
            type=self.type,
            priority=self.priority,
            complexity=self.complexity,
            constraints=self.constraints,
            estimated_tokens=self.estimated_tokens,
        )


@dataclass
class TaskNode:
    """عقدة في رسم المهام"""

    task: AtomicTask
    children: list["TaskNode"] = field(default_factory=list)
    parents: list["TaskNode"] = field(default_factory=list)
    level: int = 0

    @property
    def id(self) -> str:
        return self.task.id

    def is_ready(self, completed: set[str]) -> bool:
        """هل المهمة جاهزة للتنفيذ؟"""
        return all(dep_id in completed for dep_id in self.task.dependencies)


@dataclass
class TaskGraph:
    """رسم بياني للمهام (DAG)"""

    root_nodes: list[TaskNode] = field(default_factory=list)
    all_nodes: dict[str, TaskNode] = field(default_factory=dict)

    # الإحصائيات
    total_tasks: int = 0
    max_depth: int = 0
    critical_path: list[str] = field(default_factory=list)

    def add_task(self, task: AtomicTask) -> TaskNode:
        """إضافة مهمة"""
        node = TaskNode(task=task)
        self.all_nodes[task.id] = node
        self.total_tasks += 1
        return node

    def add_dependency(
        self, task_id: str, depends_on_id: str, dep_type: DependencyType = DependencyType.HARD
    ) -> None:
        """إضافة تبعية"""
        if task_id not in self.all_nodes or depends_on_id not in self.all_nodes:
            return

        child = self.all_nodes[task_id]
        parent = self.all_nodes[depends_on_id]

        child.parents.append(parent)
        parent.children.append(child)

        # Note: do NOT modify child.task.dependencies here.
        # resolve() iterates over it — mutating during iteration causes infinite loop.
        child.task.dependency_type[depends_on_id] = dep_type

    def get_ready_tasks(self, completed: set[str], in_progress: set[str]) -> list[AtomicTask]:
        """الحصول على المهام الجاهزة"""
        ready = []

        for node in self.all_nodes.values():
            if node.task.id in completed or node.task.id in in_progress:
                continue

            if node.is_ready(completed):
                ready.append(node.task)

        # ترتيب حسب الأولوية
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3,
            TaskPriority.BACKGROUND: 4,
        }

        ready.sort(key=lambda t: priority_order.get(t.priority, 2))

        return ready

    def calculate_levels(self) -> None:
        """حساب مستويات المهام"""
        # BFS من الجذور
        visited = set()
        queue = []

        # العثور على الجذور (مهام بدون تبعيات)
        for node in self.all_nodes.values():
            if not node.parents:
                node.level = 0
                queue.append(node)
                self.root_nodes.append(node)

        while queue:
            current = queue.pop(0)

            if current.id in visited:
                continue

            visited.add(current.id)

            for child in current.children:
                child.level = max(child.level, current.level + 1)
                self.max_depth = max(self.max_depth, child.level)

                if child.id not in visited:
                    queue.append(child)

    def find_critical_path(self) -> list[str]:
        """العثور على المسار الحرج"""
        # تقريب: أطول مسار
        self.calculate_levels()

        # البحث عن العقد الأعمق
        max_level_node = max(self.all_nodes.values(), key=lambda n: n.level, default=None)

        if not max_level_node:
            return []

        # الرجوع للخلف (with cycle protection)
        path = [max_level_node.id]
        visited = {max_level_node.id}
        current = max_level_node

        while current.parents:
            # اختيار الأب ذو المستوى الأعلى
            parent = max(current.parents, key=lambda p: p.level)
            if parent.id in visited:
                break  # cycle detected, stop
            path.insert(0, parent.id)
            visited.add(parent.id)
            current = parent

        self.critical_path = path
        return path

    def detect_cycles(self) -> list[str]:
        """كشف الدورات"""
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(self.all_nodes, WHITE)
        cycles = []

        def dfs(node_id: str, path: list[str]) -> None:
            color[node_id] = GRAY
            path.append(node_id)

            node = self.all_nodes[node_id]
            for child in node.children:
                if color[child.id] == GRAY:
                    # دورة найдены
                    cycle_start = path.index(child.id)
                    cycles.append(path[cycle_start:] + [child.id])
                elif color[child.id] == WHITE:
                    dfs(child.id, path.copy())

            color[node_id] = BLACK

        for node_id in self.all_nodes:
            if color[node_id] == WHITE:
                dfs(node_id, [])

        return cycles

    def to_dict(self) -> dict[str, Any]:
        """تحويل لقاموس"""
        return {
            "total_tasks": self.total_tasks,
            "max_depth": self.max_depth,
            "critical_path_length": len(self.critical_path),
            "root_tasks": [n.id for n in self.root_nodes],
        }


# =============================================================================
# Dependency Resolver
# =============================================================================


class DependencyResolver:
    """حل التبعيات"""

    def __init__(self):
        self._logger = get_logger("gaap.layer2.resolver")

    def resolve(self, tasks: list[AtomicTask]) -> TaskGraph:
        """حل التبعيات وبناء الرسم البياني"""
        graph = TaskGraph()

        # إضافة جميع المهام
        for task in tasks:
            graph.add_task(task)

        # إضافة التبعيات (iterate over copy to be safe)
        for task in tasks:
            for dep_id in list(task.dependencies):
                graph.add_dependency(task.id, dep_id)

        # كشف الدورات
        cycles = graph.detect_cycles()
        if cycles:
            self._logger.error(f"Circular dependencies detected: {cycles}")
            # يمكن إضافة معالجة هنا

        # حساب المستويات
        graph.calculate_levels()

        # العثور على المسار الحرج
        graph.find_critical_path()

        return graph

    def topological_sort(self, graph: TaskGraph) -> list[str]:
        """ترتيب طوبولوجي"""
        # Kahn's algorithm
        in_degree = dict.fromkeys(graph.all_nodes, 0)

        for node in graph.all_nodes.values():
            for child in node.children:
                in_degree[child.id] += 1

        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for child in graph.all_nodes[current].children:
                in_degree[child.id] -= 1
                if in_degree[child.id] == 0:
                    queue.append(child.id)

        return result


# =============================================================================
# Tactical Decomposer
# =============================================================================


class TacticalDecomposer:
    """
    المحلل التكتيكي الذكي (LLM-Powered)

    يحول المواصفات المعمارية إلى مهام ذرية باستخدام الـ LLM:
    - تحليل الطلب الأصلي والمعمارية
    - توليد مهام مخصصة ومرتبطة بالطلب فعلياً
    - تحديد التبعيات بين المهام
    - تقدير الموارد لكل مهمة
    - Fallback ذكي في حالة فشل الـ LLM
    """

    # تعيين الفئات من نص إلى Enum
    CATEGORY_MAP = {
        "setup": TaskCategory.SETUP,
        "database": TaskCategory.DATABASE,
        "api": TaskCategory.API,
        "frontend": TaskCategory.FRONTEND,
        "testing": TaskCategory.TESTING,
        "documentation": TaskCategory.DOCUMENTATION,
        "security": TaskCategory.SECURITY,
        "integration": TaskCategory.INTEGRATION,
        "infrastructure": TaskCategory.INFRASTRUCTURE,
    }

    PRIORITY_MAP = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW,
    }

    COMPLEXITY_MAP = {
        "simple": TaskComplexity.SIMPLE,
        "moderate": TaskComplexity.MODERATE,
        "complex": TaskComplexity.COMPLEX,
    }

    TYPE_MAP = {
        "code_generation": TaskType.CODE_GENERATION,
        "code_review": TaskType.CODE_REVIEW,
        "debugging": TaskType.DEBUGGING,
        "testing": TaskType.TESTING,
        "documentation": TaskType.DOCUMENTATION,
        "analysis": TaskType.ANALYSIS,
        "refactoring": TaskType.REFACTORING,
    }

    # البرومبت الهيكلي لتفكيك المهام
    DECOMPOSITION_PROMPT = """You are an expert task decomposition system. Your job is to break down a user's request into atomic, actionable subtasks.

## Original User Request
{original_text}

## Architecture Context
- Paradigm: {paradigm}
- Data Strategy: {data_strategy}
- Communication: {communication}
- Goals: {goals}
- Intent Type: {intent_type}

## Instructions
Break this request into {max_tasks} or fewer atomic subtasks. Each subtask should be:
1. **Specific** to the actual request (NOT generic templates)
2. **Actionable** — clear what needs to be done
3. **Atomic** — can be executed independently or with clear dependencies
4. Ordered logically with dependencies

## Output Format
Return ONLY a valid JSON array. Each element must have these fields:
```json
[
  {{
    "name": "Short task name (max 60 chars)",
    "description": "Detailed description of what to implement/do. Be specific about the actual code, logic, or content needed.",
    "category": "one of: setup|database|api|frontend|testing|documentation|security|integration|infrastructure",
    "type": "one of: code_generation|code_review|debugging|testing|documentation|analysis|refactoring",
    "priority": "one of: critical|high|normal|low",
    "complexity": "one of: simple|moderate|complex",
    "depends_on": [0],
    "estimated_minutes": 15
  }}
]
```

The `depends_on` field contains 0-based indices of tasks that must complete first. An empty array means no dependencies.

CRITICAL: Tasks must be SPECIFIC to "{original_text}". Do NOT generate generic tasks like "Initialize project" or "Setup linting". Generate tasks that directly address the user's actual request.

Return ONLY the JSON array, no markdown fences, no explanation."""

    def __init__(self, max_subtasks: int = 50, provider=None):
        self.max_subtasks = max_subtasks
        self._provider = provider
        self._logger = get_logger("gaap.layer2.decomposer")
        self._task_counter = 0
        self._llm_decompositions = 0
        self._fallback_decompositions = 0

    async def decompose(self, spec: ArchitectureSpec, goals: list[str]) -> list[AtomicTask]:
        """
        تفكيك المواصفات لمهام ذرية

        يستخدم الـ LLM أولاً، وإذا فشل يرجع لتفكيك ذكي هيوريستيكي
        """
        # استخراج النص الأصلي من الـ metadata
        original_intent = spec.metadata.get("original_intent", {})
        original_text = original_intent.get("original_text", "")
        intent_type = original_intent.get("intent_type", "UNKNOWN")
        explicit_goals = original_intent.get("explicit_goals", goals)

        # محاولة التفكيك بالـ LLM
        if self._provider and original_text:
            try:
                tasks = await self._llm_decompose(spec, original_text, intent_type, explicit_goals)
                if tasks:
                    self._llm_decompositions += 1
                    self._logger.info(f"LLM decomposition successful: {len(tasks)} tasks")
                    return tasks
            except Exception as e:
                self._logger.warning(f"LLM decomposition failed, using smart fallback: {e}")

        # Fallback: تفكيك ذكي بدون LLM
        tasks = await self._smart_fallback_decompose(
            spec, original_text, intent_type, explicit_goals
        )
        self._fallback_decompositions += 1
        return tasks

    async def _llm_decompose(
        self, spec: ArchitectureSpec, original_text: str, intent_type: str, goals: list[str]
    ) -> list[AtomicTask] | None:
        """تفكيك باستخدام الـ LLM"""
        # بناء البرومبت
        prompt = self.DECOMPOSITION_PROMPT.format(
            original_text=original_text,
            paradigm=spec.paradigm.value,
            data_strategy=spec.data_strategy.value,
            communication=spec.communication.value,
            goals=", ".join(goals) if goals else "Not specified",
            intent_type=intent_type,
            max_tasks=self.max_subtasks,
        )

        # إرسال للـ LLM
        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a precise task decomposition engine. Output only valid JSON arrays.",
            ),
            Message(role=MessageRole.USER, content=prompt),
        ]

        response = await self._provider.chat_completion(
            messages=messages,
            model=self._provider.default_model,
        )

        raw_response = response.choices[0].message.content

        # تنظيف واستخراج الـ JSON
        tasks_data = self._parse_llm_response(raw_response)

        if not tasks_data:
            self._logger.warning("Failed to parse LLM response into tasks")
            return None

        # تحويل لـ AtomicTask objects
        tasks = self._convert_to_atomic_tasks(tasks_data, spec)

        # التحقق من صحة التبعيات
        tasks = self._validate_and_fix_dependencies(tasks)

        return tasks

    def _parse_llm_response(self, raw: str) -> list[dict] | None:
        """تنظيف واستخراج JSON من رد الـ LLM"""
        if not raw:
            return None

        # إزالة أي markdown fences
        cleaned = raw.strip()

        # إزالة thinking tags إذا موجودة
        thinking_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)
        cleaned = thinking_pattern.sub("", cleaned).strip()

        # إزالة markdown code fences
        if cleaned.startswith("```"):
            # إزالة أول سطر (```json أو ```)
            lines = cleaned.split("\n")
            start_idx = 1
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        # محاولة parse مباشرة
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "tasks" in data:
                return data["tasks"]
        except json.JSONDecodeError:
            pass

        # محاولة استخراج JSON array من النص
        bracket_match = re.search(r"\[[\s\S]*\]", cleaned)
        if bracket_match:
            try:
                data = json.loads(bracket_match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        self._logger.error(f"Could not parse LLM response: {cleaned[:200]}...")
        return None

    def _convert_to_atomic_tasks(
        self, tasks_data: list[dict], spec: ArchitectureSpec
    ) -> list[AtomicTask]:
        """تحويل بيانات JSON لـ AtomicTask objects"""
        tasks = []
        task_id_map = {}  # index → task_id

        for idx, data in enumerate(tasks_data):
            if idx >= self.max_subtasks:
                break

            task_id = self._generate_task_id()
            task_id_map[idx] = task_id

            # تعيين الفئة
            category_str = data.get("category", "api").lower().strip()
            category = self.CATEGORY_MAP.get(category_str, TaskCategory.API)

            # تعيين النوع
            type_str = data.get("type", "code_generation").lower().strip()
            task_type = self.TYPE_MAP.get(type_str, TaskType.CODE_GENERATION)

            # تعيين الأولوية
            priority_str = data.get("priority", "normal").lower().strip()
            priority = self.PRIORITY_MAP.get(priority_str, TaskPriority.NORMAL)

            # تعيين التعقيد
            complexity_str = data.get("complexity", "moderate").lower().strip()
            complexity = self.COMPLEXITY_MAP.get(complexity_str, TaskComplexity.MODERATE)

            # تقدير الوقت
            est_minutes = data.get("estimated_minutes", 15)
            if not isinstance(est_minutes, (int, float)):
                est_minutes = 15

            task = AtomicTask(
                id=task_id,
                name=str(data.get("name", f"Task {idx + 1}"))[:80],
                description=str(data.get("description", data.get("name", f"Task {idx + 1}"))),
                category=category,
                type=task_type,
                priority=priority,
                complexity=complexity,
                estimated_time_minutes=int(est_minutes),
                estimated_tokens=int(est_minutes * 80),
                constraints={
                    "language": spec.tech_stack.get("language", "python"),
                    "framework": spec.tech_stack.get("framework", ""),
                },
                metadata={
                    "source": "llm_decomposition",
                    "original_index": idx,
                    "raw_depends_on": data.get("depends_on", []),
                },
            )
            tasks.append(task)

        # ربط التبعيات باستخدام الـ index map
        for task in tasks:
            raw_deps = task.metadata.get("raw_depends_on", [])
            if isinstance(raw_deps, list):
                for dep_idx in raw_deps:
                    if isinstance(dep_idx, int) and dep_idx in task_id_map:
                        dep_id = task_id_map[dep_idx]
                        if dep_id != task.id:  # لا تعتمد على نفسها
                            task.dependencies.append(dep_id)

        return tasks

    def _validate_and_fix_dependencies(self, tasks: list[AtomicTask]) -> list[AtomicTask]:
        """التحقق وإصلاح التبعيات (إزالة الدورات)"""
        valid_ids = {t.id for t in tasks}

        for task in tasks:
            # إزالة تبعيات غير موجودة
            task.dependencies = [d for d in task.dependencies if d in valid_ids]
            # إزالة تبعيات ذاتية
            task.dependencies = [d for d in task.dependencies if d != task.id]

        # كشف وإزالة الدورات (بسيط)
        task_map = {t.id: t for t in tasks}

        def has_cycle(task_id: str, visited: set, stack: set) -> bool:
            visited.add(task_id)
            stack.add(task_id)
            for dep_id in task_map.get(
                task_id, AtomicTask(id="", name="", description="", category=TaskCategory.SETUP)
            ).dependencies:
                if dep_id not in visited:
                    if has_cycle(dep_id, visited, stack):
                        return True
                elif dep_id in stack:
                    # إزالة التبعية الدائرية
                    task_map[task_id].dependencies.remove(dep_id)
                    self._logger.warning(f"Removed circular dependency: {task_id} → {dep_id}")
                    return False
            stack.discard(task_id)
            return False

        visited = set()
        for task in tasks:
            if task.id not in visited:
                has_cycle(task.id, visited, set())

        return tasks

    async def _smart_fallback_decompose(
        self, spec: ArchitectureSpec, original_text: str, intent_type: str, goals: list[str]
    ) -> list[AtomicTask]:
        """
        تفكيك ذكي هيوريستيكي (Fallback)

        بدلاً من القوالب الثابتة، يحلل النص ويولد مهام مرتبطة
        """
        tasks = []

        # تحليل النص لاستخراج كلمات مفتاحية ونوع العمل
        text_lower = (original_text or "").lower()

        if intent_type == "CODE_GENERATION" or any(
            kw in text_lower for kw in ["write", "create", "implement", "build", "اكتب", "أنشئ"]
        ):
            tasks = self._decompose_code_generation(original_text, spec, goals)
        elif intent_type == "DEBUGGING" or any(
            kw in text_lower for kw in ["debug", "fix", "error", "bug", "صلح", "خطأ"]
        ):
            tasks = self._decompose_debugging(original_text, spec)
        elif intent_type == "CODE_REVIEW" or any(
            kw in text_lower for kw in ["review", "analyze", "check", "راجع"]
        ):
            tasks = self._decompose_code_review(original_text, spec)
        elif intent_type == "ARCHITECTURE" or any(
            kw in text_lower for kw in ["design", "architect", "system", "صمم", "معمارية"]
        ):
            tasks = self._decompose_architecture(original_text, spec, goals)
        elif intent_type == "REFACTORING" or any(
            kw in text_lower for kw in ["refactor", "improve", "optimize", "حسن"]
        ):
            tasks = self._decompose_refactoring(original_text, spec)
        else:
            # تفكيك عام مبني على الأهداف
            tasks = self._decompose_general(original_text, spec, goals)

        # تقليم إذا تجاوز الحد
        if len(tasks) > self.max_subtasks:
            tasks = tasks[: self.max_subtasks]

        return tasks

    def _decompose_code_generation(
        self, original_text: str, spec: ArchitectureSpec, goals: list[str]
    ) -> list[AtomicTask]:
        """تفكيك مهمة برمجية"""
        tasks = []

        # المهمة الأساسية: تحليل المتطلبات
        analyze_task = AtomicTask(
            id=self._generate_task_id(),
            name=f"Analyze requirements: {original_text[:50]}",
            description=f"Analyze the requirements for: {original_text}\n\nIdentify:\n- Input/output specifications\n- Edge cases\n- Performance requirements\n- Data structures needed",
            category=TaskCategory.SETUP,
            type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=5,
            estimated_tokens=400,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(analyze_task)

        # المهمة الرئيسية: التنفيذ
        impl_task = AtomicTask(
            id=self._generate_task_id(),
            name=f"Implement: {original_text[:50]}",
            description=(
                f"Implement the following:\n{original_text}\n\nRequirements:\n"
                + "\n".join(f"- {g}" for g in goals)
                if goals
                else original_text
            ),
            category=TaskCategory.API,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=20,
            estimated_tokens=1500,
            dependencies=[analyze_task.id],
            constraints={
                "language": spec.tech_stack.get("language", "python"),
            },
            metadata={"source": "smart_fallback"},
        )
        tasks.append(impl_task)

        # مهمة الاختبار
        test_task = AtomicTask(
            id=self._generate_task_id(),
            name=f"Write tests for: {original_text[:40]}",
            description=f"Write comprehensive tests for the implementation of: {original_text}\n\nInclude:\n- Unit tests for core logic\n- Edge case tests\n- Input validation tests",
            category=TaskCategory.TESTING,
            type=TaskType.TESTING,
            priority=TaskPriority.NORMAL,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=15,
            estimated_tokens=1000,
            dependencies=[impl_task.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(test_task)

        return tasks

    def _decompose_debugging(self, original_text: str, spec: ArchitectureSpec) -> list[AtomicTask]:
        """تفكيك مهمة تصحيح أخطاء"""
        tasks = []

        # تحليل الخطأ
        analyze = AtomicTask(
            id=self._generate_task_id(),
            name=f"Analyze error: {original_text[:50]}",
            description=f"Analyze and diagnose the following issue:\n{original_text}\n\nSteps:\n1. Identify the root cause\n2. Trace the execution flow\n3. Identify affected components",
            category=TaskCategory.API,
            type=TaskType.DEBUGGING,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=10,
            estimated_tokens=800,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(analyze)

        # تطبيق الإصلاح
        fix = AtomicTask(
            id=self._generate_task_id(),
            name=f"Fix: {original_text[:55]}",
            description=f"Implement the fix for:\n{original_text}\n\nProvide:\n1. The corrected code\n2. Explanation of what was wrong\n3. How the fix resolves the issue",
            category=TaskCategory.API,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=15,
            estimated_tokens=1200,
            dependencies=[analyze.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(fix)

        # اختبار التصحيح
        verify = AtomicTask(
            id=self._generate_task_id(),
            name=f"Verify fix: {original_text[:50]}",
            description=f"Verify that the fix for '{original_text[:60]}' works correctly:\n1. Test the fix with the original failing case\n2. Test edge cases\n3. Ensure no regression",
            category=TaskCategory.TESTING,
            type=TaskType.TESTING,
            priority=TaskPriority.NORMAL,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=10,
            estimated_tokens=600,
            dependencies=[fix.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(verify)

        return tasks

    def _decompose_code_review(
        self, original_text: str, spec: ArchitectureSpec
    ) -> list[AtomicTask]:
        """تفكيك مهمة مراجعة كود"""
        tasks = []

        review = AtomicTask(
            id=self._generate_task_id(),
            name=f"Review: {original_text[:55]}",
            description=f"Perform a thorough code review:\n{original_text}\n\nCheck:\n1. Code quality and readability\n2. Potential bugs and edge cases\n3. Performance issues\n4. Security vulnerabilities\n5. Best practices compliance",
            category=TaskCategory.API,
            type=TaskType.CODE_REVIEW,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=20,
            estimated_tokens=1500,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(review)

        suggestions = AtomicTask(
            id=self._generate_task_id(),
            name="Improvement suggestions",
            description=f"Based on the review of:\n{original_text}\n\nProvide:\n1. Specific improvement suggestions with code examples\n2. Refactoring recommendations\n3. Performance optimization tips",
            category=TaskCategory.DOCUMENTATION,
            type=TaskType.ANALYSIS,
            priority=TaskPriority.NORMAL,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=10,
            estimated_tokens=800,
            dependencies=[review.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(suggestions)

        return tasks

    def _decompose_architecture(
        self, original_text: str, spec: ArchitectureSpec, goals: list[str]
    ) -> list[AtomicTask]:
        """تفكيك مهمة تصميم معماري"""
        tasks = []

        # تحليل المتطلبات
        requirements = AtomicTask(
            id=self._generate_task_id(),
            name=f"Requirements analysis: {original_text[:40]}",
            description=f"Analyze requirements for:\n{original_text}\n\nGoals:\n"
            + "\n".join(f"- {g}" for g in goals)
            + "\n\nIdentify:\n1. Functional requirements\n2. Non-functional requirements\n3. Scale and performance needs\n4. Integration points",
            category=TaskCategory.SETUP,
            type=TaskType.ANALYSIS,
            priority=TaskPriority.CRITICAL,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=15,
            estimated_tokens=1000,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(requirements)

        # التصميم المعماري
        design = AtomicTask(
            id=self._generate_task_id(),
            name=f"Architecture design: {original_text[:40]}",
            description=f"Design the architecture for:\n{original_text}\n\nUsing paradigm: {spec.paradigm.value}\nData strategy: {spec.data_strategy.value}\n\nDeliver:\n1. Component diagram\n2. Data flow description\n3. Technology choices with justification\n4. API design",
            category=TaskCategory.INFRASTRUCTURE,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.COMPLEX,
            estimated_time_minutes=30,
            estimated_tokens=2000,
            dependencies=[requirements.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(design)

        # خطة التنفيذ
        plan = AtomicTask(
            id=self._generate_task_id(),
            name="Implementation plan",
            description=f"Create implementation plan for:\n{original_text}\n\nBased on the architecture design, create:\n1. Development phases\n2. Task breakdown per phase\n3. Timeline estimates\n4. Risk mitigation strategies",
            category=TaskCategory.DOCUMENTATION,
            type=TaskType.DOCUMENTATION,
            priority=TaskPriority.NORMAL,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=15,
            estimated_tokens=1200,
            dependencies=[design.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(plan)

        return tasks

    def _decompose_refactoring(
        self, original_text: str, spec: ArchitectureSpec
    ) -> list[AtomicTask]:
        """تفكيك مهمة إعادة هيكلة"""
        tasks = []

        analyze = AtomicTask(
            id=self._generate_task_id(),
            name=f"Analyze for refactoring: {original_text[:40]}",
            description=f"Analyze the current code for refactoring:\n{original_text}\n\nIdentify:\n1. Code smells and anti-patterns\n2. Duplication and coupling issues\n3. Performance bottlenecks\n4. Testability improvements",
            category=TaskCategory.API,
            type=TaskType.CODE_REVIEW,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=15,
            estimated_tokens=1000,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(analyze)

        refactor = AtomicTask(
            id=self._generate_task_id(),
            name=f"Refactor: {original_text[:50]}",
            description=f"Refactor the code:\n{original_text}\n\nApply:\n1. Clean code principles\n2. SOLID principles where applicable\n3. Proper error handling\n4. Clear naming and documentation",
            category=TaskCategory.API,
            type=TaskType.REFACTORING,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=25,
            estimated_tokens=1500,
            dependencies=[analyze.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(refactor)

        test = AtomicTask(
            id=self._generate_task_id(),
            name="Test refactored code",
            description="Verify the refactored code works correctly:\n1. Run existing tests\n2. Add tests for new structure\n3. Verify no regression in behavior",
            category=TaskCategory.TESTING,
            type=TaskType.TESTING,
            priority=TaskPriority.NORMAL,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=10,
            estimated_tokens=800,
            dependencies=[refactor.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(test)

        return tasks

    def _decompose_general(
        self, original_text: str, spec: ArchitectureSpec, goals: list[str]
    ) -> list[AtomicTask]:
        """تفكيك عام للمهام غير المصنفة"""
        tasks = []

        # مهمة التحليل
        analyze = AtomicTask(
            id=self._generate_task_id(),
            name=f"Analyze: {original_text[:55]}",
            description=f"Analyze and understand the following request:\n{original_text}\n\n"
            + (
                "Goals:\n" + "\n".join(f"- {g}" for g in goals)
                if goals
                else "Identify the key objectives and deliverables."
            ),
            category=TaskCategory.SETUP,
            type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=10,
            estimated_tokens=600,
            metadata={"source": "smart_fallback"},
        )
        tasks.append(analyze)

        # مهمة التنفيذ الرئيسية
        execute = AtomicTask(
            id=self._generate_task_id(),
            name=f"Execute: {original_text[:55]}",
            description=f"Execute the main task:\n{original_text}\n\nProvide a comprehensive, high-quality response that directly addresses the request.",
            category=TaskCategory.API,
            type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH,
            complexity=TaskComplexity.MODERATE,
            estimated_time_minutes=20,
            estimated_tokens=1500,
            dependencies=[analyze.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(execute)

        # مهمة التلخيص
        summarize = AtomicTask(
            id=self._generate_task_id(),
            name="Review and summarize results",
            description=f"Review the execution results for:\n{original_text}\n\nSummarize:\n1. What was accomplished\n2. Key findings or deliverables\n3. Any limitations or next steps",
            category=TaskCategory.DOCUMENTATION,
            type=TaskType.DOCUMENTATION,
            priority=TaskPriority.LOW,
            complexity=TaskComplexity.SIMPLE,
            estimated_time_minutes=5,
            estimated_tokens=400,
            dependencies=[execute.id],
            metadata={"source": "smart_fallback"},
        )
        tasks.append(summarize)

        return tasks

    def _generate_task_id(self) -> str:
        """توليد معرف مهمة"""
        self._task_counter += 1
        return f"task_{int(time.time() * 1000)}_{self._task_counter}"

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات المحلل"""
        return {
            "llm_decompositions": self._llm_decompositions,
            "fallback_decompositions": self._fallback_decompositions,
            "total": self._llm_decompositions + self._fallback_decompositions,
        }


# =============================================================================
# Execution Queue
# =============================================================================


class ExecutionQueue:
    """طابور التنفيذ"""

    def __init__(self, max_parallel: int = 10):
        self.max_parallel = max_parallel
        self._queue: list[AtomicTask] = []
        self._completed: set[str] = set()
        self._in_progress: set[str] = set()
        self._logger = get_logger("gaap.layer2.queue")

    def enqueue(self, tasks: list[AtomicTask]) -> None:
        """إضافة مهام للطابور"""
        self._queue.extend(tasks)
        self._logger.info(f"Enqueued {len(tasks)} tasks, total: {len(self._queue)}")

    def get_next_batch(self, graph: TaskGraph) -> list[AtomicTask]:
        """الحصول على الدفعة التالية"""
        ready = graph.get_ready_tasks(self._completed, self._in_progress)

        # تحديد الحجم
        available_slots = self.max_parallel - len(self._in_progress)
        batch = ready[:available_slots]

        # تحديث حالة
        for task in batch:
            self._in_progress.add(task.id)
            task.status = ExecutionStatus.QUEUED

        return batch

    def mark_completed(self, task_id: str) -> None:
        """تحديد مهمة كمكتملة"""
        self._in_progress.discard(task_id)
        self._completed.add(task_id)

    def mark_failed(self, task_id: str) -> None:
        """تحديد مهمة كفاشلة"""
        self._in_progress.discard(task_id)

    def get_progress(self) -> dict[str, Any]:
        """الحصول على التقدم"""
        total = len(self._queue)
        return {
            "total": total,
            "completed": len(self._completed),
            "in_progress": len(self._in_progress),
            "pending": total - len(self._completed) - len(self._in_progress),
            "progress_percent": (len(self._completed) / total * 100) if total > 0 else 0,
        }


# =============================================================================
# Layer 2 Tactical
# =============================================================================


class Layer2Tactical(BaseLayer):
    """
    طبقة التنظيم التكتيكي

    المسؤوليات:
    - تفكيك المواصفات المعمارية لمهام ذرية
    - حل التبعيات وبناء رسم المهام
    - إنشاء طابور التنفيذ
    - تقدير الموارد
    """

    def __init__(self, max_subtasks: int = 50, max_parallel: int = 10, provider=None):
        super().__init__(LayerType.TACTICAL)

        self.decomposer = TacticalDecomposer(max_subtasks=max_subtasks, provider=provider)
        self.resolver = DependencyResolver()
        self.queue = ExecutionQueue(max_parallel=max_parallel)

        self._logger = get_logger("gaap.layer2")

        # الإحصائيات
        self._tasks_created = 0
        self._graphs_built = 0

    async def process(self, input_data: Any) -> TaskGraph:
        """معالجة المدخل"""
        start_time = time.time()

        # استخراج البيانات
        if isinstance(input_data, ArchitectureSpec):
            spec = input_data
            goals = spec.metadata.get("plan", {}).get("phases", [])
        elif isinstance(input_data, dict):
            spec = input_data.get("spec")
            goals = input_data.get("goals", [])
        else:
            raise ValueError("Expected ArchitectureSpec or dict")

        self._logger.info("Tactical decomposition started")

        # 1. تفكيك المهام
        tasks = await self.decomposer.decompose(spec, goals)
        self._tasks_created += len(tasks)

        # 2. حل التبعيات وبناء الرسم
        graph = self.resolver.resolve(tasks)
        self._graphs_built += 1

        # 3. إضافة للطابور
        self.queue.enqueue(tasks)

        elapsed = (time.time() - start_time) * 1000
        self._logger.info(
            f"Task graph created: {len(tasks)} tasks, {graph.max_depth} levels, {elapsed:.0f}ms"
        )

        return graph

    def get_next_tasks(self, graph: TaskGraph) -> list[AtomicTask]:
        """الحصول على المهام التالية"""
        return self.queue.get_next_batch(graph)

    def complete_task(self, task_id: str) -> None:
        """إكمال مهمة"""
        self.queue.mark_completed(task_id)

    def fail_task(self, task_id: str) -> None:
        """فشل مهمة"""
        self.queue.mark_failed(task_id)

    def get_progress(self) -> dict[str, Any]:
        """الحصول على التقدم"""
        return self.queue.get_progress()

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "layer": "L2_Tactical",
            "tasks_created": self._tasks_created,
            "graphs_built": self._graphs_built,
            "current_progress": self.get_progress(),
            "decomposer": self.decomposer.get_stats(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_tactical_layer(
    max_subtasks: int = 50, max_parallel: int = 10, provider=None
) -> Layer2Tactical:
    """إنشاء طبقة تكتيكية"""
    return Layer2Tactical(max_subtasks=max_subtasks, max_parallel=max_parallel, provider=provider)
