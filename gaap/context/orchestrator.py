# Context Orchestrator
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from gaap.core.base import ContextManager
from gaap.core.types import (
    ContextBudget,
    ContextLevel,
    ContextWindow,
    Task,
    TaskComplexity,
    TaskPriority,
    TaskType,
)

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


class ContextStrategy(Enum):
    """استراتيجيات إدارة السياق"""

    HCL = auto()  # Hierarchical Context Loading
    PKG = auto()  # Project Knowledge Graph
    TERRITORY = auto()  # Territory Mapping
    SMART_CHUNKING = auto()  # Smart Chunking
    EXTERNAL_BRAIN = auto()  # External Brain
    HYBRID = auto()  # مزيج من الاستراتيجيات


class ProjectSize(Enum):
    """حجم المشروع"""

    TINY = auto()  # < 10k tokens
    SMALL = auto()  # 10k - 100k tokens
    MEDIUM = auto()  # 100k - 1M tokens
    LARGE = auto()  # 1M - 10M tokens
    HUGE = auto()  # 10M - 100M tokens
    MASSIVE = auto()  # 100M+ tokens


class TaskScope(Enum):
    """نطاق المهمة"""

    LOCAL = auto()  # ملف واحد
    MODULE = auto()  # وحدة واحدة
    CROSS_MODULE = auto()  # عدة وحدات
    SYSTEM_WIDE = auto()  # النظام بالكامل


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ContextDecision:
    """قرار إدارة السياق"""

    strategy: ContextStrategy
    budget: ContextBudget
    estimated_tokens: int
    confidence: float
    reasoning: str
    alternatives: list[ContextStrategy] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectProfile:
    """ملف تعريف المشروع"""

    name: str
    total_files: int = 0
    total_lines: int = 0
    estimated_tokens: int = 0
    size: ProjectSize = ProjectSize.SMALL
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    main_components: list[str] = field(default_factory=list)
    last_analyzed: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "estimated_tokens": self.estimated_tokens,
            "size": self.size.name,
            "languages": self.languages,
            "frameworks": self.frameworks,
            "main_components": self.main_components,
            "last_analyzed": self.last_analyzed.isoformat() if self.last_analyzed else None,
        }


@dataclass
class AgentTerritory:
    """منطقة وكيل"""

    agent_id: str
    zone_name: str
    files: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    interfaces: list[str] = field(default_factory=list)  # ملفات التفاعل مع مناطق أخرى


# =============================================================================
# Context Orchestrator
# =============================================================================


class ContextOrchestrator:
    """
    منسق السياق - العقل الذي يوزع المزايا بذكاء

    يقرر:
    - أي استراتيجية استخدام
    - كيفية توزيع الميزانية
    - متى يتم تحديث السياق
    """

    # حدود الميزانية الافتراضية
    DEFAULT_BUDGETS = {
        "minimal": 10_000,
        "low": 20_000,
        "medium": 50_000,
        "high": 80_000,
        "critical": 120_000,
        "unlimited": 500_000,
    }

    # مضاعفات الأولوية
    PRIORITY_MULTIPLIERS = {
        TaskPriority.CRITICAL: 2.0,
        TaskPriority.HIGH: 1.5,
        TaskPriority.NORMAL: 1.0,
        TaskPriority.LOW: 0.6,
        TaskPriority.BACKGROUND: 0.4,
    }

    def __init__(
        self,
        project_path: str | None = None,
        default_budget_level: str = "medium",
        enable_caching: bool = True,
    ):
        self.project_path = project_path
        self.default_budget_level = default_budget_level
        self.enable_caching = enable_caching

        self._logger = get_logger("gaap.context.orchestrator")
        self._project_profile: ProjectProfile | None = None
        self._territories: dict[str, AgentTerritory] = {}
        self._context_cache: dict[str, ContextWindow] = {}
        self._decision_history: list[ContextDecision] = []

        # المكونات الفرعية (ستُحمّل عند الحاجة)
        self._hcl = None
        self._pkg_agent = None
        self._smart_chunker = None
        self._external_brain = None

    # =========================================================================
    # Project Analysis
    # =========================================================================

    async def analyze_project(self, force: bool = False) -> ProjectProfile:
        """تحليل المشروع"""
        if self._project_profile and not force:
            return self._project_profile

        if not self.project_path:
            self._project_profile = ProjectProfile(name="unknown")
            return self._project_profile

        self._logger.info(f"Analyzing project: {self.project_path}")

        profile = ProjectProfile(name=os.path.basename(self.project_path))

        # تحليل الملفات
        for root, dirs, files in os.walk(self.project_path):
            # تجاهل المجلدات المخفية و node_modules
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d != "node_modules" and d != "__pycache__"
            ]

            for file in files:
                if file.startswith("."):
                    continue

                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()

                try:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        lines = content.count("\n") + 1
                        tokens = len(content.split()) * 1.5  # تقدير تقريبي

                        profile.total_files += 1
                        profile.total_lines += lines
                        profile.estimated_tokens += int(tokens)

                        # تحديد اللغة
                        lang_map = {
                            ".py": "Python",
                            ".js": "JavaScript",
                            ".ts": "TypeScript",
                            ".jsx": "React",
                            ".tsx": "React",
                            ".java": "Java",
                            ".go": "Go",
                            ".rs": "Rust",
                            ".rb": "Ruby",
                            ".php": "PHP",
                            ".cs": "C#",
                            ".cpp": "C++",
                            ".c": "C",
                            ".swift": "Swift",
                            ".kt": "Kotlin",
                        }
                        lang = lang_map.get(ext)
                        if lang and lang not in profile.languages:
                            profile.languages.append(lang)

                except Exception as e:
                    self._logger.debug(f"Could not read {file_path}: {e}")

        # تحديد حجم المشروع
        profile.size = self._determine_project_size(profile.estimated_tokens)
        profile.last_analyzed = datetime.now()

        self._project_profile = profile

        self._logger.info(
            f"Project analyzed: {profile.total_files} files, "
            f"{profile.estimated_tokens:,} tokens, size={profile.size.name}"
        )

        return profile

    def _determine_project_size(self, tokens: int) -> ProjectSize:
        """تحديد حجم المشروع"""
        if tokens < 10_000:
            return ProjectSize.TINY
        elif tokens < 100_000:
            return ProjectSize.SMALL
        elif tokens < 1_000_000:
            return ProjectSize.MEDIUM
        elif tokens < 10_000_000:
            return ProjectSize.LARGE
        elif tokens < 100_000_000:
            return ProjectSize.HUGE
        else:
            return ProjectSize.MASSIVE

    # =========================================================================
    # Strategy Decision
    # =========================================================================

    async def decide_strategy(
        self, task: Task, budget_override: int | None = None
    ) -> ContextDecision:
        """
        تحديد أفضل استراتيجية لإدارة السياق

        Args:
            task: المهمة المراد تنفيذها
            budget_override: تجاوز الميزانية (اختياري)

        Returns:
            قرار السياق
        """
        # تحليل المشروع إذا لم يكن تم
        if not self._project_profile:
            await self.analyze_project()

        # تحليل المهمة
        task_scope = self._analyze_task_scope(task)

        # حساب الميزانية
        base_budget = self.DEFAULT_BUDGETS.get(self.default_budget_level, 50_000)
        priority_mult = self.PRIORITY_MULTIPLIERS.get(task.priority, 1.0)
        budget_amount = int(base_budget * priority_mult)

        if budget_override:
            budget_amount = budget_override

        # تحديد الاستراتيجية المثلى
        strategy, reasoning = self._select_strategy(
            project_size=self._project_profile.size,
            task_scope=task_scope,
            task_type=task.type,
            budget=budget_amount,
        )

        # تقدير الرموز المطلوبة
        estimated_tokens = self._estimate_required_tokens(
            strategy=strategy, task_scope=task_scope, task=task
        )

        # بناء القرار
        budget = ContextBudget(total=budget_amount, level=self._get_context_level(task_scope))

        decision = ContextDecision(
            strategy=strategy,
            budget=budget,
            estimated_tokens=estimated_tokens,
            confidence=self._calculate_confidence(strategy, task_scope),
            reasoning=reasoning,
            alternatives=self._get_alternative_strategies(strategy),
            metadata={
                "project_size": self._project_profile.size.name,
                "task_scope": task_scope.name,
                "task_type": task.type.name,
            },
        )

        self._decision_history.append(decision)

        self._logger.info(
            f"Strategy decided: {strategy.name} "
            f"(confidence: {decision.confidence:.0%}, "
            f"budget: {budget_amount:,} tokens)"
        )

        return decision

    def _analyze_task_scope(self, task: Task) -> TaskScope:
        """تحليل نطاق المهمة"""
        # من سياق المهمة
        if task.context:
            deps = task.context.dependencies
            if len(deps) == 0:
                return TaskScope.LOCAL
            elif len(deps) <= 3:
                return TaskScope.MODULE
            elif len(deps) <= 10:
                return TaskScope.CROSS_MODULE
            else:
                return TaskScope.SYSTEM_WIDE

        # من تعقيد المهمة
        scope_from_complexity = {
            TaskComplexity.TRIVIAL: TaskScope.LOCAL,
            TaskComplexity.SIMPLE: TaskScope.LOCAL,
            TaskComplexity.MODERATE: TaskScope.MODULE,
            TaskComplexity.COMPLEX: TaskScope.CROSS_MODULE,
            TaskComplexity.ARCHITECTURAL: TaskScope.SYSTEM_WIDE,
        }

        return scope_from_complexity.get(task.complexity, TaskScope.MODULE)

    def _select_strategy(
        self, project_size: ProjectSize, task_scope: TaskScope, task_type: TaskType, budget: int
    ) -> tuple[ContextStrategy, str]:
        """اختيار الاستراتيجية المثلى"""

        # المشاريع الضخمة تحتاج PKG
        if project_size in (ProjectSize.HUGE, ProjectSize.MASSIVE):
            if task_scope == TaskScope.SYSTEM_WIDE:
                return ContextStrategy.PKG, "PKB needed for massive project system-wide task"
            return ContextStrategy.HYBRID, "Hybrid approach for massive project"

        # المشاريع الكبيرة
        if project_size == ProjectSize.LARGE:
            if task_scope in (TaskScope.CROSS_MODULE, TaskScope.SYSTEM_WIDE):
                return ContextStrategy.TERRITORY, "Territory mapping for cross-module task"
            return ContextStrategy.HCL, "HCL for large project local task"

        # المشاريع المتوسطة
        if project_size == ProjectSize.MEDIUM:
            if task_type in (TaskType.PLANNING, TaskType.ANALYSIS):
                return ContextStrategy.EXTERNAL_BRAIN, "External brain for analysis task"
            return ContextStrategy.HCL, "HCL for medium project"

        # المشاريع الصغيرة
        if project_size in (ProjectSize.SMALL, ProjectSize.TINY):
            return ContextStrategy.SMART_CHUNKING, "Smart chunking sufficient for small project"

        # الافتراضي
        return ContextStrategy.HCL, "Default HCL strategy"

    def _estimate_required_tokens(
        self, strategy: ContextStrategy, task_scope: TaskScope, task: Task
    ) -> int:
        """تقدير الرموز المطلوبة"""
        base_estimates = {
            TaskScope.LOCAL: 5_000,
            TaskScope.MODULE: 20_000,
            TaskScope.CROSS_MODULE: 50_000,
            TaskScope.SYSTEM_WIDE: 100_000,
        }

        base = base_estimates.get(task_scope, 20_000)

        # تعديل بناءً على الاستراتيجية
        strategy_multipliers = {
            ContextStrategy.HCL: 1.0,
            ContextStrategy.PKG: 0.3,  # PKG يقلل السياق المطلوب
            ContextStrategy.TERRITORY: 0.5,
            ContextStrategy.SMART_CHUNKING: 1.2,
            ContextStrategy.EXTERNAL_BRAIN: 0.7,
            ContextStrategy.HYBRID: 0.6,
        }

        mult = strategy_multipliers.get(strategy, 1.0)

        return int(base * mult)

    def _get_context_level(self, task_scope: TaskScope) -> "ContextLevel":
        """الحصول على مستوى السياق"""
        level_map = {
            TaskScope.LOCAL: ContextLevel.LEVEL_3_FULL,
            TaskScope.MODULE: ContextLevel.LEVEL_2_FILE,
            TaskScope.CROSS_MODULE: ContextLevel.LEVEL_1_MODULE,
            TaskScope.SYSTEM_WIDE: ContextLevel.LEVEL_0_OVERVIEW,
        }
        return level_map.get(task_scope, ContextLevel.LEVEL_2_FILE)

    def _calculate_confidence(self, strategy: ContextStrategy, task_scope: TaskScope) -> float:
        """حساب الثقة في القرار"""
        # ثقة عالية للتطابقات الواضحة
        confidence_map = {
            (ContextStrategy.PKG, TaskScope.SYSTEM_WIDE): 0.95,
            (ContextStrategy.TERRITORY, TaskScope.CROSS_MODULE): 0.90,
            (ContextStrategy.HCL, TaskScope.MODULE): 0.90,
            (ContextStrategy.HCL, TaskScope.LOCAL): 0.95,
            (ContextStrategy.SMART_CHUNKING, TaskScope.LOCAL): 0.85,
        }

        return confidence_map.get((strategy, task_scope), 0.75)

    def _get_alternative_strategies(self, primary: ContextStrategy) -> list[ContextStrategy]:
        """الحصول على الاستراتيجيات البديلة"""
        alternatives_map = {
            ContextStrategy.PKG: [ContextStrategy.HYBRID, ContextStrategy.TERRITORY],
            ContextStrategy.HYBRID: [ContextStrategy.HCL, ContextStrategy.TERRITORY],
            ContextStrategy.TERRITORY: [ContextStrategy.HCL, ContextStrategy.PKG],
            ContextStrategy.HCL: [ContextStrategy.SMART_CHUNKING, ContextStrategy.EXTERNAL_BRAIN],
            ContextStrategy.SMART_CHUNKING: [ContextStrategy.HCL],
            ContextStrategy.EXTERNAL_BRAIN: [ContextStrategy.HCL, ContextStrategy.PKG],
        }

        return alternatives_map.get(primary, [ContextStrategy.HCL])

    # =========================================================================
    # Context Loading
    # =========================================================================

    async def load_context(
        self, decision: ContextDecision, task: Task, focus_files: list[str] | None = None
    ) -> ContextManager:
        """تحميل السياق بناءً على القرار"""
        budget = decision.budget
        manager = ContextManager(budget=budget)

        # تحميل حسب الاستراتيجية
        if decision.strategy == ContextStrategy.HCL:
            await self._load_hcl(manager, task, focus_files)
        elif decision.strategy == ContextStrategy.SMART_CHUNKING:
            await self._load_smart_chunks(manager, task, focus_files)
        elif decision.strategy == ContextStrategy.EXTERNAL_BRAIN:
            await self._load_from_brain(manager, task)
        elif decision.strategy == ContextStrategy.TERRITORY:
            await self._load_territory(manager, task)
        elif decision.strategy == ContextStrategy.PKG:
            await self._load_from_pkg(manager, task)
        elif decision.strategy == ContextStrategy.HYBRID:
            await self._load_hybrid(manager, task, focus_files)

        return manager

    async def _load_hcl(
        self, manager: ContextManager, task: Task, focus_files: list[str] | None
    ) -> None:
        """تحميل باستخدام HCL"""
        if self._hcl is None:
            from gaap.context.hcl import HierarchicalContextLoader

            self._hcl = HierarchicalContextLoader(self.project_path)

        # تحميل المستوى 0 دائماً
        overview = await self._hcl.load_level(ContextLevel.LEVEL_0_OVERVIEW)
        if overview:
            window = ContextWindow(
                id="project_overview",
                content=overview.content,
                token_count=overview.token_count,
                level=ContextLevel.LEVEL_0_OVERVIEW,
                priority=100,
            )
            manager.add_window(window)

        # تحميل الملفات المطلوبة
        if focus_files:
            for file_path in focus_files[:5]:  # حد أقصى 5 ملفات
                node = await self._hcl.load_file_context(file_path)
                if node:
                    window = ContextWindow(
                        id=f"file_{file_path}",
                        content=node.content,
                        token_count=node.token_count,
                        level=ContextLevel.LEVEL_3_FULL,
                        source=file_path,
                    )
                    manager.add_window(window)

    async def _load_smart_chunks(
        self, manager: ContextManager, task: Task, focus_files: list[str] | None
    ) -> None:
        """تحميل باستخدام Smart Chunking"""
        if self._smart_chunker is None:
            from gaap.context.smart_chunking import SmartChunker

            self._smart_chunker = SmartChunker(self.project_path)

        if focus_files:
            for file_path in focus_files[:3]:
                chunks = await self._smart_chunker.chunk_file(file_path)
                for chunk in chunks[:10]:  # حد أقصى 10 chunks
                    window = ContextWindow(
                        id=f"chunk_{chunk.id}",
                        content=chunk.content,
                        token_count=chunk.token_count,
                        level=ContextLevel.LEVEL_3_FULL,
                        source=file_path,
                        metadata={"chunk_type": chunk.chunk_type.name},
                    )
                    manager.add_window(window)

    async def _load_from_brain(self, manager: ContextManager, task: Task) -> None:
        """تحميل من External Brain"""
        if self._external_brain is None:
            from gaap.context.external_brain import ExternalBrain

            self._external_brain = ExternalBrain(self.project_path)

        # البحث عن السياق ذي الصلة
        query = task.description
        results = await self._external_brain.search(query, limit=5)

        for result in results:
            window = ContextWindow(
                id=f"brain_{result.id}",
                content=result.content,
                token_count=result.token_count,
                level=ContextLevel.LEVEL_2_FILE,
                source=result.source,
                metadata={"relevance_score": result.relevance_score},
            )
            manager.add_window(window)

    async def _load_territory(self, manager: ContextManager, task: Task) -> None:
        """تحميل منطقة معينة"""
        # تحديد المنطقة من المهمة
        territory = self._find_relevant_territory(task)

        if territory:
            for file_path in territory.files[:10]:
                if os.path.exists(file_path):
                    try:
                        with open(file_path) as f:
                            content = f.read()
                            tokens = len(content.split()) * 1.5

                            window = ContextWindow(
                                id=f"territory_{file_path}",
                                content=content[:50000],  # حد
                                token_count=int(tokens),
                                level=ContextLevel.LEVEL_2_FILE,
                                source=file_path,
                            )
                            manager.add_window(window)
                    except Exception as e:
                        self._logger.warning(f"Could not load {file_path}: {e}")

    async def _load_from_pkg(self, manager: ContextManager, task: Task) -> None:
        """تحميل من PKG"""
        if self._pkg_agent is None:
            from gaap.context.pkg_agent import PKGAgent

            self._pkg_agent = PKGAgent(self.project_path)

        # الحصول على الرسم البياني ذي الصلة
        relevant_nodes = await self._pkg_agent.find_relevant_nodes(task.description)

        for node in relevant_nodes[:20]:
            window = ContextWindow(
                id=f"pkg_{node.id}",
                content=node.summary or node.name,
                token_count=len((node.summary or node.name).split()) * 1.5,
                level=ContextLevel.LEVEL_1_MODULE,
                source=node.file_path,
                metadata={"node_type": node.node_type},
            )
            manager.add_window(window)

    async def _load_hybrid(
        self, manager: ContextManager, task: Task, focus_files: list[str] | None
    ) -> None:
        """تحميل هجين"""
        # مزج من عدة مصادر
        await self._load_hcl(manager, task, focus_files)
        await self._load_from_brain(manager, task)

    def _find_relevant_territory(self, task: Task) -> AgentTerritory | None:
        """العثور على المنطقة ذات الصلة"""
        # تحليل المهمة للعثور على منطقة
        for territory in self._territories.values():
            if any(
                keyword.lower() in " ".join(territory.files).lower()
                for keyword in task.description.split()[:5]
            ):
                return territory

        return None

    # =========================================================================
    # Territory Management
    # =========================================================================

    def define_territory(
        self, agent_id: str, zone_name: str, files: list[str], interfaces: list[str] | None = None
    ) -> AgentTerritory:
        """تحديد منطقة وكيل"""
        territory = AgentTerritory(
            agent_id=agent_id, zone_name=zone_name, files=files, interfaces=interfaces or []
        )

        # تقدير الرموز
        total_tokens = 0
        for file_path in files:
            try:
                with open(file_path) as f:
                    total_tokens += len(f.read().split()) * 1.5
            except Exception:
                pass

        territory.estimated_tokens = int(total_tokens)
        self._territories[agent_id] = territory

        self._logger.info(
            f"Territory defined: {zone_name} for agent {agent_id} "
            f"({len(files)} files, ~{territory.estimated_tokens:,} tokens)"
        )

        return territory

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """الحصول على الإحصائيات"""
        return {
            "project": self._project_profile.to_dict() if self._project_profile else None,
            "territories": len(self._territories),
            "cache_size": len(self._context_cache),
            "decisions_made": len(self._decision_history),
            "default_budget_level": self.default_budget_level,
        }

    def get_project_profile(self) -> ProjectProfile | None:
        """الحصول على ملف المشروع"""
        return self._project_profile


# =============================================================================
# Convenience Functions
# =============================================================================


def create_orchestrator(project_path: str, budget_level: str = "medium") -> ContextOrchestrator:
    """إنشاء منسق سياق"""
    return ContextOrchestrator(project_path=project_path, default_budget_level=budget_level)
