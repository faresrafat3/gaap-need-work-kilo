# Hierarchical Context Layer
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# =============================================================================
# Context Level
# =============================================================================


class ContextLevel(Enum):
    """مستويات تحميل السياق"""

    LEVEL_0_OVERVIEW = 0  # نظرة عامة (~100 tokens)
    LEVEL_1_MODULE = 1  # نظرة وحدة (~500 tokens)
    LEVEL_2_FILE = 2  # نظرة ملف (~2k tokens)
    LEVEL_3_FULL = 3  # محتوى كامل (~20k+ tokens)
    LEVEL_4_DEPENDENCIES = 4  # مع التبعيات


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ContextNode:
    """عقدة سياق"""

    id: str
    name: str
    level: ContextLevel
    content: str
    token_count: int
    file_path: str = ""
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "token_count": self.token_count,
            "file_path": self.file_path,
            "children_count": len(self.children),
        }


@dataclass
class FileSummary:
    """ملخص ملف"""

    path: str
    name: str
    language: str
    overview: str  # ~50 tokens
    imports: list[str]  # ~50 tokens
    exports: list[str]  # ~50 tokens
    main_components: list[str]  # ~100 tokens
    line_count: int
    estimated_tokens: int


# =============================================================================
# Hierarchical Context Loader
# =============================================================================


class HierarchicalContextLoader:
    """
    المحمل الهرمي للسياق

    الميزات:
    - تحميل تدريجي حسب الحاجة
    - تلخيص ذكي للملفات
    - التوسع عند الطلب
    - تخزين مؤقت
    """

    # حدود الرموز لكل مستوى
    LEVEL_LIMITS = {
        ContextLevel.LEVEL_0_OVERVIEW: 100,
        ContextLevel.LEVEL_1_MODULE: 500,
        ContextLevel.LEVEL_2_FILE: 2000,
        ContextLevel.LEVEL_3_FULL: 50000,
        ContextLevel.LEVEL_4_DEPENDENCIES: 100000,
    }

    def __init__(self, project_path: str) -> None:
        self.project_path = project_path
        self._logger = logging.getLogger("gaap.context.hcl")

        # التخزين المؤقت
        self._cache: dict[str, ContextNode] = {}
        self._file_summaries: dict[str, FileSummary] = {}
        self._module_structure: dict[str, list[str]] = {}

        # العقدة الجذر
        self._root: ContextNode | None = None

    # =========================================================================
    # Loading Methods
    # =========================================================================

    async def load_level(
        self, level: ContextLevel, node_id: str | None = None
    ) -> ContextNode | None:
        """تحميل مستوى معين"""
        if node_id is None:
            # تحميل من الجذر
            if level == ContextLevel.LEVEL_0_OVERVIEW:
                return await self._load_project_overview()
            elif level == ContextLevel.LEVEL_1_MODULE:
                return await self._load_module_overview()

        # تحميل عقدة محددة
        return await self._load_node(node_id, level)

    async def _load_project_overview(self) -> ContextNode:
        """تحميل نظرة عامة على المشروع"""
        cache_key = "project_overview"

        if cache_key in self._cache:
            return self._cache[cache_key]

        self._logger.info("Loading project overview (Level 0)")

        # جمع المعلومات الأساسية
        total_files = 0
        languages: dict[str, int] = {}
        modules = []

        for root, dirs, files in os.walk(self.project_path):
            # تجاهل المجلدات المخفية
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "venv", ".git"]
            ]

            # اكتشاف الوحدات
            rel_path = os.path.relpath(root, self.project_path)
            if rel_path != "." and "/" not in rel_path:
                modules.append(rel_path)

            for file in files:
                if file.startswith("."):
                    continue

                total_files += 1
                ext = os.path.splitext(file)[1].lower()

                # تحديد اللغة
                lang_map = {
                    ".py": "Python",
                    ".js": "JavaScript",
                    ".ts": "TypeScript",
                    ".java": "Java",
                    ".go": "Go",
                    ".rs": "Rust",
                }
                lang = lang_map.get(ext, "Other")
                languages[lang] = languages.get(lang, 0) + 1

        # بناء النظرة العامة
        overview = f"# Project: {os.path.basename(self.project_path)}\n\n"
        overview += f"- Total files: {total_files}\n"
        overview += f"- Modules: {len(modules)}\n"
        overview += f"- Languages: {', '.join(f'{k}({v})' for k, v in sorted(languages.items(), key=lambda x: -x[1])[:5])}\n"

        if modules:
            overview += "\n## Main Modules\n"
            for m in modules[:10]:
                overview += f"- {m}\n"

        node = ContextNode(
            id="project_root",
            name=os.path.basename(self.project_path),
            level=ContextLevel.LEVEL_0_OVERVIEW,
            content=overview,
            token_count=int(len(overview.split()) * 1.5),
            children=[f"module:{m}" for m in modules],
        )

        self._cache[cache_key] = node
        self._root = node

        return node

    async def _load_module_overview(self) -> ContextNode:
        """تحميل نظرة على وحدة"""
        # إنشاء نظرة مجمعة للوحدات
        modules_content = "# Modules Overview\n\n"

        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "venv", ".git"]
            ]

            rel_path = os.path.relpath(root, self.project_path)
            if rel_path == "." or "/" in rel_path:
                continue

            # تلخيص الوحدة
            module_files = [f for f in files if not f.startswith(".")]
            if module_files:
                modules_content += f"\n## {rel_path}\n"
                modules_content += f"Files: {len(module_files)}\n"

                # أهم الملفات
                for f in sorted(module_files)[:5]:
                    modules_content += f"- {f}\n"

        node = ContextNode(
            id="modules_overview",
            name="Modules",
            level=ContextLevel.LEVEL_1_MODULE,
            content=modules_content,
            token_count=int(len(modules_content.split()) * 1.5),
        )

        return node

    async def _load_node(self, node_id: str, level: ContextLevel) -> ContextNode | None:
        """تحميل عقدة محددة"""
        if node_id in self._cache:
            return self._cache[node_id]

        # تحديد نوع العقدة
        if node_id.startswith("module:"):
            return await self._load_module_node(node_id, level)
        elif node_id.startswith("file:"):
            return await self._load_file_node(node_id, level)

        return None

    async def _load_module_node(self, node_id: str, level: ContextLevel) -> ContextNode | None:
        """تحميل عقدة وحدة"""
        module_name = node_id.replace("module:", "")
        module_path = os.path.join(self.project_path, module_name)

        if not os.path.isdir(module_path):
            return None

        content = f"# Module: {module_name}\n\n"
        children = []

        for item in os.listdir(module_path):
            item_path = os.path.join(module_path, item)

            if os.path.isfile(item) and not item.startswith("."):
                if level.value >= ContextLevel.LEVEL_2_FILE.value:
                    # تلخيص الملف
                    summary = await self._summarize_file(item_path)
                    content += f"\n## {item}\n{summary.overview}\n"
                children.append(f"file:{item_path}")

            elif os.path.isdir(item) and not item.startswith("."):
                children.append(f"module:{module_name}/{item}")

        node = ContextNode(
            id=node_id,
            name=module_name,
            level=level,
            content=content,
            token_count=int(len(content.split()) * 1.5),
            children=children,
        )

        self._cache[node_id] = node
        return node

    async def _load_file_node(self, node_id: str, level: ContextLevel) -> ContextNode | None:
        """تحميل عقدة ملف"""
        file_path = node_id.replace("file:", "")

        if not os.path.isfile(file_path):
            return None

        content = ""

        if level == ContextLevel.LEVEL_2_FILE:
            # ملخص الملف
            summary = await self._summarize_file(file_path)
            content = self._build_file_summary_content(summary)
        elif level == ContextLevel.LEVEL_3_FULL:
            # المحتوى الكامل
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                content = f"Error reading file: {e}"

        node = ContextNode(
            id=node_id,
            name=os.path.basename(file_path),
            level=level,
            content=content,
            token_count=int(len(content.split()) * 1.5),
            file_path=file_path,
        )

        self._cache[node_id] = node
        return node

    # =========================================================================
    # File Summarization
    # =========================================================================

    async def _summarize_file(self, file_path: str) -> FileSummary:
        """تلخيص ملف"""
        if file_path in self._file_summaries:
            return self._file_summaries[file_path]

        ext = os.path.splitext(file_path)[1].lower()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            return FileSummary(
                path=file_path,
                name=os.path.basename(file_path),
                language="unknown",
                overview="Could not read file",
                imports=[],
                exports=[],
                main_components=[],
                line_count=0,
                estimated_tokens=0,
            )

        # تحليل حسب نوع الملف
        if ext == ".py":
            summary = self._summarize_python(file_path, content)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            summary = self._summarize_javascript(file_path, content)
        else:
            summary = self._summarize_generic(file_path, content)

        self._file_summaries[file_path] = summary
        return summary

    def _summarize_python(self, path: str, content: str) -> FileSummary:
        """تلخيص ملف Python"""
        lines = content.split("\n")

        # استخراج imports
        imports = []
        for line in lines:
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line.strip()[:100])

        # استخراج المكونات الرئيسية
        components = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("class "):
                components.append(stripped.split("(")[0].replace("class ", ""))
            elif stripped.startswith("def ") or stripped.startswith("async def "):
                func_name = stripped.split("(")[0].replace("def ", "").replace("async ", "")
                components.append(func_name)

        # نظرة عامة
        overview_lines = []
        for line in lines[:20]:
            if (
                line.strip().startswith("#")
                or line.strip().startswith('"""')
                or line.strip().startswith("'''")
            ):
                overview_lines.append(line.strip("#'\" "))

        return FileSummary(
            path=path,
            name=os.path.basename(path),
            language="Python",
            overview=" ".join(overview_lines[:3])[:200] or "Python module",
            imports=imports[:10],
            exports=components[:10],
            main_components=components[:20],
            line_count=len(lines),
            estimated_tokens=int(len(content.split()) * 1.5),
        )

    def _summarize_javascript(self, path: str, content: str) -> FileSummary:
        """تلخيص ملف JavaScript/TypeScript"""
        lines = content.split("\n")

        # استخراج imports
        imports = []
        import_pattern = re.compile(r'import\s+.*?from\s+[\'"](.+?)[\'"]')
        for line in lines:
            match = import_pattern.search(line)
            if match:
                imports.append(match.group(1))

        # استخراج exports
        exports = []
        export_pattern = re.compile(r"export\s+(?:default\s+)?(?:function|class|const)\s+(\w+)")
        for line in lines:
            match = export_pattern.search(line)
            if match:
                exports.append(match.group(1))

        # استخراج المكونات
        components = []
        func_pattern = re.compile(
            r"(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\()"
        )
        for match in func_pattern.finditer(content):
            name = match.group(1) or match.group(2)
            if name:
                components.append(name)

        return FileSummary(
            path=path,
            name=os.path.basename(path),
            language="TypeScript" if ".ts" in path else "JavaScript",
            overview=f"Module with {len(components)} functions/components",
            imports=imports[:10],
            exports=exports[:10],
            main_components=components[:20],
            line_count=len(lines),
            estimated_tokens=int(len(content.split()) * 1.5),
        )

    def _summarize_generic(self, path: str, content: str) -> FileSummary:
        """تلخيص ملف عام"""
        lines = content.split("\n")

        return FileSummary(
            path=path,
            name=os.path.basename(path),
            language="generic",
            overview=f"File with {len(lines)} lines",
            imports=[],
            exports=[],
            main_components=[],
            line_count=len(lines),
            estimated_tokens=int(len(content.split()) * 1.5),
        )

    def _build_file_summary_content(self, summary: FileSummary) -> str:
        """بناء محتوى ملخص الملف"""
        content = f"# {summary.name}\n\n"
        content += f"Language: {summary.language}\n"
        content += f"Lines: {summary.line_count}\n\n"

        if summary.overview:
            content += f"## Overview\n{summary.overview}\n\n"

        if summary.imports:
            content += f"## Imports ({len(summary.imports)})\n"
            for imp in summary.imports[:5]:
                content += f"- {imp}\n"
            content += "\n"

        if summary.main_components:
            content += f"## Components ({len(summary.main_components)})\n"
            for comp in summary.main_components[:10]:
                content += f"- {comp}\n"

        return content

    # =========================================================================
    # Utility Methods
    # =========================================================================

    async def load_file_context(
        self, file_path: str, include_dependencies: bool = False
    ) -> ContextNode | None:
        """تحميل سياق ملف"""
        node_id = f"file:{file_path}"
        level = (
            ContextLevel.LEVEL_4_DEPENDENCIES if include_dependencies else ContextLevel.LEVEL_3_FULL
        )
        return await self._load_file_node(node_id, level)

    def clear_cache(self) -> None:
        """مسح التخزين المؤقت"""
        self._cache.clear()
        self._file_summaries.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """إحصائيات التخزين المؤقت"""
        return {
            "cached_nodes": len(self._cache),
            "cached_summaries": len(self._file_summaries),
            "total_tokens_cached": sum(n.token_count for n in self._cache.values()),
        }
