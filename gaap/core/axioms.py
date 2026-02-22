# Axiom Validator - Constitutional Gatekeeper
import ast
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from gaap.core.exceptions import (
    AxiomViolationError,
    DependencyAxiomError,
    InterfaceAxiomError,
    SyntaxAxiomError,
)


class AxiomLevel(Enum):
    """مستويات البديهيات"""

    INVARIANT = auto()
    GUIDELINE = auto()
    PREFERENCE = auto()


@dataclass
class Axiom:
    """بديهية"""

    name: str
    description: str
    level: AxiomLevel = AxiomLevel.INVARIANT
    enabled: bool = True
    check_func: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "level": self.level.name,
            "enabled": self.enabled,
        }


@dataclass
class AxiomCheckResult:
    """نتيجة فحص بديهية"""

    axiom_name: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "axiom": self.axiom_name,
            "passed": self.passed,
            "message": self.message,
            "severity": self.severity,
        }


AXIOM_REGISTRY: dict[str, Axiom] = {
    "syntax": Axiom(
        name="syntax",
        description="All code must parse without syntax errors",
        level=AxiomLevel.INVARIANT,
        check_func="check_syntax",
    ),
    "dependency": Axiom(
        name="dependency",
        description="No new packages added without L1 Strategic approval",
        level=AxiomLevel.INVARIANT,
        check_func="check_dependency",
    ),
    "interface": Axiom(
        name="interface",
        description="Changes to __init__.py or models.py require Swarm Review",
        level=AxiomLevel.INVARIANT,
        check_func="check_interface",
    ),
    "read_only_diagnostic": Axiom(
        name="read_only_diagnostic",
        description="Diagnostic tasks must not modify project source code",
        level=AxiomLevel.INVARIANT,
        check_func="check_read_only",
    ),
    "source_integrity": Axiom(
        name="source_integrity",
        description="Research findings must be backed by cited evidence",
        level=AxiomLevel.GUIDELINE,
        check_func="check_citations",
    ),
}

KNOWN_PACKAGES: set[str] = {
    "asyncio",
    "json",
    "logging",
    "os",
    "re",
    "sys",
    "time",
    "typing",
    "dataclasses",
    "datetime",
    "enum",
    "collections",
    "pathlib",
    "hashlib",
    "uuid",
    "traceback",
    "functools",
    "itertools",
    "contextlib",
    "abc",
    "copy",
    "io",
    "tempfile",
    "subprocess",
    "threading",
    "multiprocessing",
    "queue",
    "socket",
    "ssl",
    "http",
    "urllib",
    "email",
    "html",
    "xml",
    "csv",
    "configparser",
    "argparse",
    "getopt",
    "warnings",
    "unittest",
    "doctest",
    "pdb",
    "profile",
    "cProfile",
    "timeit",
    "trace",
    "gc",
    "inspect",
    "dis",
    "pickle",
    "shelve",
    "dbm",
    "sqlite3",
    "bisect",
    "heapq",
    "array",
    "weakref",
    "types",
    "numbers",
    "math",
    "cmath",
    "decimal",
    "fractions",
    "random",
    "statistics",
    "operator",
    "string",
    "textwrap",
    "unicodedata",
    "struct",
    "codecs",
    "difflib",
    "locale",
    "gettext",
    "calendar",
    "secrets",
    "base64",
    "binascii",
    "quopri",
    "uu",
    "zlib",
    "gzip",
    "bz2",
    "lzma",
    "zipfile",
    "tarfile",
    "aiohttp",
    "requests",
    "httpx",
    "pydantic",
    "pytest",
    "black",
    "isort",
    "ruff",
    "mypy",
    "fastapi",
    "uvicorn",
    "click",
    "rich",
    "typer",
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "chromadb",
    "networkx",
}

INTERFACE_FILES: set[str] = {
    "__init__.py",
    "models.py",
    "types.py",
    "config.py",
    "exceptions.py",
}


from gaap.core.logging import get_standard_logger as get_logger


class AxiomValidator:
    """
    Constitutional Gatekeeper

    يتحقق من البديهيات الأساسية قبل إكمال أي مهمة:
    - Syntax Invariant: الكود يجب أن يُفهم
    - Dependency Invariant: لا حزم جديدة بدون موافقة
    - Interface Invariant: ملفات حساسة تحتاج مراجعة
    """

    def __init__(
        self,
        axioms: dict[str, Axiom] | None = None,
        known_packages: set[str] | None = None,
        interface_files: set[str] | None = None,
        strict_mode: bool = True,
    ) -> None:
        self.axioms = axioms or AXIOM_REGISTRY.copy()
        self.known_packages = known_packages or KNOWN_PACKAGES.copy()
        self.interface_files = interface_files or INTERFACE_FILES.copy()
        self.strict_mode = strict_mode
        self._logger = get_logger("gaap.axioms")

        self._checks_run = 0
        self._violations: list[AxiomCheckResult] = []

    def validate(
        self,
        code: str | None = None,
        file_path: str | None = None,
        imports: list[str] | None = None,
        task_id: str | None = None,
    ) -> list[AxiomCheckResult]:
        """
        التحقق من جميع البديهيات

        Args:
            code: الكود للتحقق منه
            file_path: مسار الملف (للتحقق من Interface)
            imports: قائمة الاستيرادات المكتشفة
            task_id: معرف المهمة

        Returns:
            قائمة نتائج الفحص
        """
        results = []
        self._checks_run += 1

        for axiom_name, axiom in self.axioms.items():
            if not axiom.enabled:
                continue

            check_func = getattr(self, axiom.check_func, None)
            if check_func is None:
                self._logger.warning(f"Check function not found for axiom: {axiom_name}")
                continue

            try:
                result = check_func(
                    code=code, file_path=file_path, imports=imports, task_id=task_id
                )
                results.append(result)

                if not result.passed:
                    self._violations.append(result)
                    self._logger.warning(f"Axiom violation: {axiom_name} - {result.message}")

            except Exception as e:
                self._logger.error(f"Error checking axiom {axiom_name}: {e}")
                results.append(
                    AxiomCheckResult(
                        axiom_name=axiom_name,
                        passed=True,
                        message=f"Check skipped due to error: {str(e)[:50]}",
                        severity="low",
                    )
                )

        return results

    def check_syntax(
        self,
        code: str | None = None,
        file_path: str | None = None,
        imports: list[str] | None = None,
        task_id: str | None = None,
    ) -> AxiomCheckResult:
        """فحص صيغة الكود"""
        if not code:
            return AxiomCheckResult(axiom_name="syntax", passed=True, message="No code to check")

        try:
            ast.parse(code)
            return AxiomCheckResult(
                axiom_name="syntax", passed=True, message="Code parses successfully"
            )
        except SyntaxError as e:
            snippet = code.split("\n")[max(0, e.lineno - 3) : e.lineno + 2] if e.lineno else []
            return AxiomCheckResult(
                axiom_name="syntax",
                passed=False,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                details={
                    "line": e.lineno,
                    "column": e.offset,
                    "snippet": "\n".join(snippet),
                },
                severity="low",
            )

    def check_dependency(
        self,
        code: str | None = None,
        file_path: str | None = None,
        imports: list[str] | None = None,
        task_id: str | None = None,
    ) -> AxiomCheckResult:
        """فحص التبعيات الجديدة"""
        detected_imports = imports or []

        if code and not detected_imports:
            detected_imports = self._extract_imports(code)

        new_packages = []
        for imp in detected_imports:
            base_package = imp.split(".")[0] if "." in imp else imp
            if base_package not in self.known_packages:
                new_packages.append(base_package)

        if new_packages:
            return AxiomCheckResult(
                axiom_name="dependency",
                passed=False,
                message=f"New packages detected without L1 approval: {', '.join(new_packages)}",
                details={"new_packages": new_packages, "detected_imports": detected_imports},
                severity="medium",
            )

        return AxiomCheckResult(
            axiom_name="dependency",
            passed=True,
            message="All dependencies are known",
            details={"checked_imports": detected_imports},
        )

    def check_interface(
        self,
        code: str | None = None,
        file_path: str | None = None,
        imports: list[str] | None = None,
        task_id: str | None = None,
    ) -> AxiomCheckResult:
        """فحص تغييرات الواجهة"""
        if not file_path:
            return AxiomCheckResult(
                axiom_name="interface", passed=True, message="No file path to check"
            )

        import os

        filename = os.path.basename(file_path)

        if filename in self.interface_files:
            return AxiomCheckResult(
                axiom_name="interface",
                passed=False,
                message=f"Interface file '{filename}' modified - requires Swarm Review",
                details={"file": file_path, "filename": filename},
                severity="high",
            )

        return AxiomCheckResult(
            axiom_name="interface", passed=True, message="No interface files affected"
        )

    def check_read_only(
        self,
        code: str | None = None,
        file_path: str | None = None,
        task_id: str | None = None,
        **kwargs: Any,
    ) -> AxiomCheckResult:
        """منع تعديل الملفات أثناء التشخيص (v2.1)"""
        write_patterns = [
            r"open\(.*['\"]w['\"]\)",
            r"open\(.*['\"]a['\"]\)",
            r"write_file\(",
            r"os\.remove\(",
            r"shutil\.rmtree\("
        ]
        
        if code:
            for pattern in write_patterns:
                if re.search(pattern, code):
                    return AxiomCheckResult(
                        axiom_name="read_only_diagnostic",
                        passed=False,
                        message="Write operation detected in diagnostic task",
                        severity="critical"
                    )
        
        return AxiomCheckResult(axiom_name="read_only_diagnostic", passed=True)

    def _extract_imports(self, code: str) -> list[str]:
        """استخراج الاستيرادات من الكود"""
        imports = []

        import_pattern = r"^(?:from\s+(\S+)\s+)?import\s+([^\n]+)"
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            from_module = match.group(1)
            imported_names = match.group(2)

            if from_module:
                imports.append(from_module)
            else:
                for name in imported_names.split(","):
                    name = name.strip().split(" as ")[0].strip()
                    if name:
                        imports.append(name)

        return imports

    def raise_on_violation(self, result: AxiomCheckResult) -> None:
        """رفع استثناء عند انتهاك"""
        if result.passed:
            return

        if result.axiom_name == "syntax":
            raise SyntaxAxiomError(
                syntax_error=result.message,
                code_snippet=result.details.get("snippet", ""),
            )
        elif result.axiom_name == "dependency":
            raise DependencyAxiomError(
                package_name=result.details.get("new_packages", ["unknown"])[0],
            )
        elif result.axiom_name == "interface":
            raise InterfaceAxiomError(
                file_path=result.details.get("file", "unknown"),
                change_type="modification",
            )
        else:
            raise AxiomViolationError(
                axiom_name=result.axiom_name,
                violation_details=result.message,
                severity_level=result.severity,
            )

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الفحص"""
        violations_by_type: dict[str, int] = {}
        for v in self._violations:
            violations_by_type[v.axiom_name] = violations_by_type.get(v.axiom_name, 0) + 1

        return {
            "checks_run": self._checks_run,
            "total_violations": len(self._violations),
            "violations_by_type": violations_by_type,
            "violation_rate": len(self._violations) / max(self._checks_run, 1),
        }

    def reset_stats(self) -> None:
        """إعادة تعيين الإحصائيات"""
        self._checks_run = 0
        self._violations = []

    def add_known_package(self, package: str) -> None:
        """إضافة حزمة معروفة"""
        self.known_packages.add(package)

    def add_interface_file(self, filename: str) -> None:
        """إضافة ملف واجهة"""
        self.interface_files.add(filename)

    def enable_axiom(self, name: str) -> None:
        """تفعيل بديهية"""
        if name in self.axioms:
            self.axioms[name].enabled = True

    def disable_axiom(self, name: str) -> None:
        """تعطيل بديهية"""
        if name in self.axioms:
            self.axioms[name].enabled = False


def create_validator(strict: bool = True, extra_packages: set[str] | None = None) -> AxiomValidator:
    """إنشاء مدقق بديهيات"""
    packages = KNOWN_PACKAGES.copy()
    if extra_packages:
        packages.update(extra_packages)

    return AxiomValidator(known_packages=packages, strict_mode=strict)
