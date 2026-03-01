"""
Behavioral Validator - Dynamic Execution Validation
Implements: docs/evolution_plan_2026/41_VALIDATORS_AUDIT_SPEC.md

Features:
- Dynamic execution in sandbox
- Test generation for code
- Pass/fail reporting
- Safety checks
"""

from __future__ import annotations

import ast
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.validators.behavioral")


@dataclass
class TestResult:
    test_name: str
    passed: bool
    message: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "message": self.message,
            "duration_ms": self.duration_ms,
        }


@dataclass
class BehavioralReport:
    is_valid: bool
    tests_passed: int = 0
    tests_failed: int = 0
    test_results: list[TestResult] = field(default_factory=list)
    execution_errors: list[str] = field(default_factory=list)
    total_duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "total_duration_ms": self.total_duration_ms,
            "tests": [t.to_dict() for t in self.test_results],
            "errors": self.execution_errors,
        }


@dataclass
class BehavioralConfig:
    timeout_seconds: int = 10
    max_output_size: int = 10000
    use_sandbox: bool = True
    generate_tests: bool = True

    @classmethod
    def default(cls) -> BehavioralConfig:
        return cls()

    @classmethod
    def strict(cls) -> BehavioralConfig:
        return cls(timeout_seconds=5, use_sandbox=True)

    @classmethod
    def fast(cls) -> BehavioralConfig:
        return cls(timeout_seconds=3, use_sandbox=False, generate_tests=False)


class BehavioralValidator:
    """
    Validates code behavior through dynamic execution.

    Features:
    - Runs code in sandbox
    - Generates tests automatically
    - Checks for runtime errors
    - Safe execution environment

    Usage:
        validator = BehavioralValidator()
        report = validator.validate("def add(a, b): return a + b")
        print(f"Valid: {report.is_valid}")
    """

    def __init__(self, config: BehavioralConfig | None = None) -> None:
        self.config = config or BehavioralConfig.default()
        self._logger = logger
        self._sandbox_available = False

        if self.config.use_sandbox:
            try:
                from gaap.security import get_sandbox

                self._sandbox = get_sandbox(use_docker=False)
                self._sandbox_available = True
            except Exception as e:
                self._logger.warning(f"Sandbox not available: {e}")
                self._sandbox_available = False

    def validate(
        self,
        code: str,
        tests: list[dict[str, Any]] | None = None,
        entry_point: str | None = None,
    ) -> BehavioralReport:
        import time

        start_time = time.time()

        results: list[TestResult] = []
        errors: list[str] = []

        syntax_check = self._check_syntax(code)
        results.append(syntax_check)
        if not syntax_check.passed:
            return BehavioralReport(
                is_valid=False,
                tests_failed=1,
                test_results=results,
                execution_errors=[syntax_check.message],
            )

        if entry_point:
            entry_check = self._check_entry_point(code, entry_point)
            results.append(entry_check)
            if not entry_check.passed:
                return BehavioralReport(
                    is_valid=False,
                    tests_failed=1,
                    test_results=results,
                    execution_errors=[entry_check.message],
                )

        if tests or self.config.generate_tests:
            test_results = self._run_tests(code, tests or [])
            results.extend(test_results)

        if not tests and not self.config.generate_tests:
            import_result = self._test_import(code)
            results.append(import_result)

        total_duration = (time.time() - start_time) * 1000

        tests_passed = sum(1 for r in results if r.passed)
        tests_failed = sum(1 for r in results if not r.passed)

        is_valid = tests_failed == 0

        return BehavioralReport(
            is_valid=is_valid,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_results=results,
            execution_errors=errors,
            total_duration_ms=total_duration,
        )

    def _check_syntax(self, code: str) -> TestResult:
        import time

        start = time.time()

        try:
            ast.parse(code)
            return TestResult(
                test_name="syntax_check",
                passed=True,
                message="Code parses successfully",
                duration_ms=(time.time() - start) * 1000,
            )
        except SyntaxError as e:
            return TestResult(
                test_name="syntax_check",
                passed=False,
                message=f"Syntax error at line {e.lineno}: {e.msg}",
                duration_ms=(time.time() - start) * 1000,
            )

    def _check_entry_point(self, code: str, entry_point: str) -> TestResult:
        import time

        start = time.time()

        try:
            tree = ast.parse(code)
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]

            if entry_point in functions:
                return TestResult(
                    test_name="entry_point_check",
                    passed=True,
                    message=f"Entry point '{entry_point}' found",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return TestResult(
                    test_name="entry_point_check",
                    passed=False,
                    message=f"Entry point '{entry_point}' not found. Available: {functions}",
                    duration_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return TestResult(
                test_name="entry_point_check",
                passed=False,
                message=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    def _run_tests(
        self,
        code: str,
        tests: list[dict[str, Any]],
    ) -> list[TestResult]:
        results: list[TestResult] = []

        if not tests:
            tests = self._generate_basic_tests(code)

        test_code = self._build_test_code(code, tests)

        if self._sandbox_available and self.config.use_sandbox:
            result = self._execute_in_sandbox(test_code)
        else:
            result = self._execute_locally(test_code)

        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            passed = f"test_{i}: PASSED" in result.get("output", "")
            message = test.get("description", "")

            results.append(
                TestResult(
                    test_name=test_name,
                    passed=passed,
                    message=message,
                )
            )

        return results

    def _generate_basic_tests(self, code: str) -> list[dict[str, Any]]:
        tests: list[dict[str, Any]] = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith("_"):
                        continue

                    tests.append(
                        {
                            "name": f"test_{node.name}_exists",
                            "type": "existence",
                            "function": node.name,
                            "description": f"Function '{node.name}' exists and is callable",
                        }
                    )
        except Exception:
            pass

        return tests

    def _build_test_code(
        self,
        code: str,
        tests: list[dict[str, Any]],
    ) -> str:
        test_lines = [
            code,
            "",
            "import sys",
            "",
        ]

        for i, test in enumerate(tests):
            test_name = test.get("name", f"test_{i}")
            func_name = test.get("function", "")
            test_type = test.get("type", "existence")

            if test_type == "existence" and func_name:
                test_lines.extend(
                    [
                        f"def {test_name}():",
                        f"    assert callable({func_name}), '{func_name} is not callable'",
                        f"    print('test_{i}: PASSED')",
                        "",
                    ]
                )
            elif test_type == "execution" and func_name:
                args = test.get("args", [])
                expected = test.get("expected")
                test_lines.extend(
                    [
                        f"def {test_name}():",
                        f"    result = {func_name}(*{args})",
                    ]
                )
                if expected is not None:
                    test_lines.append(
                        f"    assert result == {repr(expected)}, f'Expected {expected}, got {{result}}'"
                    )
                test_lines.append(f"    print('test_{i}: PASSED')")
                test_lines.append("")

        test_lines.extend(
            [
                "if __name__ == '__main__':",
                "    tests_to_run = ["
                + ", ".join(f"'{t.get('name', f'test_{i}')}'" for i, t in enumerate(tests))
                + "]",
                "    for t in tests_to_run:",
                "        try:",
                "            globals()[t]()",
                "        except Exception as e:",
                "            print(f'{t}: FAILED - {e}')",
            ]
        )

        return "\n".join(test_lines)

    def _execute_in_sandbox(self, code: str) -> dict[str, Any]:
        if not self._sandbox_available:
            return {"output": "", "error": "Sandbox not available"}

        try:
            import asyncio

            result = asyncio.run(self._sandbox.execute(code, "python"))
            return {
                "output": result.output,
                "error": result.error,
                "exit_code": result.exit_code,
            }
        except Exception as e:
            return {"output": "", "error": str(e)}

    def _execute_locally(self, code: str) -> dict[str, Any]:
        import subprocess
        import sys

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            return {
                "output": result.stdout,
                "error": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"output": "", "error": "Execution timed out", "exit_code": -1}
        except Exception as e:
            return {"output": "", "error": str(e), "exit_code": -1}
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _test_import(self, code: str) -> TestResult:
        import time

        start = time.time()

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
        ) as f:
            f.write(code)
            temp_path = f.name

        try:
            import importlib.util

            spec = importlib.util.spec_from_file_location("test_module", temp_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                return TestResult(
                    test_name="import_test",
                    passed=True,
                    message="Module imported successfully",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return TestResult(
                    test_name="import_test",
                    passed=False,
                    message="Failed to create module spec",
                    duration_ms=(time.time() - start) * 1000,
                )
        except Exception as e:
            return TestResult(
                test_name="import_test",
                passed=False,
                message=f"Import error: {str(e)}",
                duration_ms=(time.time() - start) * 1000,
            )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def get_stats(self) -> dict[str, Any]:
        return {
            "config": {
                "timeout_seconds": self.config.timeout_seconds,
                "use_sandbox": self.config.use_sandbox,
                "generate_tests": self.config.generate_tests,
            },
            "sandbox_available": self._sandbox_available,
        }


def create_behavioral_validator(
    timeout: int = 10,
    use_sandbox: bool = True,
) -> BehavioralValidator:
    config = BehavioralConfig(
        timeout_seconds=timeout,
        use_sandbox=use_sandbox,
    )
    return BehavioralValidator(config)
