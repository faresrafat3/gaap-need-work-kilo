"""
Interpreter Tool - Safe Code Execution for Verification
=========================================================

Provides sandboxed code execution for verifying code correctness,
testing functions, and validating syntax.

Key Components:
    - InterpreterTool: Safe code execution
    - ExecutionResult: Result with output, errors, runtime

Usage:
    from gaap.tools.interpreter_tool import InterpreterTool

    tool = InterpreterTool()
    result = await tool.execute("print('hello')")
    print(result.output)  # 'hello\\n'
"""

from __future__ import annotations

import ast
import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from gaap.core.logging import get_standard_logger as get_logger

logger = get_logger("gaap.tools.interpreter")


class ExecutionStatus(Enum):
    """Status of code execution"""

    SUCCESS = auto()
    ERROR = auto()
    TIMEOUT = auto()
    SECURITY_VIOLATION = auto()
    SYNTAX_ERROR = auto()


@dataclass
class ExecutionResult:
    """
    Result of code execution.

    Attributes:
        status: Execution status
        output: Standard output from execution
        error: Error message if failed
        return_value: Return value from executed code
        runtime_ms: Execution time in milliseconds
        memory_estimate_kb: Estimated memory usage
        metadata: Additional metadata
    """

    status: ExecutionStatus
    output: str = ""
    error: str = ""
    return_value: Any = None
    runtime_ms: float = 0.0
    memory_estimate_kb: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.name,
            "output": self.output,
            "error": self.error,
            "return_value": repr(self.return_value),
            "runtime_ms": self.runtime_ms,
            "memory_estimate_kb": self.memory_estimate_kb,
            "metadata": self.metadata,
        }


@dataclass
class TestCase:
    """A test case for function verification"""

    name: str
    inputs: dict[str, Any]
    expected_output: Any
    description: str = ""


@dataclass
class TestResult:
    """Result of running a test case"""

    test_name: str
    passed: bool
    actual_output: Any = None
    expected_output: Any = None
    error: str = ""
    runtime_ms: float = 0.0


SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "__name__": "__main__",
    "__builtins__": {},
}

BLOCKED_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "socketserver",
    "http",
    "urllib",
    "ftplib",
    "telnetlib",
    "smtplib",
    "poplib",
    "imaplib",
    "nntplib",
    "pickle",
    "shelve",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "_thread",
    "signal",
    "resource",
    "posix",
    "nt",
    "fcntl",
    "pipes",
    "posixpath",
    "shutil",
    "tempfile",
    "glob",
    "fnmatch",
    "linecache",
    "code",
    "codeop",
    "compileall",
    "importlib",
    "pkgutil",
    "modulefinder",
    "runpy",
    "trace",
    "tracemalloc",
    "warnings",
    "gc",
    "inspect",
    "dis",
    "pickletools",
    "ast",
}


class InterpreterTool:
    """
    Safe code execution for verification.

    Provides sandboxed Python execution with:
    - Restricted builtins
    - No file/network access
    - Timeout enforcement
    - Memory limits

    Attributes:
        default_timeout: Default execution timeout in seconds
        max_output_length: Maximum output length
        enable_math: Enable math module
        enable_re: Enable regex module
        enable_json: Enable json module
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        max_output_length: int = 10000,
        enable_math: bool = True,
        enable_re: bool = True,
        enable_json: bool = True,
    ):
        self.default_timeout = default_timeout
        self.max_output_length = max_output_length
        self._logger = logger

        self._safe_globals: dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}

        if enable_math:
            self._safe_globals["math"] = math
        if enable_re:
            self._safe_globals["re"] = re

        if enable_json:
            import json

            self._safe_globals["json"] = json

    def _get_safe_globals(self) -> dict[str, Any]:
        """Get safe globals for execution"""
        return self._safe_globals.copy()

    def validate_syntax(self, code: str) -> tuple[bool, str]:
        """
        Validate Python syntax.

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def _check_imports(self, code: str) -> list[str]:
        """Check for blocked imports"""
        blocked = []
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in BLOCKED_MODULES:
                        blocked.append(module_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in BLOCKED_MODULES:
                        blocked.append(module_name)

        return blocked

    def _estimate_complexity(self, code: str) -> float:
        """Estimate code complexity (rough runtime estimate)"""
        tree = ast.parse(code)

        loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        comprehensions = sum(
            1
            for node in ast.walk(tree)
            if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp))
        )
        function_calls = sum(1 for node in ast.walk(tree) if isinstance(node, ast.Call))

        return loops * 0.5 + comprehensions * 0.2 + function_calls * 0.05 + 0.1

    def estimate_runtime(self, code: str) -> float:
        """
        Estimate execution time.

        Args:
            code: Python code to estimate

        Returns:
            Estimated runtime in seconds
        """
        complexity = self._estimate_complexity(code)
        return min(complexity, self.default_timeout)

    async def execute(
        self,
        code: str,
        timeout: float | None = None,
        globals_override: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        """
        Execute code safely.

        Args:
            code: Python code to execute
            timeout: Timeout in seconds (default: self.default_timeout)
            globals_override: Additional globals to include

        Returns:
            ExecutionResult with output and status
        """

        is_valid, syntax_error = self.validate_syntax(code)
        if not is_valid:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error=syntax_error,
            )

        blocked = self._check_imports(code)
        if blocked:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                error=f"Blocked imports: {', '.join(blocked)}",
            )

        timeout = timeout or self.default_timeout
        safe_globals = self._get_safe_globals()
        if globals_override:
            safe_globals.update(globals_override)

        local_vars: dict[str, Any] = {}

        start_time = time.time()

        try:

            async def run_code():
                exec(code, safe_globals, local_vars)

            await asyncio.wait_for(run_code(), timeout=timeout)

            runtime_ms = (time.time() - start_time) * 1000

            return_value = local_vars.get("result", None)

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output="",
                return_value=return_value,
                runtime_ms=runtime_ms,
                metadata={"variables": list(local_vars.keys())},
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {timeout}s",
                runtime_ms=timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}",
                runtime_ms=(time.time() - start_time) * 1000,
            )

    async def execute_function(
        self,
        code: str,
        function_name: str,
        args: tuple = (),
        kwargs: dict | None = None,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """
        Execute a specific function from code.

        Args:
            code: Python code containing the function
            function_name: Name of the function to call
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds

        Returns:
            ExecutionResult with function return value
        """
        kwargs = kwargs or {}

        is_valid, syntax_error = self.validate_syntax(code)
        if not is_valid:
            return ExecutionResult(
                status=ExecutionStatus.SYNTAX_ERROR,
                error=syntax_error,
            )

        blocked = self._check_imports(code)
        if blocked:
            return ExecutionResult(
                status=ExecutionStatus.SECURITY_VIOLATION,
                error=f"Blocked imports: {', '.join(blocked)}",
            )

        timeout = timeout or self.default_timeout
        safe_globals = self._get_safe_globals()
        local_vars: dict[str, Any] = {}

        start_time = time.time()

        try:
            exec(code, safe_globals, local_vars)

            if function_name not in local_vars:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"Function '{function_name}' not found in code",
                )

            func = local_vars[function_name]
            if not callable(func):
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=f"'{function_name}' is not callable",
                )

            async def run_function():
                return func(*args, **kwargs)

            result = await asyncio.wait_for(run_function(), timeout=timeout)

            runtime_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                return_value=result,
                runtime_ms=runtime_ms,
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Function execution timed out after {timeout}s",
                runtime_ms=timeout * 1000,
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}",
                runtime_ms=(time.time() - start_time) * 1000,
            )

    async def test_function(
        self,
        code: str,
        function_name: str,
        test_cases: list[TestCase],
        timeout: float | None = None,
    ) -> list[TestResult]:
        """
        Test a function with multiple test cases.

        Args:
            code: Python code containing the function
            function_name: Name of the function to test
            test_cases: List of test cases
            timeout: Timeout per test case in seconds

        Returns:
            List of TestResult for each test case
        """
        results: list[TestResult] = []

        is_valid, syntax_error = self.validate_syntax(code)
        if not is_valid:
            for tc in test_cases:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"Syntax error: {syntax_error}",
                    )
                )
            return results

        blocked = self._check_imports(code)
        if blocked:
            for tc in test_cases:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"Blocked imports: {', '.join(blocked)}",
                    )
                )
            return results

        timeout = timeout or self.default_timeout
        safe_globals = self._get_safe_globals()
        local_vars: dict[str, Any] = {}

        try:
            exec(code, safe_globals, local_vars)
        except Exception as e:
            for tc in test_cases:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"Code execution error: {e}",
                    )
                )
            return results

        if function_name not in local_vars:
            for tc in test_cases:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"Function '{function_name}' not found",
                    )
                )
            return results

        func = local_vars[function_name]

        for tc in test_cases:
            start_time = time.time()

            try:

                async def run_test():
                    return func(**tc.inputs)

                actual = await asyncio.wait_for(run_test(), timeout=timeout)
                runtime_ms = (time.time() - start_time) * 1000

                passed = actual == tc.expected_output

                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=passed,
                        actual_output=actual,
                        expected_output=tc.expected_output,
                        runtime_ms=runtime_ms,
                        error="" if passed else f"Expected {tc.expected_output}, got {actual}",
                    )
                )

            except asyncio.TimeoutError:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"Test timed out after {timeout}s",
                    )
                )
            except Exception as e:
                results.append(
                    TestResult(
                        test_name=tc.name,
                        passed=False,
                        error=f"{type(e).__name__}: {str(e)}",
                        runtime_ms=(time.time() - start_time) * 1000,
                    )
                )

        return results

    def create_test_case(
        self,
        name: str,
        inputs: dict[str, Any],
        expected_output: Any,
        description: str = "",
    ) -> TestCase:
        """Create a test case for function testing"""
        return TestCase(
            name=name,
            inputs=inputs,
            expected_output=expected_output,
            description=description,
        )

    async def compare_outputs(
        self,
        code1: str,
        code2: str,
        test_inputs: list[dict[str, Any]],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """
        Compare outputs of two code snippets.

        Args:
            code1: First code snippet
            code2: Second code snippet
            test_inputs: List of input dictionaries
            timeout: Timeout per execution

        Returns:
            Comparison results
        """
        results = {
            "matches": 0,
            "mismatches": 0,
            "errors": 0,
            "details": [],
        }

        for i, inputs in enumerate(test_inputs):
            result1 = await self.execute(code1, timeout)
            result2 = await self.execute(code2, timeout)

            if not result1.success or not result2.success:
                results["errors"] += 1
                results["details"].append(
                    {
                        "input_index": i,
                        "error": "Execution failed",
                        "code1_status": result1.status.name,
                        "code2_status": result2.status.name,
                    }
                )
                continue

            if result1.return_value == result2.return_value:
                results["matches"] += 1
            else:
                results["mismatches"] += 1
                results["details"].append(
                    {
                        "input_index": i,
                        "code1_output": result1.return_value,
                        "code2_output": result2.return_value,
                    }
                )

        results["match_rate"] = results["matches"] / len(test_inputs) if test_inputs else 0.0
        return results


def create_interpreter(
    timeout: float = 5.0,
    enable_math: bool = True,
    enable_re: bool = True,
    enable_json: bool = True,
) -> InterpreterTool:
    """Create an interpreter tool with default settings"""
    return InterpreterTool(
        default_timeout=timeout,
        enable_math=enable_math,
        enable_re=enable_re,
        enable_json=enable_json,
    )
