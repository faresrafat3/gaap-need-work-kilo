"""
Comprehensive Tests for GAAP Behavioral Validator Module

This module tests the BehavioralValidator class and related components
to achieve 80%+ line coverage.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaap.validators.behavioral import (
    BehavioralConfig,
    BehavioralReport,
    BehavioralValidator,
    TestResult,
    create_behavioral_validator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Return a default BehavioralConfig instance."""
    return BehavioralConfig.default()


@pytest.fixture
def fast_config():
    """Return a fast BehavioralConfig instance (no sandbox)."""
    return BehavioralConfig.fast()


@pytest.fixture
def strict_config():
    """Return a strict BehavioralConfig instance."""
    return BehavioralConfig.strict()


@pytest.fixture
def validator_no_sandbox(fast_config):
    """Return a BehavioralValidator with sandbox disabled."""
    return BehavioralValidator(fast_config)


@pytest.fixture
def valid_python_code():
    """Return valid Python code for testing."""
    return '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b

class Calculator:
    """A simple calculator class."""

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''


@pytest.fixture
def code_with_syntax_error():
    """Return Python code with syntax errors."""
    return """
def broken_function(
    # Missing closing parenthesis and colon
    return "never reached"
"""


@pytest.fixture
def code_with_imports():
    """Return Python code with various imports."""
    return """
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

def process_data(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = defaultdict(list)
    for item in data:
        key = item.get('key', 'default')
        result[key].append(item)
    return dict(result)
"""


@pytest.fixture
def code_with_async():
    """Return Python code with async functions."""
    return """
import asyncio

async def fetch_data(url: str) -> str:
    await asyncio.sleep(0.1)
    return f"Data from {url}"

async def main():
    result = await fetch_data("https://example.com")
    return result

class AsyncProcessor:
    async def process(self, item):
        return await fetch_data(item)
"""


@pytest.fixture
def unicode_code():
    """Return Python code with unicode characters."""
    return '''
# -*- coding: utf-8 -*-
def greet(name: str) -> str:
    """Greet someone in multiple languages."""
    greetings = {
        "english": f"Hello, {name}!",
        "spanish": f"¡Hola, {name}!",
        "french": f"Bonjour, {name} !",
        "chinese": f"你好，{name}！",
        "arabic": f"مرحبا، {name}!",
        "emoji": f"👋 {name}! 🎉",
    }
    return greetings.get("english", "Hello!")

class UnicodeProcessor:
    """Process unicode text."""

    PI = "π"
    EMOJI_MATH = "🧮"

    def process_unicode(self, text: str) -> str:
        return text.upper()
'''


@pytest.fixture
def empty_code():
    """Return empty Python code."""
    return ""


@pytest.fixture
def whitespace_code():
    """Return code with only whitespace."""
    return "   \n\n   \n"


@pytest.fixture
def custom_tests():
    """Return custom test definitions."""
    return [
        {
            "name": "test_add_positive",
            "type": "execution",
            "function": "add",
            "args": [2, 3],
            "expected": 5,
            "description": "Test adding positive numbers",
        },
        {
            "name": "test_add_negative",
            "type": "execution",
            "function": "add",
            "args": [-1, -1],
            "expected": -2,
            "description": "Test adding negative numbers",
        },
        {
            "name": "test_subtract",
            "type": "execution",
            "function": "subtract",
            "args": [5, 3],
            "expected": 2,
            "description": "Test subtraction",
        },
    ]


@pytest.fixture
def existence_tests():
    """Return existence test definitions."""
    return [
        {
            "name": "test_add_exists",
            "type": "existence",
            "function": "add",
            "description": "Function 'add' exists",
        },
        {
            "name": "test_calculator_exists",
            "type": "existence",
            "function": "Calculator",
            "description": "Class 'Calculator' exists",
        },
    ]


# =============================================================================
# TestResult Tests
# =============================================================================


class TestTestResult:
    """Tests for the TestResult dataclass."""

    def test_default_values(self):
        """Test TestResult with default values."""
        result = TestResult(test_name="basic_test", passed=True)
        assert result.test_name == "basic_test"
        assert result.passed is True
        assert result.message == ""
        assert result.duration_ms == 0.0

    def test_full_values(self):
        """Test TestResult with all values specified."""
        result = TestResult(
            test_name="complex_test",
            passed=False,
            message="Test failed due to assertion error",
            duration_ms=150.5,
        )
        assert result.test_name == "complex_test"
        assert result.passed is False
        assert "assertion error" in result.message
        assert result.duration_ms == 150.5

    def test_to_dict_complete(self):
        """Test to_dict with complete data."""
        result = TestResult(
            test_name="dict_test",
            passed=True,
            message="Success",
            duration_ms=42.0,
        )
        d = result.to_dict()
        assert d == {
            "test_name": "dict_test",
            "passed": True,
            "message": "Success",
            "duration_ms": 42.0,
        }

    def test_to_dict_minimal(self):
        """Test to_dict with minimal data."""
        result = TestResult(test_name="minimal", passed=False)
        d = result.to_dict()
        assert d["test_name"] == "minimal"
        assert d["passed"] is False
        assert d["message"] == ""
        assert d["duration_ms"] == 0.0


# =============================================================================
# BehavioralReport Tests
# =============================================================================


class TestBehavioralReport:
    """Tests for the BehavioralReport dataclass."""

    def test_default_values(self):
        """Test BehavioralReport with default values."""
        report = BehavioralReport(is_valid=True)
        assert report.is_valid is True
        assert report.tests_passed == 0
        assert report.tests_failed == 0
        assert report.test_results == []
        assert report.execution_errors == []
        assert report.total_duration_ms == 0.0

    def test_with_test_results(self):
        """Test BehavioralReport with test results."""
        results = [
            TestResult(test_name="test1", passed=True),
            TestResult(test_name="test2", passed=False),
        ]
        report = BehavioralReport(
            is_valid=False,
            tests_passed=1,
            tests_failed=1,
            test_results=results,
            execution_errors=["Error 1"],
            total_duration_ms=100.0,
        )
        assert report.is_valid is False
        assert report.tests_passed == 1
        assert report.tests_failed == 1
        assert len(report.test_results) == 2
        assert report.execution_errors == ["Error 1"]
        assert report.total_duration_ms == 100.0

    def test_to_dict(self):
        """Test to_dict method."""
        results = [
            TestResult(test_name="t1", passed=True),
            TestResult(test_name="t2", passed=False, message="Failed"),
        ]
        report = BehavioralReport(
            is_valid=False,
            tests_passed=1,
            tests_failed=1,
            test_results=results,
            execution_errors=["Test error"],
            total_duration_ms=50.0,
        )
        d = report.to_dict()
        assert d["is_valid"] is False
        assert d["tests_passed"] == 1
        assert d["tests_failed"] == 1
        assert d["total_duration_ms"] == 50.0
        assert len(d["tests"]) == 2
        assert d["tests"][0]["test_name"] == "t1"
        assert d["tests"][1]["message"] == "Failed"
        assert d["errors"] == ["Test error"]


# =============================================================================
# BehavioralConfig Tests
# =============================================================================


class TestBehavioralConfig:
    """Tests for the BehavioralConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = BehavioralConfig()
        assert config.timeout_seconds == 10
        assert config.max_output_size == 10000
        assert config.use_sandbox is True
        assert config.generate_tests is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BehavioralConfig(
            timeout_seconds=30,
            max_output_size=50000,
            use_sandbox=False,
            generate_tests=False,
        )
        assert config.timeout_seconds == 30
        assert config.max_output_size == 50000
        assert config.use_sandbox is False
        assert config.generate_tests is False

    def test_default_classmethod(self):
        """Test the default() classmethod."""
        config = BehavioralConfig.default()
        assert config.timeout_seconds == 10
        assert config.use_sandbox is True
        assert config.generate_tests is True

    def test_strict_classmethod(self):
        """Test the strict() classmethod."""
        config = BehavioralConfig.strict()
        assert config.timeout_seconds == 5
        assert config.use_sandbox is True
        assert config.generate_tests is True

    def test_fast_classmethod(self):
        """Test the fast() classmethod."""
        config = BehavioralConfig.fast()
        assert config.timeout_seconds == 3
        assert config.use_sandbox is False
        assert config.generate_tests is False


# =============================================================================
# BehavioralValidator - Initialization Tests
# =============================================================================


class TestBehavioralValidatorInit:
    """Tests for BehavioralValidator initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        validator = BehavioralValidator()
        assert validator.config is not None
        assert validator.config.timeout_seconds == 10

    def test_init_custom_config(self, fast_config):
        """Test initialization with custom config."""
        validator = BehavioralValidator(fast_config)
        assert validator.config.timeout_seconds == 3
        assert validator.config.use_sandbox is False

    def test_init_sandbox_disabled(self, fast_config):
        """Test initialization with sandbox disabled."""
        validator = BehavioralValidator(fast_config)
        assert validator._sandbox_available is False

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_available(self, mock_get_sandbox):
        """Test initialization when sandbox is available."""
        mock_sandbox = MagicMock()
        mock_get_sandbox.return_value = mock_sandbox

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is True
        assert validator._sandbox is mock_sandbox

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_import_error(self, mock_get_sandbox):
        """Test initialization when sandbox import fails."""
        mock_get_sandbox.side_effect = ImportError("No sandbox module")

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is False

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_generic_error(self, mock_get_sandbox):
        """Test initialization when sandbox raises generic error."""
        mock_get_sandbox.side_effect = RuntimeError("Sandbox error")

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is False

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_available(self, mock_get_sandbox):
        """Test initialization when sandbox is available."""
        mock_sandbox = MagicMock()
        mock_get_sandbox.return_value = mock_sandbox

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is True
        assert validator._sandbox is mock_sandbox

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_import_error(self, mock_get_sandbox):
        """Test initialization when sandbox import fails."""
        mock_get_sandbox.side_effect = ImportError("No sandbox module")

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is False

    @patch("gaap.security.get_sandbox")
    def test_init_sandbox_generic_error(self, mock_get_sandbox):
        """Test initialization when sandbox raises generic error."""
        mock_get_sandbox.side_effect = RuntimeError("Sandbox error")

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        assert validator._sandbox_available is False


# =============================================================================
# BehavioralValidator - Syntax Checking Tests
# =============================================================================


class TestBehavioralValidatorSyntax:
    """Tests for syntax checking functionality."""

    def test_check_syntax_valid(self, validator_no_sandbox, valid_python_code):
        """Test syntax check with valid code."""
        result = validator_no_sandbox._check_syntax(valid_python_code)
        assert result.passed is True
        assert "parses successfully" in result.message
        assert result.duration_ms >= 0

    def test_check_syntax_invalid(self, validator_no_sandbox, code_with_syntax_error):
        """Test syntax check with invalid code."""
        result = validator_no_sandbox._check_syntax(code_with_syntax_error)
        assert result.passed is False
        assert "Syntax error" in result.message
        assert "line" in result.message

    def test_check_syntax_empty(self, validator_no_sandbox, empty_code):
        """Test syntax check with empty code."""
        result = validator_no_sandbox._check_syntax(empty_code)
        assert result.passed is True  # Empty code is valid Python

    def test_check_syntax_whitespace_only(self, validator_no_sandbox, whitespace_code):
        """Test syntax check with whitespace-only code."""
        result = validator_no_sandbox._check_syntax(whitespace_code)
        assert result.passed is True

    def test_check_syntax_unicode(self, validator_no_sandbox, unicode_code):
        """Test syntax check with unicode code."""
        result = validator_no_sandbox._check_syntax(unicode_code)
        assert result.passed is True

    def test_check_syntax_various_errors(self, validator_no_sandbox):
        """Test syntax check with various syntax errors."""
        errors = [
            "def func(\n    pass",  # Missing closing paren
            "if True\n    print(1)",  # Missing colon
            "for i in range(10)\n    pass",  # Missing colon
            "class MyClass\n    pass",  # Missing colon
        ]
        for code in errors:
            result = validator_no_sandbox._check_syntax(code)
            assert result.passed is False
            assert "Syntax error" in result.message


# =============================================================================
# BehavioralValidator - Entry Point Tests
# =============================================================================


class TestBehavioralValidatorEntryPoint:
    """Tests for entry point validation."""

    def test_check_entry_point_exists(self, validator_no_sandbox):
        """Test entry point check when function exists."""
        code = "def main(): pass"
        result = validator_no_sandbox._check_entry_point(code, "main")
        assert result.passed is True
        assert "found" in result.message

    def test_check_entry_point_missing(self, validator_no_sandbox):
        """Test entry point check when function doesn't exist."""
        code = "def helper(): pass"
        result = validator_no_sandbox._check_entry_point(code, "main")
        assert result.passed is False
        assert "not found" in result.message
        assert "helper" in result.message  # Should list available functions

    def test_check_entry_point_multiple_functions(self, validator_no_sandbox):
        """Test entry point check with multiple functions."""
        code = "def func1(): pass\ndef func2(): pass\ndef func3(): pass"
        result = validator_no_sandbox._check_entry_point(code, "func2")
        assert result.passed is True

    def test_check_entry_point_async_function(self, validator_no_sandbox):
        """Test entry point check with async function."""
        code = "async def main(): pass"
        result = validator_no_sandbox._check_entry_point(code, "main")
        assert result.passed is True

    def test_check_entry_point_class_not_function(self, validator_no_sandbox):
        """Test entry point check when entry point is a class."""
        code = "class Main: pass"
        result = validator_no_sandbox._check_entry_point(code, "Main")
        assert result.passed is False  # Classes aren't detected by entry point check

    def test_check_entry_point_invalid_code(self, validator_no_sandbox):
        """Test entry point check with invalid code."""
        code = "def broken("
        result = validator_no_sandbox._check_entry_point(code, "main")
        assert result.passed is False

    def test_check_entry_point_empty_code(self, validator_no_sandbox):
        """Test entry point check with empty code."""
        result = validator_no_sandbox._check_entry_point("", "main")
        assert result.passed is False


# =============================================================================
# BehavioralValidator - Test Generation Tests
# =============================================================================


class TestBehavioralValidatorTestGeneration:
    """Tests for automatic test generation."""

    def test_generate_basic_tests(self, validator_no_sandbox, valid_python_code):
        """Test basic test generation."""
        tests = validator_no_sandbox._generate_basic_tests(valid_python_code)
        assert len(tests) > 0
        test_names = [t["name"] for t in tests]
        assert "test_add_exists" in test_names
        assert "test_subtract_exists" in test_names

    def test_generate_basic_tests_private_functions(self, validator_no_sandbox):
        """Test that private functions are excluded from test generation."""
        code = "def _private(): pass\ndef public(): pass"
        tests = validator_no_sandbox._generate_basic_tests(code)
        test_names = [t["name"] for t in tests]
        assert "test_public_exists" in test_names
        assert "test__private_exists" not in test_names

    def test_generate_basic_tests_invalid_code(self, validator_no_sandbox):
        """Test test generation with invalid code."""
        tests = validator_no_sandbox._generate_basic_tests("invalid syntax")
        assert tests == []

    def test_generate_basic_tests_empty_code(self, validator_no_sandbox):
        """Test test generation with empty code."""
        tests = validator_no_sandbox._generate_basic_tests("")
        assert tests == []

    def test_generate_basic_tests_no_functions(self, validator_no_sandbox):
        """Test test generation with code that has no functions."""
        code = "x = 1\ny = 2"
        tests = validator_no_sandbox._generate_basic_tests(code)
        assert tests == []


# =============================================================================
# BehavioralValidator - Test Code Building Tests
# =============================================================================


class TestBehavioralValidatorBuildTestCode:
    """Tests for building test code."""

    def test_build_test_code_existence(self, validator_no_sandbox):
        """Test building test code for existence tests."""
        code = "def add(a, b): return a + b"
        tests = [{"name": "test_add_exists", "type": "existence", "function": "add"}]
        test_code = validator_no_sandbox._build_test_code(code, tests)
        assert "def test_add_exists():" in test_code
        assert "assert callable(add)" in test_code
        assert "test_0: PASSED" in test_code

    def test_build_test_code_execution(self, validator_no_sandbox):
        """Test building test code for execution tests."""
        code = "def add(a, b): return a + b"
        tests = [
            {
                "name": "test_add",
                "type": "execution",
                "function": "add",
                "args": [1, 2],
                "expected": 3,
            }
        ]
        test_code = validator_no_sandbox._build_test_code(code, tests)
        assert "def test_add():" in test_code
        assert "result = add(*[1, 2])" in test_code
        assert "assert result == 3" in test_code

    def test_build_test_code_execution_no_expected(self, validator_no_sandbox):
        """Test building test code for execution tests without expected value."""
        code = "def process(): return 'done'"
        tests = [{"name": "test_process", "type": "execution", "function": "process", "args": []}]
        test_code = validator_no_sandbox._build_test_code(code, tests)
        assert "def test_process():" in test_code
        assert "result = process(*[])" in test_code
        assert "assert result ==" not in test_code

    def test_build_test_code_multiple_tests(self, validator_no_sandbox):
        """Test building test code with multiple tests."""
        code = "def func(): pass"
        tests = [
            {"name": "test1", "type": "existence", "function": "func"},
            {"name": "test2", "type": "existence", "function": "func"},
        ]
        test_code = validator_no_sandbox._build_test_code(code, tests)
        assert "def test1():" in test_code
        assert "def test2():" in test_code
        assert "if __name__ == '__main__':" in test_code

    def test_build_test_code_empty_tests(self, validator_no_sandbox):
        """Test building test code with empty tests list."""
        code = "def func(): pass"
        tests = []
        test_code = validator_no_sandbox._build_test_code(code, tests)
        assert code in test_code
        assert "import sys" in test_code


# =============================================================================
# BehavioralValidator - Local Execution Tests
# =============================================================================


class TestBehavioralValidatorLocalExecution:
    """Tests for local code execution."""

    def test_execute_locally_valid_code(self, validator_no_sandbox):
        """Test executing valid code locally."""
        code = "print('Hello, World!')"
        result = validator_no_sandbox._execute_locally(code)
        assert "output" in result
        assert "Hello, World!" in result["output"]
        assert result.get("exit_code") == 0

    def test_execute_locally_with_error(self, validator_no_sandbox):
        """Test executing code that raises an error."""
        code = "raise ValueError('Test error')"
        result = validator_no_sandbox._execute_locally(code)
        assert "error" in result
        assert "exit_code" in result

    def test_execute_locally_timeout(self, validator_no_sandbox):
        """Test executing code that times out."""
        code = "import time\ntime.sleep(100)"
        result = validator_no_sandbox._execute_locally(code)
        assert "error" in result
        assert "timed out" in result.get("error", "").lower()
        assert result.get("exit_code") == -1

    def test_execute_locally_empty_code(self, validator_no_sandbox):
        """Test executing empty code."""
        result = validator_no_sandbox._execute_locally("")
        assert "output" in result
        assert "exit_code" in result


# =============================================================================
# BehavioralValidator - Import Tests
# =============================================================================


class TestBehavioralValidatorImport:
    """Tests for import validation."""

    def test_test_import_valid(self, validator_no_sandbox, valid_python_code):
        """Test import of valid code."""
        result = validator_no_sandbox._test_import(valid_python_code)
        assert result.passed is True
        assert "successfully" in result.message

    def test_test_import_with_imports(self, validator_no_sandbox, code_with_imports):
        """Test import of code with imports."""
        result = validator_no_sandbox._test_import(code_with_imports)
        assert result.passed is True

    def test_test_import_syntax_error(self, validator_no_sandbox, code_with_syntax_error):
        """Test import of code with syntax error."""
        result = validator_no_sandbox._test_import(code_with_syntax_error)
        assert result.passed is False
        assert "Import error" in result.message

    def test_test_import_runtime_error(self, validator_no_sandbox):
        """Test import of code that raises at import time."""
        code = "raise RuntimeError('Import error')"
        result = validator_no_sandbox._test_import(code)
        assert result.passed is False

    def test_test_import_empty_code(self, validator_no_sandbox, empty_code):
        """Test import of empty code."""
        result = validator_no_sandbox._test_import(empty_code)
        assert result.passed is True  # Empty code imports successfully


# =============================================================================
# BehavioralValidator - Full Validation Tests
# =============================================================================


class TestBehavioralValidatorValidate:
    """Tests for the main validate method."""

    def test_validate_valid_code(self, validator_no_sandbox, valid_python_code):
        """Test validation of valid code."""
        report = validator_no_sandbox.validate(valid_python_code)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0
        assert report.is_valid is True

    def test_validate_syntax_error(self, validator_no_sandbox, code_with_syntax_error):
        """Test validation with syntax error."""
        report = validator_no_sandbox.validate(code_with_syntax_error)
        assert report.is_valid is False
        assert report.tests_failed >= 1
        assert len(report.execution_errors) > 0

    def test_validate_empty_code(self, validator_no_sandbox, empty_code):
        """Test validation with empty code."""
        report = validator_no_sandbox.validate(empty_code)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0  # Empty code passes syntax check

    def test_validate_with_entry_point_exists(self, validator_no_sandbox):
        """Test validation with existing entry point."""
        code = "def main(): return 42"
        report = validator_no_sandbox.validate(code, entry_point="main")
        assert report.tests_passed >= 2  # syntax + entry point

    def test_validate_with_entry_point_missing(self, validator_no_sandbox):
        """Test validation with missing entry point."""
        code = "def helper(): return 42"
        report = validator_no_sandbox.validate(code, entry_point="main")
        assert report.is_valid is False
        assert report.tests_failed >= 1

    def test_validate_with_custom_tests(self, validator_no_sandbox, custom_tests):
        """Test validation with custom tests."""
        code = "def add(a, b): return a + b\ndef subtract(a, b): return a - b"
        report = validator_no_sandbox.validate(code, tests=custom_tests)
        assert isinstance(report, BehavioralReport)
        assert len(report.test_results) > 0

    def test_validate_no_generate_tests(self):
        """Test validation without test generation."""
        config = BehavioralConfig(use_sandbox=False, generate_tests=False)
        validator = BehavioralValidator(config)
        code = "def func(): pass"
        report = validator.validate(code)
        assert isinstance(report, BehavioralReport)
        # Should do syntax check and import test

    def test_validate_duration_recorded(self, validator_no_sandbox, valid_python_code):
        """Test that duration is recorded."""
        report = validator_no_sandbox.validate(valid_python_code)
        assert report.total_duration_ms > 0

    def test_validate_results_count(self, validator_no_sandbox, valid_python_code):
        """Test that test results are properly counted."""
        report = validator_no_sandbox.validate(valid_python_code)
        passed = sum(1 for r in report.test_results if r.passed)
        failed = sum(1 for r in report.test_results if not r.passed)
        assert passed == report.tests_passed
        assert failed == report.tests_failed
        assert passed + failed == len(report.test_results)


# =============================================================================
# BehavioralValidator - Sandbox Execution Tests
# =============================================================================


class TestBehavioralValidatorSandbox:
    """Tests for sandbox execution."""

    @patch("gaap.security.get_sandbox")
    def test_execute_in_sandbox_available(self, mock_get_sandbox):
        """Test execution when sandbox is available."""
        mock_sandbox_result = MagicMock()
        mock_sandbox_result.output = "test output"
        mock_sandbox_result.error = ""
        mock_sandbox_result.exit_code = 0

        mock_sandbox = MagicMock()

        # Create a proper async function that returns the mock result
        async def async_execute(*args, **kwargs):
            return mock_sandbox_result

        mock_sandbox.execute = async_execute
        mock_get_sandbox.return_value = mock_sandbox

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)
        validator._sandbox_available = True
        validator._sandbox = mock_sandbox

        result = validator._execute_in_sandbox("print('test')")
        assert result["output"] == "test output"
        assert result["error"] == ""
        assert result["exit_code"] == 0

    def test_execute_in_sandbox_not_available(self, validator_no_sandbox):
        """Test execution when sandbox is not available."""
        result = validator_no_sandbox._execute_in_sandbox("print('test')")
        assert "error" in result
        assert "not available" in result["error"].lower()

    @patch("gaap.security.get_sandbox")
    def test_execute_in_sandbox_exception(self, mock_get_sandbox):
        """Test sandbox execution with exception."""
        mock_sandbox = MagicMock()

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Sandbox error")

        mock_sandbox.execute = raise_error
        mock_get_sandbox.return_value = mock_sandbox

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)
        validator._sandbox_available = True
        validator._sandbox = mock_sandbox

        result = validator._execute_in_sandbox("print('test')")
        assert "error" in result
        assert "not available" in result["error"].lower()

    @patch("gaap.security.get_sandbox")
    def test_execute_in_sandbox_exception(self, mock_get_sandbox):
        """Test sandbox execution with exception."""
        mock_sandbox = MagicMock()

        async def raise_error(*args, **kwargs):
            raise RuntimeError("Sandbox error")

        mock_sandbox.execute = raise_error
        mock_get_sandbox.return_value = mock_sandbox

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)
        validator._sandbox_available = True
        validator._sandbox = mock_sandbox

        result = validator._execute_in_sandbox("print('test')")
        assert "error" in result


# =============================================================================
# BehavioralValidator - Stats Tests
# =============================================================================


class TestBehavioralValidatorStats:
    """Tests for validator statistics."""

    def test_get_stats(self, validator_no_sandbox):
        """Test getting validator stats."""
        stats = validator_no_sandbox.get_stats()
        assert "config" in stats
        assert "sandbox_available" in stats
        assert stats["config"]["timeout_seconds"] == 3
        assert stats["config"]["use_sandbox"] is False
        assert stats["config"]["generate_tests"] is False

    @patch("gaap.security.get_sandbox")
    def test_get_stats_with_sandbox(self, mock_get_sandbox):
        """Test getting stats when sandbox is available."""
        mock_get_sandbox.return_value = MagicMock()

        config = BehavioralConfig(use_sandbox=True)
        validator = BehavioralValidator(config)

        stats = validator.get_stats()
        assert stats["sandbox_available"] is True
        assert stats["config"]["use_sandbox"] is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestCreateBehavioralValidator:
    """Tests for the create_behavioral_validator helper function."""

    def test_create_with_defaults(self):
        """Test creating validator with default parameters."""
        validator = create_behavioral_validator()
        assert isinstance(validator, BehavioralValidator)
        assert validator.config.timeout_seconds == 10
        assert validator.config.use_sandbox is True

    def test_create_with_custom_timeout(self):
        """Test creating validator with custom timeout."""
        validator = create_behavioral_validator(timeout=5)
        assert validator.config.timeout_seconds == 5

    def test_create_with_sandbox_disabled(self):
        """Test creating validator with sandbox disabled."""
        validator = create_behavioral_validator(use_sandbox=False)
        assert validator.config.use_sandbox is False

    def test_create_with_all_params(self):
        """Test creating validator with all custom parameters."""
        validator = create_behavioral_validator(timeout=20, use_sandbox=False)
        assert validator.config.timeout_seconds == 20
        assert validator.config.use_sandbox is False


# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================


class TestBehavioralValidatorEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_validate_unicode_code(self, validator_no_sandbox, unicode_code):
        """Test validation of code with unicode characters."""
        report = validator_no_sandbox.validate(unicode_code)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0

    def test_validate_async_code(self, validator_no_sandbox, code_with_async):
        """Test validation of code with async functions."""
        report = validator_no_sandbox.validate(code_with_async)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0

    def test_validate_code_with_imports(self, validator_no_sandbox, code_with_imports):
        """Test validation of code with various imports."""
        report = validator_no_sandbox.validate(code_with_imports)
        assert isinstance(report, BehavioralReport)

    def test_validate_large_code(self, validator_no_sandbox):
        """Test validation of relatively large code."""
        code = "\n\n".join([f"def func_{i}(): return {i}" for i in range(100)])
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0

    def test_validate_code_with_comments_only(self, validator_no_sandbox):
        """Test validation of code with only comments."""
        code = "# This is a comment\n# Another comment\n"
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_docstrings_only(self, validator_no_sandbox):
        """Test validation of code with only docstrings."""
        code = '"""Module docstring"""\n'
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_nested_functions(self, validator_no_sandbox):
        """Test validation of code with nested functions."""
        code = """
def outer():
    def inner():
        return "inner"
    return inner()

def standalone():
    return "standalone"
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0

    def test_validate_code_with_decorators(self, validator_no_sandbox):
        """Test validation of code with decorators."""
        code = """
def decorator(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def decorated():
    return "decorated"
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_lambda(self, validator_no_sandbox):
        """Test validation of code with lambda functions."""
        code = """
add = lambda a, b: a + b
multiply = lambda x: x * 2

def use_lambdas():
    return add(1, 2) + multiply(3)
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_comprehensions(self, validator_no_sandbox):
        """Test validation of code with list/dict comprehensions."""
        code = """
def process_list(items):
    return [x * 2 for x in items if x > 0]

def process_dict(items):
    return {k: v for k, v in items.items() if v > 0}

def process_set(items):
    return {x for x in items}
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_classes(self, validator_no_sandbox):
        """Test validation of code with classes."""
        code = """
class Base:
    def method(self):
        return "base"

class Derived(Base):
    def method(self):
        return "derived"

    @classmethod
    def class_method(cls):
        return "class"

    @staticmethod
    def static_method():
        return "static"
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_exception_handling(self, validator_no_sandbox):
        """Test validation of code with exception handling."""
        code = """
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")

def with_finally():
    try:
        return "try"
    finally:
        print("finally")
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_context_managers(self, validator_no_sandbox):
        """Test validation of code with context managers."""
        code = """
from contextlib import contextmanager

@contextmanager
def my_context():
    yield "context"

def use_context():
    with my_context() as ctx:
        return ctx
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_code_with_generator(self, validator_no_sandbox):
        """Test validation of code with generator functions."""
        code = """
def count_up_to(n):
    count = 0
    while count < n:
        yield count
        count += 1

def use_generator():
    return list(count_up_to(5))
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_multiple_validation_calls(self, validator_no_sandbox):
        """Test that validator can be used multiple times."""
        code1 = "def func1(): return 1"
        code2 = "def func2(): return 2"

        report1 = validator_no_sandbox.validate(code1)
        report2 = validator_no_sandbox.validate(code2)

        assert report1.is_valid is True
        assert report2.is_valid is True
        assert report1 != report2

    def test_validate_with_mixed_passing_failing_tests(self, validator_no_sandbox):
        """Test validation with mix of passing and failing tests."""
        code = "def add(a, b): return a + b"
        tests = [
            {"name": "test_pass", "type": "existence", "function": "add"},
            {"name": "test_fail", "type": "existence", "function": "nonexistent"},
        ]
        report = validator_no_sandbox.validate(code, tests=tests)
        assert isinstance(report, BehavioralReport)
        # Some tests may pass, some may fail


# =============================================================================
# Performance and Stress Tests
# =============================================================================


class TestBehavioralValidatorPerformance:
    """Tests for performance characteristics."""

    def test_validate_small_code(self, validator_no_sandbox):
        """Test validation of very small code."""
        code = "x = 1"
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_single_function(self, validator_no_sandbox):
        """Test validation of single function."""
        code = "def f(): pass"
        report = validator_no_sandbox.validate(code)
        assert report.is_valid is True

    def test_validate_multiple_functions(self, validator_no_sandbox):
        """Test validation of multiple functions."""
        code = "\n".join([f"def f{i}(): return {i}" for i in range(50)])
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_deeply_nested_code(self, validator_no_sandbox):
        """Test validation of deeply nested code."""
        code = """
def level1():
    def level2():
        def level3():
            def level4():
                def level5():
                    return "deep"
                return level5()
            return level4()
        return level3()
    return level2()
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBehavioralValidatorErrorHandling:
    """Tests for error handling."""

    def test_validate_indentation_error(self, validator_no_sandbox):
        """Test validation with indentation error."""
        code = "def f():\n  pass\n   pass"  # Indentation error
        report = validator_no_sandbox.validate(code)
        assert report.is_valid is False

    def test_validate_name_error(self, validator_no_sandbox):
        """Test validation where code has name errors (caught at runtime)."""
        code = "def f(): return undefined_variable"
        report = validator_no_sandbox.validate(code)
        # Syntax is valid, so this should pass syntax check
        assert isinstance(report, BehavioralReport)

    def test_validate_type_error(self, validator_no_sandbox):
        """Test validation where code has type errors (caught at runtime)."""
        code = "def f(): return 'string' + 5"
        report = validator_no_sandbox.validate(code)
        # Syntax is valid, so this should pass syntax check
        assert isinstance(report, BehavioralReport)

    def test_validate_recursion_error(self, validator_no_sandbox):
        """Test validation with potential recursion issues."""
        code = """
def recursive(n):
    if n <= 0:
        return 0
    return n + recursive(n - 1)
"""
        report = validator_no_sandbox.validate(code)
        # Should pass syntax validation
        assert isinstance(report, BehavioralReport)


# =============================================================================
# Test Pattern Matching (if applicable in the module)
# =============================================================================


class TestBehavioralValidatorPatternMatching:
    """Tests for pattern matching capabilities in code."""

    def test_validate_match_statement(self, validator_no_sandbox):
        """Test validation of code with match statement (Python 3.10+)."""
        code = """
def http_status(status):
    match status:
        case 200:
            return "OK"
        case 404:
            return "Not Found"
        case _:
            return "Unknown"
"""
        report = validator_no_sandbox.validate(code)
        # May or may not pass depending on Python version
        assert isinstance(report, BehavioralReport)

    def test_validate_walrus_operator(self, validator_no_sandbox):
        """Test validation of code with walrus operator (Python 3.8+)."""
        code = """
def process(items):
    if (n := len(items)) > 0:
        return n
    return 0
"""
        report = validator_no_sandbox.validate(code)
        assert isinstance(report, BehavioralReport)

    def test_validate_f_strings(self, validator_no_sandbox):
        """Test validation of code with f-strings."""
        code = """
def format_info(name, age):
    return f"{name} is {age} years old"

def format_expression(x, y):
    return f"Sum: {x + y}"
"""
        report = validator_no_sandbox.validate(code)
        assert report.is_valid is True

    def test_validate_type_hints(self, validator_no_sandbox):
        """Test validation of code with type hints."""
        code = """
from typing import List, Dict, Optional

def process(data: List[int]) -> Dict[str, int]:
    return {"count": len(data)}

def maybe(value: Optional[str]) -> str:
    return value or "default"
"""
        report = validator_no_sandbox.validate(code)
        assert report.is_valid is True


# =============================================================================
# Main Validation Integration Tests
# =============================================================================


class TestBehavioralValidatorMainIntegration:
    """Integration tests for the main validate flow."""

    def test_full_validation_flow(self):
        """Test complete validation flow with real code."""
        validator = BehavioralValidator(config=BehavioralConfig.fast())

        code = '''
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    if length < 0 or width < 0:
        raise ValueError("Dimensions must be non-negative")
    return length * width

def calculate_perimeter(length: float, width: float) -> float:
    """Calculate the perimeter of a rectangle."""
    return 2 * (length + width)

class Rectangle:
    """A rectangle shape."""

    def __init__(self, length: float, width: float):
        self.length = length
        self.width = width

    def area(self) -> float:
        return calculate_area(self.length, self.width)

    def perimeter(self) -> float:
        return calculate_perimeter(self.length, self.width)
'''

        tests = [
            {
                "name": "test_area_calculation",
                "type": "execution",
                "function": "calculate_area",
                "args": [5.0, 3.0],
                "expected": 15.0,
            },
            {
                "name": "test_perimeter_calculation",
                "type": "execution",
                "function": "calculate_perimeter",
                "args": [5.0, 3.0],
                "expected": 16.0,
            },
        ]

        report = validator.validate(code, tests=tests, entry_point="calculate_area")

        assert isinstance(report, BehavioralReport)
        assert report.tests_passed > 0
        assert report.total_duration_ms >= 0
        assert len(report.test_results) > 0

        # Verify stats
        stats = validator.get_stats()
        assert stats["config"]["use_sandbox"] is False

    def test_validation_with_entry_point_and_tests(self):
        """Test validation combining entry point and custom tests."""
        validator = BehavioralValidator(config=BehavioralConfig.fast())

        code = """
def main(data):
    return process_data(data)

def process_data(items):
    return [item * 2 for item in items]
"""

        tests = [
            {
                "name": "test_process_data",
                "type": "existence",
                "function": "process_data",
            },
        ]

        report = validator.validate(code, tests=tests, entry_point="main")
        assert isinstance(report, BehavioralReport)

    def test_validation_without_tests_or_generation(self):
        """Test validation when no tests and no generation."""
        config = BehavioralConfig(use_sandbox=False, generate_tests=False)
        validator = BehavioralValidator(config)

        code = "def func(): pass"
        report = validator.validate(code, tests=[], entry_point=None)

        assert isinstance(report, BehavioralReport)
        # Should only do syntax and import checks
