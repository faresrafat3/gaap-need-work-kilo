"""
Tests for GAAP Validators Module
"""

import pytest

from gaap.validators.behavioral import (
    BehavioralConfig,
    BehavioralReport,
    BehavioralValidator,
    TestResult,
)


class TestTestResult:
    def test_defaults(self):
        result = TestResult(test_name="test", passed=True)
        assert result.test_name == "test"
        assert result.passed is True
        assert result.message == ""
        assert result.duration_ms == 0.0

    def test_to_dict(self):
        result = TestResult(
            test_name="test",
            passed=True,
            message="Success",
            duration_ms=100.0,
        )
        d = result.to_dict()
        assert d["test_name"] == "test"
        assert d["passed"] is True
        assert d["message"] == "Success"
        assert d["duration_ms"] == 100.0


class TestBehavioralReport:
    def test_defaults(self):
        report = BehavioralReport(is_valid=True)
        assert report.is_valid is True
        assert report.tests_passed == 0
        assert report.tests_failed == 0

    def test_to_dict(self):
        report = BehavioralReport(
            is_valid=True,
            tests_passed=5,
            tests_failed=1,
            total_duration_ms=500.0,
        )
        d = report.to_dict()
        assert d["is_valid"] is True
        assert d["tests_passed"] == 5
        assert d["tests_failed"] == 1
        assert d["total_duration_ms"] == 500.0


class TestBehavioralConfig:
    def test_defaults(self):
        config = BehavioralConfig()
        assert config.timeout_seconds == 10
        assert config.max_output_size == 10000
        assert config.use_sandbox is True

    def test_default_classmethod(self):
        config = BehavioralConfig.default()
        assert config.timeout_seconds == 10

    def test_strict_classmethod(self):
        config = BehavioralConfig.strict()
        assert config.timeout_seconds == 5
        assert config.use_sandbox is True

    def test_fast_classmethod(self):
        config = BehavioralConfig.fast()
        assert config.timeout_seconds == 3
        assert config.use_sandbox is False
        assert config.generate_tests is False


class TestBehavioralValidator:
    def test_init_default_config(self):
        validator = BehavioralValidator()
        assert validator.config is not None

    def test_init_custom_config(self):
        config = BehavioralConfig(timeout_seconds=5)
        validator = BehavioralValidator(config)
        assert validator.config.timeout_seconds == 5

    def test_validate_valid_code(self):
        validator = BehavioralValidator(config=BehavioralConfig(use_sandbox=False))
        code = "def add(a, b): return a + b"
        report = validator.validate(code)
        assert report.tests_passed >= 1

    def test_validate_syntax_error(self):
        validator = BehavioralValidator(config=BehavioralConfig(use_sandbox=False))
        code = "def invalid syntax here"
        report = validator.validate(code)
        assert report.is_valid is False
        assert report.tests_failed >= 1

    def test_validate_with_entry_point_exists(self):
        validator = BehavioralValidator(config=BehavioralConfig(use_sandbox=False))
        code = """
def main():
    return "hello"

def helper():
    return main()
"""
        report = validator.validate(code, entry_point="main")
        assert report.tests_passed >= 1

    def test_validate_with_entry_point_missing(self):
        validator = BehavioralValidator(config=BehavioralConfig(use_sandbox=False))
        code = """
def helper():
    return 42
"""
        report = validator.validate(code, entry_point="main")
        assert report.tests_failed >= 1
        assert report.is_valid is False

    def test_validate_with_custom_tests(self):
        validator = BehavioralValidator(config=BehavioralConfig(use_sandbox=False))
        code = """
def add(a, b):
    return a + b
"""
        tests = [
            {"name": "test_add_positive", "input": {"a": 1, "b": 2}, "expected": 3},
            {"name": "test_add_negative", "input": {"a": -1, "b": -1}, "expected": -2},
        ]
        report = validator.validate(code, tests=tests)
        assert len(report.test_results) > 0

    def test_check_syntax_valid(self):
        validator = BehavioralValidator()
        code = "def foo(): return 1"
        result = validator._check_syntax(code)
        assert result.passed is True

    def test_check_syntax_invalid(self):
        validator = BehavioralValidator()
        code = "def foo( return 1"
        result = validator._check_syntax(code)
        assert result.passed is False
        assert "Syntax error" in result.message

    def test_check_entry_point_exists(self):
        validator = BehavioralValidator()
        code = "def main(): pass"
        result = validator._check_entry_point(code, "main")
        assert result.passed is True

    def test_check_entry_point_missing(self):
        validator = BehavioralValidator()
        code = "def helper(): pass"
        result = validator._check_entry_point(code, "main")
        assert result.passed is False
