"""
Tests for GAAP Validators
"""

import pytest

from gaap.validators.base import BaseValidator, QualityGate, Severity, ValidationResult


class MockValidator(BaseValidator):
    @property
    def gate(self):
        return QualityGate.SYNTAX_CHECK

    async def validate(self, artifact, context=None):
        if artifact is None:
            result = ValidationResult.failed("Artifact is None")
        elif isinstance(artifact, str) and len(artifact) == 0:
            result = ValidationResult.failed("Empty artifact", suggestions=["Provide content"])
        else:
            result = ValidationResult.passed("Valid")
        return self._record_result(result)


class TestValidationResult:
    def test_passed_factory(self):
        result = ValidationResult.passed("All good")

        assert result.passed is True
        assert result.severity == Severity.INFO
        assert result.message == "All good"

    def test_warning_factory(self):
        result = ValidationResult.warning("Minor issue", suggestions=["Fix this"])

        assert result.passed is True
        assert result.severity == Severity.WARNING
        assert result.message == "Minor issue"
        assert result.suggestions == ["Fix this"]

    def test_failed_factory(self):
        result = ValidationResult.failed("Critical error", suggestions=["Check input"])

        assert result.passed is False
        assert result.severity == Severity.ERROR
        assert result.message == "Critical error"
        assert result.suggestions == ["Check input"]

    def test_with_details(self):
        result = ValidationResult.passed("OK", details={"count": 5})

        assert result.details["count"] == 5


class TestBaseValidator:
    @pytest.fixture
    def validator(self):
        return MockValidator()

    def test_init(self, validator):
        assert validator.enabled is True
        assert validator.fail_on_error is True

    def test_gate_property(self, validator):
        assert validator.gate == QualityGate.SYNTAX_CHECK

    @pytest.mark.asyncio
    async def test_validate_valid(self, validator):
        result = await validator.validate("Some content")

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_validate_invalid(self, validator):
        result = await validator.validate(None)

        assert result.passed is False
        assert "None" in result.message

    @pytest.mark.asyncio
    async def test_validate_with_suggestions(self, validator):
        result = await validator.validate("")

        assert result.passed is False
        assert len(result.suggestions) > 0

    def test_stats(self, validator):
        assert validator._validation_count == 0
        assert validator._failures == 0

    @pytest.mark.asyncio
    async def test_stats_tracking(self, validator):
        await validator.validate("valid")
        await validator.validate("")
        await validator.validate(None)

        stats = validator.get_stats()

        assert stats["validations"] == 3
        assert stats["failures"] == 2
        assert stats["enabled"] is True


class TestSeverity:
    def test_severity_values(self):
        assert Severity.ERROR.value < Severity.WARNING.value
        assert Severity.WARNING.value < Severity.INFO.value


class TestQualityGate:
    def test_all_gates_defined(self):
        gates = [
            QualityGate.SYNTAX_CHECK,
            QualityGate.SECURITY_SCAN,
            QualityGate.PERFORMANCE_CHECK,
            QualityGate.STYLE_CHECK,
            QualityGate.COMPLIANCE_CHECK,
        ]

        assert len(gates) == 5


class TestDisabledValidator:
    def test_disabled_validator(self):
        validator = MockValidator(enabled=False)

        assert validator.enabled is False

    @pytest.mark.asyncio
    async def test_disabled_still_validates(self):
        validator = MockValidator(enabled=False)
        result = await validator.validate("test")

        assert result is not None


class TestFailOnError:
    def test_fail_on_error_false(self):
        validator = MockValidator(fail_on_error=False)

        assert validator.fail_on_error is False
