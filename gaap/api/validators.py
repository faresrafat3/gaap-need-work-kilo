"""
Validators API - Code Validation Endpoints
========================================

Provides endpoints for:
- AST Guard (security scanning)
- Performance validation
- Behavioral testing
- Axiom compliance
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from gaap.validators import (
    ASTGuard,
    AxiomComplianceValidator,
    BehavioralValidator,
    PerformanceValidator,
)

router = APIRouter(prefix="/api/validators", tags=["validators"])


class CodeValidationRequest(BaseModel):
    """Request for code validation."""

    code: str = Field(..., min_length=1)
    language: str = Field(default="python")
    options: dict = Field(default_factory=dict)
    file_path: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response for validation."""

    valid: bool
    issues: list[dict] = Field(default_factory=list)
    report: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


@router.post("/ast", response_model=ValidationResponse)
async def validate_ast(request: CodeValidationRequest) -> ValidationResponse:
    """Validate code using AST Guard (security scanning)."""
    try:
        guard = ASTGuard()
        result = guard.scan(request.code, filename=request.file_path or "<string>")

        return ValidationResponse(
            valid=result.is_safe,
            issues=[
                {
                    "type": issue.issue_type.value,
                    "severity": issue.severity,
                    "message": issue.message,
                    "line": issue.line,
                    "column": issue.column,
                }
                for issue in result.issues
            ],
            report={
                "scan_time_ms": result.scan_time_ms,
                "lines_scanned": result.lines_scanned,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/performance", response_model=ValidationResponse)
async def validate_performance(request: CodeValidationRequest) -> ValidationResponse:
    """Validate code performance (complexity, maintainability)."""
    try:
        validator = PerformanceValidator()
        report = validator.validate(request.code, filename=request.file_path or "<string>")

        return ValidationResponse(
            valid=report.is_acceptable,
            report={
                "maintainability_index": report.maintainability_index,
                "total_loc": report.total_loc,
                "comment_ratio": report.comment_ratio,
                "avg_complexity": report.avg_complexity,
                "max_complexity": report.max_complexity,
                "scan_time_ms": report.scan_time_ms,
            },
            warnings=[issue.get("message", "") for issue in report.issues],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/axiom", response_model=ValidationResponse)
async def validate_axiom(request: CodeValidationRequest) -> ValidationResponse:
    """Validate code against project axioms/constitution."""
    try:
        validator = AxiomComplianceValidator()
        result = validator.validate(request.code, filename=request.file_path or "<string>")

        return ValidationResponse(
            valid=result.is_compliant,
            issues=[
                {
                    "type": "axiom_violation",
                    "constraint": c.constraint_name,
                    "message": c.message,
                    "line": c.line,
                    "severity": c.severity,
                }
                for c in result.issues
            ],
            report={
                "is_compliant": result.is_compliant,
                "checks_passed": result.checks_passed,
                "checks_failed": result.checks_failed,
                "scan_time_ms": result.scan_time_ms,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/behavioral", response_model=ValidationResponse)
async def validate_behavioral(request: CodeValidationRequest) -> ValidationResponse:
    """Validate code behavior through testing."""
    try:
        validator = BehavioralValidator()
        report = validator.validate(request.code)

        return ValidationResponse(
            valid=report.is_valid,
            report={
                "tests_passed": report.tests_passed,
                "tests_failed": report.tests_failed,
                "total_duration_ms": report.total_duration_ms,
            },
            warnings=report.execution_errors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
