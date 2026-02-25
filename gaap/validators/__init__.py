"""
GAAP Validators Module
======================

Implements: docs/evolution_plan_2026/41_VALIDATORS_AUDIT_SPEC.md

Multi-layer validation system:

AST Guard (Security v2.0):
    - AST pattern matching for dangerous code
    - Call chain tracing
    - Dangerous import detection

Behavioral Validator:
    - Dynamic execution in sandbox
    - Test generation
    - Pass/fail reporting

Performance Validator:
    - Radon integration (Cyclomatic Complexity)
    - Maintainability Index
    - Quality thresholds

Axiom Compliance:
    - Project constitution validation
    - Positive/Negative constraint checking

Usage:
    from gaap.validators import ASTGuard, PerformanceValidator

    guard = ASTGuard()
    issues = guard.scan(code)

    perf = PerformanceValidator()
    report = perf.validate(code)
"""

from .ast_guard import (
    ASTGuard,
    ASTIssue,
    ASTIssueType,
    ASTScanResult,
    create_ast_guard,
)

from .performance import (
    PerformanceValidator,
    PerformanceReport,
    ComplexityResult,
    create_performance_validator,
)

from .axiom_compliance import (
    AxiomComplianceValidator,
    ComplianceResult,
    ConstraintType,
    create_axiom_validator,
)

from .behavioral import (
    BehavioralValidator,
    BehavioralReport,
    BehavioralConfig,
    TestResult,
    create_behavioral_validator,
)

__all__ = [
    "ASTGuard",
    "ASTIssue",
    "ASTIssueType",
    "ASTScanResult",
    "create_ast_guard",
    "PerformanceValidator",
    "PerformanceReport",
    "ComplexityResult",
    "create_performance_validator",
    "AxiomComplianceValidator",
    "ComplianceResult",
    "ConstraintType",
    "create_axiom_validator",
    "BehavioralValidator",
    "BehavioralReport",
    "BehavioralConfig",
    "TestResult",
    "create_behavioral_validator",
]
