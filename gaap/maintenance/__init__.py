"""
Maintenance Module - Technical Debt Management
==============================================

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md

Automated technical debt detection and management:
- DebtScanner: Detect TODO/FIXME/complexity/duplicates/dead code
- InterestCalculator: Prioritize debt by impact
- RefinancingEngine: Propose and apply fixes

Usage:
    from gaap.maintenance import DebtScanner, InterestCalculator, RefinancingEngine

    scanner = DebtScanner()
    result = scanner.scan_directory("./src")

    calculator = InterestCalculator()
    prioritized = calculator.prioritize(result.items)

    engine = RefinancingEngine(llm_provider=provider)
    proposal = await engine.propose(prioritized[0])
"""

from gaap.maintenance.debt_config import (
    DebtConfig,
    DebtPriority,
    DebtType,
    ProposalStatus,
    create_debt_config,
)
from gaap.maintenance.debt_scanner import (
    DebtItem,
    DebtScanner,
    ScanResult,
    create_scanner,
)
from gaap.maintenance.interest_calculator import (
    InterestCalculator,
    InterestFactors,
    InterestReport,
    create_interest_calculator,
)
from gaap.maintenance.refinancing import (
    RefactoringProposal,
    RefinancingEngine,
    RefinancingResult,
    create_refinancing_engine,
)

__all__ = [
    "DebtConfig",
    "DebtPriority",
    "DebtType",
    "ProposalStatus",
    "create_debt_config",
    "DebtItem",
    "DebtScanner",
    "ScanResult",
    "create_scanner",
    "InterestCalculator",
    "InterestFactors",
    "InterestReport",
    "create_interest_calculator",
    "RefactoringProposal",
    "RefinancingEngine",
    "RefinancingResult",
    "create_refinancing_engine",
]
