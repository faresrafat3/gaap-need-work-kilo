# GAAP Evolution: The Technical Debt Collector (v1.0)

**Focus:** Automated Maintenance & Code Health.

## 1. The Problem: The "Rot"
Codebases rot over time. Developers leave `TODOs` that never get done. Hacks become permanent.
**Target:** **Active Code Hygiene.**

## 2. Architecture: The Debt Agent

A background agent that wakes up when the system is idle.

### 2.1 The "Debt Scanner"
Scans the codebase for semantic markers:
- **Explicit Markers:** `# TODO`, `# FIXME`, `# XXX`.
- **Implicit Markers:** Functions with Complexity > 15 (Cyclomatic Complexity), Duplicate code blocks (Dry/Copy-Paste).

### 2.2 The "Interest Calculator"
Not all debt is equal. The agent calculates "Interest":
- A `# FIXME` in `auth.py` (Critical Path) has **High Interest**.
- A `# TODO` in `tests/utils.py` has **Low Interest**.

### 2.3 The "Refinancing" Workflow
1.  **Identify:** "Found a complex function `process_data` (Complexity 25)."
2.  **Plan:** "I can split this into 3 smaller functions."
3.  **Propose:** Create a `refactor/process-data` branch and push it.
4.  **Notify:** "Hey, I refactored that messy function. Tests passed. Merge?"

## 3. Implementation Plan
1.  **Phase 1:** Integrate `radon` (for complexity metrics) and `pylint` (for code smells).
2.  **Phase 2:** Build the `DebtScanner` module.
3.  **Phase 3:** Create the `RefactoringProposal` template for the GitHub integration.

## 4. Axiom
**Safety First:** The Debt Agent NEVER pushes to `main`. It always works on a side branch.
