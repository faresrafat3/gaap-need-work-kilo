# TECHNICAL SPECIFICATION: Validators Evolution (Dynamic & Axiomatic QA)

**Target:** `gaap/validators/`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Regex-Only Security:** Blind to obfuscated payloads and advanced injection patterns.
- **Static Inactivity:** Only checks how code looks, not how it behaves.
- **Context-Free Rules:** Ignorant of Project Axioms and architectural constraints.

## 2. Refactoring Requirements

### 2.1 The Semantic AST Guard (Security v2.0)
Replace/Augment regex with **AST Pattern Matching**.
- **Action:** Use Python's `ast` module to trace call chains.
- **Logic:** Flag any execution of `builtins.eval`, `builtins.exec`, or `subprocess` calls where `shell=True` is a variable, not a literal.

### 2.2 Dynamic Execution Validation (The "Prover")
Introduce `BehavioralValidator`.
- **Requirement:** For any `TaskType.CODE_GENERATION`, the validator must:
    1.  Generate a test suite (using a secondary cheap model).
    2.  Run the code + tests in the `Sandbox`.
    3.  Report `is_passed` based on test results.

### 2.3 Axiom Compliance Gate
Create `AxiomValidator` linked to `docs/evolution_plan_2026/16_AXIOMATIC_CORE.md`.
- **Logic:** Scans for "Positive Constraints" (e.g., "Must use async/await") and "Negative Constraints" (e.g., "Do not use requests, use aiohttp").

### 2.4 Performance Profiling (radon integration)
Update `PerformanceValidator`.
- **Action:** Integrate `radon` library.
- **Metric:** Fail if Cyclomatic Complexity > 10 for a single function or Maintainability Index < 20.

## 3. Implementation Steps
1.  **Add** `ast-grep` or `bandit` to dependencies for deeper static analysis.
2.  **Update** `QualityPipeline` in `Layer3` to include the Dynamic Execution step.
3.  **Implement** the `AxiomValidator` using Pydantic schemas.

---
**Handover Status:** Ready. Code Agent must integrate 'Bandit' for security and 'Radon' for performance immediately.
