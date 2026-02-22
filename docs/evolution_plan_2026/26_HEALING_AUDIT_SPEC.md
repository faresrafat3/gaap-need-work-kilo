# TECHNICAL SPECIFICATION: Healing Evolution (Reflexion & Semantic Analysis)

**Target:** `gaap/healing/healer.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Template-Based Refinement:** Generic error messages lead to generic fixes.
- **Regex Classification:** Misses semantic/logic errors.
- **Linear Escalation:** Wastes resources on futile retries for critical errors.

## 2. Refactoring Requirements

### 2.1 Implementing Reflexion (Verbal Reinforcement)
Replace `PromptRefiner` with a `ReflexionEngine`.
- **Workflow:**
    1.  Error occurs.
    2.  Agent generates **Self-Reflection**: "I failed because [Reason]. Plan: [Action]."
    3.  New Prompt = Original Prompt + "## PREVIOUS ATTEMPT FAILED DUE TO:
" + Reflection.

### 2.2 Semantic Error Classifier
Use a lightweight LLM call to classify the error if Regex fails.
- **Prompt:** "Here is the error trace. Is this a Syntax Error, Logic Error, or Environment Error?"
- **Action:** Map the semantic category to the correct Healing Level directly (Skip L1 if Logic Error).

### 2.3 The "Post-Mortem" Memory
If L4 (Strategy Shift) fails, store the entire failure trace in `EpisodicMemory` with a negative weight.
- **Future Impact:** When `Layer1` plans next time, it sees this failure and avoids the same path.

## 3. Implementation Steps
1.  **Create** `ReflexionEngine` class.
2.  **Update** `_attempt_level` to store and inject reflections.
3.  **Refactor** `_determine_start_level` to be smarter (Semantic checks).

---
**Handover Status:** Ready. Code Agent must implement Reflexion.
