# TECHNICAL SPECIFICATION: Meta-Learning Evolution (Recursive Wisdom)

**Target:** `gaap/meta_learning/`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Keyword Clustering:** Mistakes statistical correlation for logical causation.
- **Disconnected Recommendations:** Advice is generated but not enforced as project axioms.
- **Weak Failure Modeling:** Doesn't capture the "Anti-Patterns" specific to this user/project.

## 2. Refactoring Requirements

### 2.1 The Wisdom Distiller (v2.0)
Upgrade `PatternExtractor` to use **LLM-Summarized Abstraction**.
- **Action:** During the "Dream Cycle", take 5 similar successful episodes and ask an LLM: "What is the universal engineering principle used here?"
- **Output:** A `ProjectHeuristic` (e.g., "Always use `contextlib.suppress` when dealing with ephemeral file cleanup").
- **Integration:** Inject this heuristic into the **Prompt Gene Pool** (File 49) as a new 'Constraint Trait' for future generations.

### 2.2 Automated Axiom Injection
Link `RecommendationEngine` directly to `.gaap/constitution.yaml`.
- **Logic:** If a recommendation (e.g., "Use async for I/O") has a >90% success rate across 10 tasks, the system should propose adding it as a **Hard Axiom**.

### 2.3 Contrastive Experience Store
Implement a dual-entry memory for failures.
- **Structure:** 
    - `FailedTrace`: What the agent thought + the error.
    - `CorrectiveAction`: The specific fix that worked.
- **Retrieval:** When a new task starts, search the `FailureStore` for "similar pitfalls".

### 2.4 Self-tuning Hyperparameters
The MetaLearner should recommend changes to **System Settings**.
- **Action:** Analyze if `temperature=0.7` is causing too many syntax errors in L3. 
- **Recommendation:** "Lower temperature to 0.2 for TaskType.CODE_GEN".

## 3. Implementation Steps
1.  **Add** `distill_wisdom()` method to `MetaLearner`.
2.  **Refactor** `experience_analyzer.py` to support contrastive (Positive/Negative) pairs.
3.  **Create** a bridge to the `AxiomaticCore` for automated rule proposal.

---
**Handover Status:** Ready. Code Agent must implement 'Wisdom Distillation' during the next Dream Cycle.
