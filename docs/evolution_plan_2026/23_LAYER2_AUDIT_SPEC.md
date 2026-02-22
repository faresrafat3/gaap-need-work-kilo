# TECHNICAL SPECIFICATION: Layer 2 Evolution (Rolling Wave Planning)

**Target:** `gaap/layers/layer2_tactical.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Waterfall Planning:** Generates 50 tasks upfront, making adaptation impossible.
- **Fragile Dependencies:** Relies on LLM's hallucinated dependency lists.
- **Vague Handover:** Tasks lack structural definitions (Inputs/Outputs/Tools).

## 2. Refactoring Requirements

### 2.1 Implementing Rolling Wave Planning
Replace `TacticalDecomposer.decompose` with a **JIT Decomposer**.

**Logic:**
1.  **Initial Plan:** Create high-level "Phases" (Epics).
2.  **Detailed Plan:** Break down *only* the current Phase into Atomic Tasks.
3.  **Feedback Loop:** When a Phase completes, re-evaluate the next Phase based on the actual code state.

### 2.2 Semantic Dependency & Environment Resolver
Enhance `DependencyResolver` to use **Logic** and **Environment Scans**:
- *Rule:* Testing tasks ALWAYS depend on their Implementation tasks.
- *System Check:* If a task requires compilation (e.g. `pip install numpy`), check for system tools (`gcc`, `cargo`) first. If missing, inject a `setup_environment` task automatically.
- *Rule:* If Task A modifies `file_x` and Task B reads `file_x`, then B depends on A.

### 2.3 Structural Task Definitions (DSPy Style)
Update `AtomicTask` dataclass to include:
```python
@dataclass
class AtomicTask:
    # ... existing ...
    input_schema: dict[str, str]  # e.g. {"user_model": "path/to/user.py"}
    output_schema: dict[str, str] # e.g. {"test_file": "path/to/test_user.py"}
    recommended_tools: list[str]  # e.g. ["write_file", "run_pytest"]
```

## 3. Implementation Steps
1.  **Refactor** `TaskGraph` to support "Placeholder Nodes" (unexpanded tasks).
2.  **Implement** the `RollingPlanner` class.
3.  **Update** `AtomicTask` schema to be stricter.

---
**Handover Status:** Ready. Code Agent must implement `RollingPlanner`.
