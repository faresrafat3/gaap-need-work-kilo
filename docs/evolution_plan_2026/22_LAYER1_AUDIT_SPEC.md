# TECHNICAL SPECIFICATION: Layer 1 Evolution (Graph of Thoughts)

**Target:** `gaap/layers/layer1_strategic.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Linearity:** Decisions follow a fixed hierarchy (Paradigm -> DB -> Comm).
- **Isolation:** No memory of past architectural failures.
- **Subjectivity:** Critics vote with arbitrary numbers instead of proofs.

## 2. Refactoring Requirements

### 2.1 Implementing Graph of Thoughts (GoT)
Replace `ToTStrategic` with `GoTStrategic`.

**New Data Structure:**
```python
class ThoughtNode:
    id: str
    content: ArchitectureDecision
    parents: list[ThoughtNode]  # Allows merging
    children: list[ThoughtNode]
    score: float
    valid: bool
```

**New Operations:**
- `Generate`: Create 3 diverse paradigms.
- `Aggregate`: Take the "Security" from Paradigm A and "Speed" from Paradigm B.
- `Refine`: Improve a specific node (e.g., "Switch DB from SQL to NoSQL").

### 2.2 Memory-Augmented & Research-Driven Planning
Before generating options, perform an **Epistemic Check**:
1.  **Memory:** "What architectural patterns failed in the past for [Intent]?"
2.  **Research Trigger:** If the request involves unknown terms (e.g. "CryptoLib_2026"), the system **MUST** pause planning and delegate a `DeepResearchTask` to Layer 0.5.
3.  **Action:** Add failures as "Negative Constraints" and Research Findings as "Context" to the System Prompt.

### 2.3 Evidence-Based Criticism
Update `MADArchitecturePanel` to require **Evidence**.
- *Old:* "Score: 0.6 because it might be slow."
- *New:* "Score: 0.6. Evidence: Microservices add ~20ms latency per hop. For a real-time app (Constraint #2), this exceeds the 50ms budget."

## 3. Implementation Steps
1.  **Refactor** `ArchitectureSpec` to support a graph of decisions, not just a list.
2.  **Implement** the `GoT` engine with `merge()` and `refine()` methods.
3.  **Inject** `HierarchicalMemory` into `Layer1Strategic.__init__`.

---
**Handover Status:** Ready. Code Agent must implement `GoTStrategic`.
