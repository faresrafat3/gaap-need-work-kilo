# TECHNICAL SPECIFICATION: GAAP Engine Evolution (v2.0)

**Target:** `gaap/gaap_engine.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws Identified
The current engine follows a linear `L1 -> L2 -> L3` pipeline. This is fragile.
- **Flaw 1:** No back-propagation of strategic errors (L3 failure doesn't inform L1).
- **Flaw 2:** Passive memory usage (Doesn't actively fetch "lessons" before tasks).
- **Flaw 3:** Missing "Constitutional Gatekeeper" (Accepts technically correct but axiomatically wrong code).

## 2. Refactoring Requirements

### 2.1 The Recursive Loop (The "OODA" Loop)
Refactor `process()` to implement an **Observe-Orient-Decide-Act (OODA)** loop instead of a waterfall.

```python
async def process(self, request):
    # ... existing init ...
    while not goal_achieved:
        # 1. Observe: Check environment state & memory
        current_state = await self.observer.scan()
        
        # 2. Orient: Update TaskGraph based on new state
        # If L3 failed a critical task, L1 must re-plan here.
        if self.needs_replanning(current_state):
             await self.layer1.replan()
        
        # 3. Decide: Pick next best task (using Graph of Thoughts)
        next_task = self.layer2.get_next_task()
        
        # 4. Act: Execute with Axiom Enforcement
        result = await self.layer3.execute(next_task)
        
        # 5. Learn: Immediate Reflection
        self.memory.working.add(reflection(result))
```

### 2.2 Constitutional Integration
Inject the `AxiomValidator` before marking any task as complete.
- **Requirement:** `if not axiom_validator.validate(result.code): raise AxiomViolationError`.

### 2.3 Dynamic Few-Shot Injection
Update `_execute_with_healing` to fetch relevant examples from `VectorMemory`.
- **Logic:** `context['examples'] = self.memory.vector.search(task.description, k=3)`

## 3. New Components to Import
- `from gaap.core.axioms import AxiomValidator`
- `from gaap.memory.memorag import MemoRAG`
- `from gaap.meta_learning.reflection import RealTimeReflector`

## 4. Metrics & Telemetry
- Log `strategic_replan_count` (How often L3 failures caused L1 to change mind).
- Log `axiom_violation_rate` (How often code was rejected for style/safety).

---
**Handover Status:** Ready. Code Agent must implement the OODA loop structure.
