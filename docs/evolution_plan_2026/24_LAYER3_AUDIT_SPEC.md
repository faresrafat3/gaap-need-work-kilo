# TECHNICAL SPECIFICATION: Layer 3 Evolution (Zero-Trust Execution)

**Target:** `gaap/layers/layer3_execution.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Unsafe Parsing:** Regex-based tool parsing is fragile and dangerous.
- **Passive Learning:** Writes lessons but never reads them.
- **Over-Privileged:** Executor has access to the full host environment by default.

## 2. Refactoring Requirements

### 2.1 Implementing Zero-Trust Sandbox
Replace `subprocess` calls with a **Container Interface**.
- **Default:** Wasmtime (for pure logic/math).
- **Fallback:** Docker (for complex dependencies like pandas/numpy).
- **Network:** Disabled by default.

### 2.2 Native Function Calling (MCP Ready)
Abandon regex parsing (`CALL: tool(...)`).
- Use the Provider's native `tools` API (OpenAI/Gemini).
- If the model doesn't support tools, use a **Structured JSON Schema** output.

### 2.3 Active Lesson Injection
Update `ExecutorPool.execute`:
```python
# Before execution
lessons = self.memory.retrieve(task.description, k=3)
prompt += "

## LESSONS FROM PAST FAILURES:
" + "
".join(lessons)
```

### 2.4 The "Auditor" Role
Add a step *after* code generation but *before* execution:
- **Static Analysis:** Run `ruff` and `bandit` on the generated code.
- **Policy Check:** Ensure no banned imports (e.g., `socket`, `subprocess`) unless explicitly allowed by the Task Capability Token.

## 3. Implementation Steps
1.  **Integrate** `gaap.security.sandbox` properly (make it mandatory).
2.  **Refactor** `ToolRegistry` to export JSON schemas for LLMs.
3.  **Implement** the `PreFlightCheck` (Lesson retrieval + Policy check).

---
**Handover Status:** Ready. Code Agent must enforce Sandboxing.
