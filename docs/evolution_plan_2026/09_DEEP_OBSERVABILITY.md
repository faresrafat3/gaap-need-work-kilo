# GAAP Evolution: Deep Observability & Time-Travel Debugging

**Focus:** "See everything, rewind anytime."

## 1. The Observability Gap
Standard logs (`logging.info`) are insufficient for complex cognitive processes involving multiple LLM calls, tool executions, and memory retrievals.

**Target:** Full **Distributed Tracing** (Jaeger/OpenTelemetry) for every thought.

## 2. Architecture: The Flight Recorder

We will implement **OpenTelemetry (OTEL)** across the entire GAAP stack.

### 2.1 Tracing Span Structure

Every user request generates a `TraceID`.
Every layer generates a `Span`.

- **Span: Layer 1 (Strategic)**
  - **Attributes:** `intent_type`, `complexity_score`, `model_name`.
  - **Event:** `ToT_Branch_Generated` (Payload: JSON of the thought).
  - **Event:** `Critic_Vote` (Payload: Vote reasoning).
  
- **Span: Layer 3 (Execution)**
  - **Attributes:** `tool_name`, `sandbox_id`, `execution_time`.
  - **Event:** `Tool_Call` (Payload: Arguments).
  - **Event:** `Tool_Result` (Payload: Output).

### 2.2 Time-Travel Debugging (The "Replay" Feature)

This is the killer feature. Since we log every LLM input/output and tool result:
**We can deterministically replay any past session.**

#### Scenario:
1.  User runs a complex refactoring task. It fails at Step 15.
2.  User types: `gaap debug --session-id=xyz --step=14`.
3.  GAAP loads the exact state of memory/context at Step 14.
4.  User can now:
    - Change the prompt manually.
    - Inspect the memory retrieval.
    - Force a different tool output.
    - Resume execution from there to fix the bug.

## 3. Implementation Plan
1.  **Phase 1:** Add `opentelemetry-sdk` and instrument `GAAPEngine`.
2.  **Phase 2:** Set up a local Jaeger/Zipkin instance (Docker).
3.  **Phase 3:** Build the `SessionReplay` engine in `gaap/core/replay.py`.

## 4. Telemetry Dashboard
Integration with **Grafana** to show:
- Token usage per task type.
- Cost per success.
- Average "Thought Depth" (Layer 1 recursion depth).
- Failure rate per Tool.
