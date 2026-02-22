# GAAP Evolution: Operations, CI/CD & Feedback Loops (v1.0)

**Focus:** Transforming GAAP from a "Script" to a "Production-Grade Product".

## 1. The Challenge: Testing the Unpredictable
Standard unit tests (`assert x == 5`) fail with Agents because LLMs give different answers every time.
**Target:** **Semantic Eval & Deterministic Replay**.

## 2. Architecture: The GAAP CI Pipeline

We need a dedicated GitHub Action workflow that runs **"Evaluations"** not just "Tests".

### 2.1 The Evaluation Matrix
Run the agent against the `tests/scenarios/adversarial_cases.json`.
- **Pass:** Code works AND Agent detected the trap.
- **Fail:** Code fails OR Agent fell for the trap.
- **Flake:** Code works sometimes. (Flag for "Reflexion Tuning").

### 2.2 Cost & Latency Guardrails
CI fails if:
- Average Task Cost > $0.05.
- Average Latency > 10s.
*Why?* An AGI that bankrupts the user is a failed product.

## 3. Feedback Loop: The "Complaint Box" (v1.0)

Users need a way to say "You were stupid here" without opening a GitHub Issue.

### 3.1 The `gaap feedback` Command
- **User:** Runs `gaap feedback --last-task --rating 1 --comment "You deleted my .env file!"`
- **System:**
    1.  Captures the Trace ID.
    2.  Creates a **Negative Episodic Memory**.
    3.  Adds a high-priority "Constraint" to the `VectorMemory`: *"NEVER delete .env files"*.
    4.  Triggers an immediate **Dream Cycle** to consolidate this rule.

## 4. Monetization & Deployment (Future)
- **GAAP Cloud:** A hosted version where we manage the Docker Containers.
- **GAAP Local:** The current version, but packaged as a single binary (using PyInstaller) to avoid dependency hell.

## 5. Roadmap
1.  **Phase 1:** Create `.github/workflows/agent_eval.yml`.
2.  **Phase 2:** Implement `gaap feedback` CLI command.
3.  **Phase 3:** Build `scripts/cost_monitor.py` to track token usage.
