# GAAP Evolution: The State-Transition World Model (v2.0)

**Focus:** Moving from "Textual Imagination" to "State-Transition Prediction".

## 1. The Core Problem: Lack of Counterfactuals
Current GAAP doesn't ask "What if this command fails halfway?". It assumes linear success.
**New Paradigm:** **The Predictive State Model (PSM)**.

## 2. Architecture: The Delta-Simulation Loop

Instead of just predicting the outcome, the simulator predicts the **State Delta** (ΔS).

### 2.1 The Components
1.  **State Snapshot (S0):** Capture the relevant metadata of the current environment (FS tree, environment variables, installed packages).
2.  **Action Projector (A):** The proposed command/code.
3.  **Delta Predictor (Δ):** An LLM + Heuristic engine that outputs:
    - `FilesCreated: []`
    - `FilesDeleted: []`
    - `SecurityChanges: []`
4.  **Verification Oracle:** Compares ΔS against the **Project Constitution**. (e.g., "Predicting deletion of `main.py` -> VIOLATION of Axiom #2").

## 3. High-Stakes Rehearsal: The "GhostFS"

For filesystem operations, we implement a **Virtual File System (VFS)** in-memory.

### 3.1 Workflow
1.  **Virtualization:** Map the project files into a Python-based VFS (like `PyFilesystem2`).
2.  **Execution:** Run the agent's logic against the VFS.
3.  **Audit:** Inspect the VFS after execution.
4.  **Commit:** If the user approves, apply the changes from the VFS to the Real FS.

## 4. Counterfactual Reasoning & Deterministic Mocking
The simulator will run **Adversarial Scenarios**:
- "What if the network fails during `pip install`?"
- "What if the disk is full during `db_migration`?"

### 4.1 Deterministic Mocking
To ensure simulations are reproducible:
- **Randomness:** The Simulator forces a fixed seed for `random` and `uuid`.
- **Time:** The Simulator freezes `datetime.now()` to a static value.
- **Goal:** Predictable state transitions, preventing "Flaky Simulations".

## 5. Roadmap
1.  **Phase 1:** Build the `StateSnapshot` tool using `git ls-files` and `os.walk`.
2.  **Phase 2:** Implement the `GhostFS` for safe file-writing rehearsals.
3.  **Phase 3:** Integrate "Adversarial State Checks" into Layer 2 (Tactical).

