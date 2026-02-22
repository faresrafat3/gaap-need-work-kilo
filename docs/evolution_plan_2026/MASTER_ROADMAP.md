# GAAP Master Roadmap: Path to AGI (2026)

**Goal:** Transform GAAP from a CLI tool into a self-evolving, autonomous Swarm Intelligence.

## 1. Phase 1: The Cognitive Upgrade (Weeks 1-4)
**Objective:** Give the agent "Long-term Memory" and "Imagination".

*   [ ] **Week 1: Vector Memory Core**
    *   Install `chromadb`.
    *   Create `VectorMemoryStore` class.
    *   Update `Layer3` to query memory before execution.
*   [ ] **Week 2: The Dreaming Cycle**
    *   Implement `DreamProcessor`.
    *   Create cron job for nightly "sleep" (log consolidation).
*   [ ] **Week 3: Semantic Simulation (World Model v0.5)**
    *   Implement LLM-based `predict_outcome` function.
    *   Add "Thinking" step in `Layer2` before task assignment.
*   [ ] **Week 4: Integration Testing**
    *   Verify that "lessons learned" on Day 1 are applied on Day 2.

## 2. Phase 2: The Safety Shield (Weeks 5-8)
**Objective:** Secure the agent so it can run autonomously without supervision.

*   [ ] **Week 5: Docker Sandboxing**
    *   Implement `DockerSandbox`.
    *   Update `ExecutorPool` to use Docker instead of `subprocess`.
*   [ ] **Week 6: Resource Limits & Firewalls**
    *   Implement strict CPU/RAM limits.
    *   Add network whitelist (PyPI only).
*   [ ] **Week 7: MCP Integration**
    *   Implement MCP Client.
    *   Connect to local filesystem via MCP (instead of direct access).
*   [ ] **Week 8: Security Audit**
    *   Red Team attack: Try to break out of the sandbox.

## 3. Phase 3: The Swarm & Evolution (Weeks 9-12+)
**Objective:** Scale intelligence via multiple agents and dynamic tools.

*   [ ] **Week 9: Dynamic Tool Synthesis**
    *   Build `ToolSynthesizer`.
    *   Allow agent to write its own python helpers.
*   [ ] **Week 10: Fractal Swarm Prototype**
    *   Launch 2 sub-agents (Coder + Critic) communicating via memory.
*   [ ] **Week 11: Full Autonomy Loop**
    *   Connect Swarm + Memory + Sandbox.
    *   Run "Project Genesis": Ask GAAP to build a simple To-Do app entirely on its own.
*   [ ] **Week 12+: Optimization**
    *   Fine-tune local SLM on the accumulated Vector Memory data.

## 4. Immediate Action Items (Next 24 Hours)

1.  Review `00_GAP_ANALYSIS.md` to confirm priorities.
2.  Set up the `dev` branch for Phase 1.
3.  Install initial dependencies: `pip install chromadb docker`.

---
**Signed:** *Gemini CLI Agent - Architect Mode*
