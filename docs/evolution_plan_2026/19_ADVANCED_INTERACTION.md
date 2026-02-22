# GAAP Evolution: Advanced LLM Interaction & Dynamic Personas (v1.0)

**Focus:** Elevating LLM performance via Dynamic Role-Playing, Semantic Pressure, and Contrastive Reasoning.

## 1. Dynamic Persona Engine (DPE)
Instead of a static system prompt, GAAP will use a **Tiered Persona System**.

### 1.1 The Tiered Structure
1.  **Core Identity (The Soul):** Persistent values from the Project Constitution (The Strategic Architect).
2.  **Adaptive Mask (The Persona):** Switches based on the `IntentType` (from Layer 0).
    - *Intent: DEBUG* -> Mask: **The Forensic Pathologist** (Focus on root cause, side effects).
    - *Intent: ARCHITECT* -> Mask: **The Civil Engineer** (Focus on longevity, structural integrity).
    - *Intent: SECURITY* -> Mask: **The Thief** (Focus on breaking trust, finding leaks).

## 2. Context Management: The "Chain of Density"
To prevent performance degradation in long tasks, we implement **Incremental Summary Buffering**.

### 2.1 Logic
- Every 5 turns, the agent triggers a `Self-Distillation` task.
- It compresses the last 5 turns into a **Semantic Matrix** (Facts, Decisions, Pending Risks).
- The old turns are moved to `Episodic Memory`, and only the Matrix stays in the active context.

## 3. Reasoning Patterns: Contrastive CoT
We force the model into **"Dual-Track Thinking"**.

### 3.1 The Prompt Template Update
For every complex decision, the prompt must structure the output as:
1.  **Path A (Proposed):** The logical solution.
2.  **Path B (Adversarial):** Why Path A might fail, what are its hidden costs?
3.  **Synthesis:** The final decision after weighing A vs B.

## 4. Semantic Pressure (The "Hard-Mode" Prompt)
We use linguistic constraints to prevent "lazy" outputs.
- *Constraint:* "Do not use the words 'ensure', 'robust', or 'efficient' without providing a quantitative metric (e.g., latency in ms, memory in MB)."
- *Goal:* Forcing the model to move from vague marketing language to precise engineering language.

## 5. Implementation Roadmap
1.  **Phase 1:** Update `gaap/mad/critic_prompts.py` with Contrastive templates.
2.  **Phase 2:** Implement the `PersonaSwitcher` in `gaap/gaap_engine.py`.
3.  **Phase 3:** Build the `SemanticDistiller` middleware to manage context window health.
