# GAAP Evolution: Intelligence Accumulation & SOTA Integration (v2.0)

**Focus:** Integrating breakthroughs from DSPy, MetaGPT, CrewAI, and Microsoft Research.

## 1. The "Programmatic Prompting" Core (Inspired by DSPy)
We are moving from **Static Strings** to **Declarative Modules**.
- **The Signature:** Every task defines its input/output schema.
- **The Teleprompter:** A module that automatically optimizes prompts based on the success/failure history in `Episodic Memory`.

## 2. Structural Role-Playing (Inspired by MetaGPT)
Agents are no longer "Chatty Assistants"; they are **Formal Roles**.
- **Role Backstory:** Defined in `.gaap/roles/`. Each role has a specific "Training Manual" (SOP).
- **Artifact-Centric:** Agents do not communicate via chat; they communicate via **Artifacts** (PRs, Specs, Test Results).

## 3. Contextual Expertise (Inspired by Medprompt)
We implement **Dynamic Few-Shot Selection (DFS)**.
- Before executing Layer 3, the `MemoryManager` retrieves the top 3 most similar successful task "trajectories" and injects them as examples.
- This effectively "Fine-tunes" the model on-the-fly for the specific task at hand.

## 4. Self-Evolving Profiles (Inspired by MorphAgent)
Agents in the Swarm can **update their own identity**.
- If a "Coder Fractal" consistently succeeds at SQL but fails at CSS, it updates its `Reputation Profile` to "SQL Specialist".
- The Swarm Marketplace then routes SQL tasks to it automatically.

## 5. Implementation Roadmap (Research Integration)
1.  **Phase 1 (DSPy):** Refactor `gaap/core/base.py` to support Declarative Signatures.
2.  **Phase 2 (MetaGPT):** Create the `SOP_Manager` in `gaap/layers/`.
3.  **Phase 3 (Medprompt):** Implement the `FewShotRetriever` in `gaap/memory/`.

## 6. Curated Bibliography (The Lab)
- *DSPy: Compiling Declarative Language Model Calls.*
- *MetaGPT: Meta Programming for Multi-Agent Systems.*
- *Medprompt: The Power of Prompting for Generalist Models.*
- *Reflexion: Language Agents with Iterative Self-Reflection.*
