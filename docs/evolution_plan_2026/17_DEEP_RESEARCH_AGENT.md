# GAAP Evolution: Deep Research & Discovery Agent (v2.0 - STORM Enhanced)

**Focus:** Implementing State-of-the-Art (SOTA) Research Architectures (STORM, Co-STORM, Adversarial Verification).

## 1. Core Philosophy: Research as "Multi-Perspective Synthesis"
We are moving away from simple "Search & Summarize" to the **STORM Architecture** (Synthesis of Topic Outlines through Retrieval and Multiperspective question asking).

## 2. Architecture: The STORM Loop

The Research Agent is not a single entity, but a **Cluster of 4 Sub-Agents**:

### 2.1 The Perspective Generator (The Questioner)
Instead of searching for "How to fix bug X", it generates diverse perspectives:
- *Security perspective:* "Does fixing bug X introduce a vulnerability?"
- *Performance perspective:* "Is the fix O(n) or O(n^2)?"
- *Maintenance perspective:* "Is this fix standard practice or a hack?"

### 2.2 The Expert Hunter (The Retriever)
Executes targeted searches for *each* perspective generated above.
- Uses **Adaptive Query Generation**: If a search yields low-confidence results, it automatically refines the query keywords based on semantic expansion.

### 2.3 The Adversarial Critic (The Skeptic)
**New SOTA Feature:** Implementation of **Adversarial Factuality**.
- After finding a solution, this agent explicitly searches for: *"Why solution X is dangerous/wrong"*.
- It calculates a **PCC Score** (Probabilistic Certainty & Consistency). If the score is < 0.8, the finding is flagged as "Unverified".

### 2.4 The Knowledge Synthesizer (The Writer)
Compiles all verified findings into a structured report (Markdown/JSON) referencing sources.
- It builds a temporary **Knowledge Graph** of the findings to detect contradictions before writing the final output.

## 3. Data Flow
```
User Request
    |
    v
[Perspective Generator] --> Generates 5 distinct sub-questions
    |
    v
[Expert Hunter] x 5 Parallel Threads --> Searches Web/Docs/Memory
    |
    v
[Adversarial Critic] --> Attacks the findings
    |
    v
[Synthesizer] --> Resolves conflicts & Writes Report
```

## 4. Implementation Tech Stack
- **Orchestration:** `LangGraph` or Custom State Machine (in `gaap/research/engine.py`).
- **Retrieval:** `Google Search API` (or Serper) + `Trafilatura` (for scraping content).
- **Verification:** `Cross-Encoder` models for checking entailment (Does source A actually support claim B?).

## 5. Roadmap
1.  **Phase 1:** Implement the `PerspectiveGenerator` class using ToT (Tree of Thoughts).
2.  **Phase 2:** Build the `AdversarialCritic` loop.
3.  **Phase 3:** Integrate **STORM** logic into `Layer1_Strategic` for architectural decisions.
