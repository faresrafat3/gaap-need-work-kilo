# GAAP Evolution: Metacognition & Epistemic Uncertainty

**Focus:** Giving the agent the ability to "know what it doesn't know".

## 1. The Epistemic Gap
Currently, an LLM will often confidently hallucinate a solution for a library it doesn't understand. 
**Target:** The agent must perform an "Internal Knowledge Audit" before execution.

## 2. Confidence Scoring Engine (CSE)

Before moving from Layer 1 to Layer 2, the system calculates a `ConfidenceScore`.

### 2.1 Variables
- **Similarity Score (S):** Distance to the nearest successful episodic memory (Vector search).
- **Task Novelty (N):** Presence of unknown keywords or libraries (compared to Procedural Memory).
- **Consensus Variance (V):** Disagreement level between Strategic critics (MAD Panel).

### 2.2 Formula (Heuristic)
`FinalConfidence = (S * 0.5) + ((1 - N) * 0.3) + ((1 - V) * 0.2)`

- **If Confidence < 40%:** The agent MUST trigger a **"Research Task"** (search docs, read code) before planning.
- **If Confidence 40-70%:** Proceed but increase **Genetic Twin** verification frequency.
- **If Confidence > 70%:** Direct execution.

## 3. Real-Time Auditor (The Inner Monologue)

We add a middleware in `Layer3_Execution` that monitors the stream of thoughts.

### 3.1 Audit Patterns
- **Circular Reasoning:** Detection of repeating sentences/patterns.
- **Complexity Spike:** If the generated code is significantly more complex than the planned architecture.
- **Safety Violation:** Real-time regex scan for forbidden patterns (e.g., hardcoded keys).

### 3.2 Intervention Logic
When the Auditor detects an issue, it injects a **System Interrupt**:
`"System Message: Your last thought seems circular. Re-evaluate your approach to the 'Authentication' module using a simpler logic."`

## 4. Implementation Plan
1.  **Phase 1:** Update `gaap/meta_learning/experience_analyzer.py` to include a `KnowledgeMap` generator.
2.  **Phase 2:** Integrate the `ConfidenceScorer` in `Layer0_Interface`.
3.  **Phase 3:** Build the `StreamingAuditor` for `Layer3`.
