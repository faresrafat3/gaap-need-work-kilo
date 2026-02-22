# TECHNICAL SPECIFICATION: MCTS Strategic Planning (v1.0)

**Focus:** Probabilistic decision making and long-term planning (Inspired by SWE-Search and AlphaZero).

## 1. Beyond Tree of Thoughts
ToT is a simple branch-and-prune approach. **Monte Carlo Tree Search (MCTS)** adds "Simulation" and "Value Estimation" to find the global optimum.

## 2. Architecture: The MCTS Loop

`Layer 1 Strategic` will be upgraded to run 4 phases per decision:

### 2.1 Selection
Traverse the current Thought Graph using the **UCT (Upper Confidence Bound for Trees)** formula to balance exploration vs. exploitation.

### 2.2 Expansion
Generate 3 new "Move" proposals (Architecture Decisions).

### 2.3 Simulation (Rollout)
Ask a cheap SLM (e.g. Llama-3-8B) to "Fast-forward" the plan: "If we pick this DB, what happens 5 steps later in deployment?"

### 2.4 Backpropagation
Update the `ValueScore` of all parent nodes based on the simulation outcome.

## 3. The Value Agent (The Oracle)
A specialized sub-agent that predicts the "Success Probability" of a node.
- *Input:* ThoughtNode context.
- *Output:* Score from 0.0 to 1.0.

## 4. Implementation Plan
1.  **Add** `mcts_logic.py` to `gaap/layers/`.
2.  **Refactor** `Layer1Strategic` to use the MCTS loop for high-complexity tasks (Complexity > MODERATE).
3.  **Integrate** with `Simulator` (File 03) for the Rollout phase.

## 5. Axiom: Depth First for Criticality
For `CRITICAL` priority tasks, the MCTS must run at least 50 iterations to ensure no "catastrophic failure" branch was missed.
