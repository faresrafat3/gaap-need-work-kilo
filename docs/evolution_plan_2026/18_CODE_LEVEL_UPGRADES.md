# GAAP Technical Upgrade Specification: SOTA Integration (v1.0)

**Focus:** Mapping Academic Papers to Code Refactoring Tasks.

## 1. Upgrade Target: Cognitive Engine (Layer 1)
**Current File:** `gaap/layers/layer1_strategic.py`
**Target Methodology:** **Graph of Thoughts (GoT)** [Besta et al., 2024]

### Refactoring Logic
1.  **Replace** `TreeOfThoughts` class with `GraphOfThoughts`.
2.  **Introduce** `ThoughtNode` structure allowing multiple parents (for merging).
3.  **Implement** new Operations:
    - `Aggregation`: Combine 3 weak architectural ideas into 1 strong one.
    - `Refinement`: Loop on a single node to improve detail.
4.  **Metric:** Reduction in total tokens used for complex planning by ~30% via branch merging.

## 2. Upgrade Target: Memory System
**Current File:** `gaap/memory/hierarchical.py`
**Target Methodology:** **RAPTOR (Recursive Abstractive Retrieval)** [Sarthi et al., 2024]

### Refactoring Logic
1.  **Implement** `SummaryTree` structure using `LanceDB`.
2.  **Process:**
    - Leaf Nodes = Original chunks (Code snippets).
    - Parent Nodes = LLM Summaries of children.
    - Root Node = Project-level summary.
3.  **Retrieval Logic:** Implement `CollapsedTreeRetrieval` (Search across layers depending on query abstractness).

## 3. Upgrade Target: Self-Healing
**Current File:** `gaap/healing/healer.py`
**Target Methodology:** **Reflexion** [Shinn et al., 2023]

### Refactoring Logic
1.  **Add** `ShortTermMemory` buffer for "Reflections".
2.  **Loop Change:**
    - *Old:* Error -> Retry.
    - *New:* Error -> `Reflect()` -> Store Reflection -> Append to Context -> Retry.
3.  **Constraint:** The reflection MUST explicitly state the *cause* of the error, not just "I will try again".

## 4. Upgrade Target: MAD Critic
**Current File:** `gaap/mad/critic_prompts.py`
**Target Methodology:** **CRITIC (Tool-Interactive)** [Gou et al., 2024]

### Refactoring Logic
1.  **Deprecate** Pure-LLM prompts for verification.
2.  **Inject Tools:** Give the Critic access to:
    - `Interpreter`: To verify code snippets actually run.
    - `Search`: To verify API existence.
3.  **Workflow:** Critic generates a "Verification Plan", executes tools, compares output vs expectation.

## 5. Implementation Order
1.  **Healer (Reflexion):** Highest ROI, easiest to implement.
2.  **Critic (Tools):** Critical for preventing bugs.
3.  **Memory (RAPTOR):** Required for deep context.
4.  **Layer 1 (GoT):** Required for complex architectural innovation.
