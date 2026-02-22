# GAAP Evolution: Cognitive Consolidation & Associative Memory (v2.0)

**Focus:** Moving from "Retrieval" to "Understanding" via Hybrid RAG & Knowledge Graphs.

## 1. The Cognitive Shift: From Files to Graph
Storing experiences as isolated JSON blobs leads to "Context Fragmentation".
**New Paradigm:** **Episodic Knowledge Graph (EKG)**.
- Every "Episode" (Task) is a node.
- Relationships (Depends on, Caused by, Solved by) are edges.

## 2. Architecture: The Hybrid MemoRAG Core

We will implement a dual-stream memory system:

### 2.1 The Semantic Vector Stream (Fuzzy Recall)
- **Tech:** `LanceDB` (for performance) + `Ollama/BGE-M3` embeddings.
- **Function:** Handles "I think I've seen something like this before" queries.

### 2.2 The Graph Reasoning Stream (Logical Recall)
- **Tech:** `NetworkX` (Local) or `Neo4j` (Optional).
- **Function:** Resolves complex dependencies. (e.g., "Find all past failures related to 'Database Migrations' in 'PostgreSQL' context").

## 3. The "Dreaming" Cycle: Active Consolidation (v2.0)

This is no longer just a log-processing script. It is an **Architectural Review Process**.

### 3.1 The REAP Algorithm (Review, Extract, Abstract, Prune)
1.  **Review:** Cluster daily episodes using semantic similarity.
2.  **Extract:** Identify the "Invariant" in successful tasks. (e.g., "Every time we used `peewee` with `Postgres`, we needed `psycopg2`").
3.  **Abstract:** Create a **Strategic Heuristic** (A new rule for Layer 1).
4.  **Prune:** Delete low-confidence episodes to prevent "Memory Bloat" and "Semantic Noise".

## 4. Real-Time Memory Repair (Self-Corrective RAG)
If `Layer3` finds that a retrieved "Memory" leads to a failure:
1.  The Memory is flagged as **"Contradicted"**.
2.  During the next Dream Cycle, the system performs an **Adversarial Audit** to see why the previous rule failed and updates the Knowledge Graph.

## 5. Implementation Roadmap
1.  **Phase 1:** Implement `GraphMemory` to map project file dependencies.
2.  **Phase 2:** Integrate `LanceDB` for vectorized episodic retrieval.
3.  **Phase 3:** Deploy the `REAP` Dream Engine.

