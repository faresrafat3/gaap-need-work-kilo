# TECHNICAL SPECIFICATION: Memory Evolution (Vector & Graph)

**Target:** `gaap/memory/hierarchical.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Keyword Dependency:** Fails to retrieve semantically related concepts (e.g. Auth vs Login).
- **Flat Structure:** No relationship mapping between Rules and Episodes.
- **Context Noise:** Retrieves low-quality matches due to simple keyword overlap.

## 2. Refactoring Requirements

### 2.1 Implementing Vector Store
Replace `_extract_keywords` logic with **Embedding-Based Retrieval**.
- **Tech:** Use `ChromaDB` (or `LanceDB`) as the backend for `EpisodicMemoryStore` and `SemanticMemoryStore`.
- **Logic:** `query_vector = model.encode(text)`. `results = db.query(query_vector)`.

### 2.2 Implementing Knowledge Graph (Lightweight)
Introduce a `NetworkX` graph to track relationships.
- **Nodes:** Episodes, Rules, Code Files.
- **Edges:** `CREATED`, `MODIFIED`, `CAUSED_ERROR`, `FIXED`.
- **Query:** When retrieving an Episode, perform a 1-hop traversal to find related Code Files.

### 2.3 Context Ranking (Re-ranking)
Implement a **Cross-Encoder Re-ranker**.
- After retrieving top 20 results via Vectors, pass them through a Cross-Encoder to sort them by *actual relevance* to the query.
- Keep only the top 5 for the Context Window.

## 3. Implementation Steps
1.  **Install** `chromadb` and `sentence-transformers`.
2.  **Refactor** `store()` and `retrieve()` methods to be async (DB operations).
3.  **Migrate** existing JSON data to the Vector Store (Migration Script).

---
**Handover Status:** Ready. Code Agent must implement Vector Logic.
