# TECHNICAL SPECIFICATION: Context Evolution (Semantic Graph & Skeleton Indexing)

**Target:** `gaap/context/smart_chunking.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Keyword Retrieval:** Fails to find code based on intent/meaning.
- **Local-Only Dependencies:** Ignores connections between different files.
- **Context Fragmentation:** Separated methods lose their parent class attributes.

## 2. Refactoring Requirements

### 2.1 Implementing Semantic Vector Indexing
Integrate with `gaap/memory/vector_store.py`.
- **Action:** For every `CodeChunk`, generate an Embedding.
- **Metadata:** Store `file_path`, `chunk_type`, and `signature` as metadata in the Vector DB.
- **Search:** Replace string matching with Vector Similarity search.

### 2.2 Cross-File Call Graph (v2.0)
Build a global map of the project structure.
- **Action:** Use `Tree-Sitter` to extract all function calls.
- **Logic:** Create a `NetworkX` graph where nodes are functions and edges are calls.
- **Retrieval:** When a function is retrieved, also fetch its **Upstream Callers** and **Downstream Dependencies** (1-hop).

### 2.3 The "Skeleton" Inbound Context
When serving a chunk to an LLM, use the **Skeleton Pattern**.
- **Rule:** Never send a standalone method.
- **Structure:** 
    ```python
    class {ClassName}:
        # ... other methods hidden ...
        {TargetMethodContent}
    ```
- **Benefit:** LLM understands the `self` context and class variables perfectly.

### 2.4 Multi-Language Support (LSP Integration)
Move beyond custom Regex for JS/TS.
- **Requirement:** Support Python, Javascript, Typescript, and Rust via **Language Server Protocol (LSP)** or unified Tree-Sitter grammars.

## 3. Implementation Steps
1.  **Refactor** `SmartChunker` to be a generator that yields chunks directly to the Vector Indexer.
2.  **Implement** the `GlobalMap` builder script.
3.  **Update** `ContextOrchestrator` to use the Skeleton Pattern when building prompts.

---
**Handover Status:** Ready. Code Agent must implement the Skeleton Context wrapper to improve execution quality.
