# TECHNICAL SPECIFICATION: The Library Eater (Repository Ingestion Engine)

**Focus:** Transforming GAAP into an agent that learns new libraries instantly by reading their source code.

## 1. The Core Problem: Outdated Training Data
LLMs hallucinate non-existent parameters for libraries that updated last week.
**Target:** **Grounding via Codebase Indexing.**

## 2. Architecture: The Neural Code Cartographer

We will build a module `gaap/knowledge/ingestion.py` that processes local or remote repositories.

### 2.1 The "Tree-Sitter" Parser
Unlike Regex, **Tree-Sitter** builds a concrete syntax tree. It understands that `def login():` is a function definition, regardless of formatting.
- **Action:** Parse `.py` files to identify:
    - Classes & Inheritance.
    - Public Methods & Signatures (Arguments, Types).
    - Docstrings.

### 2.2 The "Usage Miner"
The agent scans the `tests/` and `examples/` folders of the target library.
- **Assumption:** Tests contain the *truth* of how code is supposed to be run.
- **Output:** A collection of `(Code Snippet, Intent)` pairs stored in `VectorMemory`.

### 2.3 The "Cheat Sheet" Generator
After ingestion, the system generates a `reference_card.json` for the library.
- **Content:**
    - "Top 10 Most Used Functions"
    - "Common Patterns"
    - "Breaking Changes" (detected by comparing vs internal knowledge).

## 3. Workflow: "Learn this Repo"
1.  **User:** `gaap learn https://github.com/pydantic/pydantic`
2.  **System:**
    - Clones repo to temp.
    - Walks AST.
    - Extracts `BaseModel` patterns.
    - Indexes `v2` changes.
    - Saves to `gaap/.kilocode/memory/libraries/pydantic.json`.
3.  **Future Task:** User asks "Create a Pydantic model".
    - Agent loads `pydantic.json` into context.
    - Writes perfect v2 code.

## 4. Implementation Plan
1.  **Install** `tree-sitter` and `tree-sitter-languages`.
2.  **Build** the `RepoWalker` class.
3.  **Create** the `SyntheticDocumentation` generator.
