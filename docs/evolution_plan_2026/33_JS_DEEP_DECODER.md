# TECHNICAL SPECIFICATION: Advanced JS Deep Decoder (v1.0)

**Author:** Strategic Architect
**Target:** Code Agent
**Focus:** Reverse-engineering production Javascript bundles to extract logic and endpoints.

## 1. Requirement: Beyond Static Search
Most production JS is minified and obfuscated. We need a structural approach to understand the **Functional Surface**.

## 2. Logic Flow: The Decoder Pipeline

### 2.1 Chunk Aggregation
- **Action:** Identify and fetch all `.js` and `.js.map` files referenced in the HTML and Manifest files.
- **Goal:** Reconstruct the original directory structure if source maps are exposed (a common vulnerability).

### 2.2 AST-Based Extraction
Use `tree-sitter-javascript` to perform:
- **Identifier Renaming:** Use LLM to "Guess" the original names of minified variables based on their usage (e.g., `function a(b, c)` where `b` is used in `auth` header -> `b` is `token`).
- **Route Discovery:** Search the AST for string patterns matching `/api/v[0-9]/` or URL object constructions.
- **Schema Mapping:** Identify JSON structure expected by `POST` requests by analyzing the object keys passed to network functions.

### 2.3 Hardcoded Secret Detection
- Scan AST for high-entropy strings, API keys (`AIza...`, `sk_live...`), and environment variable injections.

## 3. Implementation Tools (For Code Agent)
- **Primary:** `esprima` or `acorn` for Javascript parsing.
- **Secondary:** `shittier` (the reverse-prettier tool) to normalize code.
- **GAAP Link:** Store discovered endpoints in `WebLogicMapper` (File 31).

## 4. Axiom: Performance
De-bundling must happen in a separate thread to avoid blocking the main OODA loop. Max bundle size for analysis: 10MB.
