# TECHNICAL SPECIFICATION: Storage Evolution (Atomic & Structured Persistence)

**Target:** `gaap/storage/json_store.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Non-Atomic Writes:** Risks total data loss if process crashes during save.
- **Full-File Rewrites:** Inefficient $O(N)$ operations for simple appends.
- **Unstructured Data:** No validation of stored schemas, leading to potential runtime crashes.

## 2. Refactoring Requirements

### 2.1 Implementing Atomic Saves
Replace direct `open(path, 'w')` with an atomic pattern.
- **Logic:**
    1. Write data to `{name}.json.tmp`.
    2. Ensure data is flushed to disk (`f.flush()`, `os.fsync()`).
    3. Rename `{name}.json.tmp` to `{name}.json`.
- **Benefit:** Guarantees that the file is either the old version or the complete new version, never corrupted.

### 2.2 Migration to SQLite (For Stats & History)
Replace `JSONStore` with a hybrid model.
- **Config:** Keep in JSON (small, easy to edit).
- **Stats/History:** Move to `gaap.db` (SQLite).
- **Benefit:** Instant lookups by ID, efficient appending, and ACID compliance.

### 2.3 Pydantic Model Enforcement
Integrate with `gaap/core/types.py`.
- **Requirement:** Every `append` or `save` must validate against a Pydantic model.
- **Action:** If validation fails, log a **Critical Security Violation** (Data Integrity Breach).

### 2.4 Automatic Backups
Implement a rotation policy for critical files (`config.json`, `stats.json`).
- **Action:** Before a major update, copy the file to `.gaap/backups/{timestamp}_{name}.json`.

## 3. Implementation Steps
1.  **Add** `pydantic` and `sqlite3` (built-in) to the storage workflow.
2.  **Refactor** `JSONStore` to include an `atomic_write` helper.
3.  **Create** a `MigrationUtility` to move history from JSON to SQLite.

---
**Handover Status:** Ready. Code Agent must implement 'Atomic Writes' immediately to prevent data corruption.
