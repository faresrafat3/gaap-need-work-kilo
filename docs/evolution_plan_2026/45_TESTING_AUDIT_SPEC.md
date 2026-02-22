# TECHNICAL SPECIFICATION: Testing Evolution (Cognitive & E2E)

**Target:** `tests/`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Mock Reliance:** Tests pass even if real APIs break.
- **Semantic Blindness:** Asserts types, not intelligence.
- **Fragmentation:** No full-cycle testing of the OODA loop.

## 2. Refactoring Requirements

### 2.1 Implementing VCR (Cassette Replay)
Integrate `pytest-vcr` or `vcrpy`.
- **Action:** Record real interaction with Gemini/Groq once.
- **Benefit:** Deterministic tests that reflect real-world JSON schemas.

### 2.2 LLM-as-a-Judge Fixture
Create a `semantic_assert` helper.
- **Usage:** `assert_semantically_similar(result, expected, threshold=0.9)`
- **Tech:** Use Embeddings (Cosine Similarity) or a cheap LLM to grade the answer.

### 2.3 The "Gauntlet" (E2E Scenario Runner)
Create a new test category `tests/gauntlet/`.
- **Scenario:** "Create a Flask App with Auth".
- **Check:** Does the final folder structure contain `app.py`, `requirements.txt`, and `auth.py`? Do `pytest` runs on the generated code pass?

### 2.4 Chaos Testing (The Monkey)
Introduce a fixture that randomly fails network calls or corrupts memory.
- **Goal:** Verify `SelfHealingSystem` actually heals.

## 3. Implementation Steps
1.  **Install** `pytest-vcr`, `vcrpy`.
2.  **Create** `tests/gauntlet/test_e2e_code_gen.py`.
3.  **Refactor** `conftest.py` to support VCR cassettes.

---
**Handover Status:** Ready. Code Agent must implement 'The Gauntlet' to verify system stability.
