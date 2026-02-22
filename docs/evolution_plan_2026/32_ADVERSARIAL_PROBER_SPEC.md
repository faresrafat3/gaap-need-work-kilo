# TECHNICAL SPECIFICATION: Adversarial Web Prober (v1.0)

**Author:** Strategic Architect
**Target:** Code Agent
**Focus:** Executing safe, logical security probes based on hypotheses.

## 1. Overview
The `AdversarialProber` takes the hypotheses from the `WebLogicMapper` and attempts to prove them using minimal, non-destructive payloads.

## 2. Probing Strategies

### 2.1 Parameter Tampering (IDOR/Privilege Escalation)
- **Action:** Identify resource IDs in URLs or JSON bodies (e.g., `user_id=505`).
- **Experiment:** Attempt to access the same resource with `user_id=506` or `user_id=admin`.
- **Analysis:** If the response size or status code remains `200 OK` and contains different data, flag as **CRITICAL IDOR**.

### 2.2 Blind Logic Probing (Time-Based)
- **Action:** Test for Race Conditions or slow DB queries.
- **Experiment:** Send 10 simultaneous requests to an expensive endpoint (e.g., `/search`) and measure response drift.

### 2.3 Semantic Fuzzing
- **Action:** Instead of random chars, send "Meaningful Garbage".
- **Examples:**
    - If a field expects a "Country", send `../../etc/passwd`.
    - If a field expects "Price", send `-1`.
- **Analysis:** Use LLM to read the *error message*. An error like `"SQL Syntax near..."` is a huge win.

## 3. The "Stealth" Protocol (WAF Evasion)
1. **Header Rotation:** Rotate User-Agents and use standard browser headers.
2. **Contextual Pacing:** Space out requests to mimic human browsing behavior.
3. **Cookie Integrity:** Maintain consistent session cookies to avoid "Session Fixation" alerts.

## 4. Safety & The PoC (Proof of Concept)
When a probe succeeds:
1. **Immediate Stop:** Stop all active probes on that endpoint.
2. **Verification:** Repeat the success *once* to ensure it wasn't a fluke.
3. **PoC Generation:** Generate a standalone `curl` command or Python script that reproduces the bug.
4. **Impact Assessment:** Explain *why* this matters (e.g., "An attacker can download all user invoices").

## 5. Integration with Layer 3 (Execution)
The Prober acts as an "Executor" in the GAAP pipeline, returning `SecurityTaskResult` objects to the main engine.
