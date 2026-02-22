# TECHNICAL SPECIFICATION: Security Evolution (Cognitive Shield & DLP)

**Target:** `gaap/security/firewall.py`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Regex Fragility:** Simple string matching fails against encoded or semantic attacks.
- **Inbound-Only focus:** No protection against the agent accidentally leaking secrets in its response.
- **Memory Volatility:** Audit logs are lost on process exit.

## 2. Refactoring Requirements

### 2.1 Implementing Semantic Shield (L4 Upgrade)
Integrate an LLM-based guardrail (e.g., Llama-Guard or specialized small model).
- **Workflow:** `if score > threshold: run_semantic_check(input)`.
- **Logic:** The guardrail evaluates if the intent violates the **Project Constitution**.

### 2.2 Outbound DLP (Data Loss Prevention)
Add a mandatory `scan_output()` step in the Firewall.
- **Entropy Check:** Detect high-entropy strings (Potential API Keys).
- **Pattern Match:** Scan for Emails, Credit Cards, and Internal Paths.
- **Action:** Auto-redact `[REDACTED_SECRET]` before output.

### 2.3 Cryptographic Persistence
Update `AuditTrail` to save to disk using a **Forward-Secure Logging** pattern.
- **Action:** Every 10 entries, flush to `.gaap/audit/log.jsonl`.
- **Integrity:** Store the final Hash in a separate "State" file to detect tampering upon restart.

### 2.4 Capability-Locked IO
Integrate `CapabilityManager` with `Layer3_Execution`.
- **Requirement:** `ExecutorPool` must verify a `CapabilityToken` before calling any tool that touches the filesystem or network.
- **No Token = No Tool.**

## 3. Implementation Steps
1.  **Add** `pii-identifier` or similar lightweight library to dependencies.
2.  **Refactor** `AuditTrail` to support async disk writing.
3.  **Update** `PromptFirewall.scan` to include the `scan_output` phase.

---
**Handover Status:** Ready. Code Agent must prioritize Outbound DLP to prevent credential leaks.
