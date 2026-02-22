# TECHNICAL SPECIFICATION: Web Logic Mapper & Recon Agent (v1.0)

**Author:** Strategic Architect
**Target:** Code Agent
**Focus:** Understanding Web Application Flow and Logic.

## 1. Overview
The `WebLogicMapper` is a specialized GAAP sub-agent that converts a raw URL into a **Functional Logic Graph**. It doesn't just find endpoints; it understands their purpose (e.g., "This endpoint handles sensitive user PII").

## 2. Component Specifications

### 2.1 The "JS Archaeologist" (analyzer.py)
*   **Input:** URL.
*   **Action:** 
    1. Fetches all Javascript bundles.
    2. Uses **Tree-Sitter** (File 28) to extract hidden API routes, hardcoded keys, and logic flows.
    3. De-obfuscates and prettifies code to find "Dead Logic" (unused code that might still have active endpoints).

### 2.2 The "Semantic Traffic Proxy" (proxy.py)
*   **Function:** Sits between GAAP and the target.
*   **Action:** 
    1. Captures requests/responses.
    2. Tags endpoints semantically: `[AUTH]`, `[PAYMENT]`, `[PROFILE]`, `[UPLOAD]`.
    3. Identifies **State Dependencies**: "To access `/get_data`, I must first call `/login` and then `/session_init`."

### 2.3 The "Logic Flaw Hypothesis" Generator
Based on the map, GAAP generates **Hypotheses**:
- *Hypothesis:* "The password reset token might be predictable based on the timestamp."
- *Hypothesis:* "The `/api/v1/delete_user` endpoint might not check if the user is deleting themselves or someone else (IDOR)."

## 3. Tooling Integration (MCP Servers)
GAAP will connect to existing security tools via MCP (File 02):
- **Burp Suite MCP:** To control the professional interceptor.
- **HTTPX MCP:** For fast probing.
- **Waymore MCP:** To find historical endpoints in Wayback Machine.

## 4. Axiomatic Constraints
1. **Rate Limiting:** Maximum 2 requests per second to avoid triggering WAF/DDoS protection.
2. **Scope Guard:** Every request is checked against the `scope.json` provided by the bounty program.
3. **Passive First:** Prioritize reading JS and public docs before active probing.

## 5. Output for Layer 2 (Tactical)
A JSON `AttackSurfaceMap` containing:
- High-value targets (Endpoints with complex logic).
- Suspected vulnerabilities (The Hypotheses).
- Authentication state required for each.
