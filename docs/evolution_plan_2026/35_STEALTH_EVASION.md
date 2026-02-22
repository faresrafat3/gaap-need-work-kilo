# TECHNICAL SPECIFICATION: Stealth & WAF Evasion Protocol (v1.0)

**Author:** Strategic Architect
**Target:** Code Agent
**Focus:** Ensuring GAAP operations are not blocked by Web Application Firewalls.

## 1. Requirement: Adaptive Stealth
Most WAFs use rate-limiting and signature-based detection. We must behave like a human user.

## 2. Stealth Layers

### 2.1 The Browser Fingerprint
- **Action:** Use `Playwright-Stealth` to mask automated signatures.
- **Header Entropy:** Randomly select User-Agents, Viewport sizes, and Timezones from a "Legitimate Human" dataset.
- **TLS Fingerprinting:** Use `curl_cffi` to mimic Chrome's TLS handshake (JA3 fingerprint).

### 2.2 Traffic Pacing
- **Action:** Implement a **Poisson Distribution** for request delays. Humans don't click every 1.0 seconds; they wait 2s, then 5s, then 0.5s.
- **Session Lifecycle:** Always start with "Warm-up" requests (reading `robots.txt`, home page, static CSS) before touching the `/api/` endpoints.

### 2.3 Payload Mutation (The Alchemist)
If a payload like `SELECT * FROM users` is blocked:
- **Encoding:** Try `S%45LECT`, `UN%49ON`, or Double-URL encoding.
- **Splitting:** Send data in multiple smaller chunks if the API supports it.
- **Polyglots:** Use payloads that are valid in multiple languages/contexts simultaneously to confuse signature-based filters.

## 3. Implementation Tools
- **TLS:** `curl_cffi` (mandatory).
- **Automation:** `playwright` with `stealth` plugin.
- **Proxy:** Integrate with `Tor` or residential proxy rotation.

## 4. Axiom: The "Back-off" Rule
If GAAP receives more than 2 `403 Forbidden` or `429 Too Many Requests` in 60 seconds, it MUST **Hibernate** for 15 minutes and switch its proxy IP.
