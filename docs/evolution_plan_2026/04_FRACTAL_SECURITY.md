# GAAP Evolution: Zero-Trust Fractal Security (v2.0)

**Focus:** Moving from "Sandboxing" to "Capability-Based Execution".

## 1. The Core Problem: Over-Privileged Agents
A standard Docker sandbox often has full read-write access to the mounted volume.
**Target:** **The Token-Gate Security Model**.

## 2. Architecture: Capability-Based Sandboxing

Every "Fractal" (Sub-agent) is born with **Zero Permissions**.

### 2.1 The Permission Handshake
1.  **Request:** Fractal A needs to write to `gaap/core/logic.py`.
2.  **Grant:** Orchestrator issues a **Temporary File Token (TFT)**.
3.  **Access:** The Sandbox Mount only exposes that *specific file* for that *specific time*.

### 2.2 Micro-Kernel Execution (Wasm-First)
For non-IO logic (like calculations or data parsing), we will prefer **WebAssembly (Wasm)** over Docker.
- **Why:** Wasm has a smaller attack surface and instant startup.
- **Python-in-Wasm:** Use `Pyodide` or `Wasmtime` to run isolated Python logic.

## 3. The "Honey-Pot" Monitor
If an agent attempts to access a forbidden path (e.g., `/etc/passwd` or `~/.ssh`):
1.  **Detection:** Immediate halt of the Fractal.
2.  **Quarantine:** The entire task is moved to a **Security Audit Queue**.
3.  **Sanction:** The Fractal's Identity is destroyed and recreated.

## 4. Resource Hardening (v2.0)
- **Seccomp Profiles:** Restrict syscalls allowed by the Docker container.
- **Network Air-Gap:** By default, no network access. Explicit tokens required for `pip` or `search`.

## 5. Roadmap
1.  **Phase 1:** Implement the `PermissionManager` to handle file access tokens.
2.  **Phase 3:** Integrate `Wasmtime` for logic-only execution.
3.  **Phase 4:** Build the `Honey-Pot` detection layer.

