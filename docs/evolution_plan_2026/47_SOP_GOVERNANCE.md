# TECHNICAL SPECIFICATION: SOP Governance & Role Enforcement (v1.0)

**Focus:** Moving from "Agent Chat" to "Process-Driven Intelligence" (Inspired by MetaGPT).

## 1. Requirement: Reliability through Rigor
Intelligence is not enough; process is key. Every role in GAAP must follow a **Standard Operating Procedure (SOP)**.

## 2. The Role Schema (`.gaap/roles/`)

Every agent identity must load a `.yaml` file defining its SOP.

### Example: `security_critic.yaml`
```yaml
role: Security Analyst
mission: Identify vulnerabilities in proposed code.
sop_steps:
  1. Scan for hardcoded credentials.
  2. Check for unsafe library imports (pickle, subprocess shell=True).
  3. Analyze data flow for SQLi/XSS.
  4. Generate a 'Security Risk Table'.
mandatory_artifacts:
  - security_report.md
  - cvss_score
```

## 3. The SOP Gatekeeper
`Layer3_Execution` will not accept a task as "done" unless the **Mandatory Artifacts** defined in the role's SOP are present and valid.

## 4. Implementation Steps
1.  **Create** `gaap/core/governance.py` to handle role definitions.
2.  **Update** `GAAPEngine` to inject the relevant SOP into the beginning of every thought cycle.
3.  **Implement** artifact validation in the `QualityPipeline`.

## 5. Axiom: Standardized Output
If an SOP step is skipped, the system must trigger an internal `Reflexion` (File 26) asking the model why it deviated from the protocol.
