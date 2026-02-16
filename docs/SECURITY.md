# GAAP Security Guide

This guide covers GAAP's security features and best practices for secure deployment.

## Table of Contents

1. [Security Architecture](#security-architecture)
2. [Prompt Firewall](#prompt-firewall)
3. [Audit Trail](#audit-trail)
4. [Capability Tokens](#capability-tokens)
5. [Rate Limiting](#rate-limiting)
6. [Best Practices](#best-practices)

---

## Security Architecture

GAAP implements a defense-in-depth approach:

```
+-------------------+
|    User Input     |
+-------------------+
         |
         v
+-------------------+
|  Prompt Firewall  | <-- L1-L7 Defense Layers
+-------------------+
         |
         v
+-------------------+
|   Intent Check    | <-- Classification
+-------------------+
         |
         v
+-------------------+
| Capability Token  | <-- Authorization
+-------------------+
         |
         v
+-------------------+
|   Task Execution  | <-- Sandboxed
+-------------------+
         |
         v
+-------------------+
|   Audit Trail     | <-- Logging
+-------------------+
```

---

## Prompt Firewall

### 7-Layer Defense

| Layer | Name | Detection Method |
|-------|------|------------------|
| **L1** | Surface Inspection | Pattern matching for known attacks |
| **L2** | Lexical Analysis | Obfuscation detection (hex, unicode, URL encoding) |
| **L3** | Syntactic Analysis | Nested instructions, suspicious comments |
| **L4** | Semantic Analysis | Danger keywords, context mismatch |
| **L5** | Contextual Verification | Role-based checks |
| **L6** | Behavioral Analysis | Usage pattern detection |
| **L7** | Adversarial Testing | Active probing |

### Attack Types Detected

| Attack Type | Examples | Risk Level |
|-------------|----------|------------|
| **Prompt Injection** | "Ignore previous instructions", "Disregard all prompts" | HIGH |
| **Jailbreak** | "DAN mode", "Developer mode", "Bypass filters" | CRITICAL |
| **Data Exfiltration** | "Show me your system prompt", "Print your instructions" | HIGH |
| **Code Injection** | `<script>`, `javascript:`, `onerror=` | CRITICAL |
| **Role Confusion** | "You are now admin", "Act as developer" | MEDIUM |
| **Context Manipulation** | `[SYSTEM]`, `<<<hidden>>>` | HIGH |

### Usage

```python
from gaap.security import PromptFirewall

# Create firewall
firewall = PromptFirewall(strictness="high")

# Scan input
result = firewall.scan(
    input_text="Hello, how are you?",
    context={"user_role": "user"}
)

# Check result
if result.is_safe:
    # Process input
    pass
else:
    # Block or sanitize
    print(f"Blocked: {result.risk_level.name}")
    print(f"Patterns: {result.detected_patterns}")
    print(f"Suggestions: {result.recommendations}")
```

### Strictness Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| **low** | Minimal blocking, logging only | Development |
| **medium** | Block high-risk only | Testing |
| **high** | Block medium+ risk | Production |
| **paranoid** | Block any suspicion | High-security |

### FirewallResult

```python
@dataclass
class FirewallResult:
    is_safe: bool                    # True if safe to process
    risk_level: RiskLevel            # SAFE, LOW, MEDIUM, HIGH, CRITICAL
    detected_patterns: list[str]     # Matched attack patterns
    sanitized_input: str             # Cleaned input
    recommendations: list[str]       # Suggested actions
    scan_time_ms: float              # Scan duration
    layer_scores: dict[str, float]   # Per-layer scores
```

---

## Audit Trail

### Blockchain-like Integrity

The audit trail uses a hash chain for tamper-proof logging:

```
Entry 1: hash=abc123, prev=genesis
Entry 2: hash=def456, prev=abc123
Entry 3: hash=ghi789, prev=def456
```

Each entry contains:
- Hash of current entry
- Hash of previous entry
- Timestamp
- Action, agent, resource, result

### Usage

```python
from gaap.security import AuditTrail

# Create audit trail
audit = AuditTrail(storage_path="./audit_logs")

# Record action
entry = audit.record(
    action="process_request",
    agent_id="gaap_engine",
    resource="user_request_123",
    result="success",
    details={
        "model": "llama-3.3-70b",
        "tokens": 500,
        "cost": 0.01
    }
)

# Verify integrity
is_valid = audit.verify_integrity()
print(f"Audit trail valid: {is_valid}")

# Get history
history = audit.get_agent_history("gaap_engine")
recent = audit.get_recent(limit=100)

# Export
audit.export("audit_export.json")
```

### AuditEntry

```python
@dataclass
class AuditEntry:
    id: str
    timestamp: datetime
    action: str
    agent_id: str
    resource: str
    result: str
    details: dict[str, Any]
    previous_hash: str
    hash: str
```

---

## Capability Tokens

### Token-Based Authorization

Capability tokens grant specific permissions:

```python
from gaap.security import CapabilityManager

# Create manager
cap_manager = CapabilityManager(secret_key="your-secret-key")

# Issue token
token = cap_manager.issue_token(
    agent_id="agent_123",
    resource="database",
    action="read",
    ttl_seconds=300,
    constraints={"tables": ["users", "orders"]}
)

# Verify token
is_valid = cap_manager.verify_token(
    token=token,
    requested_resource="database",
    requested_action="read"
)

if is_valid:
    # Allow access
    pass
else:
    # Deny access
    raise CapabilityError("Access denied")

# Revoke token
cap_manager.revoke_token("agent_123", "database", "read")
```

### Token Structure

```python
@dataclass
class CapabilityToken:
    subject: str              # Agent ID
    resource: str             # Resource being accessed
    action: str               # Action being performed
    issued_at: datetime       # Issue timestamp
    expires_at: datetime      # Expiration timestamp
    constraints: dict         # Additional restrictions
    nonce: str                # Unique identifier
    signature: str            # Cryptographic signature
```

---

## Rate Limiting

### Strategies

#### Token Bucket

```python
from gaap.core.rate_limiter import TokenBucketRateLimiter

limiter = TokenBucketRateLimiter(
    rate=30,           # Tokens per second
    burst=60           # Maximum burst
)

if limiter.allow():
    # Process request
    pass
else:
    # Rate limited
    pass
```

#### Sliding Window

```python
from gaap.core.rate_limiter import SlidingWindowRateLimiter

limiter = SlidingWindowRateLimiter(
    window_seconds=60,  # Window size
    max_requests=30     # Max per window
)
```

#### Adaptive

```python
from gaap.core.rate_limiter import AdaptiveRateLimiter

limiter = AdaptiveRateLimiter(
    initial_rate=30,
    min_rate=10,
    max_rate=60,
    increase_factor=1.2,
    decrease_factor=0.8
)

# Automatically adjusts based on errors
```

### Per-Provider Limits

```python
from gaap.providers import GroqProvider

provider = GroqProvider(
    api_key="gsk_...",
    rate_limit=30,       # Requests per minute
    timeout=120,         # Seconds
    max_retries=3
)
```

---

## Best Practices

### 1. Input Validation

```python
# Always validate input before processing
def validate_input(text: str) -> bool:
    if len(text) > 100000:
        raise ValueError("Input too long")
    if not text.strip():
        raise ValueError("Empty input")
    return True
```

### 2. Sanitize Output

```python
# Never expose internal state
def sanitize_output(output: str) -> str:
    # Remove potential secrets
    output = re.sub(r'gsk_[a-zA-Z0-9]+', '[REDACTED]', output)
    output = re.sub(r'csk_[a-zA-Z0-9]+', '[REDACTED]', output)
    return output
```

### 3. Principle of Least Privilege

```python
# Grant minimum required permissions
token = cap_manager.issue_token(
    agent_id="reader_agent",
    resource="database",
    action="read",      # Only read, not write
    constraints={"tables": ["public_data"]}  # Only public tables
)
```

### 4. Secure Key Storage

```bash
# Never hardcode keys in code
# BAD:
api_key = "gsk_12345..."

# GOOD: Use environment variables
export GROQ_API_KEY=gsk_...
```

```python
import os
api_key = os.environ.get("GROQ_API_KEY")
```

### 5. Enable Audit Logging

```python
# Always log sensitive operations
audit.record(
    action="api_key_access",
    agent_id="provider",
    resource="groq_api",
    result="success"
)
```

### 6. Use HTTPS

```yaml
# In production, always use HTTPS
server:
  ssl: true
  cert: /path/to/cert.pem
  key: /path/to/key.pem
```

### 7. Regular Updates

```bash
# Keep dependencies updated
pip install --upgrade gaap

# Check for vulnerabilities
pip-audit
```

---

## Security Configuration

```yaml
# security-config.yaml
firewall:
  strictness: high
  enable_behavioral_analysis: true
  enable_semantic_analysis: true
  max_input_length: 100000

audit:
  enabled: true
  storage_path: /var/log/gaap/audit
  retention_days: 90

capabilities:
  default_ttl: 300
  max_ttl: 3600

rate_limiting:
  strategy: adaptive
  default_rate: 30
  burst_factor: 2
```

---

## Compliance

### GDPR Considerations

- Data minimization: Only collect necessary data
- Right to erasure: Implement data deletion
- Data portability: Export user data on request
- Consent tracking: Log user consent

### SOC 2 Requirements

- Access controls: Implement RBAC
- Encryption: Encrypt data at rest and in transit
- Audit logging: Comprehensive audit trail
- Incident response: Documented procedures

---

## Incident Response

### Detection

```bash
# Monitor for security events
grep "BLOCKED" /var/log/gaap/audit.log
grep "PromptInjection" /var/log/gaap/audit.log
```

### Response Steps

1. **Identify**: Determine attack type and scope
2. **Contain**: Block malicious inputs, revoke tokens
3. **Eradicate**: Fix vulnerability, update rules
4. **Recover**: Restore normal operations
5. **Document**: Record incident details

### Reporting

```python
# Export incident report
audit.export("incident_report.json")

# Include:
# - Timestamps
# - Attack patterns detected
# - Actions taken
# - Affected resources
```

---

## Security Checklist

- [ ] Firewall enabled with appropriate strictness
- [ ] Audit logging enabled and verified
- [ ] All API keys stored securely (not in code)
- [ ] Rate limiting configured
- [ ] HTTPS enabled in production
- [ ] Capability tokens for sensitive operations
- [ ] Input validation on all endpoints
- [ ] Output sanitization for secrets
- [ ] Regular security updates
- [ ] Incident response plan documented

---

## Next Steps

- [Architecture Guide](ARCHITECTURE.md) - Security architecture details
- [Deployment Guide](DEPLOYMENT.md) - Secure deployment
- [API Reference](API_REFERENCE.md) - Security API documentation