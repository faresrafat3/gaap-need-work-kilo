# GAAP API Endpoints

Complete reference for all GAAP API endpoints.

---

## Table of Contents

1. [Health & System](#health--system)
2. [Chat](#chat)
3. [Sessions](#sessions)
4. [Providers](#providers)
5. [Memory](#memory)
6. [Research](#research)
7. [Budget](#budget)
8. [Configuration](#configuration)
9. [Healing](#healing)
10. [Swarm](#swarm)
11. [Knowledge](#knowledge)
12. [Maintenance](#maintenance)
9. [Healing](#healing)
10. [Swarm](#swarm)
11. [Knowledge](#knowledge)
12. [Maintenance](#maintenance)

---

## Health & System

### GET /api/health

Health check endpoint for monitoring system status.

**Method:** `GET`  
**Path:** `/api/health`  
**Auth:** None

#### Response

```json
{
  "status": "healthy",
  "timestamp": 1704067200.123,
  "uptime_seconds": 3600.45,
  "version": "1.0.0",
  "system": {
    "memory": {
      "total_mb": 16384,
      "available_mb": 8192,
      "percent_used": 50.0
    },
    "cpu_percent": 25.5,
    "disk": {
      "total_gb": 512,
      "free_gb": 256,
      "percent_used": 50.0
    }
  },
  "websocket": {
    "total_connections": 5,
    "channels": {
      "events": 3,
      "ooda": 1,
      "steering": 1
    }
  },
  "database": {
    "status": "connected",
    "tables": 12
  },
  "providers": {
    "kimi": "available",
    "deepseek": "available",
    "glm": "not_initialized"
  }
}
```

#### Status Values

| Status | Meaning |
|--------|---------|
| `healthy` | All systems operational |
| `degraded` | Some components experiencing issues |
| `critical` | Critical issues detected |

**Error Codes:**
- `500` - Internal server error

---

### GET /api/system/metrics

Get system metrics and performance data.

**Method:** `GET`  
**Path:** `/api/system/metrics`  
**Auth:** Required

#### Response

```json
{
  "requests_total": 15234,
  "requests_per_minute": 45,
  "average_latency_ms": 125.5,
  "error_rate": 0.02,
  "active_sessions": 12,
  "provider_usage": {
    "kimi": 8500,
    "deepseek": 4234,
    "glm": 2500
  }
}
```

---

## Chat

### POST /api/chat

Send a chat message to the AI assistant.

**Method:** `POST`  
**Path:** `/api/chat`  
**Auth:** Optional (depends on config)  
**Rate Limit:** 30 req/min

#### Request Schema

```json
{
  "message": "string (required, 1-50000 chars)",
  "context": "object (optional, max 100KB)",
  "provider": "string (optional, default: 'kimi')"
}
```

#### Request Example

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Explain quantum computing in simple terms",
    "context": {
      "session_id": "sess-123",
      "user_preference": "detailed"
    },
    "provider": "kimi"
  }'
```

#### Response Schema

```json
{
  "response": "string"
}
```

#### Response Example

```json
{
  "response": "Quantum computing is a type of computation that harnesses the principles of quantum mechanics..."
}
```

#### Error Responses

```json
// 400 Bad Request
{
  "error": "Invalid request",
  "detail": "Message must be between 1 and 50000 characters"
}

// 429 Rate Limit
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Please try again later."
}

// 503 Service Unavailable
{
  "error": "All providers failed",
  "detail": "Timeout"
}
```

**Error Codes:**
- `400` - Invalid request body
- `429` - Rate limit exceeded
- `503` - All providers unavailable

---

### GET /api/chat/providers

Get available chat providers and their status.

**Method:** `GET`  
**Path:** `/api/chat/providers`  
**Auth:** None

#### Response

```json
[
  {
    "name": "kimi",
    "model": "kimi-k2.5-thinking",
    "status": "available",
    "is_default": true,
    "cache_age_seconds": 3600
  },
  {
    "name": "deepseek",
    "model": "deepseek-chat",
    "status": "not_initialized",
    "is_default": false,
    "cache_age_seconds": null
  }
]
```

---

### GET /api/chat/cache/stats

Get provider cache statistics.

**Method:** `GET`  
**Path:** `/api/chat/cache/stats`  
**Auth:** Required (admin)

#### Response

```json
{
  "total_cached": 2,
  "cache_ttl_seconds": 900,
  "providers": {
    "kimi": {
      "created_at": 1704063600,
      "last_accessed": 1704067200,
      "access_count": 150,
      "age_seconds": 3600,
      "idle_seconds": 0
    }
  }
}
```

---

### POST /api/chat/cache/clear

Clear the provider cache (admin only).

**Method:** `POST`  
**Path:** `/api/chat/cache/clear`  
**Auth:** Required (admin)

#### Response

```json
{
  "status": "cleared",
  "entries_removed": "2"
}
```

---

## Sessions

### GET /api/sessions

List all sessions with optional filtering.

**Method:** `GET`  
**Path:** `/api/sessions`  
**Auth:** Required

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status (pending, running, paused, completed, failed, cancelled) |
| `priority` | string | Filter by priority (low, normal, high, critical) |
| `limit` | integer | Maximum results (default: 50, max: 1000) |
| `offset` | integer | Pagination offset (default: 0) |

#### Response

```json
{
  "sessions": [
    {
      "id": "abc123def456",
      "name": "Code Review Session",
      "description": "Reviewing authentication module",
      "status": "running",
      "priority": "high",
      "tags": ["code-review", "security"],
      "config": {},
      "metadata": {},
      "created_at": "2024-01-01T10:00:00",
      "updated_at": "2024-01-01T10:30:00",
      "started_at": "2024-01-01T10:05:00",
      "completed_at": null,
      "progress": 45.5,
      "tasks_total": 10,
      "tasks_completed": 4,
      "tasks_failed": 0,
      "cost_usd": 0.25,
      "tokens_used": 15000
    }
  ],
  "total": 42
}
```

---

### POST /api/sessions

Create a new session.

**Method:** `POST`  
**Path:** `/api/sessions`  
**Auth:** Required

#### Request Schema

```json
{
  "name": "string (required, 1-200 chars)",
  "description": "string (optional)",
  "priority": "string (optional, default: 'normal')",
  "tags": "array of strings (optional)",
  "config": "object (optional)",
  "metadata": "object (optional)"
}
```

#### Request Example

```bash
curl -X POST http://localhost:8000/api/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "name": "API Documentation Task",
    "description": "Create comprehensive API docs",
    "priority": "high",
    "tags": ["documentation", "api"],
    "config": {
      "auto_save": true
    }
  }'
```

#### Response

```json
{
  "id": "sess-abc123",
  "name": "API Documentation Task",
  "description": "Create comprehensive API docs",
  "status": "pending",
  "priority": "high",
  "tags": ["documentation", "api"],
  "config": {
    "auto_save": true
  },
  "metadata": {},
  "created_at": "2024-01-01T10:00:00",
  "progress": 0.0
}
```

**Status Codes:**
- `201` - Created
- `400` - Invalid request

---

### GET /api/sessions/{id}

Get session details.

**Method:** `GET`  
**Path:** `/api/sessions/{id}`  
**Auth:** Required

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | string | Session ID |

#### Response

Same as session object in list response.

**Error Codes:**
- `404` - Session not found

---

### PUT /api/sessions/{id}

Update session.

**Method:** `PUT`  
**Path:** `/api/sessions/{id}`  
**Auth:** Required

#### Request Schema

All fields are optional:

```json
{
  "name": "string",
  "description": "string",
  "priority": "string",
  "tags": ["string"],
  "config": {},
  "metadata": {}
}
```

#### Response

Updated session object.

**Error Codes:**
- `404` - Session not found

---

### DELETE /api/sessions/{id}

Delete a session.

**Method:** `DELETE`  
**Path:** `/api/sessions/{id}`  
**Auth:** Required

#### Response

```json
{
  "success": true,
  "message": "Session abc123 deleted"
}
```

**Error Codes:**
- `404` - Session not found

---

### POST /api/sessions/{id}/pause

Pause a running session.

**Method:** `POST`  
**Path:** `/api/sessions/{id}/pause`  
**Auth:** Required

#### Response

Updated session with `status: "paused"`.

**Error Codes:**
- `400` - Can only pause running sessions
- `404` - Session not found

---

### POST /api/sessions/{id}/resume

Resume a paused session.

**Method:** `POST`  
**Path:** `/api/sessions/{id}/resume`  
**Auth:** Required

#### Response

Updated session with `status: "running"`.

**Error Codes:**
- `400` - Can only resume paused sessions
- `404` - Session not found

---

### POST /api/sessions/{id}/cancel

Cancel a session.

**Method:** `POST`  
**Path:** `/api/sessions/{id}/cancel`  
**Auth:** Required

#### Response

Updated session with `status: "cancelled"`.

**Error Codes:**
- `400` - Cannot cancel completed/cancelled sessions
- `404` - Session not found

---

### POST /api/sessions/{id}/export

Export session data.

**Method:** `POST`  
**Path:** `/api/sessions/{id}/export`  
**Auth:** Required

#### Response

```json
{
  "session": { /* session object */ },
  "tasks": [ /* task objects */ ],
  "logs": [ /* log entries */ ],
  "metrics": {
    "total_cost": 0.25,
    "total_tokens": 15000,
    "tasks_total": 10,
    "tasks_completed": 4,
    "tasks_failed": 0,
    "success_rate": 0.4
  }
}
```

---

## Providers

### GET /api/providers

List all registered LLM providers.

**Method:** `GET`  
**Path:** `/api/providers`  
**Auth:** Required

#### Response

```json
[
  {
    "name": "kimi",
    "type": "CHAT",
    "enabled": true,
    "priority": 1,
    "models": ["kimi-k2.5-thinking", "kimi-k2.5"],
    "health": "healthy",
    "stats": {
      "total_requests": 1000,
      "successful_requests": 980,
      "failed_requests": 20,
      "success_rate": 0.98,
      "average_latency_ms": 150
    }
  }
]
```

---

### POST /api/providers

Add a new provider.

**Method:** `POST`  
**Path:** `/api/providers`  
**Auth:** Required (admin)

#### Request Schema

```json
{
  "name": "string (required)",
  "provider_type": "string (default: 'chat')",
  "api_key": "string (optional)",
  "base_url": "string (optional)",
  "priority": "integer (default: 1)",
  "enabled": "boolean (default: true)",
  "models": ["string"],
  "default_model": "string",
  "max_tokens": "integer (default: 4096)",
  "temperature": "float (default: 0.7)",
  "metadata": "object"
}
```

#### Response

Provider status object.

---

### GET /api/providers/{name}

Get provider details.

**Method:** `GET`  
**Path:** `/api/providers/{name}`  
**Auth:** Required

#### Response

Provider status object.

**Error Codes:**
- `404` - Provider not found

---

### PUT /api/providers/{name}

Update provider configuration.

**Method:** `PUT`  
**Path:** `/api/providers/{name}`  
**Auth:** Required (admin)

#### Request Schema

Same as POST with all fields optional.

---

### DELETE /api/providers/{name}

Remove a provider.

**Method:** `DELETE`  
**Path:** `/api/providers/{name}`  
**Auth:** Required (admin)

#### Response

```json
{
  "success": true,
  "message": "Provider 'openai' removed"
}
```

---

### POST /api/providers/{name}/test

Test provider connection.

**Method:** `POST`  
**Path:** `/api/providers/{name}/test`  
**Auth:** Required

#### Response

```json
{
  "success": true,
  "latency_ms": 145.5,
  "error": null,
  "model_available": true
}
```

---

### POST /api/providers/{name}/enable

Enable a provider.

**Method:** `POST`  
**Path:** `/api/providers/{name}/enable`  
**Auth:** Required (admin)

---

### POST /api/providers/{name}/disable

Disable a provider.

**Method:** `POST`  
**Path:** `/api/providers/{name}/disable`  
**Auth:** Required (admin)

---

## Memory

### GET /api/memory/stats

Get memory system statistics.

**Method:** `GET`  
**Path:** `/api/memory/stats`  
**Auth:** Required

#### Response

```json
{
  "working_memory": {
    "items": 45,
    "capacity": 100,
    "utilization": 0.45
  },
  "episodic_memory": {
    "episodes": 1250,
    "size_mb": 45.5
  },
  "semantic_memory": {
    "concepts": 500,
    "relations": 2500
  },
  "procedural_memory": {
    "procedures": 150
  }
}
```

---

### POST /api/memory/search

Search memory by query.

**Method:** `POST`  
**Path:** `/api/memory/search`  
**Auth:** Required

#### Request

```json
{
  "query": "authentication patterns",
  "memory_type": "semantic",
  "limit": 10
}
```

#### Response

```json
{
  "results": [
    {
      "id": "mem-123",
      "content": "JWT authentication pattern...",
      "relevance": 0.95,
      "memory_type": "semantic"
    }
  ]
}
```

---

## Research

### POST /api/research

Start a research task.

**Method:** `POST`  
**Path:** `/api/research`  
**Auth:** Required

#### Request

```json
{
  "query": "Latest developments in quantum computing",
  "depth": "comprehensive",
  "sources": ["arxiv", "web"],
  "max_results": 50
}
```

#### Response

```json
{
  "research_id": "res-abc123",
  "status": "started",
  "estimated_duration": "5m"
}
```

---

### GET /api/research/{id}

Get research results.

**Method:** `GET`  
**Path:** `/api/research/{id}`  
**Auth:** Required

---

## Budget

### GET /api/budget

Get current budget usage.

**Method:** `GET`  
**Path:** `/api/budget`  
**Auth:** Required

#### Response

```json
{
  "daily": {
    "limit": 200.0,
    "used": 45.5,
    "remaining": 154.5,
    "percentage": 22.75
  },
  "monthly": {
    "limit": 5000.0,
    "used": 1200.0,
    "remaining": 3800.0,
    "percentage": 24.0
  },
  "alerts": [
    {
      "threshold": 0.5,
      "triggered": false
    }
  ]
}
```

---

### GET /api/budget/history

Get budget usage history.

**Method:** `GET`  
**Path:** `/api/budget/history`  
**Auth:** Required

#### Query Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `days` | integer | Number of days (default: 30) |

---

## Configuration

### GET /api/config

Get current configuration.

**Method:** `GET`  
**Path:** `/api/config`  
**Auth:** Required (admin)

#### Response

```json
{
  "environment": "production",
  "log_level": "INFO",
  "rate_limit": 60,
  "features": {
    "external_research": true,
    "tool_synthesis": true,
    "self_healing": true
  }
}
```

---

### PUT /api/config

Update configuration.

**Method:** `PUT`  
**Path:** `/api/config`  
**Auth:** Required (admin)

#### Request

Partial configuration update:

```json
{
  "log_level": "DEBUG"
}
```

---

## Healing

### GET /api/healing/config

Get self-healing configuration.

**Method:** `GET`  
**Path:** `/api/healing/config`  
**Auth:** Required (admin)

#### Response

```json
{
  "max_healing_level": 3,
  "max_retries_per_level": 3,
  "base_delay_seconds": 1.0,
  "max_delay_seconds": 60.0,
  "exponential_backoff": true,
  "jitter": true,
  "enable_learning": true,
  "enable_observability": true
}
```

---

### PUT /api/healing/config

Update healing configuration.

**Method:** `PUT`  
**Path:** `/api/healing/config`  
**Auth:** Required (admin)

---

### GET /api/healing/history

Get healing history and statistics.

**Method:** `GET`  
**Path:** `/api/healing/history`  
**Auth:** Required

---

### POST /api/healing/reset

Reset healing statistics.

**Method:** `POST`  
**Path:** `/api/healing/reset`  
**Auth:** Required (admin)

---

## Swarm

### POST /api/swarm/fractal/register

Register a new fractal agent.

**Method:** `POST`  
**Path:** `/api/swarm/fractal/register`  
**Auth:** Required

#### Request

```json
{
  "fractal_id": "fractal-001",
  "specialization": "python",
  "capabilities": ["refactoring", "testing"]
}
```

---

### POST /api/swarm/task

Process a task through swarm auction.

**Method:** `POST`  
**Path:** `/api/swarm/task`  
**Auth:** Required

---

### POST /api/swarm/bid

Submit a bid for a task.

**Method:** `POST`  
**Path:** `/api/swarm/bid`  
**Auth:** Required

---

## Knowledge

### POST /api/knowledge/parse

Parse code and extract AST structure.

**Method:** `POST`  
**Path:** `/api/knowledge/parse`  
**Auth:** Required

#### Request

```json
{
  "code": "def hello(): pass",
  "file_path": "example.py"
}
```

---

### POST /api/knowledge/mine

Mine usage patterns from code.

**Method:** `POST`  
**Path:** `/api/knowledge/mine`  
**Auth:** Required

---

### POST /api/knowledge/ingest

Ingest a library/repository.

**Method:** `POST`  
**Path:** `/api/knowledge/ingest`  
**Auth:** Required

---

### POST /api/knowledge/cheatsheet

Generate a cheat sheet for a library.

**Method:** `POST`  
**Path:** `/api/knowledge/cheatsheet`  
**Auth:** Required

---

## Maintenance

### POST /api/maintenance/scan

Scan project for technical debt.

**Method:** `POST`  
**Path:** `/api/maintenance/scan`  
**Auth:** Required

#### Request

```json
{
  "project_path": "/path/to/project",
  "include_types": ["TODO", "FIXME", "complexity"]
}
```

---

### POST /api/maintenance/refinance

Optimize technical debt.

**Method:** `POST`  
**Path:** `/api/maintenance/refinance`  
**Auth:** Required

---

## WebSocket Endpoints

See [WebSocket Documentation](./websocket.md) for real-time endpoints:

- `/ws/events` - System events
- `/ws/ooda` - OODA loop visualization
- `/ws/steering` - Steering commands

---

## SDK Examples

### Python

```python
import requests

class GAAPClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    def chat(self, message: str, provider: str = "kimi") -> dict:
        response = requests.post(
            f"{self.base_url}/api/chat",
            headers=self.headers,
            json={"message": message, "provider": provider}
        )
        response.raise_for_status()
        return response.json()

# Usage
client = GAAPClient("http://localhost:8000")
result = client.chat("Hello GAAP!")
print(result["response"])
```

### JavaScript/TypeScript

```typescript
class GAAPClient {
  constructor(private baseUrl: string, private apiKey?: string) {}

  async chat(message: string, provider: string = "kimi"): Promise<string> {
    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey && { Authorization: `Bearer ${this.apiKey}` }),
      },
      body: JSON.stringify({ message, provider }),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    return data.response;
  }
}

// Usage
const client = new GAAPClient("http://localhost:8000");
const response = await client.chat("Hello GAAP!");
console.log(response);
```

---

## Error Code Reference

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Request validation failed |
| `UNAUTHORIZED` | 401 | Authentication required |
| `FORBIDDEN` | 403 | Permission denied |
| `NOT_FOUND` | 404 | Resource not found |
| `RATE_LIMITED` | 429 | Too many requests |
| `PROVIDER_ERROR` | 502 | LLM provider error |
| `SERVICE_UNAVAILABLE` | 503 | All providers failed |
| `INTERNAL_ERROR` | 500 | Server error |
