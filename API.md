# GAAP Frontend API Documentation

## Base URL
- Development: `http://localhost:3000`
- Production: `https://yourdomain.com`

---

## Health Check

### GET /api/health

Check the health status of the frontend and backend services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "latency": 45,
  "services": {
    "nextjs": {
      "status": "healthy",
      "uptime": 3600,
      "memory": {...},
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "python": {
      "status": "healthy",
      "url": "http://localhost:8000",
      "latency": 23,
      "error": null,
      "version": "1.0.0"
    }
  },
  "message": "✅ النظام يعمل بكامل طاقته"
}
```

---

## Providers

### GET /api/providers

List all AI providers.

**Response:**
```json
{
  "providers": [
    {
      "name": "kimi",
      "type": "webchat",
      "enabled": true,
      "priority": 1,
      "models": ["kimi-k2.5-thinking"],
      "health": "healthy"
    }
  ]
}
```

### POST /api/providers

Create a new provider.

**Request Body:**
```json
{
  "name": "openai",
  "type": "api",
  "enabled": true,
  "priority": 1,
  "models": ["gpt-4"]
}
```

### GET /api/providers/live

Get live provider status with actual models from backend.

**Response:**
```json
{
  "providers": [
    {
      "name": "kimi",
      "display_name": "Kimi",
      "actual_model": "kimi-k2.5-thinking",
      "status": "active",
      "latency_ms": 150,
      "success_rate": 98.5
    }
  ],
  "last_updated": "2024-01-15T10:30:00Z"
}
```

---

## Sessions

### GET /api/sessions

List all sessions with optional filtering.

**Query Parameters:**
- `status` - Filter by status (pending, running, paused, completed, failed, cancelled)
- `priority` - Filter by priority (low, normal, high, critical)
- `limit` - Maximum number of results (default: 50)
- `offset` - Pagination offset (default: 0)

**Response:**
```json
{
  "sessions": [
    {
      "id": "uuid",
      "name": "Chat Session",
      "status": "completed",
      "priority": "normal",
      "progress": 100,
      "tasks_total": 10,
      "tasks_completed": 10,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ],
  "total": 1
}
```

### POST /api/sessions

Create a new session.

**Request Body:**
```json
{
  "name": "New Session",
  "description": "Session description",
  "priority": "normal",
  "tags": ["chat", "research"]
}
```

### GET /api/sessions/:id

Get a specific session by ID.

### PUT /api/sessions/:id

Update a session.

### DELETE /api/sessions/:id

Delete a session.

---

## Chat

### POST /api/chat

Send a chat message with streaming support.

**Request Body:**
```json
{
  "messages": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi!" }
  ],
  "provider": "kimi",
  "stream": true
}
```

**Response (Streaming):**
```
{"type":"content","content":"Hello"}
{"type":"content","content":" there"}
{"type":"done"}
```

**Rate Limiting:**
- 60 requests per minute per IP
- Returns 429 status when exceeded

---

## Error Responses

All errors follow this format:

```json
{
  "error": "Error description in Arabic",
  "detail": "Technical details (optional)"
}
```

**Status Codes:**
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `404` - Not Found
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error
