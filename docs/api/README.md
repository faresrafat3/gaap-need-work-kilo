# GAAP API Documentation

منصة الذكاء الاصطناعي العامة - General AI Assistant Platform

---

## Overview

GAAP provides a RESTful API and WebSocket endpoints for interacting with the AI assistant platform. This documentation covers authentication, rate limiting, and all available endpoints.

## Base URLs

| Environment | URL |
|------------|-----|
| Development | `http://localhost:8000` |
| Staging | `https://staging-api.gaap.io` |
| Production | `https://api.gaap.io` |

## Quick Start

```bash
# Health check
curl http://localhost:8000/api/health

# List providers
curl http://localhost:8000/api/providers

# Send chat message
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello GAAP!"}'
```

## Authentication

### API Keys

GAAP uses API keys for authentication. Include your API key in the `Authorization` header:

```bash
curl http://localhost:8000/api/providers \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Environment Variables

Configure API keys in your `.env` file:

```bash
# Required for LLM providers
GAAP_KILO_API_KEY=your_key_here
GAAP_OPENAI_API_KEY=your_key_here
GAAP_ANTHROPIC_API_KEY=your_key_here

# Required for authentication
GAAP_JWT_SECRET=your_jwt_secret
GAAP_CAPABILITY_SECRET=your_capability_secret
```

### Capability Tokens

GAAP supports capability-based authentication for fine-grained access control:

```python
from gaap.security.capability import generate_capability_token

token = generate_capability_token(
    subject="user:123",
    capabilities=["chat:read", "chat:write"],
    expires_in=3600
)
```

## Rate Limiting

### Default Limits

| Endpoint Type | Limit | Window |
|--------------|-------|--------|
| General API | 60 requests | 1 minute |
| Chat API | 30 requests | 1 minute |
| WebSocket | 100 messages | 1 minute |

### Response Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded

When rate limit is exceeded, the API returns:

```json
{
  "error": "Rate limit exceeded",
  "detail": "Too many requests. Please try again in 30 seconds.",
  "retry_after": 30
}
```

### Configuration

Configure rate limiting via environment variables:

```bash
# Requests per minute per IP
GAAP_RATE_LIMIT_PER_MINUTE=60

# Redis for distributed rate limiting (production)
GAAP_REDIS_URL=redis://localhost:6379/0
```

## Content Types

All API requests and responses use JSON:

```http
Content-Type: application/json
```

For file uploads, use `multipart/form-data`.

## Error Handling

### Error Format

All errors follow a consistent format:

```json
{
  "error": "Human-readable error message in Arabic or English",
  "detail": "Technical details for debugging",
  "code": "ERROR_CODE",
  "request_id": "req-1234567890"
}
```

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 429 | Rate Limit Exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Malformed request |
| `AUTH_REQUIRED` | Authentication required |
| `PROVIDER_UNAVAILABLE` | All AI providers failed |
| `RATE_LIMITED` | Rate limit exceeded |
| `SESSION_NOT_FOUND` | Session does not exist |

## API Versioning

The current API version is **v1**. Version is specified in the URL:

```
/api/v1/chat
```

For backward compatibility, unversioned URLs default to the latest version.

## Documentation Sections

- [Endpoints Reference](./endpoints.md) - Complete API endpoint documentation
- [WebSocket Protocol](./websocket.md) - Real-time communication

---

## Support

- 📧 Email: support@gaap.io
- 🐛 Issues: [GitHub Issues](https://github.com/gaap-system/gaap/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/gaap-system/gaap/discussions)

## License

GAAP is licensed under the MIT License.
