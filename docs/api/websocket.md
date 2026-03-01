# GAAP WebSocket Protocol

Real-time communication protocol for GAAP.

---

## Overview

GAAP provides WebSocket endpoints for real-time updates, streaming responses, and bidirectional communication. All WebSocket connections use JSON message format.

## Connection

### Endpoint

```
ws://localhost:8000/ws/{channel}
```

For production with TLS:

```
wss://api.gaap.io/ws/{channel}
```

### Available Channels

| Channel | Path | Purpose |
|---------|------|---------|
| Events | `/ws/events` | System events and notifications |
| OODA | `/ws/ooda` | OODA loop visualization |
| Steering | `/ws/steering` | User steering commands |

### Connection Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/events');

ws.onopen = () => {
  console.log('Connected to GAAP events');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Disconnected from GAAP');
};
```

---

## Message Format

### General Message Structure

All messages follow this structure:

```json
{
  "type": "message_type",
  "timestamp": 1704067200,
  "data": { /* message-specific data */ },
  "id": "msg-uuid"
}
```

### Message Types

#### Client → Server

| Type | Description |
|------|-------------|
| `ping` | Keep-alive ping |
| `subscribe` | Subscribe to event type |
| `unsubscribe` | Unsubscribe from event type |
| `command` | Send command to GAAP |

#### Server → Client

| Type | Description |
|------|-------------|
| `pong` | Ping response |
| `event` | System event |
| `ooda_update` | OODA loop state update |
| `error` | Error message |
| `connected` | Connection confirmation |

---

## Channel: Events

### Subscribing to Events

```json
{
  "type": "subscribe",
  "event_type": "session_created"
}
```

### Event Types

| Event Type | Description |
|------------|-------------|
| `session_created` | New session created |
| `session_updated` | Session state changed |
| `session_completed` | Session finished |
| `task_started` | Task execution started |
| `task_completed` | Task execution completed |
| `provider_status` | Provider health changed |
| `system_alert` | System alert/warning |

### Event Message Example

```json
{
  "type": "event",
  "timestamp": 1704067200,
  "event_type": "session_created",
  "data": {
    "session_id": "sess-abc123",
    "name": "Code Review",
    "status": "pending"
  }
}
```

---

## Channel: OODA

### OODA Loop Visualization

The OODA channel provides real-time updates about the OODA (Observe-Orient-Decide-Act) loop execution.

### OODA States

```json
{
  "type": "ooda_update",
  "timestamp": 1704067200,
  "data": {
    "session_id": "sess-abc123",
    "phase": "orient",
    "layer": 1,
    "progress": 45.5,
    "state": {
      "current_node": "strategy_selection",
      "depth": 3,
      "confidence": 0.87
    }
  }
}
```

### Phase Values

| Phase | Description |
|-------|-------------|
| `observe` | Gathering information |
| `orient` | Analyzing context |
| `decide` | Making decisions |
| `act` | Executing actions |
| `learn` | Learning from results |

---

## Channel: Steering

### Sending Commands

Users can send steering commands to influence GAAP behavior.

### Command Types

| Command | Description |
|---------|-------------|
| `pause` | Pause current session |
| `resume` | Resume paused session |
| `veto` | Cancel current action |
| `redirect` | Change execution direction |

### Pause Command

```json
{
  "type": "command",
  "command": "pause",
  "session_id": "sess-abc123"
}
```

### Resume Command

```json
{
  "type": "command",
  "command": "resume",
  "session_id": "sess-abc123",
  "instruction": "Continue with approach B"
}
```

### Veto Command

```json
{
  "type": "command",
  "command": "veto",
  "session_id": "sess-abc123"
}
```

### Command Response

```json
{
  "type": "command_response",
  "command": "pause",
  "status": "success",
  "timestamp": 1704067200
}
```

---

## Keep-Alive

### Ping/Pong

Both client and server send periodic pings to maintain connection:

**Client Ping:**
```json
{
  "type": "ping",
  "timestamp": 1704067200
}
```

**Server Pong:**
```json
{
  "type": "pong",
  "timestamp": 1704067200
}
```

### Connection Timeout

- Ping interval: 30 seconds
- Receive timeout: 60 seconds
- Auto-reconnect recommended

---

## Error Handling

### Error Message Format

```json
{
  "type": "error",
  "timestamp": 1704067200,
  "error": {
    "code": "INVALID_COMMAND",
    "message": "Unknown command type: 'stop'",
    "recoverable": true
  }
}
```

### Error Codes

| Code | Description | Recoverable |
|------|-------------|-------------|
| `INVALID_MESSAGE` | Malformed JSON | Yes |
| `INVALID_COMMAND` | Unknown command | Yes |
| `SESSION_NOT_FOUND` | Session doesn't exist | No |
| `NOT_AUTHORIZED` | Permission denied | No |
| `RATE_LIMITED` | Too many messages | Yes |
| `INTERNAL_ERROR` | Server error | Yes |

---

## Reconnection

### Reconnection Strategy

```javascript
class GAAPWebSocket {
  constructor(url) {
    this.url = url;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
  }

  connect() {
    this.ws = new WebSocket(this.url);
    
    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      console.log('Connected');
    };
    
    this.ws.onclose = () => {
      this.attemptReconnect();
    };
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * this.reconnectAttempts;
      
      console.log(`Reconnecting in ${delay}ms...`);
      setTimeout(() => this.connect(), delay);
    }
  }
}
```

---

## Client Examples

### JavaScript (Browser)

```javascript
class GAAPWebSocketClient {
  constructor(baseUrl, channel = 'events') {
    this.ws = new WebSocket(`${baseUrl}/ws/${channel}`);
    this.messageHandlers = new Map();
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleMessage(data);
    };
  }

  handleMessage(data) {
    const handlers = this.messageHandlers.get(data.type) || [];
    handlers.forEach(handler => handler(data));
  }

  on(type, handler) {
    if (!this.messageHandlers.has(type)) {
      this.messageHandlers.set(type, []);
    }
    this.messageHandlers.get(type).push(handler);
  }

  send(type, data) {
    this.ws.send(JSON.stringify({ type, ...data }));
  }

  subscribe(eventType) {
    this.send('subscribe', { event_type: eventType });
  }

  pause(sessionId) {
    this.send('command', { 
      command: 'pause', 
      session_id: sessionId 
    });
  }
}

// Usage
const client = new GAAPWebSocketClient('ws://localhost:8000', 'ooda');

client.on('ooda_update', (data) => {
  console.log('OODA Phase:', data.data.phase);
  console.log('Progress:', data.data.progress);
});
```

### Python

```python
import asyncio
import json
import websockets

class GAAPWebSocketClient:
    def __init__(self, base_url: str, channel: str = "events"):
        self.url = f"{base_url}/ws/{channel}"
        self.websocket = None
        self.handlers = {}
    
    async def connect(self):
        self.websocket = await websockets.connect(self.url)
        asyncio.create_task(self._listen())
    
    async def _listen(self):
        async for message in self.websocket:
            data = json.loads(message)
            await self._handle_message(data)
    
    async def _handle_message(self, data: dict):
        msg_type = data.get("type")
        handlers = self.handlers.get(msg_type, [])
        for handler in handlers:
            await handler(data)
    
    def on(self, msg_type: str, handler):
        if msg_type not in self.handlers:
            self.handlers[msg_type] = []
        self.handlers[msg_type].append(handler)
    
    async def send(self, msg_type: str, data: dict):
        message = json.dumps({"type": msg_type, **data})
        await self.websocket.send(message)
    
    async def subscribe(self, event_type: str):
        await self.send("subscribe", {"event_type": event_type})
    
    async def pause_session(self, session_id: str):
        await self.send("command", {
            "command": "pause",
            "session_id": session_id
        })
    
    async def close(self):
        await self.websocket.close()

# Usage
async def main():
    client = GAAPWebSocketClient("ws://localhost:8000", "ooda")
    
    @client.on("ooda_update")
    async def on_ooda(data):
        print(f"Phase: {data['data']['phase']}")
        print(f"Progress: {data['data']['progress']}%")
    
    await client.connect()
    await asyncio.sleep(60)  # Listen for 60 seconds
    await client.close()

asyncio.run(main())
```

---

## Connection Management

### Connection Limits

| Resource | Limit |
|----------|-------|
| Max connections per client | 3 (one per channel) |
| Max connections per IP | 10 |
| Message size | 1MB |
| Message rate | 100/minute |

### Best Practices

1. **Reconnect on disconnect**: Network issues happen
2. **Handle ping/pong**: Keep connection alive
3. **Validate messages**: Always parse JSON safely
4. **Rate limit sends**: Don't flood the server
5. **Close properly**: Call `ws.close()` when done

---

## Security

### Authentication

WebSocket connections support the same authentication as REST API:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/events', [], {
  headers: {
    'Authorization': 'Bearer YOUR_API_KEY'
  }
});
```

Or pass token in query string:

```
ws://localhost:8000/ws/events?token=YOUR_API_KEY
```

### TLS/WSS

Always use WSS in production:

```
wss://api.gaap.io/ws/events
```
