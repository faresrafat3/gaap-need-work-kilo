# GAAP Event System

## Overview

The GAAP Event System provides a centralized, thread-safe pub/sub event bus for cross-module communication. It enables loose coupling between components and powers real-time WebSocket updates to clients.

**Key Benefits:**
- Decouples modules without direct dependencies
- Thread-safe singleton ensures consistent state
- Real-time WebSocket broadcasting for live updates
- Event history for debugging and auditing
- Supports both sync and async subscribers

## Architecture

### Singleton Pattern

The `EventEmitter` uses a thread-safe singleton pattern to ensure a single global instance:

```python
class EventEmitter:
    _instance: EventEmitter | None = None
    _lock = threading.Lock()

    def __new__(cls) -> EventEmitter:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
```

### Thread-safe Implementation

- Uses `threading.RLock()` for re-entrant locking
- All subscriber operations are atomic
- Safe for concurrent access from multiple threads

### Pub/Sub Model

```
┌─────────────┐     emit()     ┌───────────────┐     notify     ┌──────────────┐
│  Publisher  │ ──────────────▶│ EventEmitter  │ ───────────────▶│ Subscriber 1 │
└─────────────┘                └───────────────┘                 └──────────────┘
                                      │
                                      │         notify
                                      └─────────────────────────▶┌──────────────┐
                                                                │ Subscriber 2 │
                                                                └──────────────┘
```

### Event History Storage

- Stores up to 1000 events in memory (configurable via `_max_history`)
- Oldest events are removed when limit is reached
- Queryable by event type for debugging

## Event Types

The system defines 22 event types organized into 9 categories:

### Config Events

| Event | Description |
|-------|-------------|
| `CONFIG_CHANGED` | Configuration value modified |
| `CONFIG_VALIDATED` | Configuration validation completed |

```python
from gaap.core.events import EventType

# Example: Emit config change
emitter.emit(
    EventType.CONFIG_CHANGED,
    {"module": "healing", "key": "max_retries", "old_value": 3, "new_value": 5},
    source="config_api"
)
```

### OODA Events

| Event | Description |
|-------|-------------|
| `OODA_PHASE` | OODA loop phase transition (Observe/Orient/Decide/Act) |
| `OODA_ITERATION` | OODA loop iteration completed |
| `OODA_COMPLETE` | OODA loop finished |

```python
# Example: OODA phase transition
emitter.emit(
    EventType.OODA_PHASE,
    {"phase": "observe", "session_id": "sess_123", "iteration": 5},
    source="ooda_engine"
)
```

### Healing Events

| Event | Description |
|-------|-------------|
| `HEALING_STARTED` | Self-healing process initiated |
| `HEALING_LEVEL` | Healing level transition |
| `HEALING_SUCCESS` | Healing completed successfully |
| `HEALING_FAILED` | Healing failed after retries |

```python
# Example: Healing started
emitter.emit(
    EventType.HEALING_STARTED,
    {"error_type": "RateLimitError", "level": 1, "provider": "openai"},
    source="healing_module"
)
```

### Research Events

| Event | Description |
|-------|-------------|
| `RESEARCH_STARTED` | Research task initiated |
| `RESEARCH_PROGRESS` | Research progress update |
| `RESEARCH_SOURCE_FOUND` | New source discovered |
| `RESEARCH_HYPOTHESIS` | Hypothesis formed |
| `RESEARCH_COMPLETE` | Research finished |

```python
# Example: Research progress
emitter.emit(
    EventType.RESEARCH_PROGRESS,
    {"query": "best practices for async Python", "sources_found": 15, "progress": 0.65},
    source="research_agent"
)
```

### Provider Events

| Event | Description |
|-------|-------------|
| `PROVIDER_STATUS` | Provider status change |
| `PROVIDER_ERROR` | Provider error occurred |
| `PROVIDER_SWITCHED` | Active provider changed |

```python
# Example: Provider switched
emitter.emit(
    EventType.PROVIDER_SWITCHED,
    {"from": "openai", "to": "anthropic", "reason": "rate_limit"},
    source="provider_manager"
)
```

### Budget Events

| Event | Description |
|-------|-------------|
| `BUDGET_ALERT` | Budget threshold warning |
| `BUDGET_UPDATE` | Budget usage update |

```python
# Example: Budget alert
emitter.emit(
    EventType.BUDGET_ALERT,
    {"current_usd": 45.50, "limit_usd": 50.00, "percentage": 91.0},
    source="budget_tracker"
)
```

### Session Events

| Event | Description |
|-------|-------------|
| `SESSION_CREATED` | New session created |
| `SESSION_UPDATE` | Session updated |
| `SESSION_PAUSED` | Session paused |
| `SESSION_RESUMED` | Session resumed |
| `SESSION_COMPLETED` | Session finished |

```python
# Example: Session created
emitter.emit(
    EventType.SESSION_CREATED,
    {"session_id": "sess_abc123", "name": "API Integration", "priority": "high"},
    source="sessions_api"
)
```

### Steering Events

| Event | Description |
|-------|-------------|
| `STEERING_COMMAND` | Steering command received |
| `STEERING_PAUSE` | Steering pause requested |
| `STEERING_RESUME` | Steering resume requested |
| `STEERING_VETO` | Steering veto issued |

```python
# Example: Steering veto
emitter.emit(
    EventType.STEERING_VETO,
    {"action": "file_delete", "path": "/critical/config.yaml", "reason": "protected_path"},
    source="steering_controller"
)
```

### System Events

| Event | Description |
|-------|-------------|
| `SYSTEM_ERROR` | System error occurred |
| `SYSTEM_WARNING` | System warning |
| `SYSTEM_HEALTH` | Health check result |

```python
# Example: System health
emitter.emit(
    EventType.SYSTEM_HEALTH,
    {"status": "healthy", "checks": {"database": "ok", "cache": "ok", "providers": "degraded"}},
    source="health_monitor"
)
```

## EventEmitter Class

### Constructor and Singleton Access

```python
from gaap.core.events import EventEmitter

# Get singleton instance
emitter = EventEmitter.get_instance()

# Or simply instantiate (returns singleton)
emitter = EventEmitter()
```

### subscribe()

Subscribe a synchronous callback to an event type.

```python
def subscribe(
    self,
    event_type: EventType,
    callback: Callable[[Event], None],
) -> str:
```

**Returns:** Subscription ID (UUID string) for later unsubscription.

```python
def on_config_change(event: Event) -> None:
    print(f"Config changed: {event.data}")

sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, on_config_change)
```

### subscribe_async()

Subscribe an async callback to an event type.

```python
def subscribe_async(
    self,
    event_type: EventType,
    callback: Callable[[Event], Any],
) -> str:
```

```python
async def on_session_event(event: Event) -> None:
    await notify_clients(event.data)

sub_id = emitter.subscribe_async(EventType.SESSION_CREATED, on_session_event)
```

### unsubscribe()

Remove a subscription by ID.

```python
def unsubscribe(self, subscription_id: str) -> bool:
```

**Returns:** `True` if subscription was found and removed, `False` otherwise.

```python
# Later, when cleaning up
emitter.unsubscribe(sub_id)
```

### emit()

Emit an event to all synchronous subscribers.

```python
def emit(
    self,
    event_type: EventType,
    data: dict[str, Any],
    source: str = "",
) -> Event:
```

**Returns:** The created `Event` object.

```python
event = emitter.emit(
    EventType.CONFIG_CHANGED,
    {"module": "healing", "key": "enabled"},
    source="config_api"
)
```

### emit_async()

Emit an event and call all async subscribers.

```python
async def emit_async(
    self,
    event_type: EventType,
    data: dict[str, Any],
    source: str = "",
) -> Event:
```

```python
# In an async context
await emitter.emit_async(
    EventType.SESSION_CREATED,
    {"session_id": "sess_123"},
    source="sessions_api"
)
```

### get_history()

Retrieve recent events from history.

```python
def get_history(
    self,
    event_type: EventType | None = None,
    limit: int = 100,
) -> list[Event]:
```

```python
# Get all recent events
recent = emitter.get_history(limit=50)

# Get specific event type
config_events = emitter.get_history(EventType.CONFIG_CHANGED, limit=20)
```

### subscriber_count()

Count active subscribers.

```python
def subscriber_count(self, event_type: EventType | None = None) -> int:
```

```python
# Total subscribers
total = emitter.subscriber_count()

# Subscribers for specific event type
config_subs = emitter.subscriber_count(EventType.CONFIG_CHANGED)
```

## Event Dataclass

```python
@dataclass
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `EventType` | The event type enum value |
| `data` | `dict[str, Any]` | Event payload (any JSON-serializable dict) |
| `timestamp` | `datetime` | When the event was created (auto-generated) |
| `source` | `str` | Module/component that emitted the event |
| `event_id` | `str` | Unique UUID for this event |

### to_dict() Method

Converts event to a JSON-serializable dictionary:

```python
def to_dict(self) -> dict[str, Any]:
    return {
        "event_id": self.event_id,
        "type": self.type.name,
        "timestamp": self.timestamp.isoformat(),
        "source": self.source,
        "data": self.data,
    }
```

**Usage:**

```python
event = emitter.emit(EventType.SESSION_CREATED, {"session_id": "123"})
json_data = event.to_dict()
# {
#     "event_id": "550e8400-e29b-41d4-a716-446655440000",
#     "type": "SESSION_CREATED",
#     "timestamp": "2024-01-15T10:30:00.123456",
#     "source": "",
#     "data": {"session_id": "123"}
# }
```

## WebSocket Integration

### ConnectionManager Subscription

The `ConnectionManager` automatically subscribes to events for WebSocket broadcasting:

```python
class ConnectionManager:
    def __init__(self) -> None:
        self._emitter = EventEmitter.get_instance()
        # ...

    async def connect(self, websocket: WebSocket, channel: str = "events") -> None:
        if self._subscription_id is None:
            self._subscription_id = self._emitter.subscribe_async(
                EventType.CONFIG_CHANGED,  # Placeholder
                self._on_any_event,
            )
```

### Channel Routing

Events are routed to appropriate WebSocket channels:

| Channel | Events |
|---------|--------|
| `events` | All events |
| `ooda` | `OODA_PHASE`, `OODA_ITERATION`, `OODA_COMPLETE` |
| `steering` | Steering events (custom routing) |

```python
async def broadcast_event(self, event: Event) -> None:
    message = event.to_dict()
    await self.broadcast(message, "events")

    if event.type in (
        EventType.OODA_PHASE,
        EventType.OODA_ITERATION,
        EventType.OODA_COMPLETE,
    ):
        await self.broadcast(message, "ooda")
```

### Real-time Broadcasting

During app startup, all event types are subscribed for WebSocket broadcasting:

```python
async def lifespan(app: FastAPI):
    emitter = EventEmitter.get_instance()

    async def broadcast_event(event):
        await ws_manager.broadcast_event(event)

    for event_type in EventType:
        emitter.subscribe_async(event_type, broadcast_event)

    yield
```

### WebSocket Endpoints

```python
@app.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, "events")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

@app.websocket("/ws/ooda")
async def ooda_websocket(websocket: WebSocket):
    await ws_manager.connect(websocket, "ooda")
    # ... similar pattern
```

## Usage Examples

### Subscribe to Events (Synchronous)

```python
from gaap.core.events import EventEmitter, EventType

emitter = EventEmitter.get_instance()

def handle_budget_alert(event):
    alert = event.data
    if alert["percentage"] >= 90:
        send_slack_alert(f"Budget at {alert['percentage']}%!")

sub_id = emitter.subscribe(EventType.BUDGET_ALERT, handle_budget_alert)

# Later, cleanup
emitter.unsubscribe(sub_id)
```

### Subscribe to Events (Asynchronous)

```python
import asyncio
from gaap.core.events import EventEmitter, EventType

emitter = EventEmitter.get_instance()

async def handle_session_event(event):
    session_id = event.data["session_id"]
    await database.notify_clients(session_id, event.to_dict())

sub_id = emitter.subscribe_async(EventType.SESSION_CREATED, handle_session_event)
```

### Emit Events

```python
from gaap.core.events import EventEmitter, EventType

emitter = EventEmitter.get_instance()

# Synchronous emit (only sync subscribers called)
emitter.emit(
    EventType.PROVIDER_ERROR,
    {"provider": "openai", "error": "rate_limit", "retry_after": 60},
    source="provider_client"
)

# Async emit (calls both sync and async subscribers)
await emitter.emit_async(
    EventType.SESSION_COMPLETED,
    {"session_id": "sess_123", "status": "success", "duration_ms": 45000},
    source="session_manager"
)
```

### Integration with API Routes

```python
from fastapi import APIRouter, HTTPException
from gaap.core.events import EventEmitter, EventType

router = APIRouter()
emitter = EventEmitter.get_instance()

@router.post("/sessions/{session_id}/pause")
async def pause_session(session_id: str):
    # ... pause logic ...

    emitter.emit(
        EventType.SESSION_PAUSED,
        {"session_id": session_id, "paused_at": datetime.now().isoformat()},
        source="sessions_api"
    )

    return {"status": "paused"}

@router.post("/config")
async def update_config(key: str, value: Any):
    old_value = config.get(key)
    config.set(key, value)

    emitter.emit(
        EventType.CONFIG_CHANGED,
        {"key": key, "old_value": old_value, "new_value": value},
        source="config_api"
    )

    return {"key": key, "value": value}
```

### Multiple Subscribers

```python
emitter = EventEmitter.get_instance()

# Multiple components can subscribe to the same event
log_sub_id = emitter.subscribe(EventType.HEALING_FAILED, lambda e: logger.error(e.data))
metrics_sub_id = emitter.subscribe(EventType.HEALING_FAILED, lambda e: metrics.increment("healing.failed"))
alert_sub_id = emitter.subscribe(EventType.HEALING_FAILED, lambda e: alerts.send(e.data))

print(emitter.subscriber_count(EventType.HEALING_FAILED))  # 3
```

### Event History for Debugging

```python
# Get recent errors for debugging
errors = emitter.get_history(EventType.SYSTEM_ERROR, limit=10)
for error in errors:
    print(f"[{error.timestamp}] {error.source}: {error.data}")

# Get all recent events
all_events = emitter.get_history(limit=50)
```

## Best Practices

### When to Use Which Event Type

| Scenario | Event Type | Reason |
|----------|-----------|--------|
| Module config updated | `CONFIG_CHANGED` | Other modules may need to reload |
| Budget threshold crossed | `BUDGET_ALERT` | Alerting and UI updates |
| Provider fails over | `PROVIDER_SWITCHED` | Logging and metrics |
| Long operation progress | `RESEARCH_PROGRESS` | UI progress indicators |
| Session state change | `SESSION_*` | Dashboard updates |
| Critical error | `SYSTEM_ERROR` | Alerting and logging |

### Error Handling in Subscribers

Always wrap subscriber logic in try/except to prevent one subscriber from affecting others:

```python
def safe_subscriber(event: Event) -> None:
    try:
        process_event(event)
    except Exception as e:
        logger.error(f"Error processing event {event.event_id}: {e}")

emitter.subscribe(EventType.SESSION_CREATED, safe_subscriber)
```

The `EventEmitter` catches exceptions internally and logs them, but handling your own errors provides better context.

### Memory Considerations

- Event history is capped at 1000 events
- Use `clear_history()` to free memory if needed:
  ```python
  emitter.clear_history()
  ```
- Store subscription IDs and unsubscribe when components are destroyed:
  ```python
  class MyComponent:
      def __init__(self):
          self._sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, self._on_config)

      def cleanup(self):
          emitter.unsubscribe(self._sub_id)
  ```

### Async vs Sync Subscribers

- Use `subscribe()` for CPU-bound or non-async operations
- Use `subscribe_async()` for I/O operations (database, HTTP calls, WebSocket)
- `emit()` only calls sync subscribers
- `emit_async()` calls both sync and async subscribers

### Event Data Guidelines

- Keep payloads small and serializable
- Include enough context for subscribers to act
- Don't include sensitive data (passwords, tokens)
- Use consistent naming conventions:

```python
# Good
emitter.emit(
    EventType.SESSION_COMPLETED,
    {
        "session_id": "sess_123",
        "status": "success",
        "duration_ms": 45000,
        "tasks_completed": 10,
    },
    source="session_manager"
)

# Avoid
emitter.emit(
    EventType.SESSION_COMPLETED,
    {"data": session_object.to_dict()},  # Unclear structure
    source=""
)
```

### Source Naming Convention

Use the module or component name as the source:

```python
emitter.emit(EventType.CONFIG_CHANGED, data, source="config_api")
emitter.emit(EventType.HEALING_STARTED, data, source="healing_module")
emitter.emit(EventType.PROVIDER_ERROR, data, source="provider_client")
```

This helps with debugging and filtering in event history.
