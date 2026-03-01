"""
Event System - Central Event Bus for GAAP
==========================================

Provides pub/sub pattern for cross-module communication.
Used by WebSocket server for real-time updates.

Usage:
    from gaap.core.events import EventEmitter, EventType

    emitter = EventEmitter.get_instance()

    # Subscribe
    callback_id = emitter.subscribe(EventType.CONFIG_CHANGED, my_callback)

    # Emit
    emitter.emit(EventType.CONFIG_CHANGED, {"module": "healing"})

    # Unsubscribe
    emitter.unsubscribe(callback_id)
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger("gaap.core.events")


class EventType(Enum):
    """All event types in the system."""

    # Config events
    CONFIG_CHANGED = auto()
    CONFIG_VALIDATED = auto()

    # OODA loop events
    OODA_PHASE = auto()
    OODA_ITERATION = auto()
    OODA_COMPLETE = auto()

    # Healing events
    HEALING_STARTED = auto()
    HEALING_LEVEL = auto()
    HEALING_SUCCESS = auto()
    HEALING_FAILED = auto()

    # Research events
    RESEARCH_STARTED = auto()
    RESEARCH_PROGRESS = auto()
    RESEARCH_SOURCE_FOUND = auto()
    RESEARCH_HYPOTHESIS = auto()
    RESEARCH_COMPLETE = auto()

    # Provider events
    PROVIDER_STATUS = auto()
    PROVIDER_ERROR = auto()
    PROVIDER_SWITCHED = auto()

    # Budget events
    BUDGET_ALERT = auto()
    BUDGET_UPDATE = auto()

    # Session events
    SESSION_CREATED = auto()
    SESSION_UPDATE = auto()
    SESSION_PAUSED = auto()
    SESSION_RESUMED = auto()
    SESSION_COMPLETED = auto()

    # Steering events
    STEERING_COMMAND = auto()
    STEERING_PAUSE = auto()
    STEERING_RESUME = auto()
    STEERING_VETO = auto()

    # System events
    SYSTEM_ERROR = auto()
    SYSTEM_WARNING = auto()
    SYSTEM_HEALTH = auto()

    # Tool synthesis events
    TOOL_SYNTHESIS_STARTED = auto()
    TOOL_SYNTHESIS_PROGRESS = auto()
    TOOL_SYNTHESIS_COMPLETE = auto()
    TOOL_SYNTHESIS_FAILED = auto()
    TOOL_DISCOVERY_STARTED = auto()
    TOOL_DISCOVERY_COMPLETE = auto()


@dataclass
class Event:
    """Event data container."""

    type: EventType
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "event_id": self.event_id,
            "type": self.type.name,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
        }


class EventEmitter:
    """
    Central event bus using pub/sub pattern.

    Thread-safe singleton for cross-module communication.
    """

    _instance: EventEmitter | None = None
    _lock = threading.Lock()

    def __new__(cls) -> EventEmitter:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        # Sync callbacks - for synchronous event handlers
        self._sync_callbacks: dict[EventType, dict[str, Callable[[Event], None]]] = {}
        # Async callbacks - for asynchronous event handlers
        self._async_callbacks: dict[EventType, dict[str, Callable[[Event], Any]]] = {}
        self._lock = threading.RLock()
        self._event_history: list[Event] = []
        self._max_history = 1000
        self._initialized = True

        # Initialize callback dicts
        for event_type in EventType:
            self._sync_callbacks[event_type] = {}
            self._async_callbacks[event_type] = {}

    @classmethod
    def get_instance(cls) -> EventEmitter:
        """Get singleton instance."""
        return cls()

    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> str:
        """
        Subscribe to an event type with a synchronous callback.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is emitted

        Returns:
            Subscription ID for unsubscribing
        """
        sub_id = str(uuid.uuid4())
        with self._lock:
            self._sync_callbacks[event_type][sub_id] = callback
        logger.debug(f"Subscribed {sub_id} to {event_type.name}")
        return sub_id

    def subscribe_async(
        self,
        event_type: EventType,
        callback: Callable[[Event], Any],
    ) -> str:
        """
        Subscribe with async callback.

        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event is emitted

        Returns:
            Subscription ID for unsubscribing
        """
        sub_id = str(uuid.uuid4())
        with self._lock:
            self._async_callbacks[event_type][sub_id] = callback
        logger.debug(f"Async subscribed {sub_id} to {event_type.name}")
        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe()

        Returns:
            True if subscription was found and removed
        """
        with self._lock:
            for event_type in EventType:
                if subscription_id in self._sync_callbacks[event_type]:
                    del self._sync_callbacks[event_type][subscription_id]
                    logger.debug(f"Unsubscribed {subscription_id} from {event_type.name}")
                    return True
                if subscription_id in self._async_callbacks[event_type]:
                    del self._async_callbacks[event_type][subscription_id]
                    logger.debug(f"Async unsubscribed {subscription_id} from {event_type.name}")
                    return True
        return False

    def emit(
        self,
        event_type: EventType,
        data: dict[str, Any],
        source: str = "",
    ) -> Event:
        """
        Emit an event to all subscribers.

        Args:
            event_type: Type of event
            data: Event payload
            source: Source module name

        Returns:
            The emitted event
        """
        event = Event(
            type=event_type,
            data=data,
            source=source,
        )

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Call sync callbacks
        with self._lock:
            sync_cbs = self._sync_callbacks[event_type].copy()
            async_cbs = self._async_callbacks[event_type].copy()

        for sub_id, callback in sync_cbs.items():
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber {sub_id}: {e}")

        # Log for async callbacks (they'll be called via asyncio)
        if async_cbs:
            logger.debug(f"Event {event_type.name} has {len(async_cbs)} async callbacks")

        logger.debug(f"Emitted {event_type.name} to {len(sync_cbs)} sync callbacks")
        return event

    async def emit_async(
        self,
        event_type: EventType,
        data: dict[str, Any],
        source: str = "",
    ) -> Event:
        """
        Emit an event asynchronously, calling all async callbacks.

        Args:
            event_type: Type of event
            data: Event payload
            source: Source module name

        Returns:
            The emitted event
        """
        event = self.emit(event_type, data, source)

        # Call async callbacks
        with self._lock:
            async_cbs = self._async_callbacks[event_type].copy()

        for sub_id, callback in async_cbs.items():
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in async subscriber {sub_id}: {e}")

        return event

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by type (None for all)
            limit: Maximum events to return

        Returns:
            List of events (newest first)
        """
        events = self._event_history.copy()
        if event_type:
            events = [e for e in events if e.type == event_type]
        return list(reversed(events[-limit:]))

    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()

    def subscriber_count(self, event_type: EventType | None = None) -> int:
        """Count subscribers."""
        if event_type:
            return len(self._sync_callbacks[event_type]) + len(self._async_callbacks[event_type])
        return sum(
            len(self._sync_callbacks[et]) + len(self._async_callbacks[et]) for et in EventType
        )


# Global instance getter
def get_event_emitter() -> EventEmitter:
    """Get the global EventEmitter instance."""
    return EventEmitter.get_instance()
