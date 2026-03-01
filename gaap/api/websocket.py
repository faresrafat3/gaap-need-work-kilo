"""
WebSocket Manager for Real-time Updates
=======================================

Manages WebSocket connections for broadcasting events to clients.

Usage:
    from gaap.api.websocket import manager

    # In FastAPI endpoint
    @router.websocket("/ws/events")
    async def events_endpoint(websocket: WebSocket):
        await manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                # Handle client messages
        except WebSocketDisconnect:
            manager.disconnect(websocket)

    # Broadcast to all clients
    await manager.broadcast({"type": "update", "data": {...}})
"""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from datetime import datetime
from typing import Any
from weakref import WeakSet

from fastapi import WebSocket

from gaap.core.events import Event, EventEmitter, EventType

logger = logging.getLogger("gaap.api.websocket")


class ConnectionManager:
    """
    Manages WebSocket connections with event broadcasting.

    Features:
    - Multiple connection support
    - Channel-based subscriptions
    - Automatic event broadcasting
    - Connection health monitoring
    """

    def __init__(self) -> None:
        self._connections: dict[str, WeakSet[WebSocket]] = {
            "events": WeakSet(),
            "ooda": WeakSet(),
            "steering": WeakSet(),
        }

        self._all_connections: WeakSet[WebSocket] = WeakSet()

        self._emitter = EventEmitter.get_instance()
        self._subscription_id: str | None = None

        self._connection_info: dict[int, dict[str, Any]] = {}

    async def connect(
        self,
        websocket: WebSocket,
        channel: str = "events",
    ) -> None:
        """
        Accept and register a WebSocket connection.

        Args:
            websocket: WebSocket connection
            channel: Channel to subscribe to (events, ooda, steering)
        """
        await websocket.accept()

        if channel not in self._connections:
            self._connections[channel] = WeakSet()

        self._connections[channel].add(websocket)
        self._all_connections.add(websocket)

        conn_id = id(websocket)
        self._connection_info[conn_id] = {
            "channel": channel,
            "connected_at": datetime.now().isoformat(),
        }

        logger.info(f"WebSocket connected to channel '{channel}' (total: {self.connection_count})")

        if self._subscription_id is None:
            self._subscription_id = self._emitter.subscribe_async(
                EventType.CONFIG_CHANGED,
                self._on_any_event,
            )

    def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection.

        Args:
            websocket: WebSocket to disconnect
        """
        conn_id = id(websocket)

        for channel_connections in self._connections.values():
            with suppress(KeyError):
                channel_connections.discard(websocket)

        with suppress(KeyError):
            del self._connection_info[conn_id]

        logger.info(f"WebSocket disconnected (total: {self.connection_count})")

    async def broadcast(
        self,
        message: dict[str, Any],
        channel: str = "events",
    ) -> int:
        """
        Broadcast message to all connections in a channel.

        Args:
            message: Message to broadcast
            channel: Target channel

        Returns:
            Number of clients that received the message
        """
        if channel not in self._connections:
            return 0

        connections = list(self._connections[channel])
        if not connections:
            return 0

        message_json = json.dumps(message)
        sent_count = 0

        for connection in connections:
            try:
                await connection.send_text(message_json)
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send to connection: {e}")
                self.disconnect(connection)

        return sent_count

    async def broadcast_event(self, event: Event) -> None:
        """
        Broadcast an Event to all connections.

        Args:
            event: Event to broadcast
        """
        message = event.to_dict()
        await self.broadcast(message, "events")

        if event.type in (
            EventType.OODA_PHASE,
            EventType.OODA_ITERATION,
            EventType.OODA_COMPLETE,
        ):
            await self.broadcast(message, "ooda")

    async def send_personal(
        self,
        websocket: WebSocket,
        message: dict[str, Any],
    ) -> bool:
        """
        Send message to a specific connection.

        Args:
            websocket: Target connection
            message: Message to send

        Returns:
            True if sent successfully
        """
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.warning(f"Failed to send personal message: {e}")
            self.disconnect(websocket)
            return False

    async def _on_any_event(self, event: Event) -> None:
        """Handle events from EventEmitter."""
        await self.broadcast_event(event)

    @property
    def connection_count(self) -> int:
        """Total number of active connections."""
        return len(self._all_connections)

    def get_channel_count(self, channel: str) -> int:
        """Get connection count for a specific channel."""
        if channel not in self._connections:
            return 0
        return len(self._connections[channel])

    def get_connection_info(self, websocket: WebSocket) -> dict[str, Any] | None:
        """Get metadata for a connection."""
        return self._connection_info.get(id(websocket))

    async def close_all(self) -> None:
        """Close all connections."""
        for connection in list(self._all_connections):
            with suppress(Exception):
                await connection.close()

        self._connections.clear()
        self._all_connections.clear()
        self._connection_info.clear()


manager = ConnectionManager()
