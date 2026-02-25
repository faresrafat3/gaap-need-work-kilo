"""
Unit tests for GAAP Event System

Tests EventEmitter, Event, and EventType from gaap.core.events.
"""

import asyncio
import threading
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaap.core.events import Event, EventEmitter, EventType, get_event_emitter


@pytest.fixture(autouse=True)
def reset_event_emitter():
    """Reset EventEmitter singleton before and after each test."""
    EventEmitter._instance = None
    yield
    EventEmitter._instance = None


class TestEventEmitterSingleton:
    """Test EventEmitter singleton pattern."""

    def test_singleton_pattern(self):
        """Test that EventEmitter follows singleton pattern."""
        instance1 = EventEmitter()
        instance2 = EventEmitter()
        assert instance1 is instance2

    def test_get_instance(self):
        """Test get_instance returns singleton."""
        instance = EventEmitter.get_instance()
        assert isinstance(instance, EventEmitter)

    def test_same_instance(self):
        """Test that get_instance and constructor return same instance."""
        instance1 = EventEmitter.get_instance()
        instance2 = EventEmitter()
        assert instance1 is instance2


class TestEventSubscription:
    """Test event subscription functionality."""

    def test_subscribe_single(self):
        """Test subscribing a single callback."""
        emitter = EventEmitter()
        callback = MagicMock()
        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        assert sub_id is not None
        assert isinstance(sub_id, str)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

    def test_subscribe_multiple(self):
        """Test subscribing multiple callbacks to same event."""
        emitter = EventEmitter()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        sub_id1 = emitter.subscribe(EventType.CONFIG_CHANGED, callback1)
        sub_id2 = emitter.subscribe(EventType.CONFIG_CHANGED, callback2)
        sub_id3 = emitter.subscribe(EventType.CONFIG_CHANGED, callback3)

        assert sub_id1 != sub_id2 != sub_id3
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 3

    def test_subscribe_async(self):
        """Test subscribing with async callback."""
        emitter = EventEmitter()
        callback = AsyncMock()
        sub_id = emitter.subscribe_async(EventType.CONFIG_CHANGED, callback)

        assert sub_id is not None
        assert isinstance(sub_id, str)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

    def test_unsubscribe(self):
        """Test unsubscribing a callback."""
        emitter = EventEmitter()
        callback = MagicMock()
        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

        result = emitter.unsubscribe(sub_id)
        assert result is True
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 0

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing a non-existent subscription."""
        emitter = EventEmitter()
        result = emitter.unsubscribe("non-existent-id")
        assert result is False

    def test_subscriber_count(self):
        """Test subscriber count across event types."""
        emitter = EventEmitter()
        callback = MagicMock()

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        emitter.subscribe(EventType.HEALING_STARTED, callback)
        emitter.subscribe(EventType.RESEARCH_STARTED, callback)

        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1
        assert emitter.subscriber_count(EventType.HEALING_STARTED) == 1
        assert emitter.subscriber_count() == 3


class TestEventEmission:
    """Test event emission functionality."""

    def test_emit_basic(self):
        """Test basic event emission."""
        emitter = EventEmitter()
        callback = MagicMock()
        emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})

        assert event.type == EventType.CONFIG_CHANGED
        assert event.data == {"key": "value"}
        callback.assert_called_once_with(event)

    def test_emit_with_data(self):
        """Test event emission with complex data."""
        emitter = EventEmitter()
        callback = MagicMock()
        emitter.subscribe(EventType.HEALING_STARTED, callback)

        data = {
            "module": "healing",
            "level": 1,
            "details": {"error": "timeout", "retry_count": 3},
        }
        event = emitter.emit(EventType.HEALING_STARTED, data, source="test_module")

        assert event.data == data
        assert event.source == "test_module"
        callback.assert_called_once()

    async def test_emit_async(self):
        """Test async event emission."""
        emitter = EventEmitter()
        callback = AsyncMock()
        emitter.subscribe_async(EventType.CONFIG_CHANGED, callback)

        event = await emitter.emit_async(EventType.CONFIG_CHANGED, {"key": "value"})

        assert event.type == EventType.CONFIG_CHANGED
        callback.assert_called_once_with(event)

    def test_emit_to_multiple_subscribers(self):
        """Test emitting to multiple subscribers."""
        emitter = EventEmitter()
        callback1 = MagicMock()
        callback2 = MagicMock()
        callback3 = MagicMock()

        emitter.subscribe(EventType.CONFIG_CHANGED, callback1)
        emitter.subscribe(EventType.CONFIG_CHANGED, callback2)
        emitter.subscribe(EventType.CONFIG_CHANGED, callback3)

        event = emitter.emit(EventType.CONFIG_CHANGED, {"test": True})

        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        callback3.assert_called_once_with(event)

    def test_emit_no_subscribers(self):
        """Test emitting when there are no subscribers."""
        emitter = EventEmitter()
        event = emitter.emit(EventType.CONFIG_CHANGED, {"test": True})

        assert event.type == EventType.CONFIG_CHANGED
        assert event.data == {"test": True}


class TestEventHistory:
    """Test event history functionality."""

    def test_get_history(self):
        """Test getting event history."""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"id": 1})
        emitter.emit(EventType.HEALING_STARTED, {"id": 2})
        emitter.emit(EventType.RESEARCH_STARTED, {"id": 3})

        history = emitter.get_history()
        assert len(history) == 3

    def test_get_history_by_type(self):
        """Test filtering history by event type."""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"id": 1})
        emitter.emit(EventType.HEALING_STARTED, {"id": 2})
        emitter.emit(EventType.CONFIG_CHANGED, {"id": 3})
        emitter.emit(EventType.HEALING_SUCCESS, {"id": 4})

        config_history = emitter.get_history(EventType.CONFIG_CHANGED)
        assert len(config_history) == 2
        assert all(e.type == EventType.CONFIG_CHANGED for e in config_history)

    def test_get_history_limit(self):
        """Test history limit parameter."""
        emitter = EventEmitter()

        for i in range(20):
            emitter.emit(EventType.CONFIG_CHANGED, {"id": i})

        history = emitter.get_history(limit=5)
        assert len(history) == 5

    def test_clear_history(self):
        """Test clearing event history."""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"id": 1})
        emitter.emit(EventType.CONFIG_CHANGED, {"id": 2})

        assert len(emitter.get_history()) == 2

        emitter.clear_history()
        assert len(emitter.get_history()) == 0

    def test_max_history_size(self):
        """Test that history respects max size."""
        emitter = EventEmitter()
        emitter._max_history = 10

        for i in range(15):
            emitter.emit(EventType.CONFIG_CHANGED, {"id": i})

        assert len(emitter._event_history) == 10


class TestEventClass:
    """Test Event dataclass."""

    def test_event_creation(self):
        """Test creating an Event instance."""
        event = Event(
            type=EventType.CONFIG_CHANGED,
            data={"key": "value"},
            source="test_module",
        )

        assert event.type == EventType.CONFIG_CHANGED
        assert event.data == {"key": "value"}
        assert event.source == "test_module"

    def test_event_to_dict(self):
        """Test Event serialization to dict."""
        event = Event(
            type=EventType.HEALING_STARTED,
            data={"level": 1},
            source="healing_module",
        )

        result = event.to_dict()

        assert result["type"] == "HEALING_STARTED"
        assert result["data"] == {"level": 1}
        assert result["source"] == "healing_module"
        assert "event_id" in result
        assert "timestamp" in result

    def test_event_timestamp_auto(self):
        """Test that timestamp is auto-generated."""
        before = datetime.now()
        event = Event(type=EventType.CONFIG_CHANGED, data={})
        after = datetime.now()

        assert before <= event.timestamp <= after

    def test_event_id_auto(self):
        """Test that event_id is auto-generated."""
        event1 = Event(type=EventType.CONFIG_CHANGED, data={})
        event2 = Event(type=EventType.CONFIG_CHANGED, data={})

        assert event1.event_id != event2.event_id
        assert isinstance(event1.event_id, str)


class TestEventType:
    """Test EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types exist."""
        expected_types = [
            "CONFIG_CHANGED",
            "CONFIG_VALIDATED",
            "OODA_PHASE",
            "OODA_ITERATION",
            "OODA_COMPLETE",
            "HEALING_STARTED",
            "HEALING_LEVEL",
            "HEALING_SUCCESS",
            "HEALING_FAILED",
            "RESEARCH_STARTED",
            "RESEARCH_PROGRESS",
            "RESEARCH_SOURCE_FOUND",
            "RESEARCH_HYPOTHESIS",
            "RESEARCH_COMPLETE",
            "PROVIDER_STATUS",
            "PROVIDER_ERROR",
            "PROVIDER_SWITCHED",
            "BUDGET_ALERT",
            "BUDGET_UPDATE",
            "SESSION_CREATED",
            "SESSION_UPDATE",
            "SESSION_PAUSED",
            "SESSION_RESUMED",
            "SESSION_COMPLETED",
            "STEERING_COMMAND",
            "STEERING_PAUSE",
            "STEERING_RESUME",
            "STEERING_VETO",
            "SYSTEM_ERROR",
            "SYSTEM_WARNING",
            "SYSTEM_HEALTH",
            "TOOL_SYNTHESIS_STARTED",
            "TOOL_SYNTHESIS_PROGRESS",
            "TOOL_SYNTHESIS_COMPLETE",
            "TOOL_SYNTHESIS_FAILED",
            "TOOL_DISCOVERY_STARTED",
            "TOOL_DISCOVERY_COMPLETE",
        ]

        for event_name in expected_types:
            assert hasattr(EventType, event_name), f"EventType.{event_name} missing"

    def test_event_type_count(self):
        """Test the total count of event types."""
        count = len(EventType)
        assert count == 37


class TestThreadSafety:
    """Test thread safety of EventEmitter."""

    def test_concurrent_subscribe(self):
        """Test concurrent subscriptions are thread-safe."""
        emitter = EventEmitter()
        callback = MagicMock()
        sub_ids = []
        errors = []

        def subscribe_worker():
            try:
                for _ in range(100):
                    sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
                    sub_ids.append(sub_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=subscribe_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sub_ids) == 500
        assert len(set(sub_ids)) == 500

    def test_concurrent_emit(self):
        """Test concurrent emissions are thread-safe."""
        emitter = EventEmitter()
        call_count = []
        errors = []

        def callback(event):
            call_count.append(1)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        def emit_worker():
            try:
                for _ in range(50):
                    emitter.emit(EventType.CONFIG_CHANGED, {"test": True})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(call_count) == 200

    def test_concurrent_unsubscribe(self):
        """Test concurrent unsubscriptions are thread-safe."""
        emitter = EventEmitter()
        callback = MagicMock()
        errors = []
        results = []

        sub_ids = [emitter.subscribe(EventType.CONFIG_CHANGED, callback) for _ in range(100)]

        def unsubscribe_worker(ids_chunk):
            try:
                for sub_id in ids_chunk:
                    result = emitter.unsubscribe(sub_id)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        chunk_size = 20
        threads = [
            threading.Thread(target=unsubscribe_worker, args=(sub_ids[i : i + chunk_size],))
            for i in range(0, 100, chunk_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)


class TestErrorHandling:
    """Test error handling in EventEmitter."""

    def test_subscriber_exception_isolated(self):
        """Test that subscriber exceptions don't affect other subscribers."""
        emitter = EventEmitter()

        def failing_callback(event):
            raise ValueError("Test error")

        success_callback = MagicMock()
        another_callback = MagicMock()

        emitter.subscribe(EventType.CONFIG_CHANGED, failing_callback)
        emitter.subscribe(EventType.CONFIG_CHANGED, success_callback)
        emitter.subscribe(EventType.CONFIG_CHANGED, another_callback)

        emitter.emit(EventType.CONFIG_CHANGED, {"test": True})

        success_callback.assert_called_once()
        another_callback.assert_called_once()

    def test_invalid_event_type(self):
        """Test behavior with valid EventType enum."""
        emitter = EventEmitter()
        callback = MagicMock()

        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        assert sub_id is not None

        event = emitter.emit(EventType.CONFIG_CHANGED, {"test": True})
        assert event.type == EventType.CONFIG_CHANGED

    def test_callback_error_handling(self):
        """Test that errors in callbacks are logged but not raised."""
        emitter = EventEmitter()
        error_callback = MagicMock(side_effect=RuntimeError("Callback error"))
        success_callback = MagicMock()

        emitter.subscribe(EventType.CONFIG_CHANGED, error_callback)
        emitter.subscribe(EventType.CONFIG_CHANGED, success_callback)

        event = emitter.emit(EventType.CONFIG_CHANGED, {"test": True})

        error_callback.assert_called_once()
        success_callback.assert_called_once()
        assert event.type == EventType.CONFIG_CHANGED


class TestGetEventEmitter:
    """Test get_event_emitter helper function."""

    def test_get_event_emitter_returns_instance(self):
        """Test that get_event_emitter returns EventEmitter instance."""
        emitter = get_event_emitter()
        assert isinstance(emitter, EventEmitter)

    def test_get_event_emitter_returns_singleton(self):
        """Test that get_event_emitter returns singleton."""
        emitter1 = get_event_emitter()
        emitter2 = get_event_emitter()
        assert emitter1 is emitter2
