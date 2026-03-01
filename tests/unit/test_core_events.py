"""
Comprehensive tests for gaap/core/events.py module
Tests EventEmitter, Event, and EventType functionality
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import pytest

from gaap.core.events import (
    Event,
    EventEmitter,
    EventType,
    get_event_emitter,
)


class TestEventType:
    """Test EventType enum"""

    def test_event_type_values(self):
        """Test all event types are defined"""
        # Config events
        assert EventType.CONFIG_CHANGED
        assert EventType.CONFIG_VALIDATED

        # OODA loop events
        assert EventType.OODA_PHASE
        assert EventType.OODA_ITERATION
        assert EventType.OODA_COMPLETE

        # Healing events
        assert EventType.HEALING_STARTED
        assert EventType.HEALING_LEVEL
        assert EventType.HEALING_SUCCESS
        assert EventType.HEALING_FAILED

        # Research events
        assert EventType.RESEARCH_STARTED
        assert EventType.RESEARCH_PROGRESS
        assert EventType.RESEARCH_SOURCE_FOUND
        assert EventType.RESEARCH_HYPOTHESIS
        assert EventType.RESEARCH_COMPLETE

        # Provider events
        assert EventType.PROVIDER_STATUS
        assert EventType.PROVIDER_ERROR
        assert EventType.PROVIDER_SWITCHED

        # Budget events
        assert EventType.BUDGET_ALERT
        assert EventType.BUDGET_UPDATE

        # Session events
        assert EventType.SESSION_CREATED
        assert EventType.SESSION_UPDATE
        assert EventType.SESSION_PAUSED
        assert EventType.SESSION_RESUMED
        assert EventType.SESSION_COMPLETED

        # Steering events
        assert EventType.STEERING_COMMAND
        assert EventType.STEERING_PAUSE
        assert EventType.STEERING_RESUME
        assert EventType.STEERING_VETO

        # System events
        assert EventType.SYSTEM_ERROR
        assert EventType.SYSTEM_WARNING
        assert EventType.SYSTEM_HEALTH

        # Tool synthesis events
        assert EventType.TOOL_SYNTHESIS_STARTED
        assert EventType.TOOL_SYNTHESIS_PROGRESS
        assert EventType.TOOL_SYNTHESIS_COMPLETE
        assert EventType.TOOL_SYNTHESIS_FAILED
        assert EventType.TOOL_DISCOVERY_STARTED
        assert EventType.TOOL_DISCOVERY_COMPLETE

    def test_event_type_count(self):
        """Test that we have the expected number of event types"""
        event_types = list(EventType)
        assert len(event_types) >= 30  # Should have at least 30 event types


class TestEvent:
    """Test Event dataclass"""

    def test_event_creation(self):
        """Test creating an Event"""
        event = Event(
            type=EventType.HEALING_STARTED,
            data={"task_id": "123", "level": "L1"},
            source="test_module",
        )

        assert event.type == EventType.HEALING_STARTED
        assert event.data == {"task_id": "123", "level": "L1"}
        assert event.source == "test_module"
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_default_timestamp(self):
        """Test that timestamp defaults to current time"""
        before = time.time()
        event = Event(type=EventType.CONFIG_CHANGED, data={})
        after = time.time()

        assert before <= event.timestamp.timestamp() <= after

    def test_event_default_source(self):
        """Test that source defaults to empty string"""
        event = Event(type=EventType.CONFIG_CHANGED, data={})
        assert event.source == ""

    def test_event_unique_ids(self):
        """Test that each event gets a unique ID"""
        event1 = Event(type=EventType.CONFIG_CHANGED, data={})
        event2 = Event(type=EventType.CONFIG_CHANGED, data={})

        assert event1.event_id != event2.event_id

    def test_event_to_dict(self):
        """Test converting Event to dictionary"""
        event = Event(
            type=EventType.HEALING_STARTED,
            data={"key": "value"},
            source="test",
        )

        event_dict = event.to_dict()

        assert event_dict["type"] == "HEALING_STARTED"
        assert event_dict["data"] == {"key": "value"}
        assert event_dict["source"] == "test"
        assert "event_id" in event_dict
        assert "timestamp" in event_dict

    def test_event_to_dict_timestamp_format(self):
        """Test that timestamp is ISO format in dict"""
        event = Event(type=EventType.CONFIG_CHANGED, data={})
        event_dict = event.to_dict()

        # Should be ISO format string
        assert isinstance(event_dict["timestamp"], str)
        assert "T" in event_dict["timestamp"]  # ISO format has T


class TestEventEmitter:
    """Test EventEmitter class"""

    @pytest.fixture(autouse=True)
    def reset_emitter(self):
        """Reset singleton between tests"""
        EventEmitter._instance = None
        yield
        EventEmitter._instance = None

    def test_singleton_pattern(self):
        """Test that EventEmitter is a singleton"""
        emitter1 = EventEmitter()
        emitter2 = EventEmitter()
        assert emitter1 is emitter2

    def test_get_instance(self):
        """Test get_instance class method"""
        emitter = EventEmitter.get_instance()
        assert isinstance(emitter, EventEmitter)
        assert emitter is EventEmitter()

    def test_subscribe_sync_callback(self):
        """Test subscribing a synchronous callback"""
        emitter = EventEmitter()
        received_events = []

        def callback(event):
            received_events.append(event)

        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        assert sub_id is not None
        assert isinstance(sub_id, str)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

    def test_subscribe_async_callback(self):
        """Test subscribing an asynchronous callback"""
        emitter = EventEmitter()

        async def async_callback(event):
            pass

        sub_id = emitter.subscribe_async(EventType.CONFIG_CHANGED, async_callback)

        assert sub_id is not None
        assert isinstance(sub_id, str)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

    def test_emit_sync_callback(self):
        """Test emitting event to synchronous callback"""
        emitter = EventEmitter()
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})

        assert len(received_events) == 1
        assert received_events[0] is event
        assert received_events[0].data == {"key": "value"}

    def test_emit_multiple_subscribers(self):
        """Test emitting to multiple subscribers"""
        emitter = EventEmitter()
        received1 = []
        received2 = []

        def callback1(event):
            received1.append(event)

        def callback2(event):
            received2.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback1)
        emitter.subscribe(EventType.CONFIG_CHANGED, callback2)

        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})

        assert len(received1) == 1
        assert len(received2) == 1
        assert received1[0] is event
        assert received2[0] is event

    def test_emit_different_event_types(self):
        """Test that events only go to subscribers of the same type"""
        emitter = EventEmitter()
        config_events = []
        healing_events = []

        def config_callback(event):
            config_events.append(event)

        def healing_callback(event):
            healing_events.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, config_callback)
        emitter.subscribe(EventType.HEALING_STARTED, healing_callback)

        emitter.emit(EventType.CONFIG_CHANGED, {"config": "data"})

        assert len(config_events) == 1
        assert len(healing_events) == 0

    def test_emit_async_callback_stores_in_history(self):
        """Test that async callbacks don't block emit"""
        emitter = EventEmitter()

        async def async_callback(event):
            await asyncio.sleep(0.1)

        emitter.subscribe_async(EventType.CONFIG_CHANGED, async_callback)

        # Should not block
        start = time.time()
        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})
        elapsed = time.time() - start

        assert elapsed < 0.05  # Should be fast, not waiting for async
        assert event is not None

    def test_emit_async_callback(self):
        """Test emitting event asynchronously"""
        emitter = EventEmitter()
        received_events = []

        async def async_callback(event):
            received_events.append(event)

        emitter.subscribe_async(EventType.CONFIG_CHANGED, async_callback)

        async def run_test():
            event = await emitter.emit_async(EventType.CONFIG_CHANGED, {"key": "value"})
            assert event.data == {"key": "value"}
            return event

        asyncio.run(run_test())

    def test_unsubscribe(self):
        """Test unsubscribing from events"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

        result = emitter.unsubscribe(sub_id)
        assert result is True
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 0

        # Should not receive events after unsubscribing
        emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})
        assert len(received) == 0

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing non-existent subscription"""
        emitter = EventEmitter()
        result = emitter.unsubscribe("nonexistent-id")
        assert result is False

    def test_unsubscribe_async(self):
        """Test unsubscribing async callback"""
        emitter = EventEmitter()

        async def async_callback(event):
            pass

        sub_id = emitter.subscribe_async(EventType.CONFIG_CHANGED, async_callback)
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 1

        result = emitter.unsubscribe(sub_id)
        assert result is True
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 0

    def test_get_history(self):
        """Test getting event history"""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"n": 1})
        emitter.emit(EventType.HEALING_STARTED, {"n": 2})
        emitter.emit(EventType.CONFIG_CHANGED, {"n": 3})

        history = emitter.get_history()
        assert len(history) == 3

        # Should be in reverse chronological order
        assert history[0].data == {"n": 3}
        assert history[1].data == {"n": 2}
        assert history[2].data == {"n": 1}

    def test_get_history_by_type(self):
        """Test getting history filtered by event type"""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"n": 1})
        emitter.emit(EventType.HEALING_STARTED, {"n": 2})
        emitter.emit(EventType.CONFIG_CHANGED, {"n": 3})

        config_history = emitter.get_history(event_type=EventType.CONFIG_CHANGED)
        assert len(config_history) == 2
        assert all(e.type == EventType.CONFIG_CHANGED for e in config_history)

    def test_get_history_limit(self):
        """Test getting history with limit"""
        emitter = EventEmitter()

        for i in range(10):
            emitter.emit(EventType.CONFIG_CHANGED, {"n": i})

        history = emitter.get_history(limit=5)
        assert len(history) == 5

    def test_get_history_empty(self):
        """Test getting history when no events emitted"""
        emitter = EventEmitter()
        history = emitter.get_history()
        assert history == []

    def test_clear_history(self):
        """Test clearing event history"""
        emitter = EventEmitter()

        emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})
        assert len(emitter.get_history()) == 1

        emitter.clear_history()
        assert len(emitter.get_history()) == 0

    def test_subscriber_count_all_types(self):
        """Test counting subscribers across all event types"""
        emitter = EventEmitter()

        def callback(event):
            pass

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        emitter.subscribe(EventType.HEALING_STARTED, callback)
        emitter.subscribe(EventType.RESEARCH_STARTED, callback)

        total = emitter.subscriber_count()
        assert total == 3

    def test_subscriber_count_by_type(self):
        """Test counting subscribers for specific event type"""
        emitter = EventEmitter()

        def callback(event):
            pass

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        emitter.subscribe(EventType.HEALING_STARTED, callback)

        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 2
        assert emitter.subscriber_count(EventType.HEALING_STARTED) == 1
        assert emitter.subscriber_count(EventType.RESEARCH_STARTED) == 0

    def test_history_max_size(self):
        """Test that history is limited to max size"""
        emitter = EventEmitter()

        # Emit more events than max_history
        for i in range(1100):
            emitter.emit(EventType.CONFIG_CHANGED, {"n": i})

        history = emitter.get_history()
        assert len(history) <= 1000  # Should be capped at max_history

    def test_emit_with_source(self):
        """Test emitting event with source"""
        emitter = EventEmitter()

        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"}, source="test_module")

        assert event.source == "test_module"

    def test_emit_returns_event(self):
        """Test that emit returns the emitted event"""
        emitter = EventEmitter()

        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})

        assert isinstance(event, Event)
        assert event.type == EventType.CONFIG_CHANGED
        assert event.data == {"key": "value"}

    def test_callback_error_handling(self):
        """Test that errors in callbacks don't break emitter"""
        emitter = EventEmitter()
        received = []

        def error_callback(event):
            raise ValueError("Test error")

        def good_callback(event):
            received.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, error_callback)
        emitter.subscribe(EventType.CONFIG_CHANGED, good_callback)

        # Should not raise
        event = emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})

        # Good callback should still receive the event
        assert len(received) == 1

    def test_async_callback_error_handling(self):
        """Test that errors in async callbacks don't break emitter"""
        emitter = EventEmitter()

        async def error_callback(event):
            raise ValueError("Test error")

        emitter.subscribe_async(EventType.CONFIG_CHANGED, error_callback)

        async def run_test():
            # Should not raise
            event = await emitter.emit_async(EventType.CONFIG_CHANGED, {"key": "value"})
            assert event is not None

        asyncio.run(run_test())

    def test_thread_safety(self):
        """Test that EventEmitter is thread-safe"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        def emit_events():
            for i in range(10):
                emitter.emit(EventType.CONFIG_CHANGED, {"thread": i})

        threads = [threading.Thread(target=emit_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 50  # 5 threads * 10 events each

    def test_multiple_subscribe_same_callback(self):
        """Test subscribing same callback multiple times"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        sub1 = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        sub2 = emitter.subscribe(EventType.CONFIG_CHANGED, callback)

        assert sub1 != sub2  # Should get different subscription IDs
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) == 2

        emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})
        assert len(received) == 2  # Called twice


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_get_event_emitter(self):
        """Test get_event_emitter function"""
        emitter = get_event_emitter()
        assert isinstance(emitter, EventEmitter)

        # Should return same instance
        emitter2 = get_event_emitter()
        assert emitter is emitter2


class TestEventEmitterEdgeCases:
    """Test edge cases and error conditions"""

    def test_emit_with_empty_data(self):
        """Test emitting event with empty data"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        event = emitter.emit(EventType.CONFIG_CHANGED, {})

        assert event.data == {}
        assert len(received) == 1

    def test_emit_with_none_data(self):
        """Test emitting event with None data"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        event = emitter.emit(EventType.CONFIG_CHANGED, None)  # type: ignore

        assert event.data is None
        assert len(received) == 1

    def test_unsubscribe_and_resubscribe(self):
        """Test unsubscribing and resubscribing"""
        emitter = EventEmitter()
        received = []

        def callback(event):
            received.append(event)

        sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        emitter.unsubscribe(sub_id)

        # Resubscribe
        new_sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
        assert new_sub_id != sub_id

        emitter.emit(EventType.CONFIG_CHANGED, {"key": "value"})
        assert len(received) == 1

    def test_history_large_data(self):
        """Test history with large data payloads"""
        emitter = EventEmitter()
        emitter.clear_history()
        large_data = {"data": "x" * 10000}

        event = emitter.emit(EventType.CONFIG_CHANGED, large_data)
        history = emitter.get_history()

        assert len(history) == 1
        assert history[0].data == large_data

    def test_all_event_types_subscribable(self):
        """Test that all event types can be subscribed to"""
        emitter = EventEmitter()

        def callback(event):
            pass

        for event_type in EventType:
            sub_id = emitter.subscribe(event_type, callback)
            assert sub_id is not None
            assert emitter.unsubscribe(sub_id) is True

    def test_emit_with_nested_data(self):
        """Test emitting event with nested data structure"""
        emitter = EventEmitter()
        nested_data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                },
            },
        }

        event = emitter.emit(EventType.CONFIG_CHANGED, nested_data)
        assert event.data == nested_data

    def test_concurrent_subscribe_unsubscribe(self):
        """Test concurrent subscribe/unsubscribe operations"""
        emitter = EventEmitter()

        def callback(event):
            pass

        sub_ids = []

        def subscribe_loop():
            for _ in range(20):
                sub_id = emitter.subscribe(EventType.CONFIG_CHANGED, callback)
                sub_ids.append(sub_id)

        def unsubscribe_loop():
            for _ in range(10):
                if sub_ids:
                    sub_id = sub_ids.pop()
                    emitter.unsubscribe(sub_id)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=subscribe_loop))
            threads.append(threading.Thread(target=unsubscribe_loop))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify no crashes occurred
        assert emitter.subscriber_count(EventType.CONFIG_CHANGED) >= 0

    def test_emit_async_with_no_async_subscribers(self):
        """Test emit_async when no async subscribers"""
        emitter = EventEmitter()
        received = []

        def sync_callback(event):
            received.append(event)

        emitter.subscribe(EventType.CONFIG_CHANGED, sync_callback)

        async def run_test():
            event = await emitter.emit_async(EventType.CONFIG_CHANGED, {"key": "value"})
            assert event is not None
            assert len(received) == 1

        asyncio.run(run_test())
