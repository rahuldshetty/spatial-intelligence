"""Fan-out of server events to SSE subscribers."""

from __future__ import annotations

import queue
import threading


class EventBus:
    """Publishes ``(event, data)`` pairs to every subscribed SSE consumer.

    Thread-safe and independent of the session lock: tools report progress from
    worker threads while a run holds nothing, so publishing must never block on
    notebook state.
    """

    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        """Register a new subscriber queue and return it."""
        subscriber: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        """Deregister a subscriber; unknown queues are ignored."""
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def subscriber_count(self) -> int:
        """Return how many SSE consumers are currently attached."""
        with self._lock:
            return len(self._subscribers)

    def publish(self, event: str, data: dict) -> None:
        """Deliver one event to every subscriber, dropping dead queues."""
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put({"event": event, "data": data})
            except Exception:
                with self._lock:
                    try:
                        self._subscribers.remove(subscriber)
                    except ValueError:
                        pass
