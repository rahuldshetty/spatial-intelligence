"""One-shot JSON queries into the live GeoLibre iframe.

The capture broker owns image returns; this broker is the text/GeoJSON path.
A tool publishes a request over SSE, the app calls the corresponding
``bridgeRequest(method, params)``, and posts the JSON result back. Like the
capture broker, it never holds the session lock while waiting.
"""

from __future__ import annotations

import secrets
import threading
import uuid
from typing import Any

from ..contracts.errors import ToolInputError
from .bus import EventBus

#: How long a tool waits for the browser to answer one map query.
QUERY_TIMEOUT = 20.0


class _Pending:
    def __init__(self, token: str) -> None:
        self.token = token
        self.event = threading.Event()
        self.value: Any = None
        self.error: str | None = None


class MapBridgeBroker:
    """Thread-safe request/response bridge for JSON map queries."""

    def __init__(self, bus: EventBus, *, timeout: float = QUERY_TIMEOUT) -> None:
        self._bus = bus
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pending: dict[str, _Pending] = {}

    def request(
        self,
        method: str,
        params: dict | None = None,
        *,
        workspace: str | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Ask the browser for one map query result and block until it answers."""
        if self._bus.subscriber_count() == 0:
            raise ToolInputError(
                "the live map is not available: open the app before reading features "
                "the user drew"
            )
        request_id = uuid.uuid4().hex
        pending = _Pending(secrets.token_urlsafe(18))
        with self._lock:
            self._pending[request_id] = pending
        self._bus.publish(
            "map_bridge",
            {
                "id": request_id,
                "token": pending.token,
                "method": method,
                "params": params or {},
                "workspace": workspace,
            },
        )
        wait = self._timeout if timeout is None else max(1.0, float(timeout))
        if not pending.event.wait(wait):
            with self._lock:
                self._pending.pop(request_id, None)
            raise ToolInputError(
                "the map did not answer the query; make sure the app is open and "
                "the map has finished loading"
            )
        if pending.error:
            raise ToolInputError(pending.error)
        return pending.value

    def complete(
        self,
        request_id: str,
        token: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> bool:
        """Resolve one pending query; return whether it was accepted."""
        with self._lock:
            pending = self._pending.get(request_id)
        if pending is None or not secrets.compare_digest(pending.token, token):
            return False
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return False
        if error:
            pending.error = error
        else:
            pending.value = result
        pending.event.set()
        return True

    def cancel_all(self, reason: str = "the map was closed") -> None:
        """Fail every waiting query (workspace switch or shutdown)."""
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for item in pending:
            item.error = reason
            item.event.set()


__all__ = ["MapBridgeBroker", "QUERY_TIMEOUT"]
