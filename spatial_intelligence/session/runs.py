"""The run queue: queued cells, cancellation tokens, and resume payloads.

One worker drains the queue, so runs are serialized (this is what keeps the
shared live map consistent), while the notebook stays editable between runs.
"""

from __future__ import annotations

import queue
import threading

from pydantic_ai import CancellationToken


class RunQueue:
    """FIFO of cells to run, with per-cell cancellation."""

    def __init__(self) -> None:
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._tokens: dict[str, CancellationToken] = {}
        self._payloads: dict[str, dict] = {}
        self._queued: set[str] = set()
        self._cancelled: set[str] = set()
        self._idle = threading.Event()
        self._idle.set()
        self._shutdown = False

    # -- producer side ---------------------------------------------------

    def submit(self, cell_id: str, resume: dict | None = None) -> None:
        """Queue a cell, optionally resuming a paused conversation."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("the run queue is shut down; no further cell can be queued")
            self._queued.add(cell_id)
            if resume is not None:
                self._payloads[cell_id] = resume
            self._idle.clear()
        self._queue.put(cell_id)

    def stop(self, cell_id: str) -> bool:
        """Cancel a running cell, or mark a queued one as cancelled."""
        with self._lock:
            token = self._tokens.get(cell_id)
            if token is not None:
                token.cancel()
                return True
            if cell_id in self._queued:
                self._cancelled.add(cell_id)
                return True
        return False

    # -- worker side -----------------------------------------------------

    def take(self) -> str | None:
        """Block for the next cell id, or ``None`` once the queue is shut down."""
        cell_id = self._queue.get()
        if cell_id is None:
            # The sentinel ends the worker, so nothing else will mark this item
            # processed; without this ``Queue.join`` is left permanently short.
            self._queue.task_done()
        return cell_id

    def shutdown(self) -> None:
        """Release the worker: its next ``take`` returns ``None``.

        A worker that is never released outlives its owner, and an idle thread
        that still holds the workspace open is what turns a later cleanup into a
        file-lock error. The server calls this on the way down; a test harness
        calls it before it deletes the workspace tree.

        Idleness is reported only for an empty queue: work still running or
        queued keeps ``wait_idle`` blocked, so a caller that waits after
        shutting down is not told the work is over while it is not.
        """
        with self._lock:
            self._shutdown = True
            drained = not self._tokens and not self._queued
        self._queue.put(None)
        if drained:
            self._idle.set()

    def start(self, cell_id: str) -> CancellationToken | None:
        """Begin a run, returning its token, or ``None`` if it was cancelled."""
        with self._lock:
            self._queued.discard(cell_id)
            if cell_id in self._cancelled:
                self._cancelled.discard(cell_id)
                if not self._tokens and not self._queued:
                    self._idle.set()
                return None
            token = CancellationToken()
            self._tokens[cell_id] = token
            return token

    def resume_payload(self, cell_id: str) -> dict | None:
        """Pop the resume payload queued with ``cell_id``."""
        with self._lock:
            return self._payloads.pop(cell_id, None)

    def finish(self, cell_id: str) -> None:
        """Release a running cell's token and clear its cancellation mark."""
        with self._lock:
            self._tokens.pop(cell_id, None)
            self._cancelled.discard(cell_id)
            self._payloads.pop(cell_id, None)
            if not self._tokens and not self._queued:
                self._idle.set()

    def task_done(self) -> None:
        """Mark one dequeued item as processed."""
        self._queue.task_done()

    # -- queries ---------------------------------------------------------

    def is_queued(self, cell_id: str) -> bool:
        """Whether ``cell_id`` is waiting to start."""
        with self._lock:
            return cell_id in self._queued

    def is_running(self, cell_id: str) -> bool:
        """Whether ``cell_id`` currently holds a cancellation token."""
        with self._lock:
            return cell_id in self._tokens

    def active(self) -> bool:
        """Whether anything is running or waiting."""
        with self._lock:
            return bool(self._tokens or self._queued)

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no cell is running or queued.

        Callers must NOT hold the session lock while waiting: the worker needs
        it for the short mutation steps of the run it is draining.
        """
        return self._idle.wait(timeout)
