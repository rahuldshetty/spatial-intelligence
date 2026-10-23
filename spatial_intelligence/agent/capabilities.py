"""Agent capabilities: the harness hooks this app needs.

Both capabilities exist because a tool failure and a provider quirk must not
kill a run. :class:`ToolFailurePolicy` decides what a failing tool call means —
retry it in place when the network blipped, hand the model the error as the
call's result when it did not — and :class:`NormalizeDuplicateToolNames`
normalizes the ``name__name`` alias some compatible models emit before dispatch.
"""

from __future__ import annotations

import asyncio
import errno
import http.client
import socket
import ssl
import urllib.error
from contextlib import nullcontext
from dataclasses import replace

from pydantic_ai import ToolFailed
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ToolCallPart

from ..map import MAP_LOCK
from ..tools.runtime import ToolRuntime

#: Total attempts for a tool call that keeps failing with a transient error.
TRANSIENT_TOOL_ATTEMPTS = 3
#: Backoff before the second attempt; each further attempt doubles it.
TRANSIENT_TOOL_BACKOFF_SECONDS = 0.5
#: Upper bound on one transient-tool backoff.
MAX_TRANSIENT_TOOL_BACKOFF_SECONDS = 4.0

#: HTTP statuses that mean "the server is briefly unhappy"; a 404 or a 403 is
#: the server's answer about the resource, not a blip.
TRANSIENT_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 507, 509})


def _errnos(*names: str) -> frozenset[int]:
    """Collect the ``errno`` constants that exist on this platform."""
    return frozenset(
        value for name in names if isinstance(value := getattr(errno, name, None), int)
    )


#: ``errno`` values that mean the remote end or the network went away.
TRANSIENT_ERRNOS = _errnos(
    "ECONNABORTED",
    "ECONNREFUSED",
    "ECONNRESET",
    "EHOSTDOWN",
    "EHOSTUNREACH",
    "ENETDOWN",
    "ENETRESET",
    "ENETUNREACH",
    "EPIPE",
    "ETIMEDOUT",
    "EAGAIN",
    "EWOULDBLOCK",
    "EINPROGRESS",
)


def is_transient_tool_error(error: BaseException) -> bool:
    """Whether re-running the same tool call could plausibly succeed.

    Covers the transport failures a network tool can hit: a dropped or refused
    connection, a timeout, a TLS handshake failure, a truncated response, and
    the statuses that mean "the server is busy". Everything else — a missing
    file, a rejected argument, an HTTP 404 — is the tool's answer about the
    request, and repeating it would only waste a round trip.

    The exception chain is walked because the libraries wrap the original
    failure: ``urllib`` re-raises as ``URLError``, and geospatial readers keep
    the transport error as the ``__cause__``.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _is_transient(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_transient(error: BaseException) -> bool:
    """Classify one link of the exception chain."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in TRANSIENT_HTTP_STATUSES
    if isinstance(error, (urllib.error.URLError, http.client.HTTPException)):
        # A URLError carries its reason: a name that does not resolve is a typo
        # in the URL, not a blip, so asking again would fail the same way.
        return not isinstance(getattr(error, "reason", None), socket.gaierror)
    if isinstance(error, ssl.SSLCertVerificationError):
        return False
    if isinstance(error, (ConnectionError, TimeoutError, ssl.SSLError)):
        return True
    if isinstance(error, OSError):
        return error.errno in TRANSIENT_ERRNOS
    return False


def tool_failure_text(error: BaseException, *, attempts: int = 1) -> str:
    """Render a tool failure for the model.

    A transient failure that reached the model went through every attempt, so
    the message says so: the model should treat the tool as unreachable rather
    than repeat the call as if the arguments were at fault.
    """
    detail = f"{type(error).__name__}: {error}"
    if attempts > 1 and is_transient_tool_error(error):
        return (
            f"{detail} — transient remote failure, already retried "
            f"{attempts - 1} time(s); the endpoint is still unavailable."
        )
    return detail


class ToolFailurePolicy(AbstractCapability[ToolRuntime]):
    """Turn a failing tool call into something the model can act on.

    By default pydantic-ai lets a raised tool exception abort the run, and it
    answers a rejected tool call with a retry prompt charged against the tool's
    retry budget — which, once exhausted, aborts the run too. Neither is right
    here: the tools raise ordinary exceptions for recoverable conditions (bounds
    outside the Web Mercator range, a file too large to read whole, a URL that
    does not resolve), and a model that mistypes an argument has not earned a
    dead prompt.

    So every failure is classified:

    - **transient** (a dropped connection, a timeout, a 5xx) is retried in
      place, with bounded backoff, before the model sees anything — the blip
      costs no round trip and no context;
    - **deterministic** (bad arguments, a missing file, a refused download, a
      sandbox violation) becomes the call's failed result: the model reads the
      reason and adapts in the same run, with no retry and no retry budget
      spent, so it can never exhaust the budget and abort the prompt.

    Argument validation failures are the same thing said earlier: the model gets
    the schema error as the call's result instead of a retry prompt. The retry
    budget is left to pydantic-ai, which only needs it for a call this policy
    never sees — a tool the model is not allowed to call yet.
    """

    async def wrap_tool_execute(self, ctx, *, call, tool_def, args, handler):
        """Run the tool, retrying a transient failure in place.

        The final attempt sits outside the loop, so a transient failure is
        always given every attempt before it is allowed to reach the model.
        Map-writing tools additionally take the process-wide map lock: the
        browser can push a project at any time, and every such mutation is a
        read-copy-reassign that would otherwise lose a concurrent layer.
        """
        effects = set((tool_def.metadata or {}).get("si_effects") or ())
        guard = MAP_LOCK if "map_write" in effects else nullcontext()
        with guard:
            for attempt in range(1, TRANSIENT_TOOL_ATTEMPTS):
                try:
                    return await handler(args)
                except Exception as error:  # noqa: BLE001 - classified on the next line
                    # Control flow (deferrals, retries, cancellation) is never
                    # transient, so it passes through here untouched.
                    if not is_transient_tool_error(error):
                        raise
                    await asyncio.sleep(
                        min(
                            TRANSIENT_TOOL_BACKOFF_SECONDS * 2 ** (attempt - 1),
                            MAX_TRANSIENT_TOOL_BACKOFF_SECONDS,
                        )
                    )
            return await handler(args)

    async def on_tool_execute_error(self, ctx, *, call, tool_def, args, error):
        """Report a raised tool error as the call's failed result."""
        raise ToolFailed(tool_failure_text(error, attempts=TRANSIENT_TOOL_ATTEMPTS))

    async def on_tool_validate_error(self, ctx, *, call, tool_def, args, error):
        """Report rejected arguments as the call's failed result, not a retry."""
        raise ToolFailed(f"{call.tool_name} rejected these arguments: {error}")


class NormalizeDuplicateToolNames(AbstractCapability[ToolRuntime]):
    """Correct the ``name__name`` alias emitted by some compatible models."""

    async def after_model_request(self, ctx, *, request_context, response):
        parts = [
            replace(part, tool_name=deduplicate_tool_name(part.tool_name))
            if isinstance(part, ToolCallPart)
            else part
            for part in response.parts
        ]
        return replace(response, parts=parts)


def deduplicate_tool_name(name: str) -> str:
    """Collapse ``tool__tool`` into ``tool``; leave other names untouched."""
    left, separator, right = name.partition("__")
    return left if separator and left == right else name


__all__ = [
    "MAX_TRANSIENT_TOOL_BACKOFF_SECONDS",
    "TRANSIENT_HTTP_STATUSES",
    "TRANSIENT_TOOL_ATTEMPTS",
    "TRANSIENT_TOOL_BACKOFF_SECONDS",
    "NormalizeDuplicateToolNames",
    "ToolFailurePolicy",
    "deduplicate_tool_name",
    "is_transient_tool_error",
    "tool_failure_text",
]
