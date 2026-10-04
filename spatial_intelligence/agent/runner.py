"""Running one prompt cell: streaming, retries, resume, and trace persistence.

The runner owns the agent conversation lifecycle and nothing else: every effect
it has on the rest of the app goes through :class:`RunnerHooks`, so the session
layer decides what a trace step, a recorded tool call, or a plan snapshot means.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import (
    Agent,
    CancellationToken,
    DeferredToolRequests,
    DeferredToolResults,
    ModelAPIError,
    ModelHTTPError,
    RunCancelled,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_ai_harness.planning import InMemoryPlanStore, PlanItem

from ..contracts.ids import new_id
from ..tools.registry import ToolRegistry
from ..workspace import traces
from .events import step_from_event
from .interactions import interaction_from_deferred

#: HTTP statuses worth another attempt (the provider is briefly unhappy).
TRANSIENT_HTTP_STATUSES = frozenset({408, 429})
#: Upper bound on a provider-supplied ``Retry-After``.
MAX_RETRY_AFTER_SECONDS = 30.0
#: Upper bound on exponential backoff.
MAX_BACKOFF_SECONDS = 8.0

#: pydantic-ai's retry-exhaustion failure. It is the only failure a tool call
#: can still end a run with now that
#: :class:`~spatial_intelligence.agent.capabilities.ToolFailurePolicy` answers
#: every failure it sees with a result: the model kept calling a tool it may not
#: call — a name that does not exist, or a deferred tool it never discovered.
_RETRY_EXHAUSTED = re.compile(
    r"Tool '(?P<tool>[^']+)' exceeded max retries count of (?P<count>\d+)"
)


#: Phrases a provider uses when it refuses a request for length. Narrow on
#: purpose: this only decides how a failure is reported.
CONTEXT_OVERFLOW_HINTS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "max context",
    "too many tokens",
    "reduce the length",
    "input is too long",
    "prompt is too long",
)


def is_transient_run_error(error: Exception) -> bool:
    """Return whether replaying a workspace/map-safe attempt may succeed."""
    if isinstance(error, ModelHTTPError):
        return (
            error.status_code in TRANSIENT_HTTP_STATUSES or error.status_code >= 500
        )
    # Non-HTTP ModelAPIError instances represent provider/transport failures.
    return isinstance(error, ModelAPIError)


def is_context_overflow_error(error: Exception) -> bool:
    """Whether the provider refused the request because the history is too long.

    Compaction targets half the window before every request, so a refusal means
    the window is wrong for this endpoint rather than that nothing tried.
    """
    text = str(error).lower()
    return any(hint in text for hint in CONTEXT_OVERFLOW_HINTS)


def describe_run_error(error: Exception, registry: ToolRegistry) -> str:
    """Render a failed run for the cell output.

    Only the retry-exhaustion failure needs translating: its raw text is a
    pydantic-ai retry-budget complaint with a documentation link, which tells
    the user nothing about which tool the model was stuck on or why.
    """
    match = _RETRY_EXHAUSTED.search(str(error))
    if match is None:
        return str(error)
    name = match["tool"]
    reason = (
        "a deferred tool has to be discovered with search_tools before the model may call it"
        if name in registry
        else "no tool by that name exists"
    )
    return (
        f"the model kept calling {name!r} after the {match['count']} retries of that call ran "
        f"out, and {reason}; the prompt was stopped there. The earlier steps are still in the "
        "trace."
    )


def retry_delay(error: Exception, attempt: int) -> float:
    """Return bounded provider-aware backoff seconds for a retry."""
    if isinstance(error, ModelHTTPError) and error.retry_after is not None:
        return min(error.retry_after, MAX_RETRY_AFTER_SECONDS)
    return min(0.5 * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)


def latest_plan_items(steps: list[dict]) -> list[PlanItem]:
    """Rebuild the latest plan snapshot from a prompt's persisted trace."""
    for step in reversed(steps):
        if not isinstance(step, dict) or step.get("type") != "plan":
            continue
        items = step.get("items")
        if not isinstance(items, list):
            return []
        restored = []
        for item in items:
            try:
                restored.append(PlanItem.model_validate(item))
            except (TypeError, ValueError):
                continue
        return restored
    return []


@dataclass(slots=True)
class RunOutcome:
    """What one agent run did, in the terms the notebook records."""

    status: str  # done | error | stopped | waiting_for_input
    output: str | None = None
    error: str | None = None
    usage: dict | None = None
    run_id: str | None = None
    conversation_id: str | None = None
    interaction: dict | None = None


@dataclass(slots=True)
class RunnerHooks:
    """Everything the runner needs from the session, injected."""

    registry: ToolRegistry
    agent: Callable[[], Agent]
    plan_store: Callable[[], InMemoryPlanStore | None]
    model: Callable[[], str]
    settings: Callable[[], dict]
    augment_prompt: Callable[[str], str]
    trace_path: Callable[[str], Path | None]
    publish_trace: Callable[[str, dict], None]
    find_recorded_tool: Callable[[str], dict | None]
    record_tool_call: Callable[[str, dict], dict]
    record_tool_result: Callable[[dict, dict], None]
    record_agent_response: Callable[[str, str], None]
    mark_tool_waiting: Callable[[str], None]


class PromptRunner:
    """Runs one prompt cell through the agent, streaming trace steps."""

    def __init__(self, hooks: RunnerHooks) -> None:
        self.hooks = hooks

    # -- public entry points ---------------------------------------------

    def run(
        self,
        cell_id: str,
        source: str,
        trace_steps: list[dict],
        *,
        token: CancellationToken,
        resume: dict | None = None,
    ) -> RunOutcome:
        """Run (or resume) a prompt, mapping cancellation and crashes to outcomes.

        ``trace_steps`` is the cell's live step list: new steps are appended as
        they stream, and on resume its existing entries are the prior segment's
        steps (used to restore the plan snapshot).
        """
        try:
            return asyncio.run(
                self.run_async(cell_id, source, token, trace_steps, resume=resume)
            )
        except RunCancelled:
            self._finish_trace(cell_id, status="stopped")
            return RunOutcome(status="stopped")
        except Exception as exc:  # surface any failure in the cell output
            self._finish_trace(cell_id, status="error", error=str(exc))
            return RunOutcome(status="error", error=str(exc))

    async def run_async(
        self,
        cell_id: str,
        source: str,
        token: CancellationToken,
        trace_steps: list[dict],
        *,
        resume: dict | None = None,
    ) -> RunOutcome:
        """Drive the agent until it finishes, fails, pauses, or is cancelled."""
        hooks = self.hooks
        agent = hooks.agent()
        trace_path = hooks.trace_path(cell_id)
        run_id = new_id()

        if trace_path is not None:
            if resume is None:
                traces.write_run(
                    trace_path,
                    cell_id=cell_id,
                    run_id=run_id,
                    # The trace records a display label: a caller may inject a
                    # ready-made Model object, which must not reach the writer.
                    model=str(hooks.model()),
                    prompt=source,
                )
            else:
                traces.append_resume(
                    trace_path,
                    run_id=run_id,
                    conversation_id=resume.get("conversation_id"),
                    response=resume.get("answers") or {},
                )

        plan_store = hooks.plan_store()
        initial_plan_items = latest_plan_items(trace_steps) if resume else []
        if plan_store is not None:
            # The store is shared by the single worker. A different prompt may
            # run while this one waits for input, and a restart recreates the
            # in-memory store. Restore this prompt's own latest snapshot before
            # resuming instead of inheriting another run's plan.
            await plan_store.set_items(initial_plan_items)

        recorded_tools: dict[str, dict] = {}
        attempt_changed_state = False

        async def on_events(ctx, events) -> None:  # noqa: ARG001 - ctx unused
            nonlocal attempt_changed_state
            async for event in events:
                step = step_from_event(event)
                if step is not None:
                    self._publish_step(cell_id, step, trace_steps, trace_path, recorded_tools)
                    if (
                        step.get("type") == "tool_result"
                        and step.get("outcome") == "success"
                        and not hooks.registry.replay_safe(step.get("name"))
                    ):
                        attempt_changed_state = True
                if (
                    plan_store is not None
                    and getattr(event, "event_kind", None) == "function_tool_result"
                    and hooks.registry.plan_mutation(
                        getattr(getattr(event, "part", None), "tool_name", None)
                    )
                ):
                    plan_step = await self._plan_step(plan_store)
                    trace_steps.append(plan_step)
                    if trace_path is not None:
                        traces.append_step(trace_path, plan_step)
                    hooks.publish_trace(cell_id, plan_step)

        prompt = hooks.augment_prompt(source) if resume is None else None
        max_attempts = hooks.settings().get("max_retries", 5)
        max_requests = hooks.settings().get("max_requests", 200)
        last_error: Exception | None = None
        retry_block_reason: str | None = None

        for attempt in range(1, max_attempts + 1):
            attempt_changed_state = False
            if attempt > 1 and plan_store is not None:
                await plan_store.set_items(initial_plan_items)
            try:
                result = await agent.run(
                    prompt,
                    message_history=resume.get("messages") if resume else None,
                    deferred_tool_results=resume.get("deferred_results") if resume else None,
                    conversation_id=resume.get("conversation_id") if resume else None,
                    event_stream_handler=on_events,
                    cancellation_token=token,
                    run_id=run_id,
                    usage_limits=UsageLimits(request_limit=max_requests),
                )
            except RunCancelled:
                raise
            except UsageLimitExceeded as exc:
                # Stop cleanly rather than reporting the run as a failure.
                stopped = (
                    f"Stopped after {max_requests} model requests ({exc}). "
                    "Everything already done was kept. Raise GEOAI_MAX_REQUESTS "
                    "(or max_requests in settings.json) and run again to continue."
                )
                note = {"type": "text", "content": stopped}
                trace_steps.append(note)
                if trace_path is not None:
                    traces.append_step(trace_path, note)
                hooks.publish_trace(cell_id, note)
                self._finish_trace(cell_id, status="stopped", error=stopped)
                return RunOutcome(status="stopped", error=stopped)
            except Exception as exc:  # retry transient provider failures
                if is_context_overflow_error(exc):
                    stopped = (
                        "Stopped: the history no longer fits this model's context window. "
                        "Set GEOAI_CONTEXT_WINDOW to the window the endpoint really has — "
                        "compaction targets half of it — or start a new cell to begin with a "
                        f"shorter history. The work already done was kept. ({exc})"
                    )
                    note = {"type": "text", "content": stopped}
                    trace_steps.append(note)
                    if trace_path is not None:
                        traces.append_step(trace_path, note)
                    hooks.publish_trace(cell_id, note)
                    self._finish_trace(cell_id, status="stopped", error=stopped)
                    return RunOutcome(status="stopped", error=stopped)
                last_error = exc
                transient = is_transient_run_error(exc)
                retry_block_reason = (
                    "automatic replay was skipped because a tool changed the "
                    "workspace or map; those completed changes were preserved"
                    if attempt_changed_state
                    else None
                )
                if transient and not attempt_changed_state and attempt < max_attempts:
                    note = {
                        "type": "text",
                        "content": (
                            f"Attempt {attempt}/{max_attempts} failed "
                            f"({type(exc).__name__}: {exc}). Retrying."
                        ),
                    }
                    trace_steps.append(note)
                    if trace_path is not None:
                        traces.append_step(trace_path, note)
                    hooks.publish_trace(cell_id, note)
                    await asyncio.sleep(retry_delay(exc, attempt))
                    continue
                break
            else:
                return self._finish_success(
                    cell_id, result, trace_steps, trace_path, run_id
                )

        error_text = (
            describe_run_error(last_error, hooks.registry)
            if last_error is not None
            else "unknown error"
        )
        if retry_block_reason is not None:
            error_text = f"{error_text} ({retry_block_reason})"
        self._finish_trace(cell_id, status="error", error=error_text)
        return RunOutcome(status="error", error=error_text)

    # -- internals -------------------------------------------------------

    def _publish_step(
        self,
        cell_id: str,
        step: dict,
        trace_steps: list[dict],
        trace_path: Path | None,
        recorded_tools: dict[str, dict],
    ) -> None:
        """Append one streamed step to the trace, the file, and the notebook."""
        hooks = self.hooks
        tool_call_id = step.get("tool_call_id")
        # A resumed run re-emits the deferred tool call it is answering; that
        # step is already in the trace, so only its (new) result is appended.
        replayed = (
            step.get("type") == "tool_call"
            and tool_call_id is not None
            and any(
                existing.get("type") == "tool_call"
                and existing.get("tool_call_id") == tool_call_id
                for existing in trace_steps
            )
        )
        if not replayed:
            trace_steps.append(step)
            if trace_path is not None:
                traces.append_step(trace_path, step)
            hooks.publish_trace(cell_id, step)

        if not hooks.settings().get("record_agent_steps", True):
            return
        if step.get("type") == "tool_call" and tool_call_id:
            existing = hooks.find_recorded_tool(tool_call_id)
            recorded_tools[tool_call_id] = existing or hooks.record_tool_call(cell_id, step)
        elif step.get("type") == "tool_result" and tool_call_id:
            recorded = recorded_tools.get(tool_call_id)
            if recorded is not None:
                hooks.record_tool_result(recorded, step)

    async def _plan_step(self, plan_store: InMemoryPlanStore) -> dict:
        items = [item.model_dump(mode="json") for item in await plan_store.get_items()]
        return {"type": "plan", "items": items}

    def _finish_success(
        self,
        cell_id: str,
        result,
        trace_steps: list[dict],
        trace_path: Path | None,
        run_id: str,
    ) -> RunOutcome:
        """Record a completed or paused run and describe it to the session."""
        hooks = self.hooks
        usage = traces.usage_to_dict(result.usage)

        if isinstance(result.output, DeferredToolRequests):
            interaction = interaction_from_deferred(result.output)
            hooks.mark_tool_waiting(interaction["tool_call_id"])
            if trace_path is not None:
                traces.append_messages(trace_path, result.all_messages())
                traces.append_result(
                    trace_path,
                    status="waiting_for_input",
                    usage=usage,
                    conversation_id=result.conversation_id,
                )
            return RunOutcome(
                status="waiting_for_input",
                interaction=interaction,
                usage=usage,
                run_id=run_id,
                conversation_id=result.conversation_id,
            )

        usage_step = {"type": "usage", "usage": usage}
        trace_steps.append(usage_step)
        hooks.publish_trace(cell_id, usage_step)
        if trace_path is not None:
            traces.append_step(trace_path, usage_step)
            traces.append_messages(trace_path, result.new_messages())
            traces.append_result(
                trace_path,
                status="done",
                output=str(result.output),
                usage=usage,
                conversation_id=result.conversation_id,
            )
        if hooks.settings().get("record_agent_steps", True):
            hooks.record_agent_response(cell_id, str(result.output))
        return RunOutcome(
            status="done",
            output=str(result.output),
            usage=usage,
            run_id=run_id,
            conversation_id=result.conversation_id,
        )

    def _finish_trace(
        self,
        cell_id: str,
        *,
        status: str,
        output: str | None = None,
        error: str | None = None,
    ) -> None:
        path = self.hooks.trace_path(cell_id)
        if path is not None:
            traces.append_result(path, status=status, output=output, error=error)


def resume_payload(*, messages, interaction: dict, answers: dict, conversation_id, call_id: str):
    """Build the payload a paused run resumes with."""
    return {
        "messages": messages,
        "deferred_results": DeferredToolResults(calls={call_id: answers}),
        "conversation_id": conversation_id,
        "answers": answers,
        "interaction_id": interaction.get("id"),
    }
