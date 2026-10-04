"""Agent construction: one agent per session, built from the tool registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import ReinjectSystemPrompt, WebSearch
from pydantic_ai.models import Model
from pydantic_ai_harness import (
    ClampOversizedMessages,
    ClearToolResults,
    Planning,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.planning import InMemoryPlanStore

from ..tools.registry import ToolRegistry
from ..websearch import web_search
from .capabilities import NormalizeDuplicateToolNames, ToolFailurePolicy
from .model import resolve_model
from .prompt import system_prompt

#: Retry prompts one tool call may produce before the run is stopped.
#:
#: ``ToolFailurePolicy`` answers every tool failure it can see with a failed
#: result instead of a retry prompt, so this budget only ever bounds a call the
#: policy never sees: a tool name the model is not allowed to call yet (an
#: unknown name, or a deferred tool it never discovered).
TOOL_RETRIES = 3

#: Tokens one message part may reach before compaction keeps its head and tail.
CLAMP_PART_TOKENS = 32_000


@dataclass(slots=True)
class BuiltAgent:
    """An agent plus the plan store its Planning capability writes into."""

    agent: Agent
    plan_store: InMemoryPlanStore


def build_agent(
    registry: ToolRegistry,
    model: str | Model | Callable[[], str],
    *,
    tool_retries: int = TOOL_RETRIES,
    context_window: int = 0,
) -> BuiltAgent:
    """Build a pydantic-ai ``Agent`` with every registered tool.

    ``registry`` supplies both the tools and their metadata (core visibility,
    approval, timeouts, sequencing); nothing about a tool is restated here.
    ``model`` may be a model string, a ready-made ``Model``, or a zero-argument
    callable returning a string, so a settings change can be picked up without
    rebuilding this function. ``context_window`` is the model's real window in
    tokens, or 0 to resolve it from the model id.
    """
    plan_store = InMemoryPlanStore()
    agent: Agent = Agent(
        resolve_model(model() if callable(model) else model),
        system_prompt=system_prompt(),
        output_type=[str, DeferredToolRequests],
        retries={"tools": tool_retries},
        capabilities=[
            ReinjectSystemPrompt(),
            # Provider-native web search where the model has one, the local
            # DuckDuckGo client otherwise. The configured DeepSeek endpoint has
            # no native search, so ``web_search`` is what actually runs; the
            # local tool is dropped from the wire on a model that does.
            WebSearch(local=web_search),
            Planning(store=plan_store),
            NormalizeDuplicateToolNames(),
            ToolFailurePolicy(),
            # Runs before every request, and only past half the window: clear old
            # tool results (free), then summarize (one model call). The clamp is
            # first because the other tiers only drop *old* messages.
            TieredCompaction(
                tiers=[
                    ClampOversizedMessages(max_part_tokens=CLAMP_PART_TOKENS),
                    ClearToolResults(max_tokens=1, keep_pairs=3),
                    SummarizingCompaction(max_messages=1, keep_messages=20),
                ],
                target_fraction=0.5,
                # 0 means "resolve from the model id"; a proxy id resolves to
                # nothing, so the deployment can state its window instead.
                context_window=context_window or None,
            ),
        ],
    )
    registry.build(agent)
    return BuiltAgent(agent=agent, plan_store=plan_store)
