"""Conversation recall: search earlier prompts, tool calls, and results."""

from __future__ import annotations

from ...contracts.errors import ToolInputError
from ...contracts.services import CONVERSATION_SERVICE
from ...workspace.conversation import ConversationStore, query_terms
from ..runtime import ToolRuntime
from ..spec import pack, tool

#: Results one call may return.
MAX_RESULTS = 20


@pack(category="conversation", core=True)
class ConversationPack:
    """History lookup for follow-up questions in the open workspace."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime

    def _store(self) -> ConversationStore | None:
        return self._rt.services.get(CONVERSATION_SERVICE)

    @tool(
        core=True,
        summary="Search earlier prompts, tool calls, and results in this workspace",
    )
    def search_conversation_history(self, query: str, max_results: int = 6) -> dict:
        """Search the workspace's earlier prompt cells for relevant context.

        Call this only when the current request refers back to earlier work —
        "the result from before", "that file", "what did we compute", "use the
        same AOI" — and you need the exact prompt, command, or result. Do not
        call it on every turn: the live notebook already carries the visible
        conversation, so searching is for context that is no longer in view.

        ``query`` should use distinctive terms (a place, dataset, tool, index,
        file name, or number) rather than a whole sentence. Each hit names the
        prompt cell, the trace step it came from, and a short excerpt.

        Returns ``query``, ``terms``, ``count``, and ``results``. A result has
        ``reference``, ``cell_id``, ``prompt_id``, ``run_id``, ``kind``
        (``prompt``, ``tool_call``, ``tool_result``, ``response``, or ``error``),
        ``step_index``, ``tool_name``, ``hint``, ``prompt``, and ``answer``.
        """
        text = (query or "").strip()
        if not text:
            raise ToolInputError("query must not be empty")
        try:
            limit = max(1, min(MAX_RESULTS, int(max_results)))
        except (TypeError, ValueError):
            limit = 6
        store = self._store()
        terms = query_terms(text)
        hits = store.search(text, limit=limit) if store is not None else []
        payload = {
            "query": text,
            "terms": terms,
            "count": len(hits),
            "results": hits,
            "note": (
                "Matched earlier work in this workspace. Cite the prompt cell and "
                "step when you use it."
                if hits
                else (
                    "No earlier prompt, tool call, or result matched those terms."
                    if store is not None and store.open
                    else "Conversation history is unavailable for this workspace."
                )
            ),
        }
        self._rt.events.notify_recall(self._rt.run_id, payload)
        return payload


__all__ = ["ConversationPack", "MAX_RESULTS"]
