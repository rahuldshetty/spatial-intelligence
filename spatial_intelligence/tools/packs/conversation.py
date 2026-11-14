"""Conversation recall: search earlier prompts, tool calls, and results."""

from __future__ import annotations

from ...contracts.errors import ToolInputError
from ...contracts.services import CONVERSATION_SERVICE
from ...workspace.conversation import ConversationStore, search_terms
from ..runtime import ToolRuntime
from ..spec import pack, tool

#: Results one call may return.
MAX_RESULTS = 30


@pack(category="conversation", core=True)
class ConversationPack:
    """History lookup for follow-up questions in the open workspace."""

    def __init__(self, runtime: ToolRuntime) -> None:
        self._rt = runtime
        self._run: str | None = None
        self._cache: dict[tuple[tuple[str, ...], int], dict] = {}

    def _store(self) -> ConversationStore | None:
        return self._rt.services.get(CONVERSATION_SERVICE)

    @tool(
        core=True,
        summary="Search earlier prompts, tool calls, and results in this workspace",
    )
    def search_conversation_history(self, terms: list[str], max_results: int = 10) -> dict:
        """Search the workspace's earlier prompt cells for relevant context.

        Call this only when the current request refers back to earlier work —
        "the result from before", "that file", "what did we compute", "use the
        same AOI" — and you need the exact prompt, command, or result. Do not
        call it on every turn: the live notebook already carries the visible
        conversation, so searching is for context that is no longer in view.

        Generate ``terms`` from the request: 2–6 distinctive search phrases,
        most specific first. A phrase may be several words ("north field NDVI",
        "Sentinel-1 2024") or a single identifier ("results/ndvi.tif", "0.62").
        Use the place, dataset, tool, index, file, or parameter the request
        names; do not pad the list with filler words. Multi-word phrases match
        adjacent terms; single words match prefixes. One hit per earlier prompt
        is returned.

        Returns ``query`` (the terms joined for display), ``terms``, ``count``,
        and ``results``. A result has ``reference``, ``cell_id``, ``prompt_id``,
        ``run_id``, ``kind`` (``prompt``, ``assistant``, ``tool_call``,
        ``tool_result``, ``response``, or ``error``), ``step_index``,
        ``tool_name``, ``hint``, ``result`` (the matching tool result for a
        ``tool_call``), ``prompt``, ``answer``, and ``created_at``.
        """
        cleaned = search_terms(terms)
        if not cleaned:
            raise ToolInputError("terms must not be empty")
        try:
            limit = max(1, min(MAX_RESULTS, int(max_results)))
        except (TypeError, ValueError):
            limit = 10
        run = self._rt.run_id
        if run != self._run:
            self._run = run
            self._cache.clear()
        key = (tuple(term.casefold() for term in cleaned), limit)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        store = self._store()
        hits = (
            store.search(cleaned, limit=limit, exclude_cell=run)
            if store is not None
            else []
        )
        payload = {
            "query": " · ".join(cleaned),
            "terms": cleaned,
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
        self._cache[key] = payload
        self._rt.events.notify_recall(self._rt.run_id, payload)
        return payload


__all__ = ["ConversationPack", "MAX_RESULTS"]
