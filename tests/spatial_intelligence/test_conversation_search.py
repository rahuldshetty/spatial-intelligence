"""Conversation recall: the SQLite index, the tool, and the session wiring."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pydantic_ai.models.function import DeltaToolCall

from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.services import CONVERSATION_SERVICE
from spatial_intelligence.tools.packs.conversation import ConversationPack
from spatial_intelligence.tools.registry import ToolRegistry
from spatial_intelligence.tools.runtime import RuntimeEvents, ToolRuntime
from spatial_intelligence.workspace import Workspace
from spatial_intelligence.workspace.conversation import (
    MAX_UNIT_CHARS,
    ConversationStore,
    query_terms,
)

from .support import answered_tools, scripted, stream_model
from .test_session import TERMINAL_STATUSES, SessionTestCase


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.store = ConversationStore()
        self.store.open_at(self.workspace.conversation_db)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_prompt_tools_and_answer_are_indexed_but_reasoning_is_not(self):
        self.store.begin_run("c1", "r1", "Compute NDVI over the north field")
        self.store.index_step(
            "c1",
            {
                "type": "text_delta",
                "content": "I should inspect the raster first",
            },
            0,
        )
        self.store.index_step(
            "c1",
            {
                "type": "tool_call",
                "name": "spectral_index",
                "args": {"index": "NDVI", "path": "data/north.tif"},
                "tool_call_id": "t1",
            },
            2,
        )
        self.store.index_step(
            "c1",
            {
                "type": "tool_result",
                "name": "spectral_index",
                "content": "wrote results/ndvi.tif",
                "tool_call_id": "t1",
            },
            3,
        )
        self.store.index_result("c1", "done", "Mean NDVI was 0.62", None)

        self.assertEqual(self.store.unit_count("c1"), 4)

        hits = self.store.search("NDVI")
        kinds = {hit["kind"] for hit in hits}
        self.assertEqual(kinds, {"prompt", "tool_call", "tool_result", "response"})
        prompt_hit = next(hit for hit in hits if hit["kind"] == "prompt")
        for hit in hits:
            self.assertEqual(hit["cell_id"], "c1")
            self.assertEqual(hit["run_id"], "r1")
            self.assertEqual(hit["prompt_id"], prompt_hit["prompt_id"])
        call_hit = next(hit for hit in hits if hit["kind"] == "tool_call")
        self.assertEqual(call_hit["step_index"], 2)
        self.assertEqual(call_hit["tool_name"], "spectral_index")
        self.assertIn("spectral_index", call_hit["reference"])

        # The reasoning delta was not indexed.
        self.assertEqual(self.store.search("inspect"), [])

    def test_a_rerun_replaces_the_cells_rows(self):
        self.store.begin_run("c1", "r1", "first attempt with NDVI")
        self.store.begin_run("c1", "r2", "second attempt with NDWI")

        self.assertEqual(self.store.search("NDVI"), [])
        hits = self.store.search("NDWI")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["run_id"], "r2")

    def test_forget_cell_drops_only_that_prompts_rows(self):
        self.store.begin_run("c1", "r1", "NDVI for the north field")
        self.store.begin_run("c2", "r2", "NDVI for the south field")

        self.store.forget_cell("c1")

        self.assertEqual(self.store.unit_count("c1"), 0)
        hits = self.store.search("NDVI")
        self.assertEqual({hit["cell_id"] for hit in hits}, {"c2"})

    def test_a_closed_store_searches_and_indexes_nothing(self):
        self.store.close()
        self.store.begin_run("c1", "r1", "NDVI")
        self.assertEqual(self.store.search("NDVI"), [])
        self.assertEqual(self.store.unit_count(), 0)

    def test_a_huge_result_is_clipped_to_the_unit_cap(self):
        self.store.begin_run("c1", "r1", "dump the raster")
        self.store.index_step(
            "c1",
            {
                "type": "tool_result",
                "name": "read_file",
                "content": "needle " + "x" * (MAX_UNIT_CHARS + 500),
                "tool_call_id": "t1",
            },
            1,
        )
        hits = self.store.search("needle")
        self.assertEqual(len(hits), 1)
        self.assertLessEqual(len(hits[0]["hint"]), MAX_UNIT_CHARS)

    def test_query_terms_are_unique_and_bounded(self):
        self.assertEqual(query_terms("NDVI, ndvi and north"), ["NDVI", "and", "north"])


class ConversationToolTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.store = ConversationStore()
        self.store.open_at(self.workspace.conversation_db)
        self.recalls: list[tuple[str | None, dict]] = []
        self.runtime = ToolRuntime(
            workspace=self.workspace,
            run_id="c1",
            events=RuntimeEvents(
                recall=lambda run_id, payload: self.recalls.append((run_id, payload))
            ),
            services={CONVERSATION_SERVICE: self.store},
        )
        self.registry = ToolRegistry()
        self.registry.add_pack(ConversationPack, self.runtime)
        self.tool = self.registry.get("search_conversation_history")

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def test_the_tool_returns_hits_and_publishes_a_recall_event(self):
        self.store.begin_run("c0", "r0", "Compute NDVI over the north field")
        self.store.index_result("c0", "done", "Mean NDVI was 0.62", None)

        result = self.tool.callable(query="NDVI mean", max_results=3)

        self.assertEqual(result["query"], "NDVI mean")
        self.assertEqual(result["count"], len(result["results"]))
        self.assertGreaterEqual(result["count"], 1)
        self.assertIn("NDVI", result["terms"])
        self.assertEqual(self.recalls[0][0], "c1")
        self.assertEqual(self.recalls[0][1]["count"], result["count"])

    def test_an_empty_query_is_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.callable(query="   ")

    def test_an_unavailable_store_still_answers_with_a_note(self):
        self.store.close()

        result = self.tool.callable(query="NDVI")

        self.assertEqual(result["count"], 0)
        self.assertIn("unavailable", result["note"])


def follow_up_model():
    """Answer the first prompt, then recall it from the second prompt."""

    async def stream(messages, info):
        text = ""
        for message in messages:
            for part in getattr(message, "parts", ()):
                if getattr(part, "part_kind", "") == "user-prompt":
                    text = str(getattr(part, "content", "") or "")
        if "wombat" not in text:
            yield "remembered"
            return
        if answered_tools(messages):
            yield "The earlier survey computed NDVI."
            return
        yield {
            0: DeltaToolCall(
                name="search_conversation_history",
                json_args=json.dumps({"query": "wombat survey NDVI"}),
            )
        }

    return stream_model(stream)


class ConversationSessionTests(SessionTestCase):
    def test_a_run_is_indexed_and_deleting_the_prompt_clears_it(self):
        turn = SimpleNamespace(tool_name="list_files", args={})
        state = self.start(scripted(turn, final="listed"))
        cell_id = state.add_cell("prompt", "list the workspace files")["cells"][-1]["id"]

        state.run_cell(cell_id)
        self.wait_for_status(state, cell_id, TERMINAL_STATUSES)

        self.assertGreaterEqual(state.conversation.unit_count(cell_id), 3)
        hits = state.conversation.search("list_files")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["cell_id"], cell_id)

        state.delete_cell(cell_id)
        self.assertEqual(state.conversation.unit_count(cell_id), 0)
        self.assertEqual(state.conversation.search("list_files"), [])

    def test_switching_workspaces_closes_and_reopens_the_index(self):
        state = self.start(scripted(final="noted"))
        cell_id = state.add_cell("prompt", "remember the wombat survey")["cells"][-1]["id"]
        state.run_cell(cell_id)
        self.wait_for_status(state, cell_id, TERMINAL_STATUSES)
        self.assertTrue(state.conversation.search("wombat"))

        state.open_workspace("other-workspace")
        self.assertTrue(state.conversation.open)
        self.assertEqual(state.conversation.search("wombat"), [])

        state.open_workspace("session-test")
        self.assertTrue(state.conversation.search("wombat"))

    def test_a_follow_up_calls_recall_and_shows_it_in_the_trace(self):
        state = self.start(follow_up_model())
        first = state.add_cell("prompt", "remember the wombat survey used NDVI")["cells"][-1]
        state.run_cell(first["id"])
        self.wait_for_status(state, first["id"], TERMINAL_STATUSES)

        second = state.add_cell("prompt", "what did we compute for the wombat survey?")["cells"][-1]
        state.run_cell(second["id"])
        cell = self.wait_for_status(state, second["id"], TERMINAL_STATUSES)

        recall = next(step for step in cell["trace"] if step["type"] == "recall")
        self.assertEqual(recall["query"], "wombat survey NDVI")
        self.assertIn("wombat", recall["terms"])
        self.assertTrue(recall["results"])
        self.assertEqual(recall["results"][0]["cell_id"], first["id"])
        self.assertTrue(
            any(
                step.get("type") == "tool_result"
                and step.get("name") == "search_conversation_history"
                for step in cell["trace"]
            )
        )


if __name__ == "__main__":
    unittest.main()
