"""Settings, notebook serialization, and agent trace persistence."""

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.usage import RunUsage

from spatial_intelligence.settings import env, prefs
from spatial_intelligence.workspace import notebook, traces

DEFAULT_SETTINGS = {
    "model": "openai:gpt-4o-mini",
    "theme": "light",
    "dangerous_mode": False,
    "max_retries": 3,
    "record_agent_steps": True,
}


def conversation():
    """One user message, in the shape ``append_messages`` persists."""
    return ModelMessagesTypeAdapter.validate_python(
        [
            {
                "kind": "request",
                "parts": [
                    {
                        "part_kind": "user-prompt",
                        "content": "hello",
                        "timestamp": "2026-01-02T03:04:05Z",
                    }
                ],
            }
        ]
    )


def stream_output(text, name="stdout"):
    return {
        "output_type": "stream",
        "name": name,
        "text": text,
        "ename": None,
        "evalue": None,
        "traceback": None,
    }


def error_output(ename="ValueError", evalue="bad"):
    return {
        "output_type": "error",
        "name": None,
        "text": None,
        "ename": ename,
        "evalue": evalue,
        "traceback": ["frame"],
    }


class DataRootTestCase(unittest.TestCase):
    """Runs against a throwaway ``GEOAI_HOME`` data root."""

    env_vars: dict[str, str] = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        patch_home = patch.dict(os.environ, {"GEOAI_HOME": str(self.home), **self.env_vars})
        patch_home.start()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(patch_home.stop)


class EnvTests(DataRootTestCase):
    env_vars = {"GEOAI_MODEL": "openai:gpt-4o-mini", "GEOAI_MAX_RETRIES": "3"}

    def test_geoai_home_is_the_data_root(self):
        self.assertEqual(env.app_root(), self.home)

    def test_workspace_root_sits_under_the_app_root(self):
        root = env.workspace_root("x")

        self.assertTrue(root.is_absolute())
        self.assertEqual(root, env.app_root() / "workspaces" / "x")
        self.assertTrue(root.is_relative_to(env.app_root() / "workspaces"))

    def test_list_workspaces_returns_only_directories(self):
        self.assertEqual(env.list_workspaces(), [])

        base = self.home / "workspaces"
        (base / "beta").mkdir(parents=True)
        (base / "alpha").mkdir()
        (base / "notes.txt").write_text("not a workspace", encoding="utf-8")

        self.assertEqual(env.list_workspaces(), ["alpha", "beta"])

    def test_model_provider_and_key_lookup(self):
        self.assertEqual(env.model_from_env(), "openai:gpt-4o-mini")
        self.assertEqual(env.DEFAULT_MODEL, "openai:gpt-4o")
        self.assertEqual(env.model_provider("openai:gpt-4o"), "openai")
        self.assertEqual(env.model_provider("gemini-2.5-pro"), "openai")
        self.assertEqual(env.api_key_env_for("anthropic:claude-sonnet-4"), "ANTHROPIC_API_KEY")
        self.assertEqual(env.api_key_env_for("openai-chat:qwen3"), "OPENAI_API_KEY")
        self.assertEqual(env.api_key_env_for("openai-responses:gpt-5"), "OPENAI_API_KEY")
        self.assertIsNone(env.api_key_env_for("ollama:llama3"))

    def test_max_retries_defaults_and_clamps(self):
        self.assertEqual(env.max_retries(), 3)

        with patch.dict(os.environ, {"GEOAI_MAX_RETRIES": "0"}):
            self.assertEqual(env.max_retries(), 1)
        with patch.dict(os.environ, {"GEOAI_MAX_RETRIES": "oops"}):
            self.assertEqual(env.max_retries(), 5)
        with patch.dict(os.environ, {"GEOAI_MAX_RETRIES": "  "}):
            self.assertEqual(env.max_retries(), 5)

    def test_a_shadowed_env_file_value_is_reported(self):
        path = self.home / ".env"
        path.write_text(
            "OPENAI_API_KEY=from-the-file\nGEOAI_MODEL=openai:from-the-file\n",
            encoding="utf-8",
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-the-environment"}, clear=False):
            shadowed = env._shadowed_keys(path)

        # Every variable whose value actually differs counts — including one the
        # test harness itself sets (GEOAI_MODEL) — and identical values do not.
        self.assertEqual(shadowed, {"OPENAI_API_KEY", "GEOAI_MODEL"})

    def test_load_env_warns_when_the_environment_wins(self):
        (self.home / ".env").write_text("OPENAI_API_KEY=from-the-file\n", encoding="utf-8")
        buffer = io.StringIO()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "from-the-environment"}, clear=False):
            with contextlib.redirect_stderr(buffer):
                env.load_env()

            # dotenv's documented precedence: the environment wins, loudly.
            self.assertEqual(os.environ["OPENAI_API_KEY"], "from-the-environment")

        self.assertIn("OPENAI_API_KEY", buffer.getvalue())
        self.assertIn(".env", buffer.getvalue())

    def test_validate_env_refuses_to_start_without_the_provider_key(self):
        with patch.dict(
            os.environ,
            {"GEOAI_MODEL": "openai:gpt-4o", "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""},
        ):
            with self.assertRaises(SystemExit) as caught:
                env.validate_env()
        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertIn("openai:gpt-4o", str(caught.exception))

        with patch.dict(os.environ, {"GEOAI_MODEL": "openai:gpt-4o", "OPENAI_API_KEY": "sk-test"}):
            env.validate_env()
        with patch.dict(os.environ, {"GEOAI_MODEL": "ollama:llama3"}):
            env.validate_env()

    def test_validate_env_allows_a_custom_endpoint_without_a_key(self):
        # A locally served OpenAI-compatible model usually checks no key, and
        # pydantic-ai substitutes a placeholder for one; refusing to start would
        # reject the deployment that works.
        with patch.dict(
            os.environ,
            {
                "GEOAI_MODEL": "openai-chat:qwen3",
                "OPENAI_API_KEY": "",
                "OPENAI_BASE_URL": "http://127.0.0.1:8080/v1",
            },
        ):
            env.validate_env()

    def test_validate_env_still_requires_the_key_without_a_custom_endpoint(self):
        with patch.dict(
            os.environ,
            {"GEOAI_MODEL": "openai-chat:qwen3", "OPENAI_API_KEY": "", "OPENAI_BASE_URL": ""},
        ):
            with self.assertRaises(SystemExit) as caught:
                env.validate_env()

        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_validate_env_reads_the_provider_key_from_the_data_root_env_file(self):
        # Regression: start-up validated before anything read <app_root>/.env, so
        # a key that lives only in that file was reported as missing.
        (self.home / ".env").write_text(
            "OPENAI_API_KEY=key-that-is-only-in-the-file\n", encoding="utf-8"
        )
        os.environ.pop("OPENAI_API_KEY", None)
        try:
            env.validate_env()
        finally:
            os.environ.pop("OPENAI_API_KEY", None)

    def test_validate_env_resolves_the_model_from_the_settings_file_too(self):
        # ollama needs no provider key, so this passes only when the file's model
        # is read before the key requirement is decided.
        (self.home / ".env").write_text("GEOAI_MODEL=ollama:llama3\n", encoding="utf-8")
        saved = {
            name: os.environ.pop(name, None)
            for name in ("GEOAI_MODEL", "OPENAI_API_KEY")
        }
        try:
            env.validate_env()
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_validate_env_still_refuses_when_neither_source_has_the_key(self):
        (self.home / ".env").write_text("GEOAI_MODEL=openai:gpt-4o\n", encoding="utf-8")
        os.environ.pop("OPENAI_API_KEY", None)

        with self.assertRaises(SystemExit) as caught:
            env.validate_env()

        self.assertIn("OPENAI_API_KEY", str(caught.exception))
        self.assertIn(str(self.home / ".env"), str(caught.exception))

    def test_resolve_workspace_name_prefers_the_explicit_override(self):
        with patch.dict(os.environ, {"GEOAI_WORKSPACE": "from-env"}):
            self.assertEqual(env.resolve_workspace_name(), "from-env")
            self.assertEqual(env.resolve_workspace_name("  explicit  "), "explicit")

    def test_server_base_url_follows_the_configured_port(self):
        self.assertEqual(env.server_base_url(), "http://127.0.0.1:8000/")

        with patch.dict(os.environ, {"GEOAI_PORT": "9001"}):
            self.assertEqual(env.server_base_url(), "http://127.0.0.1:9001/")

    def test_load_env_reads_dotenv_without_overriding_the_environment(self):
        (self.home / ".env").write_text(
            "GEOAI_MODEL=ollama:llama3\nGEOAI_DOTENV_ONLY=yes\n", encoding="utf-8"
        )
        os.environ.pop("GEOAI_DOTENV_ONLY", None)

        env.load_env()

        self.assertEqual(os.environ["GEOAI_DOTENV_ONLY"], "yes")
        # The process environment wins over the file (dotenv's default).
        self.assertEqual(env.model_from_env(), "openai:gpt-4o-mini")


class SettingsPrefsTests(DataRootTestCase):
    env_vars = {"GEOAI_MODEL": "openai:gpt-4o-mini", "GEOAI_MAX_RETRIES": "3"}

    def test_first_run_defaults_come_from_the_environment(self):
        self.assertEqual(prefs.load_settings(), DEFAULT_SETTINGS)
        self.assertEqual(prefs.settings_path(), self.home / "settings.json")

    def test_save_normalizes_and_load_reads_it_back(self):
        saved = prefs.save_settings(
            {
                "model": "  ollama:llama3  ",
                "theme": "dark",
                "dangerous_mode": True,
                "max_retries": 4,
                "record_agent_steps": False,
            }
        )

        self.assertEqual(
            saved,
            {
                "model": "ollama:llama3",
                "theme": "dark",
                "dangerous_mode": True,
                "max_retries": 4,
                "record_agent_steps": False,
            },
        )
        self.assertEqual(prefs.load_settings(), saved)
        self.assertEqual(
            json.loads(prefs.settings_path().read_text(encoding="utf-8")), saved
        )

    def test_an_unknown_theme_and_a_low_retry_cap_are_corrected(self):
        corrected = prefs.save_settings({"theme": "solarized", "max_retries": 0})

        self.assertEqual(corrected["theme"], "light")
        self.assertEqual(corrected["max_retries"], 1)
        self.assertEqual(prefs.save_settings({"max_retries": -9})["max_retries"], 1)
        self.assertEqual(prefs.save_settings({"max_retries": "many"})["max_retries"], 3)
        self.assertEqual(
            prefs.save_settings({"model": "   ", "theme": "dark"})["model"],
            "openai:gpt-4o-mini",
        )

    def test_a_corrupt_or_partial_file_still_loads(self):
        prefs.settings_path().write_text("{not json", encoding="utf-8")
        self.assertEqual(prefs.load_settings(), DEFAULT_SETTINGS)

        prefs.settings_path().write_text("[1, 2]", encoding="utf-8")
        self.assertEqual(prefs.load_settings(), DEFAULT_SETTINGS)

        prefs.settings_path().write_text(
            json.dumps({"theme": "dark", "max_retries": 2}), encoding="utf-8"
        )
        merged = prefs.load_settings()

        self.assertEqual(merged["theme"], "dark")
        self.assertEqual(merged["max_retries"], 2)
        self.assertEqual(merged["model"], "openai:gpt-4o-mini")
        self.assertTrue(merged["record_agent_steps"])


class NotebookAtomicWriteTests(unittest.TestCase):
    """A reader must never catch the notebook mid-write.

    Regression: the document was rewritten in place, so a reader that arrived
    while a save was in flight parsed a truncated file — which surfaced as an
    empty notebook (``JSONDecodeError``) rather than as the previous one.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "notebook.ipynb"
        self.addCleanup(self._tmp.cleanup)

    def test_a_concurrent_reader_always_sees_a_complete_document(self):
        cells = [
            notebook.new_cell("prompt", "summarize the parcels " + "x" * 400),
            notebook.new_cell("python", "print('done')"),
        ]
        stop = threading.Event()
        failures: list[str] = []

        def write_repeatedly() -> None:
            while not stop.is_set():
                try:
                    notebook.write_nb(self.path, cells)
                except Exception as exc:  # reported as a failure, not as stderr noise
                    failures.append(f"write: {type(exc).__name__}: {exc}")
                    return

        writer = threading.Thread(target=write_repeatedly, daemon=True)
        writer.start()
        try:
            for _ in range(300):
                if not self.path.exists():
                    continue
                try:
                    payload = self.path.read_text(encoding="utf-8")
                except PermissionError:
                    # Windows can refuse an open for the instant the rename takes;
                    # a real reader (a refetching client) retries. What must never
                    # happen is a read that succeeds and parses short or empty.
                    continue
                try:
                    document = json.loads(payload)
                except Exception as exc:
                    failures.append(f"{type(exc).__name__}: {exc}")
                    break
                if len(document.get("cells", [])) != len(cells):
                    failures.append(f"read {len(document.get('cells', []))} cells")
                    break
        finally:
            stop.set()
            writer.join(timeout=5)

        self.assertEqual(failures, [])

    def test_writing_leaves_no_temporary_file_behind(self):
        notebook.write_nb(self.path, [notebook.new_cell("python", "x = 1")])

        leftovers = [p.name for p in self.path.parent.iterdir() if p.name != self.path.name]

        self.assertEqual(leftovers, [])
        self.assertEqual(len(json.loads(self.path.read_text(encoding="utf-8"))["cells"]), 1)


class NotebookRoundTripTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "nb" / "notebook.ipynb"
        self.addCleanup(self._tmp.cleanup)

    def build_cells(self):
        prompt = notebook.new_cell("prompt", "count the buildings", metadata={"source": "user"})
        prompt["status"] = "waiting_for_input"
        prompt["interaction_history"] = [{"role": "user", "text": "go"}]

        tool = notebook.new_cell("tool", "print(1)", metadata={"geoai": {"tool": "run_code"}})
        tool["outputs"] = [stream_output("1\n")]
        tool["execution_count"] = 3
        tool["status"] = "done"

        python = notebook.new_cell("python", "x = 1")
        python["outputs"] = [stream_output("oops", name="stderr"), error_output()]
        python["execution_count"] = 2
        python["status"] = "error"

        interaction = notebook.new_cell("interaction", "Which layer?")
        interaction["status"] = "waiting_for_input"
        interaction["interaction"] = {"id": "i1", "prompt": "pick one"}
        interaction["answers"] = {"layer": "roads"}

        markdown = notebook.new_cell("markdown", "# Notes")
        return [prompt, tool, python, interaction, markdown]

    def test_round_trip_preserves_cells(self):
        original = self.build_cells()
        notebook.write_nb(self.path, original)

        reloaded = notebook.read_nb(self.path)
        prompt, tool, python, interaction, markdown = reloaded

        self.assertEqual([cell["id"] for cell in reloaded], [cell["id"] for cell in original])
        self.assertEqual(
            [cell["kind"] for cell in reloaded],
            ["prompt", "tool", "python", "interaction", "markdown"],
        )

        self.assertEqual(prompt["source"], "count the buildings")
        self.assertEqual(prompt["status"], "waiting_for_input")
        self.assertEqual(prompt["interaction_history"], [{"role": "user", "text": "go"}])
        self.assertEqual(prompt["metadata"]["source"], "user")
        self.assertIsNone(prompt["execution_count"])

        self.assertEqual(tool["outputs"], [stream_output("1\n")])
        self.assertEqual(tool["execution_count"], 3)
        self.assertEqual(tool["status"], "done")
        self.assertEqual(tool["metadata"]["geoai"]["tool"], "run_code")

        self.assertEqual(
            python["outputs"], [stream_output("oops", name="stderr"), error_output()]
        )
        self.assertEqual(python["execution_count"], 2)

        self.assertEqual(interaction["source"], "Which layer?")
        self.assertEqual(interaction["interaction"], {"id": "i1", "prompt": "pick one"})
        self.assertEqual(interaction["answers"], {"layer": "roads"})
        self.assertEqual(interaction["status"], "waiting_for_input")

        self.assertEqual(markdown["source"], "# Notes")

    def test_status_is_runtime_only_for_code_cells(self):
        python = notebook.new_cell("python", "x = 1")
        python["status"] = "error"
        python["outputs"] = [stream_output("x")]
        python["execution_count"] = 1

        notebook.write_nb(self.path, [python])
        reloaded = notebook.read_nb(self.path)[0]

        # A code cell re-derives its status from what it actually produced.
        self.assertEqual(reloaded["status"], "done")
        self.assertEqual(notebook.read_nb(Path(str(self.path) + ".missing")), [])

    def test_written_file_is_nbformat_4_5(self):
        notebook.write_nb(self.path, self.build_cells())

        raw = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual((raw["nbformat"], raw["nbformat_minor"]), (4, 5))
        cell_types = [cell["cell_type"] for cell in raw["cells"]]
        # Prompt and interaction cells serialize as Markdown prose; tool cells
        # stay executable code so any notebook tool can read them.
        self.assertEqual(
            cell_types, ["markdown", "code", "code", "markdown", "markdown"]
        )
        self.assertEqual(raw["cells"][0]["metadata"]["geoai"]["kind"], "prompt")
        self.assertEqual(raw["cells"][1]["metadata"]["geoai"]["kind"], "tool")
        self.assertEqual(raw["cells"][2]["metadata"]["geoai"], {})
        self.assertEqual(raw["cells"][1]["outputs"], [{"output_type": "stream", "name": "stdout", "text": "1\n"}])

    def test_a_persisted_interaction_reloads_as_waiting(self):
        # A notebook written by the server while it waited for the user.
        raw = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "id": "prompt-1",
                    "metadata": {
                        "geoai": {
                            "kind": "prompt",
                            "status": "running",
                            "interaction": {"id": "i1", "prompt": "pick one"},
                            "interaction_history": [{"role": "assistant", "text": "?"}],
                        }
                    },
                    "source": ["count the buildings"],
                }
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(raw), encoding="utf-8")

        cell = notebook.read_nb(self.path)[0]

        self.assertEqual(cell["kind"], "prompt")
        self.assertEqual(cell["interaction"], {"id": "i1", "prompt": "pick one"})
        self.assertEqual(cell["status"], "waiting_for_input")
        self.assertEqual(cell["interaction_history"], [{"role": "assistant", "text": "?"}])

    def test_legacy_cell_without_geoai_metadata_gets_an_id_and_status(self):
        raw = {
            "cells": [
                {"cell_type": "code", "metadata": {}, "execution_count": 7, "outputs": [], "source": ["y = 2"]},
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": ["y"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["plain"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(raw), encoding="utf-8")

        executed, untouched, markdown = notebook.read_nb(self.path)

        self.assertEqual([cell["kind"] for cell in (executed, untouched, markdown)], ["python"] * 2 + ["markdown"])
        self.assertEqual(executed["status"], "done")
        self.assertEqual(untouched["status"], "idle")
        self.assertEqual(markdown["status"], "idle")
        self.assertTrue(all(cell["id"] for cell in (executed, untouched, markdown)))

    def test_new_cell_rejects_an_unknown_kind_and_makes_unique_ids(self):
        with self.assertRaises(ValueError):
            notebook.new_cell("sql")

        first = notebook.new_cell("python")
        second = notebook.new_cell("python")

        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(first["status"], "idle")
        self.assertEqual((first["outputs"], first["execution_count"]), ([], None))
        self.assertEqual(notebook.new_cell("markdown", metadata={"a": 1})["metadata"], {"a": 1})
        self.assertEqual(
            set(notebook.VALID_KINDS),
            {"markdown", "python", "prompt", "tool", "interaction"},
        )


class TraceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "traces" / "cell-1.jsonl"
        self.addCleanup(self._tmp.cleanup)

    def usage(self):
        return traces.usage_to_dict(
            RunUsage(requests=2, input_tokens=10, output_tokens=4, tool_calls=1)
        )

    def write_trace(self):
        usage = self.usage()
        traces.write_run(
            self.path,
            cell_id="cell-1",
            run_id="run-1",
            model="openai:gpt-4o",
            prompt="map it",
        )
        traces.append_step(self.path, {"kind": "call", "name": "list_files"})
        traces.append_step(self.path, {"kind": "result", "name": "list_files"})
        traces.append_resume(
            self.path, run_id="run-1", conversation_id="conv-1", response={"approved": True}
        )
        traces.append_messages(self.path, conversation())
        traces.append_result(
            self.path,
            status="done",
            output="ok",
            usage=usage,
            conversation_id="conv-9",
        )
        return usage

    def test_read_trace_recovers_the_run(self):
        usage = self.write_trace()

        got = traces.read_trace(self.path)

        self.assertEqual(got["run_id"], "run-1")
        self.assertEqual(got["model"], "openai:gpt-4o")
        self.assertEqual(got["prompt"], "map it")
        self.assertEqual(
            got["steps"],
            [{"kind": "call", "name": "list_files"}, {"kind": "result", "name": "list_files"}],
        )
        self.assertEqual(got["status"], "done")
        self.assertEqual(got["output"], "ok")
        self.assertIsNone(got["error"])
        self.assertEqual(got["usage"], usage)
        self.assertEqual(got["conversation_id"], "conv-9")
        self.assertEqual(got["messages"], conversation())
        self.assertEqual(traces.read_messages(self.path), conversation())

    def test_read_trace_tolerates_a_truncated_last_line(self):
        self.write_trace()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write('{"type": "step", "step": {"kind": "partial"')  # crash mid-write

        got = traces.read_trace(self.path)

        self.assertEqual(got["status"], "done")
        self.assertEqual(got["conversation_id"], "conv-9")
        self.assertEqual(len(got["steps"]), 2)

    def test_read_trace_drops_a_corrupt_message_payload(self):
        self.write_trace()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "messages", "messages": [{"bogus": True}]}) + "\n")

        got = traces.read_trace(self.path)

        self.assertEqual(got["messages"], [])
        self.assertEqual(got["status"], "done")

    def test_read_trace_of_a_missing_file_is_empty(self):
        missing = Path(self._tmp.name) / "nope.jsonl"

        got = traces.read_trace(missing)

        self.assertEqual(got["steps"], [])
        self.assertEqual(got["messages"], [])
        for key in ("run_id", "conversation_id", "model", "prompt", "status", "output", "error", "usage"):
            self.assertIsNone(got[key], key)
        self.assertEqual(traces.read_messages(missing), [])

    def test_rerunning_a_cell_replaces_its_trace(self):
        self.write_trace()

        traces.write_run(
            self.path,
            cell_id="cell-1",
            run_id="run-2",
            model="openai:gpt-4o",
            prompt="again",
        )

        got = traces.read_trace(self.path)

        self.assertEqual(got["run_id"], "run-2")
        self.assertEqual(got["prompt"], "again")
        self.assertEqual(got["steps"], [])
        self.assertIsNone(got["status"])
        self.assertIsNone(got["conversation_id"])

    def test_now_iso_is_a_utc_second_resolution_timestamp(self):
        stamp = traces.now_iso()

        parsed = datetime.fromisoformat(stamp)

        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)
        self.assertEqual(parsed.microsecond, 0)

    def test_usage_to_dict_flattens_provider_cache_details(self):
        flattened = traces.usage_to_dict(
            RunUsage(
                requests=1,
                input_tokens=5,
                details={"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 9},
            )
        )

        self.assertEqual(flattened["requests"], 1)
        self.assertEqual(flattened["input_tokens"], 5)
        self.assertEqual(flattened["output_tokens"], 0)
        self.assertEqual(flattened["total_tokens"], 5)
        self.assertEqual(flattened["tool_calls"], 0)
        self.assertEqual(flattened["cache_read_tokens"], 7)
        self.assertEqual(flattened["cache_write_tokens"], 9)
        self.assertIsNone(flattened["cost"])

    def test_usage_to_dict_prefers_reported_cache_tokens(self):
        flattened = traces.usage_to_dict(
            RunUsage(
                requests=1,
                input_tokens=5,
                cache_read_tokens=3,
                cache_write_tokens=4,
                details={"prompt_cache_hit_tokens": 7, "prompt_cache_miss_tokens": 9},
            )
        )

        self.assertEqual(flattened["cache_read_tokens"], 3)
        self.assertEqual(flattened["cache_write_tokens"], 4)

    def test_appending_creates_the_traces_directory(self):
        self.assertFalse(self.path.parent.exists())

        traces.append_step(self.path, {"kind": "call", "name": "read_file"})

        self.assertTrue(self.path.is_file())
        self.assertEqual(len(traces.read_trace(self.path)["steps"]), 1)


if __name__ == "__main__":
    unittest.main()
