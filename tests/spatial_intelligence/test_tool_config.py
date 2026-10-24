"""Tool Configuration: policy, catalog, persistence, and session application."""

from __future__ import annotations

import ast
import json
import os
import queue
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient

from spatial_intelligence.contracts.services import VISION_SERVICE
from spatial_intelligence.pythonruntime import help as api_help
from spatial_intelligence.pythonruntime.policy import (
    CORE_IMPORTS,
    METHOD_OPTIONS,
    PythonPolicy,
)
from spatial_intelligence.pythonruntime.sandbox import ALLOWED_IMPORTS, guard
from spatial_intelligence.server import deps
from spatial_intelligence.server.app import create_app
from spatial_intelligence.tools.catalog import catalog_payload, package_names
from spatial_intelligence.tools.config import (
    REQUIRED_TOOL_REASONS,
    apply_tool_config,
    normalize_tool_config,
    rejected_disables,
)
from spatial_intelligence.tools.build import declared_specs, default_registry
from spatial_intelligence.tools.runtime import ToolRuntime
from spatial_intelligence.workspace import Workspace


class _SupportedVision:
    supported = True


class ConfigModelTests(unittest.TestCase):
    def test_normalization_sorts_and_deduplicates_names(self):
        normalized = normalize_tool_config(
            {
                "disabled_tools": ["buffer", "buffer", " detect_objects ", ""],
                "python": {
                    "disabled_packages": ["numpy", "numpy"],
                    "disabled_methods": ["ws.write_text", None],
                },
            }
        )

        self.assertEqual(normalized["disabled_tools"], ["buffer", "detect_objects"])
        self.assertEqual(normalized["python"]["disabled_packages"], ["numpy"])
        self.assertEqual(normalized["python"]["disabled_methods"], ["ws.write_text"])

    def test_missing_or_malformed_values_become_the_empty_config(self):
        empty = {
            "disabled_tools": [],
            "python": {"disabled_packages": [], "disabled_methods": []},
        }
        self.assertEqual(normalize_tool_config(None), empty)
        self.assertEqual(normalize_tool_config([1, 2]), empty)
        self.assertEqual(
            normalize_tool_config({"disabled_tools": "numpy", "python": []}), empty
        )

    def test_required_tools_cannot_be_disabled(self):
        self.assertEqual(
            rejected_disables({"disabled_tools": ["run_python", "buffer"]}),
            ["run_python"],
        )
        for name in REQUIRED_TOOL_REASONS:
            with self.subTest(name=name):
                self.assertIn(name, rejected_disables({"disabled_tools": [name]}))

    def test_apply_removes_configurable_tools_but_keeps_required_and_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / "ws").create()
            runtime = ToolRuntime(
                workspace=workspace, services={VISION_SERVICE: _SupportedVision()}
            )
            registry = default_registry(runtime)
            removed = apply_tool_config(
                registry,
                {
                    "disabled_tools": [
                        "detect_objects",
                        "buffer",
                        "run_python",
                        "write_plan",
                    ]
                },
            )

            self.assertEqual(removed, ["buffer", "detect_objects"])
            self.assertNotIn("buffer", registry)
            self.assertNotIn("detect_objects", registry)
            self.assertIn("run_python", registry)
            self.assertIn("write_plan", registry)
            self.assertIn("analyze_image", registry)

    def test_catalog_lists_required_model_specific_and_python_options(self):
        payload = catalog_payload({"disabled_tools": ["detect_objects"]}, False)
        rows = {
            tool["name"]: tool
            for category in payload["categories"]
            for tool in category["tools"]
        }

        self.assertTrue(rows["run_python"]["required"])
        self.assertTrue(rows["run_python"]["enabled"])
        self.assertTrue(rows["request_user_input"]["required"])
        self.assertFalse(rows["run_python"]["configurable"])
        self.assertTrue(rows["detect_objects"]["model_specific"])
        self.assertTrue(rows["detect_objects"]["configurable"])
        self.assertFalse(rows["detect_objects"]["enabled"])
        self.assertFalse(rows["analyze_image"]["available"])
        self.assertFalse(rows["write_plan"]["configurable"])
        self.assertTrue(rows["write_plan"]["enabled"])

        self.assertEqual(
            payload["python"]["package_groups"][-1], "Optional extras"
        )
        package_rows = {row["name"]: row for row in payload["python"]["packages"]}
        self.assertTrue(package_rows["numpy"]["installed"])
        self.assertEqual(
            {row["name"] for row in payload["python"]["methods"]},
            {option.name for option in METHOD_OPTIONS},
        )

    def test_every_safe_import_and_listed_method_is_accounted_for(self):
        self.assertEqual({option.name for option in CORE_IMPORTS}, set(ALLOWED_IMPORTS))
        self.assertTrue(package_names() >= set(ALLOWED_IMPORTS))

    def test_declared_catalog_matches_the_live_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / "ws").create()
            registry = default_registry(ToolRuntime(workspace=workspace))
            live = {spec.name for spec in registry.specs()}
        declared = {spec.name for spec in declared_specs()}
        # The static catalog also carries the vision tool a text-only model omits.
        self.assertEqual(declared - live, {"analyze_image"})


class PythonPolicyTests(unittest.TestCase):
    def test_guard_refuses_disabled_core_and_optional_packages(self):
        policy = PythonPolicy.from_config(
            {"python": {"disabled_packages": ["numpy", "torch"]}}
        )

        self.assertEqual(
            guard(ast.parse("import numpy"), policy),
            "import of 'numpy' is disabled in Tool Configuration",
        )
        self.assertEqual(
            guard(ast.parse("from torch import nn"), policy),
            "import from 'torch' is disabled in Tool Configuration",
        )
        self.assertIsNone(guard(ast.parse("import pandas"), policy))

    def test_namespace_removes_disabled_packages_and_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / "ws").create()
            policy = PythonPolicy.from_config(
                {
                    "python": {
                        "disabled_packages": ["numpy"],
                        "disabled_methods": ["ws.read_text", "vision.ask", "models.path"],
                    }
                }
            )
            from spatial_intelligence.pythonruntime.executor import PythonExecutor

            executor = PythonExecutor(workspace, policy=policy)
            namespace = executor.namespace()

            self.assertNotIn("numpy", namespace)
            self.assertNotIn("np", namespace)
            self.assertIn("pandas", namespace)
            self.assertIn("ws", namespace)
            self.assertIn("models", namespace)
            self.assertNotIn("vision", namespace)
            with self.assertRaises(AttributeError):
                namespace["ws"].read_text("data/x.txt")
            with self.assertRaises(AttributeError):
                namespace["models"].path("org/repo")
            self.assertEqual(namespace["ws"].list_files("data"), [])

    def test_help_roots_and_methods_follow_the_policy(self):
        policy = PythonPolicy.from_config(
            {
                "python": {
                    "disabled_packages": ["rasterio"],
                    "disabled_methods": [
                        "ws.resolve",
                        "ws.read_text",
                        "ws.write_text",
                        "ws.list_files",
                    ],
                }
            }
        )

        roots = api_help.describe("", policy)["roots"]
        self.assertNotIn("rasterio", roots)
        self.assertNotIn("ws", roots)
        self.assertIn("models", roots)
        self.assertEqual(
            api_help.describe("rasterio", policy)["kind"],
            "error",
        )
        self.assertEqual(
            api_help.describe("ws.write_text", policy)["doc"],
            "ws is disabled in Tool Configuration",
        )

    def test_prompt_block_names_disabled_imports_once(self):
        from spatial_intelligence.agent.prompt import system_prompt

        policy = PythonPolicy.from_config(
            {"python": {"disabled_packages": ["numpy"], "disabled_methods": []}}
        )
        prompt = system_prompt(policy, ["detect_objects", "buffer"])

        self.assertIn("Tool Configuration disabled imports numpy", prompt)
        self.assertIn("disabled these tools: buffer, detect_objects", prompt)
        self.assertNotIn("Tool Configuration disabled", system_prompt())


class ToolConfigApiTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._previous_home = os.environ.get("GEOAI_HOME")
        os.environ["GEOAI_HOME"] = self._tmp.name
        deps.close_app_state()
        self.client = TestClient(create_app())

    def tearDown(self):
        deps.close_app_state()
        if self._previous_home is None:
            os.environ.pop("GEOAI_HOME", None)
        else:
            os.environ["GEOAI_HOME"] = self._previous_home
        self._tmp.cleanup()

    def open_workspace(self, name: str = "demo") -> dict:
        response = self.client.post("/api/workspace/new", json={"name": name})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_the_catalog_is_available_before_a_workspace_opens(self):
        response = self.client.get("/api/tools/config")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"], "global")
        self.assertIsNone(payload["workspace"])
        self.assertIn("categories", payload)
        self.assertIn("python", payload)
        self.assertEqual(payload["effective"]["disabled_tools"], [])
        self.assertTrue(payload["required_tools"]["run_python"])

    def test_a_global_save_persists_and_filters_the_open_session(self):
        self.open_workspace()
        state = deps.app_state()
        self.assertIn("detect_objects", state.services.registry)

        response = self.client.put(
            "/api/tools/config",
            json={
                "scope": "global",
                "config": {
                    "disabled_tools": ["detect_objects"],
                    "python": {
                        "disabled_packages": ["numpy"],
                        "disabled_methods": ["ws.write_text"],
                    },
                },
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["source"], "global")
        self.assertEqual(payload["effective"]["disabled_tools"], ["detect_objects"])
        self.assertNotIn("detect_objects", state.services.registry)
        self.assertEqual(
            state.services.runtime.python_policy.disabled_packages, {"numpy"}
        )

        persisted = json.loads(
            (Path(self._tmp.name) / "workspaces" / "tools.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(persisted["disabled_tools"], ["detect_objects"])
        settings_path = Path(self._tmp.name) / "settings.json"
        if settings_path.exists():
            self.assertNotIn(
                "tool_config",
                json.loads(settings_path.read_text(encoding="utf-8")),
            )

        executor = state.services.runtime.services["python.executor"]
        self.assertIn(
            "run_python blocked: import of 'numpy' is disabled in Tool Configuration",
            executor.run("import numpy"),
        )
        self.assertNotIn("np", executor.namespace())

    def test_a_workspace_save_overrides_global_and_uses_workspace_tools_json(self):
        self.open_workspace()
        response = self.client.put(
            "/api/tools/config",
            json={
                "scope": "workspace",
                "config": {
                    "disabled_tools": ["buffer"],
                    "python": {"disabled_packages": [], "disabled_methods": []},
                },
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["source"], "workspace")
        self.assertEqual(payload["effective"]["disabled_tools"], ["buffer"])
        self.assertIsNotNone(payload["workspace"])
        workspace_root = Path(self._tmp.name) / "workspaces" / "demo"
        persisted = json.loads(
            (workspace_root / "tools.json").read_text(encoding="utf-8")
        )
        self.assertEqual(persisted["disabled_tools"], ["buffer"])
        manifest = json.loads(
            (workspace_root / "workspace.json").read_text(encoding="utf-8")
        )
        self.assertNotIn("tool_config", manifest)
        self.assertNotIn("buffer", deps.app_state().services.registry)

        # A second workspace has no override, so it follows the global config.
        second = self.open_workspace("second")
        self.assertEqual(second["tool_config_source"], "global")
        self.assertIn("buffer", deps.app_state().services.registry)

    def test_required_tools_are_rejected_with_their_reason(self):
        response = self.client.put(
            "/api/tools/config",
            json={
                "scope": "global",
                "config": {"disabled_tools": ["run_python", "buffer"]},
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("run_python", response.json()["detail"])
        self.assertNotIn("tool_config", self.client.get("/api/settings").json())
        self.assertFalse(
            (Path(self._tmp.name) / "workspaces" / "tools.json").exists()
        )

    def test_workspace_scope_needs_an_open_workspace(self):
        response = self.client.put(
            "/api/tools/config",
            json={"scope": "workspace", "config": {"disabled_tools": ["buffer"]}},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("open a workspace", response.json()["detail"])

    def test_a_global_save_without_a_workspace_writes_the_workspaces_root_file(self):
        response = self.client.put(
            "/api/tools/config",
            json={
                "scope": "global",
                "config": {"disabled_tools": ["buffer"]},
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["source"], "global")
        path = Path(self._tmp.name) / "workspaces" / "tools.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["disabled_tools"], ["buffer"])

    def test_legacy_settings_and_manifest_configs_are_migrated(self):
        home = Path(self._tmp.name)
        settings = {
            "model": "openai:gpt-4o",
            "theme": "dark",
            "tool_config": {
                "disabled_tools": ["buffer"],
                "python": {"disabled_packages": ["numpy"], "disabled_methods": []},
            },
        }
        (home / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        workspace = home / "workspaces" / "legacy"
        workspace.mkdir(parents=True)
        manifest = {
            "name": "legacy",
            "created_at": "2026-01-01T00:00:00+00:00",
            "outputs": [],
            "version": 0,
            "tool_config": {
                "disabled_tools": ["detect_objects"],
                "python": {"disabled_packages": [], "disabled_methods": []},
            },
        }
        (workspace / "workspace.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        deps.app_state()  # the session start runs the one-time migration

        global_tools = json.loads(
            (home / "workspaces" / "tools.json").read_text(encoding="utf-8")
        )
        self.assertEqual(global_tools["disabled_tools"], ["buffer"])
        self.assertEqual(global_tools["python"]["disabled_packages"], ["numpy"])
        self.assertNotIn(
            "tool_config", json.loads((home / "settings.json").read_text(encoding="utf-8"))
        )
        workspace_tools = json.loads(
            (workspace / "tools.json").read_text(encoding="utf-8")
        )
        self.assertEqual(workspace_tools["disabled_tools"], ["detect_objects"])
        self.assertNotIn(
            "tool_config",
            json.loads((workspace / "workspace.json").read_text(encoding="utf-8")),
        )

    def test_a_save_broadcasts_the_effective_config(self):
        state = deps.app_state()
        subscriber = state.subscribe()
        self.client.put(
            "/api/tools/config",
            json={"scope": "global", "config": {"disabled_tools": ["buffer"]}},
        )

        events = []
        while True:
            try:
                events.append(subscriber.get_nowait())
            except queue.Empty:
                break
        state.unsubscribe(subscriber)

        published = [item for item in events if item["event"] == "tool_config"]
        self.assertTrue(published)
        self.assertEqual(published[-1]["data"]["source"], "global")
        self.assertEqual(
            published[-1]["data"]["effective"]["disabled_tools"], ["buffer"]
        )


if __name__ == "__main__":
    unittest.main()
