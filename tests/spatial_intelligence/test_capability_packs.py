"""Capability discovery and the interactive handoff tool.

The catalog is static metadata; the registry is the authority on what this
build can execute. These tests pin the ranking, the registry annotation, and
the structured form the deferred interaction hands to the browser.
"""

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai import CallDeferred, ToolReturn

from spatial_intelligence import discovery
from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.tools import ToolKind, ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.capabilities import CapabilityPack
from spatial_intelligence.tools.packs.files import FilesPack
from spatial_intelligence.tools.packs.interaction import (
    ChoiceOption,
    InteractionField,
    InteractionPack,
)
from spatial_intelligence.workspace import Workspace


def files_capability(found: list[dict]) -> dict:
    return next(item for item in found if item["id"] == "workspace.files")


class CapabilityCatalogTests(unittest.TestCase):
    """Scoring the static catalog, with and without a registry to check it."""

    def test_a_flood_imagery_goal_ranks_the_disaster_catalog_first(self):
        found = discovery.discover(None, "flood imagery download")

        self.assertEqual(found[0]["id"], "catalog.disaster-imagery")
        self.assertEqual(found[0]["status"], "available")
        self.assertEqual(found[0]["geolibre_plugins"], ["Vantor Open Data", "OpenAerialMap"])

    def test_limit_is_clamped_to_the_range_the_catalog_can_answer(self):
        self.assertEqual(len(discovery.discover(None, "data map raster", limit=2)), 2)
        self.assertEqual(len(discovery.discover(None, "data map raster", limit=0)), 1)
        self.assertLessEqual(len(discovery.discover(None, "anything", limit=100)), 8)

    def test_registered_tools_are_summarised_and_unknown_names_are_dropped(self):
        registry = ToolRegistry()
        registry.add_external(
            "list_files", category="files", origin="test", summary="List workspace files."
        )

        found = discovery.discover(registry, "workspace file data import")
        files = files_capability(found)

        # Only the registered tool survives, carrying the registry's summary
        # rather than a summary invented from the name.
        self.assertEqual(
            files["tools"], [{"name": "list_files", "summary": "List workspace files."}]
        )
        self.assertNotIn("download", [entry["name"] for entry in files["tools"]])

    def test_a_capability_whose_tools_are_all_unregistered_reports_none(self):
        found = discovery.discover(ToolRegistry(), "flood imagery download")

        self.assertEqual(found[0]["id"], "catalog.disaster-imagery")
        self.assertEqual(found[0]["tools"], [])

    def test_without_a_registry_names_stay_but_summaries_do_not(self):
        found = discovery.discover(None, "workspace file data import")
        files = files_capability(found)

        self.assertEqual(
            [entry["name"] for entry in files["tools"]],
            [
                "list_files",
                "find_files",
                "read_file",
                "write_file",
                "download",
                "download_files",
            ],
        )
        self.assertTrue(all(entry["summary"] is None for entry in files["tools"]))

    def test_python_execution_ranks_for_code_goals(self):
        found = discovery.discover(None, "execute Python code")

        self.assertEqual(found[0]["id"], "python.execution")
        self.assertIn("run_python", [entry["name"] for entry in found[0]["tools"]])

    def test_interactive_handoffs_carry_prose_the_agent_can_relay(self):
        found = discovery.discover(None, "compare before after terrain overture planet", limit=8)

        handoffs = [item for item in found if item["status"] == "interactive_handoff"]
        # Swipe and terrain used to be listed here; both now have tools, so only
        # the two capabilities this build genuinely cannot perform are handoffs.
        self.assertEqual(
            sorted(item["id"] for item in handoffs),
            ["catalog.overture", "catalog.planet-stac"],
        )
        for item in handoffs:
            with self.subTest(capability=item["id"]):
                self.assertTrue(item["fallback"])
                self.assertEqual(item["tools"], [])

        executable = next(item for item in found if item["id"] == "catalog.disaster-imagery")
        self.assertIsNone(executable["fallback"])


class PackTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()

    def tearDown(self):
        self._tmp.cleanup()

    def runtime(self, registry: ToolRegistry | None = None) -> ToolRuntime:
        runtime = ToolRuntime(workspace=self.workspace)
        if registry is not None:
            runtime.services["tools.registry"] = registry
        return runtime

    def pack_registry(self, pack, registry: ToolRegistry | None = None) -> ToolRegistry:
        tools = ToolRegistry()
        tools.add_pack(pack, self.runtime(registry))
        return tools


class CapabilityPackTests(PackTestCase):
    def test_discovery_annotates_results_from_the_session_registry(self):
        session = ToolRegistry()
        session.add_external(
            "list_files", category="files", origin="test", summary="List workspace files."
        )
        registry = self.pack_registry(CapabilityPack, session)

        found = registry.get("discover_capabilities").callable("workspace file data import").return_value

        files = files_capability(found)
        self.assertEqual(
            files["tools"], [{"name": "list_files", "summary": "List workspace files."}]
        )

    def test_discovery_still_ranks_before_the_session_wires_a_registry(self):
        registry = self.pack_registry(CapabilityPack)

        found = registry.get("discover_capabilities").callable("flood imagery download").return_value

        self.assertEqual(found[0]["id"], "catalog.disaster-imagery")
        self.assertTrue(all(entry["summary"] is None for entry in found[0]["tools"]))

    def test_discovery_reveals_the_tools_it_lists(self):
        registry = self.pack_registry(CapabilityPack)

        result = registry.get("discover_capabilities").callable("execute Python code")

        self.assertIsInstance(result, ToolReturn)
        self.assertEqual(result.return_value[0]["id"], "python.execution")
        self.assertIn("run_python", result.tools)

    def test_describe_tool_returns_usage_and_parameters(self):
        session = ToolRegistry()
        session.add_pack(FilesPack, self.runtime())
        registry = self.pack_registry(CapabilityPack, session)

        result = registry.get("describe_tool").callable("read_file")

        self.assertIsInstance(result, ToolReturn)
        info = result.return_value
        self.assertEqual(info["name"], "read_file")
        self.assertEqual(info["category"], "files")
        self.assertIn("Read a UTF-8 text file", info["description"])
        self.assertEqual(
            [param["name"] for param in info["parameters"]],
            ["path", "max_bytes", "offset", "limit"],
        )
        self.assertEqual(result.tools, ["read_file"])

    def test_describe_tool_without_a_name_lists_the_registry(self):
        session = ToolRegistry()
        session.add_external("list_files", category="files", origin="test", summary="List files.")
        registry = self.pack_registry(CapabilityPack, session)

        result = registry.get("describe_tool").callable()

        self.assertEqual(result["tools"][0]["name"], "list_files")

    def test_both_tools_are_core_and_read_only(self):
        registry = ToolRegistry()
        runtime = self.runtime()
        registry.add_pack(CapabilityPack, runtime)
        registry.add_pack(InteractionPack, runtime)

        self.assertEqual(
            registry.core_names(),
            frozenset({"discover_capabilities", "describe_tool", "request_user_input"}),
        )
        self.assertEqual(registry.get("discover_capabilities").effects, frozenset({Effect.READ}))
        self.assertEqual(registry.get("discover_capabilities").kind, ToolKind.SYNC)
        self.assertTrue(registry.replay_safe("discover_capabilities"))
        self.assertTrue(registry.replay_safe("request_user_input"))
        self.assertEqual(
            [spec.name for spec in registry.interactive()], ["request_user_input"]
        )
        self.assertEqual(registry.get("request_user_input").kind, ToolKind.INTERACTIVE)


class InteractionPackTests(PackTestCase):
    def test_request_user_input_defers_the_run_with_the_form(self):
        registry = self.pack_registry(InteractionPack)
        fields = [
            InteractionField(
                id="scene",
                label="Scene",
                type="radio",
                options=[
                    ChoiceOption(value="post", label="Post-event", recommended=True),
                    ChoiceOption(value="pre", label="Pre-event"),
                ],
            )
        ]

        with self.assertRaises(CallDeferred) as raised:
            registry.get("request_user_input").callable(
                "Choose imagery", "Pick a scene.", fields
            )

        form = raised.exception.metadata["interaction"]
        self.assertEqual(form["title"], "Choose imagery")
        self.assertEqual(form["prompt"], "Pick a scene.")
        self.assertEqual(form["submit_label"], "Continue")
        self.assertTrue(form["allow_cancel"])
        self.assertEqual(
            form["fields"],
            [
                {
                    "id": "scene",
                    "label": "Scene",
                    "type": "radio",
                    "description": None,
                    "required": True,
                    "options": [
                        {
                            "value": "post",
                            "label": "Post-event",
                            "description": None,
                            "recommended": True,
                            "thumbnail_url": None,
                            "metadata": {},
                        },
                        {
                            "value": "pre",
                            "label": "Pre-event",
                            "description": None,
                            "recommended": False,
                            "thumbnail_url": None,
                            "metadata": {},
                        },
                    ],
                    "default": None,
                    "placeholder": None,
                }
            ],
        )

    def test_choice_fields_require_options_but_free_text_does_not(self):
        for field_type in ("radio", "multi_select"):
            with self.subTest(field_type=field_type):
                with self.assertRaises(ValidationError) as raised:
                    InteractionField(id="scene", label="Scene", type=field_type)

                self.assertIn("requires options", str(raised.exception))

        self.assertEqual(InteractionField(id="notes", label="Notes", type="text").options, [])


if __name__ == "__main__":
    unittest.main()
