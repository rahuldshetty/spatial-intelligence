"""Tool registry: declaration resolution, derived queries, and agent wiring."""

import asyncio
import json
import tempfile
import textwrap
import types
import unittest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.tools import ToolKind, ToolRegistry, ToolRuntime
from spatial_intelligence.workspace import Workspace

MODULE_SOURCE = """
from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.tools import ToolKind, pack, tool


@tool(category="files")
def alpha(path: str, limit: int = 3) -> str:
    \"\"\"Alpha summary line.

    A longer description the model sees instead of the summary.
    \"\"\"
    return f"{path}:{limit}"


@tool(category="raster", effects=frozenset({Effect.WORKSPACE_WRITE}), core=True)
def beta(path: str) -> str:
    \"\"\"Beta summary.\"\"\"
    return path


@tool(category="catalog", capabilities=("catalog", "sar"))
def gamma(url: str) -> str:
    \"\"\"Gamma summary.\"\"\"
    return url
"""

PACK_SOURCE = """
from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.tools import ToolKind, pack, tool


@pack(category="raster", effects=frozenset({Effect.WORKSPACE_WRITE}), core=True)
class SamplePack:
    \"\"\"A pack whose defaults its methods inherit or override.\"\"\"

    def __init__(self, runtime):
        self.runtime = runtime

    @tool(effects=frozenset({Effect.READ}), core=False)
    def inspect(self, path: str) -> str:
        \"\"\"Inspect a raster.\"\"\"
        return f"inspect({self.runtime.workspace.root.name}):{path}"

    @tool(kind=ToolKind.REPORTING)
    def build(self, path: str) -> str:
        \"\"\"Build a raster.\"\"\"
        job = self.runtime.reporter.job("raster", path)
        with job:
            job.progress(1, total=1)
        return path
"""

AGENT_SOURCE = """
from spatial_intelligence.tools import tool


@tool(category="files", core=True)
def core_probe(path: str) -> str:
    \"\"\"A core tool the model always sees.\"\"\"
    return f"core_probe:{path}"


@tool(category="raster")
def deferred_probe(path: str) -> str:
    \"\"\"A deferred tool that requires discovery first.\"\"\"
    return f"deferred_probe:{path}"
"""


def load_module(source: str, name: str = "sample_tools") -> types.ModuleType:
    """Execute ``source`` as a module so its decorators run for real."""
    module = types.ModuleType(name)
    exec(compile(textwrap.dedent(source), f"<{name}>", "exec"), module.__dict__)
    return module


class RegistryTestCase(unittest.TestCase):
    """Shared temp workspace so a pack has a runtime to construct against."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(self._tmp.name).create()
        self.registry = ToolRegistry()

    def tearDown(self):
        self._tmp.cleanup()

    def runtime(self) -> ToolRuntime:
        return ToolRuntime(workspace=self.workspace)


class DeclarationTests(RegistryTestCase):
    def test_module_tools_inherit_package_defaults(self):
        specs = self.registry.add_module(load_module(MODULE_SOURCE))

        self.assertEqual([spec.name for spec in specs], ["alpha", "beta", "gamma"])
        alpha = self.registry.get("alpha")
        self.assertEqual(alpha.category, "files")
        self.assertEqual(alpha.effects, frozenset({Effect.READ}))
        self.assertIs(alpha.kind, ToolKind.SYNC)
        self.assertFalse(alpha.core)
        self.assertIsNone(alpha.timeout)
        self.assertEqual(alpha.summary, "Alpha summary line.")
        self.assertEqual(alpha.origin, "builtin")
        self.assertTrue(alpha.implemented)

    def test_declared_effects_and_core_are_honoured(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        beta = self.registry.get("beta")
        self.assertEqual(beta.effects, frozenset({Effect.WORKSPACE_WRITE}))
        self.assertTrue(beta.core)
        self.assertEqual(beta.summary, "Beta summary.")
        self.assertEqual(self.registry.core_names(), frozenset({"beta"}))

    def test_capabilities_and_categories_are_queryable(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        self.assertEqual(
            [spec.name for spec in self.registry.with_capability("sar")], ["gamma"]
        )
        self.assertEqual(
            self.registry.categories(),
            {"files": ("alpha",), "raster": ("beta",), "catalog": ("gamma",)},
        )
        self.assertEqual(self.registry.category_of("alpha"), "files")
        self.assertIsNone(self.registry.category_of("unknown_tool"))
        self.assertEqual(len(self.registry), 3)
        self.assertIn("alpha", self.registry)

    def test_replay_safety_derives_from_effects(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        self.assertTrue(self.registry.replay_safe("alpha"))
        self.assertFalse(self.registry.replay_safe("beta"))

    def test_unknown_tools_are_treated_as_state_changing(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        self.assertFalse(self.registry.replay_safe("never_registered"))
        self.assertFalse(self.registry.replay_safe(None))
        self.assertFalse(self.registry.replay_safe(""))

    def test_map_writes_are_not_replay_safe(self):
        module = load_module(
            """
from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.tools import tool


@tool(category="layers", effects=frozenset({Effect.MAP_WRITE}))
def add_layer(name: str) -> str:
    \"\"\"Add a layer.\"\"\"
    return name
"""
        )
        self.registry.add_module(module)

        self.assertFalse(self.registry.get("add_layer").replay_safe)

    def test_duplicate_names_are_rejected(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        with self.assertRaises(ValueError) as caught:
            self.registry.add_module(load_module(MODULE_SOURCE))

        self.assertIn("already registered", str(caught.exception))

    def test_missing_category_is_rejected(self):
        module = load_module(
            """
from spatial_intelligence.tools import tool


@tool()
def orphan(x: int = 1) -> int:
    \"\"\"Declared without a category.\"\"\"
    return x
"""
        )

        with self.assertRaises(TypeError) as caught:
            self.registry.add_module(module)

        self.assertIn("category", str(caught.exception))

    def test_external_tools_register_without_an_implementation(self):
        self.registry.add_external(
            "write_plan",
            category="plan",
            origin="harness",
            effects=frozenset({Effect.READ}),
        )

        spec = self.registry.get("write_plan")
        self.assertFalse(spec.implemented)
        self.assertTrue(spec.replay_safe)
        self.assertEqual(self.registry.implemented(), [])
        self.assertEqual(self.registry.category_of("write_plan"), "plan")
        self.assertEqual(self.registry.by_category("plan"), [spec])


class PackTests(RegistryTestCase):
    def test_pack_defaults_are_inherited_then_overridden(self):
        module = load_module(PACK_SOURCE)
        self.registry.add_pack(module.SamplePack, self.runtime())

        inspected = self.registry.get("inspect")
        self.assertEqual(inspected.category, "raster")  # inherited from the pack
        self.assertEqual(inspected.effects, frozenset({Effect.READ}))  # method wins
        self.assertFalse(inspected.core)

        built = self.registry.get("build")
        self.assertEqual(built.category, "raster")
        self.assertEqual(built.effects, frozenset({Effect.WORKSPACE_WRITE}))  # inherited
        self.assertTrue(built.core)  # inherited
        self.assertIs(built.kind, ToolKind.REPORTING)

    def test_pack_methods_are_bound_to_one_instance(self):
        module = load_module(PACK_SOURCE)
        self.registry.add_pack(module.SamplePack, self.runtime())

        spec = self.registry.get("inspect")

        self.assertEqual(
            spec.callable("data/x.tif"),
            f"inspect({self.workspace.root.name}):data/x.tif",
        )

    def test_undeclared_class_is_rejected(self):
        class NotAPack:
            pass

        with self.assertRaises(TypeError):
            self.registry.add_pack(NotAPack, self.runtime())

    def test_declarative_spec_cannot_be_called(self):
        self.registry.add_external("read_plan", category="plan", origin="harness")

        with self.assertRaises(TypeError):
            self.registry.get("read_plan").callable()


class AgentWiringTests(RegistryTestCase):
    def test_build_exposes_core_tools_and_defers_the_rest(self):
        self.registry.add_module(load_module(AGENT_SOURCE))
        rounds: list[list[str]] = []

        def model_fn(messages, info: AgentInfo) -> ModelResponse:
            rounds.append([definition.name for definition in (info.function_tools or [])])
            if len(rounds) == 1:
                return ModelResponse(
                    parts=[ToolCallPart("core_probe", {"path": "data/x.txt"})]
                )
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(FunctionModel(model_fn), output_type=str)
        self.registry.build(agent)
        result = asyncio.run(agent.run("go"))

        self.assertEqual(result.output, "done")
        self.assertIn("core_probe", rounds[0])
        self.assertNotIn("deferred_probe", rounds[0])
        returned = [
            part.content
            for message in result.all_messages()
            for part in getattr(message, "parts", [])
            if isinstance(part, ToolReturnPart)
        ]
        self.assertEqual(returned, ["core_probe:data/x.txt"])

    def test_model_metadata_is_json_safe(self):
        self.registry.add_module(load_module(MODULE_SOURCE))

        metadata = self.registry.get("beta").model_metadata()

        self.assertEqual(metadata["si_tool"], "beta")
        self.assertEqual(metadata["si_category"], "raster")
        self.assertEqual(metadata["si_effects"], ["workspace_write"])
        json.dumps(metadata)  # must survive a trip to the model


if __name__ == "__main__":
    unittest.main()
