"""Python sandbox: the guard, the executor, the output store, and the pack."""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from spatial_intelligence.contracts.effects import Effect
from spatial_intelligence.pythonruntime.executor import PythonExecutor
from spatial_intelligence.pythonruntime.output import OutputStore
from spatial_intelligence.pythonruntime.sandbox import dotted_name, guard
from spatial_intelligence.tools import ToolRegistry, ToolRuntime
from spatial_intelligence.tools.packs.python_ import PythonPack
from spatial_intelligence.workspace import Workspace


class PythonTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "workspace").create()
        self.addCleanup(self._tmp.cleanup)

    def executor(self, *, approved: bool = False) -> PythonExecutor:
        return PythonExecutor(self.workspace, approved=approved)

    def build_pack(self, *, approved: bool = False):
        runtime = ToolRuntime(workspace=self.workspace, approved=approved)
        registry = ToolRegistry()
        registry.add_pack(PythonPack, runtime)
        return registry, runtime

    def manifest_outputs(self) -> list[str]:
        manifest = json.loads(self.workspace.manifest_path.read_text(encoding="utf-8"))
        return manifest["outputs"]


class SandboxGuardTests(unittest.TestCase):
    def test_allows_the_geospatial_stack_and_basic_stdlib(self):
        code = (
            "import numpy as np\n"
            "import os, os.path, pathlib, json, shutil, tempfile, csv, math\n"
            "from shapely.geometry import Point\n"
            "rasterio.open('data/a.tif')\n"
            "os.path.join('a', 'b')\n"
        )

        self.assertIsNone(guard(ast.parse(code)))

    def test_allows_the_image_processing_stack(self):
        code = (
            "import skimage\n"
            "import scipy\n"
            "import PIL, PIL.Image\n"
            "import rio_cogeo\n"
            "import tifffile\n"
            "from skimage import filters, measure, morphology, segmentation\n"
            "scipy.ndimage.gaussian_filter(crop, 1.0)\n"
        )

        self.assertIsNone(guard(ast.parse(code)))

    def test_only_calls_are_blocked_not_bare_attribute_access(self):
        self.assertIsNone(guard(ast.parse("import os\nrunner = os.system")))

    def test_rejects_every_disallowed_import_root(self):
        cases = {
            "import subprocess": "import of 'subprocess' is not allowed in run_python",
            "import socket": "import of 'socket' is not allowed in run_python",
            "import os, ctypes": "import of 'ctypes' is not allowed in run_python",
            "import urllib.request": "import of 'urllib.request' is not allowed in run_python",
            "import importlib": "import of 'importlib' is not allowed in run_python",
            "from subprocess import run": "import from 'subprocess' is not allowed in run_python",
            "from socket import socket": "import from 'socket' is not allowed in run_python",
        }

        for code, message in cases.items():
            with self.subTest(code=code):
                self.assertEqual(guard(ast.parse(code)), message)

    def test_rejects_dynamic_execution_and_raw_command_calls(self):
        cases = {
            "open('results/a.txt')": "open() is not allowed in run_python",
            "__import__('os')": "__import__() is not allowed in run_python",
            "eval('1 + 1')": "eval() is not allowed in run_python",
            "exec('x = 1')": "exec() is not allowed in run_python",
            "compile('x', '<s>', 'eval')": "compile() is not allowed in run_python",
            "input()": "input() is not allowed in run_python",
            "breakpoint()": "breakpoint() is not allowed in run_python",
            "import os\nos.system('whoami')": (
                "os.system() is not allowed in run_python (use dangerous mode)"
            ),
            "import os\nos.popen('whoami')": (
                "os.popen() is not allowed in run_python (use dangerous mode)"
            ),
            "import os\nos.execv('/bin/sh', ['sh'])": (
                "os.execv() is not allowed in run_python (use dangerous mode)"
            ),
            "import os\nos.startfile('a.txt')": (
                "os.startfile() is not allowed in run_python (use dangerous mode)"
            ),
        }

        for code, message in cases.items():
            with self.subTest(code=code):
                self.assertEqual(guard(ast.parse(code)), message)

    def test_dotted_name_reconstructs_attribute_chains(self):
        call = ast.parse("os.path.join('a', 'b')").body[0].value
        self.assertEqual(dotted_name(call.func), "os.path.join")
        self.assertEqual(dotted_name(ast.parse("len(x)").body[0].value.func), "len")
        self.assertIsNone(dotted_name(ast.parse("f()()").body[0].value.func))


class PythonExecutorTests(PythonTestCase):
    def test_safe_mode_blocks_escape_vectors_with_a_clear_message(self):
        executor = self.executor()
        cases = {
            "import subprocess": "import of 'subprocess' is not allowed in run_python",
            "import os\nos.system('whoami')": (
                "os.system() is not allowed in run_python (use dangerous mode)"
            ),
            "open('results/blocked.txt', 'w')": "open() is not allowed in run_python",
            "eval('1 + 1')": "eval() is not allowed in run_python",
            "__import__('os').getcwd()": "__import__() is not allowed in run_python",
        }

        for code, message in cases.items():
            with self.subTest(code=code):
                preview = executor.run(code)

                self.assertTrue(preview.startswith("run_python blocked:"))
                self.assertIn(message, preview)
                self.assertIn("enable dangerous mode", preview)
                self.assertEqual(executor.output.text, preview)

        self.assertFalse((self.workspace.results / "blocked.txt").exists())

    def test_allowed_code_runs_against_the_geospatial_stack(self):
        executor = self.executor()

        self.assertEqual(
            executor.run("import os.path\nos.path.basename('/a/b.txt')"), "=> 'b.txt'"
        )
        self.assertEqual(executor.run("np.arange(4).tolist()"), "=> [0, 1, 2, 3]")
        self.assertEqual(
            executor.run(
                "(np.__name__, pd.__name__, gpd.__name__, rasterio.__name__, "
                "rioxarray.__name__, xr.__name__, shapely.__name__, pyproj.__name__)"
            ),
            "=> ('numpy', 'pandas', 'geopandas', 'rasterio', 'rioxarray', 'xarray', "
            "'shapely', 'pyproj')",
        )
        self.assertEqual(
            executor.run(
                "(int(pd.Series([1, 2, 3]).sum()), "
                "pyproj.CRS.from_epsg(4326).to_epsg(), "
                "gpd.GeoSeries([shapely.Point(0, 0)]).geom_type.tolist(), "
                "int(xr.DataArray(np.arange(3), dims='x').sum()))"
            ),
            "=> (6, 4326, ['Point'], 3)",
        )

    def test_relative_paths_resolve_from_the_workspace_root(self):
        executor = self.executor()
        (self.workspace.data / "countries.geojson").write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"POP_EST": 1_400_000_000},
                            "geometry": {"type": "Point", "coordinates": [0, 0]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        preview = executor.run(
            "import os\n"
            "frame = gpd.read_file('data/countries.geojson')\n"
            "(os.getcwd(), len(frame), int(frame['POP_EST'].iloc[0]))"
        )

        self.assertEqual(
            preview, "=> (" + repr(str(self.workspace.root)) + ", 1, 1400000000)"
        )

        self.assertEqual(
            executor.run(
                "gpd.read_file('data/countries.geojson')"
                ".to_file('results/copy.geojson', driver='GeoJSON')\n"
                "'written'"
            ),
            "=> 'written'",
        )
        self.assertTrue((self.workspace.results / "copy.geojson").is_file())

    def test_ws_reads_writes_and_lists_inside_the_workspace(self):
        executor = self.executor()

        preview = executor.run(
            "path = ws.write_text('results/notes.txt', 'hello')\n"
            "(ws.read_text('results/notes.txt'), ws.list_files('results'), "
            "path.endswith('notes.txt'))"
        )

        self.assertEqual(preview, "=> ('hello', ['results/notes.txt'], True)")
        self.assertEqual(
            (self.workspace.results / "notes.txt").read_text(encoding="utf-8"), "hello"
        )
        self.assertEqual(self.manifest_outputs(), ["results/notes.txt"])

    def test_ws_rejects_paths_outside_the_workspace(self):
        executor = self.executor()

        escaped = executor.run("ws.write_text('../escape.txt', 'x')")
        self.assertIn("Traceback (most recent call last)", escaped)
        self.assertIn("escapes", escaped)
        self.assertFalse((self.workspace.root.parent / "escape.txt").exists())

        outside_writable = executor.run("ws.write_text('traces/a.txt', 'x')")
        self.assertIn("write target must be under results/, maps/, or data/", outside_writable)
        self.assertFalse((self.workspace.traces / "a.txt").exists())

        missing = executor.run("ws.read_text('data/missing.txt')")
        self.assertIn("file does not exist", missing)

    def test_an_exception_comes_back_as_a_traceback_instead_of_raising(self):
        executor = self.executor()

        preview = executor.run("value = 1\nraise ValueError('boom')")

        self.assertIn("Traceback (most recent call last)", preview)
        self.assertIn("ValueError: boom", preview)

        self.assertIn("ZeroDivisionError", executor.run("1 / 0"))

    def test_stdout_and_the_last_expression_are_both_captured(self):
        executor = self.executor()

        preview = executor.run("print('rows: 2')\n{'rows': [1, 2]}")

        self.assertEqual(preview, "rows: 2\n=> {'rows': [1, 2]}")
        self.assertEqual(executor.output.stdout, "rows: 2\n")
        self.assertEqual(executor.output.value, {"rows": [1, 2]})
        self.assertTrue(executor.output.has_value)
        self.assertEqual(executor.output.value_kind, "json")

    def test_a_syntax_error_is_reported_and_nothing_runs(self):
        executor = self.executor()

        preview = executor.run("import os\nos.chdir('..')\ndef broken(:\n")

        self.assertTrue(preview.startswith("SyntaxError:"))
        self.assertEqual(executor.output.text, preview)

    def test_approved_mode_runs_what_safe_mode_blocks(self):
        blocked = self.executor().run("__import__('os').getcwd()")
        self.assertTrue(blocked.startswith("run_python blocked:"))
        self.assertIn("__import__() is not allowed", blocked)

        approved = self.executor(approved=True)

        here = "=> " + repr(str(self.workspace.root))
        self.assertEqual(approved.run("import os\nos.getcwd()"), here)
        self.assertEqual(approved.run("__import__('os').getcwd()"), here)
        self.assertEqual(approved.run("eval('1 + 1')"), "=> 2")
        self.assertEqual(approved.run("import subprocess\nsubprocess.__name__"), "=> 'subprocess'")

    def test_the_returned_preview_is_bounded_while_the_store_keeps_it_all(self):
        executor = self.executor()
        code = "for i in range(60):\n    print(f'line {i}')"

        preview = executor.run(code)

        self.assertIn("output truncated: showing 30 of 60 lines", preview)
        self.assertIn("line 29", preview)
        self.assertNotIn("line 30", preview)
        self.assertEqual(len(executor.output.text.splitlines()), 60)
        self.assertIn("line 59", executor.output.inspect(55, 10))

    def test_a_trailing_print_reports_its_none_value(self):
        executor = self.executor()

        self.assertEqual(executor.run("print('only')"), "only\n=> None")


class OutputStoreTests(unittest.TestCase):
    def test_two_stores_do_not_share_state(self):
        first = OutputStore()
        second = OutputStore()

        first.store("alpha")
        second.store('{"n": 1}', value={"n": 1}, has_value=True, stdout='{"n": 1}\n')

        self.assertEqual(first.text, "alpha")
        self.assertEqual(first.stdout, "")
        self.assertEqual(first.value, None)
        self.assertFalse(first.has_value)
        self.assertEqual(first.value_kind, "none")
        self.assertEqual(second.text, '{"n": 1}')
        self.assertEqual(second.stdout, '{"n": 1}\n')
        self.assertEqual(second.value, {"n": 1})
        self.assertEqual(second.value_kind, "json")
        self.assertIn("alpha", first.inspect())
        self.assertNotIn("alpha", second.inspect())

    def test_a_fresh_store_reports_that_it_has_no_output(self):
        store = OutputStore()

        self.assertEqual(store.inspect(), "No run_python output is available yet.")
        self.assertIn("not JSON/XML", store.query("."))
        self.assertEqual(store.truncate(""), "")
        self.assertEqual(store.store(""), "")
        self.assertEqual(store.text, "")

    def test_the_preview_is_bounded_and_inspect_pages_the_full_text(self):
        store = OutputStore()
        text = "x" * 500 + "\n" + "\n".join(f"line {i}" for i in range(50))

        preview = store.store(text)

        self.assertIn("x" * 160 + "…", preview)
        self.assertNotIn("x" * 500, preview)
        self.assertIn("output truncated: showing 30 of 51 lines", preview)
        self.assertIn("line 28", preview)
        self.assertNotIn("line 29", preview)
        self.assertEqual(store.text, text)
        self.assertEqual(store.inspect(50, 5), "[50:51] of 51 lines\nline 49")
        self.assertTrue(store.inspect(-3, 1).startswith("[0:1] of 51 lines"))

    def test_query_extracts_json_sub_values(self):
        store = OutputStore()
        store.store('{"rows": [1, 2, 3]}', value={"rows": [1, 2, 3]}, has_value=True)

        self.assertEqual(store.query(".rows[1]"), "2")
        self.assertEqual(store.query(".rows[0:2]"), "[\n  1,\n  2\n]")
        self.assertEqual(store.query(".rows.length"), "3")
        self.assertIn('"rows"', store.query("keys"))
        self.assertIn("dict with 1 keys", store.query(""))
        self.assertIn("no key 'nope'", store.query(".nope"))
        self.assertIn("query failed", store.query(".rows.nope"))

    def test_query_falls_back_to_stdout_and_handles_xml(self):
        store = OutputStore()
        store.store('{"a": 1}', stdout='{"a": 1}\n')
        self.assertEqual(store.query(".a"), "1")

        xml = "<root><item id='1'>a</item><item id='2'>b</item></root>"
        store.store(xml, value=xml, has_value=True)

        self.assertEqual(store.value_kind, "xml")
        self.assertEqual(store.query(".//item/@id"), "1\n2")
        self.assertEqual(store.query("item/text()"), "a")
        self.assertTrue(store.query("").startswith("XML root <root> with 2 child elements"))
        self.assertIn("no XML match", store.query("missing"))

    def test_query_reports_plain_text_output(self):
        store = OutputStore()
        store.store("plain text")

        self.assertIn("not JSON/XML", store.query("."))
        self.assertIn("plain text", store.inspect(0, 10))


class PythonPackTests(PythonTestCase):
    def test_registered_python_tools_expose_expected_effects(self):
        registry, _ = self.build_pack()

        self.assertEqual(
            sorted(registry.names()),
            ["inspect_output", "python_help", "query_output", "run_python"],
        )
        self.assertEqual(
            registry.categories()["python"],
            ("run_python", "inspect_output", "query_output", "python_help"),
        )
        self.assertEqual(
            registry.get("run_python").effects,
            frozenset({Effect.WORKSPACE_WRITE, Effect.PROCESS, Effect.NETWORK}),
        )
        self.assertEqual(
            registry.get("inspect_output").effects, frozenset({Effect.READ})
        )
        self.assertEqual(
            registry.get("query_output").effects, frozenset({Effect.READ})
        )
        self.assertFalse(registry.replay_safe("run_python"))
        self.assertTrue(registry.replay_safe("inspect_output"))
        self.assertTrue(registry.replay_safe("query_output"))
        self.assertTrue(registry.replay_safe("python_help"))
        self.assertEqual(registry.core_names(), frozenset({"run_python"}))
        self.assertFalse(registry.requires_approval("run_python"))

    def test_run_python_returns_a_preview_the_session_can_page(self):
        registry, _ = self.build_pack()
        run = registry.get("run_python").callable

        preview = run("for i in range(40):\n    print(f'line {i}')")

        self.assertIn("output truncated: showing 30 of 40 lines", preview)
        paged = registry.get("inspect_output").callable(35, 3)
        self.assertEqual(paged.splitlines()[0], "[35:38] of 40 lines")
        self.assertEqual(paged.splitlines()[1], "line 35")

    def test_the_agent_path_and_the_server_path_share_one_executor(self):
        registry, runtime = self.build_pack()
        run_python = registry.get("run_python").callable

        run_python("print('from the tool')")

        executor = runtime.service(
            "python.executor", lambda: self.fail("the pack already created the executor")
        )
        self.assertIn("from the tool", executor.output.text)

        executor.run("print('from a cell')")

        self.assertIn("from a cell", registry.get("inspect_output").callable(0, 10))
        self.assertEqual(run_python("{'a': 1}"), "=> {'a': 1}")
        self.assertEqual(registry.get("query_output").callable(".a"), "1")

    def test_safe_mode_blocks_through_the_tool(self):
        registry, _ = self.build_pack()

        preview = registry.get("run_python").callable("import subprocess")

        self.assertTrue(preview.startswith("run_python blocked:"))
        self.assertIn("import of 'subprocess' is not allowed", preview)
        self.assertIn(preview, registry.get("inspect_output").callable(0, 10))

    def test_an_approved_runtime_lifts_the_guard_through_the_tool(self):
        registry, _ = self.build_pack(approved=True)
        run = registry.get("run_python").callable

        self.assertEqual(
            run("import os\nos.getcwd()"), "=> " + repr(str(self.workspace.root))
        )
        self.assertNotIn("blocked", run("import subprocess\nsubprocess.__name__"))

    def test_a_snippet_can_use_skimage_and_scipy_for_imagery(self):
        _, runtime = self.build_pack()
        # The pack creates the executor itself; the dynamic import here was a
        # one-off that hid the fixture type for no reason.
        executor = runtime.services["python.executor"]

        preview = executor.run(
            "import numpy as np, skimage, scipy\n"
            "from skimage import measure, filters, morphology\n"
            "grid = np.add.outer(np.arange(8.0), np.arange(8.0))\n"
            "smoothed = filters.gaussian(grid, sigma=0.6, preserve_range=True)\n"
            "print('contours', len(measure.find_contours(smoothed, 6.0)))\n"
            "print('thickened', int(morphology.dilation(grid > 6).sum()))\n"
            "print('corr', round(float(scipy.stats.pearsonr(grid.ravel(), smoothed.ravel())[0]), 3))\n"
        )

        self.assertNotIn("blocked", preview)
        self.assertIn("contours", preview)
        self.assertIn("thickened", preview)
        self.assertIn("corr 1.0", preview)

    def test_python_help_describes_the_sandbox_namespace(self):
        registry, _ = self.build_pack()
        describe = registry.get("python_help").callable

        roots = describe()
        self.assertEqual(roots["kind"], "namespace")
        self.assertIn("ws", roots["roots"])
        self.assertIn("rasterio", roots["roots"])
        self.assertIn("numpy", roots["roots"])
        self.assertIn("skimage", roots["roots"])
        self.assertIn("scipy", roots["roots"])
        self.assertIn("rio_cogeo", roots["roots"])

        # An unknown root is an error, and the listing names what exists.
        self.assertEqual(describe("nope")["kind"], "error")

        ws = describe("ws")
        self.assertEqual(ws["kind"], "workspace helper")
        self.assertIn("ws.write_text", ws["doc"])
        self.assertIn("ws.list_files", ws["doc"])

        reproject = describe("rasterio.warp.reproject")
        self.assertEqual(reproject["kind"], "function")
        self.assertIn("signature", reproject)
        self.assertTrue(reproject["doc"])

        array = describe("numpy.ndarray")
        self.assertEqual(array["kind"], "class")
        self.assertTrue(array["members"])

        self.assertEqual(describe("numpy")["kind"], "module")
        self.assertEqual(describe("nope")["kind"], "error")
        self.assertEqual(describe("numpy.nope")["kind"], "error")


if __name__ == "__main__":
    unittest.main()
