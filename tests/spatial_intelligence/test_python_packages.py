"""The optional sandbox packages: guard, namespace, help, and prompt block."""

from __future__ import annotations

import ast
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.agent import prompt as prompt_module
from spatial_intelligence.pythonruntime import help as api_help
from spatial_intelligence.pythonruntime import packages
from spatial_intelligence.pythonruntime.executor import PythonExecutor
from spatial_intelligence.pythonruntime.sandbox import guard
from spatial_intelligence.workspace import Workspace

#: A stand-in for torchgeo, injected on sys.path so none of this needs a 2 GB install.
FAKE = "fake_geolib"
FAKE_SOURCE = textwrap.dedent(
    '''
    """A stand-in for a heavy optional package."""

    def load_model(name: str = "tiny") -> str:
        """Load a model by name."""
        return f"model:{name}"

    class Sampler:
        """Samples patches from a dataset."""

        def __init__(self, size: int = 256):
            self.size = size
    '''
).lstrip()

SPEC = packages.SandboxPackage(key=FAKE, import_name=FAKE, purpose="fake heavy library")


class FakePackageTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / f"{FAKE}.py").write_text(FAKE_SOURCE, encoding="utf-8")
        sys.path.insert(0, str(root))
        self._packages = patch.object(packages, "PACKAGES", (*packages.PACKAGES, SPEC))
        self._packages.start()

    def tearDown(self):
        self._packages.stop()
        sys.path.remove(str(self._tmp.name))
        sys.modules.pop(FAKE, None)
        self._tmp.cleanup()


class InstalledDetectionTests(FakePackageTestCase):
    def test_an_import_is_refused_when_the_package_is_absent(self):
        with patch.object(packages, "installed", lambda package: False):
            message = guard(ast.parse(f"import {FAKE}"))

        self.assertIn("not installed", message)
        self.assertIn("geoai", message)

    def test_an_unknown_module_keeps_the_plain_refusal(self):
        message = guard(ast.parse("import subprocess"))

        self.assertEqual(message, "import of 'subprocess' is not allowed in run_python")

    def test_an_installed_package_imports(self):
        self.assertIsNone(guard(ast.parse(f"import {FAKE}")))

    def test_the_namespace_binds_only_what_is_installed(self):
        executor = PythonExecutor(Workspace(Path(self._tmp.name) / "ws").create())

        self.assertIn(FAKE, executor.namespace())
        self.assertIn("models", executor.namespace())

        # Whether torch is here is this machine's business, so pin the set
        # instead of asserting on the environment: only what is installed binds.
        with patch.object(packages, "installed", lambda package: package is SPEC):
            bound = executor.namespace()

        self.assertIn(FAKE, bound)
        self.assertNotIn("torch", bound)

    def test_python_help_resolves_the_optional_package(self):
        described = api_help.describe(f"{FAKE}.load_model")

        self.assertEqual(described["kind"], "function")
        self.assertIn("name", described["signature"])

    def test_the_prompt_block_lists_the_installed_package(self):
        block = packages.prompt_block()

        self.assertIn("fake heavy library", block)
        self.assertIn("Installed on this machine", block)

    def test_the_system_prompt_grows_only_by_the_block(self):
        with patch.object(packages, "prompt_block", lambda: ""):
            lean = prompt_module.system_prompt()

        rich = prompt_module.system_prompt()

        self.assertEqual(lean, prompt_module.SYSTEM_PROMPT)
        self.assertTrue(rich.startswith(prompt_module.SYSTEM_PROMPT))
        self.assertGreater(len(rich), len(lean))


class ModelsFacadeTests(unittest.TestCase):
    def test_the_cache_view_reports_paths_and_missing_files(self):
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as wd:
            with patch.dict("os.environ", {"GEOAI_MODELS_DIR": cache}, clear=False):
                executor = PythonExecutor(Workspace(Path(wd) / "ws").create())
                models = executor.namespace()["models"]

                self.assertEqual(models.dir(), cache)
                self.assertEqual(models.list(), [])
                with self.assertRaises(FileNotFoundError):
                    models.path("org/repo", "weights.pt")

    def test_the_cache_view_finds_a_file_under_its_revision(self):
        with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as wd:
            with patch.dict("os.environ", {"GEOAI_MODELS_DIR": cache}, clear=False):
                executor = PythonExecutor(Workspace(Path(wd) / "ws").create())
                models = executor.namespace()["models"]
                weights = Path(cache) / "org/repo/abc123/commercial/weights.pt"
                weights.parent.mkdir(parents=True)
                weights.write_bytes(b"x")

                self.assertEqual(models.path("org/repo", "commercial/weights.pt"), str(weights))
                self.assertEqual(models.list("org/repo"), ["abc123"])
                with self.assertRaises(FileNotFoundError):
                    models.path("org/repo", "other.pt")


if __name__ == "__main__":
    unittest.main()
