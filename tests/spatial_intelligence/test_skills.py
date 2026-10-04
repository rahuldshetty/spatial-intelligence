"""The skills tree: generated per topic, searched, and read a page at a time."""

from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from spatial_intelligence.pythonruntime import packages, skill, spec
from spatial_intelligence.tools.packs.skills import SkillsPack
from spatial_intelligence.contracts.progress import NULL_REPORTER, Reporter
from spatial_intelligence.tools.runtime import RuntimeEvents, ToolRuntime
from spatial_intelligence.workspace import Workspace


class RecordingSink:
    """Collects the progress events one call publishes."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event) -> None:
        self.events.append(event.as_dict())

FAKE = "fake_geopkg"
SPEC = packages.SandboxPackage(key=FAKE, import_name=FAKE, purpose="fake library")

MODULE_SOURCE = '''
"""{doc}"""

def {name}(value: int = 1) -> int:
    """{summary}"""
    return value

class {cls}:
    """{cls_summary}"""
'''


def _module(doc: str, name: str, summary: str, cls: str, cls_summary: str) -> str:
    return MODULE_SOURCE.format(doc=doc, name=name, summary=summary, cls=cls, cls_summary=cls_summary)


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        package = self.root / FAKE
        package.mkdir()
        (package / "__init__.py").write_text('"""A fake package."""\n', encoding="utf-8")
        # Two topics: one plain, one bulk (a catalog topic).
        (package / "samplers.py").write_text(
            _module("Sampling.", "random_patch", "Sample a random patch.", "Sampler", "Samples patches."),
            encoding="utf-8",
        )
        (package / "datasets.py").write_text(
            "\n".join(
                [
                    '"""Benchmark datasets."""',
                    "",
                    *[
                        f'class {name}:\n    """Dataset {index}."""\n'
                        for index, name in enumerate(("Alpha", "Beta", "Gamma"))
                    ],
                ]
            ),
            encoding="utf-8",
        )
        sys.path.insert(0, str(self.root))
        self._packages = patch.object(packages, "PACKAGES", (SPEC,))
        self._packages.start()
        self.target = self.root / "skills" / FAKE

    def tearDown(self):
        self._packages.stop()
        sys.path.remove(str(self.root))
        sys.modules.pop(FAKE, None)
        self._tmp.cleanup()

    def build(self) -> dict:
        return spec.build(FAKE, target=self.target)

    def patch_skills_root(self):
        """Point the skills root at the fixture for the reader-side tests."""
        self._skills_root = patch.object(spec, "SKILLS_DIR", self.target.parent)
        self._skills_root.start()
        self._cache = patch.object(spec, "cache_dir", lambda package="": self.root / "empty")
        self._cache.start()
        return self

    def test_a_tree_is_written_per_topic(self):
        meta = self.build()

        self.assertEqual(meta["package"], FAKE)
        written = sorted(meta["files"])
        # Two topics: another package's API must not become one big file.
        self.assertEqual(written, ["datasets.catalog.md", "index.md", "samplers.md"])
        self.assertTrue((self.target / "api" / "samplers.md").is_file())
        self.assertTrue((self.target / "api" / "meta.json").is_file())

    def test_a_bulk_topic_becomes_a_catalog_not_a_file_per_class(self):
        self.build()

        catalog = (self.target / "api" / "datasets.catalog.md").read_text(encoding="utf-8")

        self.assertIn("| symbol | summary |", catalog)
        for name in ("Alpha", "Beta", "Gamma"):
            self.assertIn(f"{FAKE}.datasets.{name}", catalog)

    def test_generated_files_carry_the_banner_and_a_version(self):
        meta = self.build()

        for name in meta["files"]:
            text = (self.target / "api" / name).read_text(encoding="utf-8")
            self.assertTrue(text.startswith(spec.BANNER), name)
        recorded = json.loads((self.target / "api" / "meta.json").read_text(encoding="utf-8"))
        self.assertEqual(recorded["version"], "unknown")  # not a distribution here
        self.assertEqual(recorded["files"], meta["files"])

    def test_the_index_links_every_topic(self):
        self.build()

        index = (self.target / "api" / "index.md").read_text(encoding="utf-8")

        self.assertIn("(samplers.md)", index)
        self.assertIn("(datasets.catalog.md)", index)

    def test_an_uninstalled_package_cannot_be_described(self):
        with patch.object(packages, "installed", lambda package: False):
            with self.assertRaises(LookupError):
                spec.build(FAKE, target=self.target)


class ReaderTestCase(GeneratorTestCase):
    def setUp(self):
        super().setUp()
        self.build()
        self.patch_skills_root()

    def tearDown(self):
        self._cache.stop()
        self._skills_root.stop()
        super().tearDown()

    def test_reading_defaults_to_the_index(self):
        result = skill.read(FAKE)

        self.assertEqual(result["kind"], "file")
        self.assertEqual(result["file"], "api/index.md")

    def test_a_topic_is_read_with_paging(self):
        first = skill.read(FAKE, "api/samplers.md", count=3)
        later = skill.read(FAKE, "api/samplers.md", start=2, count=3)

        self.assertEqual(first["file"], "api/samplers.md")
        self.assertEqual(first["count"], 3)
        self.assertTrue(first["has_more"])
        self.assertTrue(first["text"].startswith(spec.BANNER))
        self.assertNotIn(spec.BANNER, later["text"])

    def test_a_query_narrows_a_file_to_matching_lines(self):
        result = skill.read(FAKE, "api/samplers.md", query="random_patch")

        self.assertIn("random_patch", result["text"])
        self.assertNotIn("Samples patches", result["text"])

    def test_a_query_with_no_match_says_so(self):
        result = skill.read(FAKE, "api/samplers.md", query="nothing-matches-this")

        self.assertEqual(result["kind"], "search")
        self.assertIn("no line matched", result["doc"])

    def test_search_reports_file_and_line(self):
        found = skill.search("Sampler", package=FAKE)
        files = {hit["file"] for hit in found["hits"]}

        self.assertIn("api/samplers.md", files)
        self.assertTrue(all(hit["line"] > 0 for hit in found["hits"]))

    def test_a_path_outside_the_tree_is_refused(self):
        result = skill.read(FAKE, "../../pyproject.toml")

        self.assertEqual(result["kind"], "error")

    def test_an_unknown_package_lists_what_exists(self):
        result = skill.read("no-such-package")

        self.assertEqual(result["kind"], "error")
        self.assertIn(FAKE, result["doc"])

    def test_available_reports_the_source_and_the_package_state(self):
        rows = skill.available()

        self.assertEqual([row["package"] for row in rows], [FAKE])
        self.assertEqual(rows[0]["source"], "repo")
        self.assertTrue(rows[0]["installed"])


class SkillToolTestCase(GeneratorTestCase):
    def setUp(self):
        super().setUp()
        self.build()
        self.patch_skills_root()
        self.workspace = Workspace(self.root / "ws").create()
        self.pack = SkillsPack(
            ToolRuntime(workspace=self.workspace, reporter=NULL_REPORTER, events=RuntimeEvents())
        )

    def tearDown(self):
        self._cache.stop()
        self._skills_root.stop()
        super().tearDown()

    def test_the_tool_lists_packages_then_reads_one(self):
        index = self.pack.skill()
        self.assertEqual(index["kind"], "index")
        self.assertEqual(index["packages"][0]["package"], FAKE)

        page = self.pack.skill(package=FAKE, path="api/index.md")
        self.assertEqual(page["kind"], "file")
        self.assertIn("samplers.md", page["text"])

        hits = self.pack.skill(query="random_patch")
        self.assertEqual(hits["kind"], "search")
        self.assertTrue(hits["hits"])

    def test_a_path_and_a_query_together_narrow_that_file(self):
        narrowed = self.pack.skill(package=FAKE, path="api/samplers.md", query="random_patch")

        self.assertEqual(narrowed["kind"], "file")
        self.assertEqual(narrowed["file"], "api/samplers.md")
        self.assertIn("random_patch", narrowed["text"])
        self.assertEqual(narrowed["matched_lines"], 1)
        self.assertGreater(narrowed["total_lines"], narrowed["matched_lines"])


class SkillJobTestCase(GeneratorTestCase):
    """Every read reports a job, which is what the browser draws as a card."""

    def setUp(self):
        super().setUp()
        self.build()
        self.patch_skills_root()
        self.sink = RecordingSink()
        self.workspace = Workspace(self.root / "ws").create()
        self.pack = SkillsPack(
            ToolRuntime(
                workspace=self.workspace,
                reporter=Reporter(self.sink, parent_id="cell-1"),
                events=RuntimeEvents(),
            )
        )

    def tearDown(self):
        self._cache.stop()
        self._skills_root.stop()
        super().tearDown()

    def job_events(self) -> list[dict]:
        return [event for event in self.sink.events if event["kind"] == "skill"]

    def test_reading_a_file_opens_and_finishes_a_job(self):
        self.pack.skill(package=FAKE, path="api/samplers.md")

        events = self.job_events()
        self.assertTrue(events)
        labels = {event["label"] for event in events}
        self.assertEqual(labels, {f"Loaded skill {FAKE}/api/samplers.md"})
        last = events[-1]
        self.assertEqual(last["status"], "done")
        self.assertEqual(last["unit"], "lines")
        self.assertEqual(last["completed"], last["total"])
        self.assertEqual(last["detail"], "api/samplers.md (repo)")
        self.assertEqual(last["parent_id"], "cell-1")

    def test_a_narrowed_read_counts_the_matching_lines(self):
        page = self.pack.skill(package=FAKE, path="api/samplers.md", query="random_patch")

        last = self.job_events()[-1]
        self.assertIn("matching", last["label"])
        self.assertEqual(last["unit"], "lines")
        self.assertEqual(last["total"], page["matched_lines"])
        self.assertEqual(last["detail"], "api/samplers.md (repo)")

    def test_the_index_and_a_tree_search_are_labelled_too(self):
        self.pack.skill()
        self.pack.skill(package=FAKE, query="random_patch")

        index, search = (event for event in self.job_events() if event["status"] == "done")
        self.assertEqual(index["label"], "Loaded skill index")
        self.assertEqual(index["unit"], "trees")
        self.assertEqual(index["detail"], FAKE)
        self.assertEqual(search["label"], f"Searched {FAKE} for 'random_patch'")
        self.assertEqual(search["unit"], "hits")
        self.assertEqual(search["total"], 1)

    def test_a_missing_file_fails_the_card_instead_of_lying(self):
        self.pack.skill(package=FAKE, path="api/nope.md")

        last = self.job_events()[-1]
        self.assertEqual(last["status"], "error")
        self.assertIn("no such file", last["error"])


class ShippedTreeTestCase(unittest.TestCase):
    """The committed trees must stay well-formed without any package installed."""

    def test_every_tree_has_an_index_and_resolves(self):
        names = spec.packages_with_skills()

        self.assertIn("torchgeo", names)
        self.assertIn("terratorch", names)
        for name in names:
            tree = skill.resolve(name)
            self.assertIsNotNone(tree, name)
            text = (tree.root / "index.md").read_text(encoding="utf-8")
            self.assertTrue(text.strip(), name)

    def test_index_links_point_at_files_that_exist(self):
        import re

        for name in spec.packages_with_skills():
            tree = skill.resolve(name)
            text = (tree.root / "index.md").read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)#]+)\)", text):
                if target.startswith(("http", "mailto")):
                    continue
                self.assertTrue((tree.root / target).exists(), f"{name}: {target}")

    def test_generated_api_files_carry_the_banner(self):
        for name in spec.packages_with_skills():
            tree = skill.resolve(name)
            for path in tree.root.rglob("api/*.md"):
                self.assertTrue(
                    path.read_text(encoding="utf-8").startswith(spec.BANNER), str(path)
                )


if __name__ == "__main__":
    unittest.main()
