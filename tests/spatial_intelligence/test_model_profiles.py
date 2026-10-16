"""Deciding what a checkpoint can do, from metadata and, when needed, a verdict.

These tests are about the answers the wizard shows *before* a run: whether a
model does the task at all, which labels it already knows, and whether it takes
prompts instead. The Hub is always stubbed, so nothing here touches the network.
"""

import unittest
from unittest import mock

from spatial_intelligence.ai import profiles, hub
from spatial_intelligence.contracts.errors import ToolInputError

#: A TerraTorch foundation model repo: no pipeline tag, no labels.
PRITHVI = {
    "repo": "ibm-nasa-geospatial/Prithvi-EO-2.0-300M",
    "revision": "abc123",
    "pipeline": None,
    "library": "terratorch",
    "tags": ["terratorch", "Earth Observation", "license:apache-2.0"],
    "architectures": [],
    "labels": [],
    "files": ["Prithvi_EO_V2_300M.pt", "config.json"],
}

#: A promptable SAM checkpoint in a library this machine does not have.
SAM = {
    "repo": "nielsr/slimsam-50-uniform",
    "revision": "def456",
    "pipeline": "mask-generation",
    "library": "transformers",
    "tags": ["transformers", "safetensors", "sam", "mask-generation"],
    "architectures": ["SamModel"],
    "labels": [],
    "files": ["model.safetensors", "config.json"],
}

#: A fixed-class ONNX detector with its labels declared.
DETECTOR = {
    "repo": "Xenova/yolos-tiny",
    "revision": "e2f9c76",
    "pipeline": "object-detection",
    "library": "transformers.js",
    "tags": ["onnx", "object-detection"],
    "architectures": ["YolosForObjectDetection"],
    "labels": ["N/A", "car", "boat", "car"],
    "files": ["onnx/model.onnx"],
}


class ClassificationTests(unittest.TestCase):
    def test_a_pipeline_tag_names_the_task(self):
        self.assertEqual(profiles.classify_hub(SAM, "hf:x/y").tasks, ("segmentation",))
        self.assertEqual(profiles.classify_hub(DETECTOR, "hf:x/y").tasks, ("detection",))

    def test_a_tag_names_the_task_when_the_pipeline_does_not(self):
        info = dict(PRITHVI, tags=["torchgeo", "image-segmentation"])

        self.assertEqual(profiles.classify_hub(info, "hf:x/y").tasks, ("segmentation",))

    def test_labels_come_from_the_model_and_lose_placeholders_and_duplicates(self):
        profile = profiles.classify_hub(DETECTOR, "hf:x/y")

        self.assertEqual(profile.labels, ("car", "boat"))

    def test_a_sam_checkpoint_is_promptable_and_a_fixed_detector_is_not(self):
        self.assertTrue(profiles.classify_hub(SAM, "hf:x/y").promptable)
        self.assertFalse(profiles.classify_hub(DETECTOR, "hf:x/y").promptable)

    def test_the_library_decides_the_route(self):
        self.assertEqual(profiles.classify_hub(PRITHVI, "hf:x/y").route, "terratorch")
        self.assertEqual(profiles.classify_hub(SAM, "hf:x/y").route, "transformers")
        self.assertEqual(
            profiles.classify_hub(
                dict(PRITHVI, library="torchgeo", tags=["torchgeo"]), "hf:x/y"
            ).route,
            "torchgeo",
        )

    def test_a_repo_with_only_onnx_weights_is_not_wired_up(self):
        profile = profiles.classify_hub(dict(DETECTOR, library=None, tags=[]), "hf:x/y")

        self.assertEqual(profile.route, "onnx")
        self.assertFalse(profile.runnable)
        self.assertIn("only the built-in models have a wired runner", profile.reason)

    def test_a_repo_with_no_weights_is_unusable(self):
        profile = profiles.classify_hub(dict(DETECTOR, files=["README.md"], tags=[]), "hf:x/y")

        self.assertEqual(profile.route, "unusable")
        self.assertFalse(profile.runnable)

    def test_a_missing_library_says_how_to_install_it(self):
        profile = profiles.classify_hub(SAM, "hf:x/y")

        if not profile.runnable:
            self.assertIn("geoai", profile.reason)


class SupportTests(unittest.TestCase):
    def test_a_detector_does_not_support_segmentation(self):
        profile = profiles.classify_hub(DETECTOR, "hf:x/y")

        self.assertTrue(profile.supports("detection"))
        self.assertFalse(profile.supports("segmentation"))

    def test_a_promptable_model_supports_both_geospatial_tasks(self):
        profile = profiles.classify_hub(SAM, "hf:x/y")

        self.assertTrue(profile.supports("segmentation"))
        self.assertTrue(profile.supports("detection"))

    def test_an_unrequested_task_is_always_supported(self):
        self.assertTrue(profiles.classify_hub(DETECTOR, "hf:x/y").supports(""))

    def test_the_catalog_knows_its_own_labels_and_prompts(self):
        slimsam = profiles.from_catalog("slimsam-77")
        yolos = profiles.from_catalog("yolos-tiny")

        self.assertTrue(slimsam.promptable)
        self.assertTrue(slimsam.runnable)
        self.assertEqual(slimsam.route, "onnx")
        self.assertEqual(len(yolos.labels), 80)
        self.assertIn("car", yolos.labels)
        self.assertNotIn("N/A", yolos.labels)

    def test_an_unknown_built_in_model_is_named(self):
        self.assertIsNone(profiles.from_catalog("nope"))


class ReferenceTests(unittest.TestCase):
    def test_a_built_in_id_is_recognised(self):
        self.assertEqual(profiles.parse_ref("slimsam-77"), ("slimsam-77", "", ""))

    def test_a_hub_reference_carries_its_repository_and_revision(self):
        self.assertEqual(
            profiles.parse_ref("hf:owner/name@abc123"), ("", "owner/name", "abc123")
        )
        self.assertEqual(profiles.parse_ref("owner/name"), ("", "owner/name", ""))

    def test_an_empty_reference_is_refused(self):
        with self.assertRaises(ToolInputError):
            profiles.parse_ref("  ")


class SearchRowTests(unittest.TestCase):
    def test_a_search_row_never_claims_a_repo_is_unusable(self):
        row = {
            "ref": "hf:acme/things",
            "repo": "acme/things",
            "library": None,
            "pipeline": "image-segmentation",
            "tags": ["license:mit"],
        }

        profile = profiles.guess_from_search(row, "segmentation")

        self.assertEqual(profile.route, "research")
        self.assertEqual(profile.tasks, ("segmentation",))
        self.assertIn("not inspected", profile.reason)
        self.assertTrue(profile.supports("segmentation"))

    def test_a_search_row_reads_the_library(self):
        row = {
            "ref": "hf:owner/name",
            "repo": "owner/name",
            "library": "terratorch",
            "pipeline": None,
            "tags": ["terratorch"],
        }

        self.assertEqual(profiles.guess_from_search(row, "segmentation").route, "terratorch")


class HintTests(unittest.TestCase):
    def test_a_curated_hint_outranks_the_hub(self):
        profile = profiles.classify_hub(PRITHVI, "hf:x/y")

        merged = profiles.apply_hint(
            profile,
            {"route": "terratorch", "tasks": ["segmentation"], "runnable": True, "source": "curated"},
        )

        self.assertEqual(merged.tasks, ("segmentation",))
        self.assertEqual(merged.source, "curated")
        self.assertTrue(merged.supports("segmentation"))

    def test_the_hint_for_a_shipped_option_is_found(self):
        hint = __import__(
            "spatial_intelligence.workflows.library", fromlist=["option_hint"]
        ).option_hint("hf:torchgeo/ftw")

        self.assertIsNotNone(hint)
        self.assertEqual(hint["tasks"], ["segmentation"])
        self.assertTrue(hint["runnable"])


class LinkTests(unittest.TestCase):
    """A pasted repository link is a first-class way to name a model."""

    def test_a_plain_repository_link(self):
        self.assertEqual(
            profiles.parse_ref("https://huggingface.co/nielsr/slimsam-50-uniform"),
            ("", "nielsr/slimsam-50-uniform", ""),
        )

    def test_a_link_with_a_tree_revision(self):
        self.assertEqual(
            profiles.parse_ref("https://huggingface.co/owner/name/tree/main"),
            ("", "owner/name", "main"),
        )

    def test_a_link_with_a_nested_path_keeps_the_revision(self):
        self.assertEqual(
            profiles.parse_ref("https://huggingface.co/owner/name/blob/v2/weights/model.safetensors"),
            ("", "owner/name", "v2"),
        )

    def test_the_short_host_and_the_models_prefix(self):
        self.assertEqual(profiles.parse_ref("https://hf.co/owner/name"), ("", "owner/name", ""))
        self.assertEqual(
            profiles.parse_ref("https://huggingface.co/models/owner/name"), ("", "owner/name", "")
        )

    def test_a_schemeless_link_is_still_a_link(self):
        self.assertEqual(
            profiles.parse_ref("huggingface.co/owner/name"), ("", "owner/name", "")
        )

    def test_a_trailing_git_suffix_is_dropped(self):
        self.assertEqual(profiles.parse_ref("https://huggingface.co/owner/name.git"), ("", "owner/name", ""))

    def test_another_host_is_refused_by_name(self):
        with self.assertRaises(ToolInputError) as raised:
            profiles.parse_ref("https://github.com/owner/name")

        self.assertIn("only Hugging Face", str(raised.exception))

    def test_a_dataset_link_is_refused(self):
        with self.assertRaises(ToolInputError) as raised:
            profiles.parse_ref("https://huggingface.co/datasets/owner/name")

        self.assertIn("not a model repository", str(raised.exception))

    def test_a_link_without_a_name_is_refused(self):
        with self.assertRaises(ToolInputError):
            profiles.parse_ref("https://huggingface.co/owner")

    def test_a_link_resolves_like_its_reference(self):
        profiles.clear_cache()

        with mock.patch.object(profiles.hub, "repo_info", return_value=DETECTOR) as info:
            from_link = profiles.resolve(
                "https://huggingface.co/Xenova/yolos-tiny", "detection"
            )

        info.assert_called_once_with("Xenova/yolos-tiny")
        self.assertEqual(from_link.repo, "Xenova/yolos-tiny")
        self.assertTrue(from_link.supports("detection"))

    def test_a_curated_hint_is_found_through_a_link(self):
        library = __import__(
            "spatial_intelligence.workflows.library", fromlist=["option_hint"]
        )

        hint = library.option_hint("https://huggingface.co/torchgeo/ftw")

        self.assertIsNotNone(hint)
        self.assertEqual(hint["tasks"], ["segmentation"])


class ResolveTests(unittest.TestCase):
    def setUp(self):
        profiles.clear_cache()

    def test_a_catalog_model_needs_no_network(self):
        with mock.patch.object(profiles.hub, "repo_info") as info:
            profile = profiles.resolve("slimsam-77", "segmentation")

        info.assert_not_called()
        self.assertEqual(profile.ref, "slimsam-77")
        self.assertEqual(profile.source, "catalog")

    def test_a_hub_model_is_described_from_its_metadata(self):
        with mock.patch.object(profiles.hub, "repo_info", return_value=DETECTOR) as info:
            profile = profiles.resolve("hf:Xenova/yolos-tiny", "detection")

        info.assert_called_once()
        self.assertEqual(profile.repo, "Xenova/yolos-tiny")
        self.assertEqual(profile.revision, "e2f9c76")
        self.assertIn("car", profile.labels)

    def test_a_curated_hint_stops_the_verdict_from_being_asked(self):
        verdict = mock.Mock(return_value={"supported": True, "task": "segmentation"})
        hint = {"tasks": ["segmentation"], "runnable": True, "source": "curated"}

        with mock.patch.object(profiles.hub, "repo_info", return_value=PRITHVI):
            profile = profiles.resolve("hf:torchgeo/ftw", "segmentation", verdict=verdict, hint=hint)

        verdict.assert_not_called()
        self.assertTrue(profile.supports("segmentation"))

    def test_an_undeclared_task_is_given_to_the_verdict(self):
        verdict = mock.Mock(
            return_value={
                "supported": True,
                "task": "segmentation",
                "labels": ["field"],
                "promptable": False,
                "reason": "A segmentation U-Net for field boundaries.",
            }
        )

        with mock.patch.object(profiles.hub, "repo_info", return_value=PRITHVI):
            profile = profiles.resolve("hf:ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "segmentation", verdict=verdict)

        verdict.assert_called_once()
        self.assertEqual(profile.source, "agent")
        self.assertEqual(profile.labels, ("field",))
        self.assertTrue(profile.supports("segmentation"))
        self.assertIn("field boundaries", profile.reason)

    def test_a_failed_verdict_leaves_the_metadata_answer_standing(self):
        verdict = mock.Mock(side_effect=RuntimeError("provider down"))

        with mock.patch.object(profiles.hub, "repo_info", return_value=PRITHVI):
            profile = profiles.resolve("hf:ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "segmentation", verdict=verdict)

        self.assertEqual(profile.tasks, ())

    def test_the_resolution_is_cached(self):
        with mock.patch.object(profiles.hub, "repo_info", return_value=PRITHVI) as info:
            profiles.resolve("hf:ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "segmentation")
            profiles.resolve("hf:ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "segmentation")

        info.assert_called_once()

    def test_a_reference_that_cannot_be_resolved_still_reports_its_repo(self):
        hint = {"tasks": ["segmentation"], "runnable": True, "source": "curated"}

        with mock.patch.object(
            profiles.hub, "repo_info", side_effect=ToolInputError("the model hub could not be reached")
        ):
            profile = profiles.resolve("hf:torchgeo/ftw", "segmentation", hint=hint)

        self.assertEqual(profile.repo, "torchgeo/ftw")
        self.assertTrue(profile.supports("segmentation"))

    def test_without_a_hint_an_unreachable_hub_is_an_error(self):
        with mock.patch.object(
            profiles.hub, "repo_info", side_effect=ToolInputError("the model hub could not be reached")
        ):
            with self.assertRaises(ToolInputError):
                profiles.resolve("hf:torchgeo/ftw", "segmentation")


class HubTests(unittest.TestCase):
    def test_weight_files_groups_by_kind(self):
        grouped = hub.weight_files(
            ["onnx/model.onnx", "model.safetensors", "weights.pt", "README.md"]
        )

        self.assertEqual(grouped["onnx"], ["onnx/model.onnx"])
        self.assertEqual(grouped["safetensors"], ["model.safetensors"])
        self.assertEqual(grouped["torch"], ["weights.pt"])

    def test_search_needs_a_term(self):
        with self.assertRaises(ToolInputError):
            hub.search_models("   ")


class VerdictTests(unittest.TestCase):
    """The model verdict: JSON text in, a plain dict out, never raising."""

    def setUp(self):
        from spatial_intelligence.agent import verdict as verdict_module

        self.module = verdict_module
        self.module.clear_cache()

    def _run(self, answer: str, facts=None):
        prompts: list[str] = []
        self.prompts = prompts

        class FakeResult:
            output = answer

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs

            def run_sync(self, prompt):
                prompts.append(prompt)
                return FakeResult()

        with mock.patch.object(self.module, "Agent", FakeAgent), mock.patch.object(
            self.module, "resolve_model", lambda model: model
        ):
            return self.module.classify("openai:test", "hf:owner/name", facts or {"repo": "o/n"}, "segmentation")

    def test_a_json_answer_is_read(self):
        answer = self._run(
            '{"supported": true, "task": "Segmentation", "labels": ["field"], '
            '"promptable": false, "reason": "Fine for this."}'
        )

        self.assertTrue(answer["supported"])
        self.assertEqual(answer["task"], "segmentation")
        self.assertEqual(answer["labels"], ["field"])
        self.assertEqual(answer["reason"], "Fine for this.")

    def test_a_fenced_or_chatty_answer_still_parses(self):
        answer = self._run(
            'Sure!\n```json\n{"supported": false, "task": "", "reason": "No head."}\n```'
        )

        self.assertFalse(answer["supported"])
        self.assertEqual(answer["reason"], "No head.")

    def test_an_answer_without_json_is_ignored(self):
        self.assertIsNone(self._run("I cannot tell."))

    def test_the_verdict_is_cached_per_checkpoint(self):
        calls = []

        class FakeResult:
            output = '{"supported": true, "task": "detection", "reason": "ok"}'

        class FakeAgent:
            def __init__(self, *args, **kwargs):
                pass

            def run_sync(self, prompt):
                calls.append(prompt)
                return FakeResult()

        with mock.patch.object(self.module, "Agent", FakeAgent), mock.patch.object(
            self.module, "resolve_model", lambda model: model
        ):
            self.module.classify("openai:test", "hf:o/n", {"repo": "o/n"}, "detection")
            self.module.classify("openai:test", "hf:o/n", {"repo": "o/n"}, "detection")

        self.assertEqual(len(calls), 1)

    def test_facts_reach_the_prompt(self):
        self._run('{"supported": true}', facts={"repo": "acme/thing", "library": "terratorch"})

        self.assertIn("acme/thing", self.prompts[0])
        self.assertIn("segmentation", self.prompts[0])


if __name__ == "__main__":
    unittest.main()
