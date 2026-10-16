"""Workflow wizards: declarations, prompt rendering, seeding, and their routes.

The prompt is the whole handoff, so most of this file is about what the agent
will read: every placeholder must be a declared field, every answer must render
into something, and an area of interest must become a real file before the
prompt points at it.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starlette.testclient import TestClient

from spatial_intelligence import workflows
from spatial_intelligence.ai.profiles import Profile
from spatial_intelligence.contracts.errors import ToolInputError
from spatial_intelligence.contracts.forms import InteractionField
from spatial_intelligence.server import deps
from spatial_intelligence.server.app import create_app
from spatial_intelligence.session.notebook_session import NotebookSession
from spatial_intelligence.tools.build import default_registry
from spatial_intelligence.tools.runtime import ToolRuntime
from spatial_intelligence.workspace import Workspace
from spatial_intelligence.workflows import resolve
from spatial_intelligence.workflows.spec import WorkflowSpec

DRAWN = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [75.80, 30.88],
                        [75.92, 30.88],
                        [75.92, 30.96],
                        [75.80, 30.96],
                        [75.80, 30.88],
                    ]
                ],
            },
        }
    ],
}

SEGMENT_PARAMS = {
    "model": {"ref": "slimsam-77", "target": "water bodies", "labels": []},
    "aoi": {"mode": "bbox", "bbox": "75.80,30.88,75.92,30.96", "label": "Pune test"},
    "source": "",
    "notes": "",
}


class SpecTests(unittest.TestCase):
    def test_a_prompt_placeholder_must_be_a_declared_field(self):
        with self.assertRaises(ValueError) as raised:
            WorkflowSpec(
                id="broken",
                label="Broken",
                summary="",
                params=(InteractionField(id="aoi", label="AOI", type="aoi"),),
                prompt="Use {aoi} and {nowhere}",
            )

        self.assertIn("nowhere", str(raised.exception))

    def test_duplicate_field_ids_are_rejected(self):
        with self.assertRaises(ValueError) as raised:
            WorkflowSpec(
                id="dupe",
                label="Dupe",
                summary="",
                params=(
                    InteractionField(id="aoi", label="AOI", type="aoi"),
                    InteractionField(id="aoi", label="Again", type="text"),
                ),
                prompt="{aoi}",
            )

        self.assertIn("duplicate", str(raised.exception))

    def test_every_shipped_workflow_promises_tools_that_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / "w").create()
            registry = default_registry(ToolRuntime(workspace=workspace))
            known = set(registry.names())

            for spec in workflows.WORKFLOWS:
                missing = set(spec.requires) - known
                self.assertEqual(missing, set(), f"{spec.id} promises unknown tools")

    def test_the_blank_workflow_comes_first_and_asks_nothing(self):
        self.assertEqual(workflows.WORKFLOWS[0].id, "blank")
        self.assertEqual(workflows.WORKFLOWS[0].params, ())
        self.assertEqual(workflows.WORKFLOWS[0].prompt, "")

    def test_an_unknown_workflow_is_named_in_the_error(self):
        with self.assertRaises(KeyError) as raised:
            workflows.get("nope")

        self.assertIn("segmentation", str(raised.exception))


class TargetTests(unittest.TestCase):
    """The target comes from the model, never from an invented default."""

    def setUp(self):
        self.spec = workflows.get("detection")
        self.field = next(param for param in self.spec.params if param.type == "model")

    def _prompt(self, model, profile):
        with mock.patch.object(resolve.profiles, "resolve", return_value=profile):
            return resolve.prompt_text(self.spec, dict(SEGMENT_PARAMS, model=model))

    def test_a_chosen_class_is_the_only_thing_reported(self):
        text = self._prompt(
            {"ref": "yolos-tiny", "labels": ["car", "boat"], "target": ""},
            Profile(
                ref="yolos-tiny",
                repo="Xenova/yolos-tiny",
                route="onnx",
                tasks=("detection",),
                labels=("person", "car", "boat"),
                runnable=True,
            ),
        )

        self.assertIn("What to find: car, boat", text)
        self.assertIn("classes=[...]", text)
        self.assertIn(resolve.OWN_OUTPUT_RULE, text)

    def test_no_choice_leaves_the_target_to_the_model(self):
        text = self._prompt(
            {"ref": "yolos-tiny", "labels": [], "target": ""},
            Profile(
                ref="yolos-tiny",
                repo="Xenova/yolos-tiny",
                route="onnx",
                tasks=("detection",),
                labels=("car", "boat"),
                runnable=True,
            ),
        )

        self.assertIn("What to find: the classes this checkpoint predicts — car, boat", text)
        self.assertIn(resolve.OWN_OUTPUT_RULE, text)
        self.assertNotIn("every class", text)

    def test_a_long_label_set_is_summarised(self):
        labels = tuple(f"class{index}" for index in range(40))

        summary = resolve.label_summary(labels)

        self.assertEqual(summary.count(","), resolve.MAX_PROMPT_LABELS - 1)
        self.assertIn("(and 16 more)", summary)

    def test_a_typed_class_the_model_lacks_is_called_out(self):
        text = self._prompt(
            {"ref": "yolos-tiny", "labels": [], "target": "roads"},
            Profile(
                ref="yolos-tiny",
                repo="Xenova/yolos-tiny",
                route="onnx",
                tasks=("detection",),
                labels=("car", "boat"),
                runnable=True,
            ),
        )

        self.assertIn("What to find: roads", text)
        self.assertIn("not one of this checkpoint's classes", text)
        self.assertIn("car, boat", text)
        self.assertIn("instead of substituting a class of your own", text)

    def test_a_typed_class_the_model_has_is_left_alone(self):
        text = self._prompt(
            {"ref": "yolos-tiny", "labels": [], "target": "boats"},
            Profile(
                ref="yolos-tiny",
                repo="Xenova/yolos-tiny",
                route="onnx",
                tasks=("detection",),
                labels=("car", "boat"),
                runnable=True,
            ),
        )

        self.assertIn("What to find: boats", text)
        self.assertNotIn("not one of this checkpoint's classes", text)

    def test_a_promptable_model_is_described_as_class_agnostic(self):
        text = self._prompt(
            {"ref": "slimsam-77", "labels": [], "target": "water"},
            Profile(
                ref="slimsam-77",
                repo="Xenova/slimsam-77-uniform",
                route="onnx",
                tasks=("segmentation",),
                promptable=True,
                runnable=True,
            ),
        )

        self.assertIn("class-agnostic", text)
        self.assertIn("name them as the user did", text)

    def test_an_unknown_checkpoint_must_be_read_before_it_is_run(self):
        text = self._prompt(
            {"ref": "hf:acme/thing", "labels": [], "target": ""},
            Profile(
                ref="hf:acme/thing",
                repo="acme/thing",
                route="research",
                tasks=("detection",),
                runnable=True,
            ),
        )

        self.assertIn("read its own config or model card", text)
        self.assertIn(resolve.OWN_OUTPUT_RULE, text)

    def test_label_matching_tolerates_case_and_plurals(self):
        # Nothing typed is the "continue as is" case, not a mismatch.
        self.assertFalse(resolve.label_mismatch("", ("car",)))
        self.assertFalse(resolve.label_mismatch("Car", ("car", "boat")))
        self.assertFalse(resolve.label_mismatch("boats", ("car", "boat")))
        self.assertTrue(resolve.label_mismatch("roads", ("car", "boat")))
        self.assertTrue(resolve.label_mismatch("cars", ("bus",)))

    def test_the_workflow_prompts_defer_to_the_model(self):
        detection = workflows.get("detection").prompt
        segmentation = workflows.get("segmentation").prompt

        self.assertIn("what the model below is trained to detect", detection)
        self.assertIn("what the model below can find", segmentation)


class PromptTests(unittest.TestCase):
    def test_the_prompt_fills_every_placeholder(self):
        text = resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertNotIn("{", text)
        self.assertIn("75.8, 30.88, 75.92, 30.96", text)
        self.assertIn("Xenova/slimsam-77-uniform", text)
        self.assertIn("water bodies", text)
        self.assertIn(resolve.AOI_PATH, text)

    def test_the_model_block_carries_what_the_user_asked_for(self):
        text = resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertIn("slimsam-77 — built in [slimsam-77] (repo Xenova/slimsam-77-uniform; route onnx)", text)
        self.assertIn("What to find: water bodies", text)

    def test_a_built_in_model_is_run_with_its_tool(self):
        text = resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertIn('How to run it: run segment_image(model_id="slimsam-77").', text)

    def test_a_sandbox_model_says_how_to_load_it(self):
        with mock.patch.object(
            resolve.profiles,
            "resolve",
            return_value=Profile(
                ref="hf:acme/segmenter",
                repo="acme/segmenter",
                route="transformers",
                tasks=("segmentation",),
                runnable=True,
            ),
        ):
            text = resolve.prompt_text(
                workflows.get("segmentation"),
                dict(SEGMENT_PARAMS, model={"ref": "hf:acme/segmenter", "target": "water"}),
            )

        self.assertIn("ai_fetch_model", text)
        self.assertIn('skill("transformers")', text)
        self.assertNotIn("run segment_image(", text)

    def test_a_pasted_link_reads_as_its_repository(self):
        with mock.patch.object(
            resolve.profiles,
            "resolve",
            return_value=Profile(
                ref="hf:acme/segmenter",
                repo="acme/segmenter",
                route="transformers",
                tasks=("segmentation",),
                runnable=True,
            ),
        ):
            text = resolve.prompt_text(
                workflows.get("segmentation"),
                dict(
                    SEGMENT_PARAMS,
                    model={"ref": "https://huggingface.co/acme/segmenter", "target": "water"},
                ),
            )

        self.assertIn("Model: acme/segmenter [hf:acme/segmenter]", text)
        self.assertNotIn("https://huggingface.co/acme/segmenter [", text)

    def test_chosen_labels_replace_the_free_text_target(self):
        params = dict(
            SEGMENT_PARAMS,
            model={"ref": "yolos-tiny", "labels": ["car", "boat"], "target": "ignored"},
        )

        with mock.patch.object(
            resolve.profiles,
            "resolve",
            return_value=Profile(ref="yolos-tiny", repo="Xenova/yolos-tiny", route="onnx", tasks=("detection", "segmentation"), runnable=True),
        ):
            text = resolve.prompt_text(workflows.get("segmentation"), params)

        self.assertIn("What to find: car, boat", text)
        self.assertNotIn("ignored", text)

    def test_a_model_that_cannot_do_the_task_is_refused_before_the_run(self):
        with mock.patch.object(
            resolve.profiles,
            "resolve",
            return_value=Profile(
                ref="yolos-tiny",
                repo="Xenova/yolos-tiny",
                route="onnx",
                tasks=("detection",),
                runnable=True,
                reason="It detects objects; it does not segment.",
            ),
        ):
            with self.assertRaises(ToolInputError) as raised:
                resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertIn("cannot do segmentation", str(raised.exception))
        self.assertIn("does not segment", str(raised.exception))

    def test_a_model_that_cannot_run_here_is_refused_with_the_reason(self):
        with mock.patch.object(
            resolve.profiles,
            "resolve",
            return_value=Profile(
                ref="hf:owner/name",
                repo="owner/name",
                route="transformers",
                tasks=("segmentation",),
                runnable=False,
                reason="needs the transformers library.",
            ),
        ):
            with self.assertRaises(ToolInputError) as raised:
                resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertIn("cannot run here yet", str(raised.exception))

    def test_an_empty_optional_answer_reads_as_none(self):
        text = resolve.prompt_text(workflows.get("segmentation"), SEGMENT_PARAMS)

        self.assertIn("Input data: none", text)
        self.assertIn("Extra instructions: none", text)

    def test_a_missing_required_answer_is_refused(self):
        params = dict(SEGMENT_PARAMS)
        params.pop("aoi")

        with self.assertRaises(ToolInputError) as raised:
            resolve.prompt_text(workflows.get("segmentation"), params)

        self.assertIn("Area of interest is required", str(raised.exception))

    def test_a_workflow_without_a_prompt_cannot_be_run(self):
        with self.assertRaises(ToolInputError):
            resolve.prompt_text(workflows.get("blank"), {})


class AoiTests(unittest.TestCase):
    def test_a_bounding_box_string_is_accepted(self):
        answer = resolve.aoi_answer("75.8, 30.88, 75.92, 30.96")

        self.assertEqual(answer["mode"], "bbox")
        self.assertEqual(answer["bbox"], [75.8, 30.88, 75.92, 30.96])

    def test_a_bare_string_becomes_a_place_lookup(self):
        with mock.patch.object(
            resolve.places,
            "geocode",
            return_value={"label": "Pune, India", "bbox": [73.7, 18.4, 73.9, 18.6]},
        ) as geocode:
            answer = resolve.aoi_answer("Pune")

        geocode.assert_called_once_with("Pune")
        self.assertEqual(answer["mode"], "place")
        self.assertEqual(answer["label"], "Pune, India")

    def test_a_drawn_shape_supplies_its_own_bounding_box(self):
        answer = resolve.aoi_answer({"mode": "drawn", "geojson": DRAWN})

        self.assertEqual(answer["bbox"], [75.8, 30.88, 75.92, 30.96])

    def test_an_inverted_box_is_refused(self):
        with self.assertRaises(ToolInputError) as raised:
            resolve.parse_bbox([75.92, 30.88, 75.80, 30.96])

        self.assertIn("inverted", str(raised.exception))

    def test_a_box_outside_the_world_is_refused(self):
        with self.assertRaises(ToolInputError):
            resolve.parse_bbox([0, 0, 200, 10])

    def test_a_mode_without_a_box_or_shape_is_refused(self):
        with self.assertRaises(ToolInputError) as raised:
            resolve.aoi_answer({"mode": "view"})

        self.assertIn("needs a bounding box", str(raised.exception))

    def test_a_drawn_shape_is_kept_as_a_feature_collection(self):
        answer = resolve.aoi_answer({"mode": "drawn", "geojson": DRAWN})

        self.assertEqual(resolve.aoi_geometry(answer), DRAWN)

    def test_a_box_becomes_a_rectangle_geometry(self):
        answer = resolve.aoi_answer({"mode": "bbox", "bbox": "75.8,30.88,75.92,30.96"})

        geometry = resolve.aoi_geometry(answer)
        ring = geometry["features"][0]["geometry"]["coordinates"][0]

        self.assertEqual(geometry["features"][0]["geometry"]["type"], "Polygon")
        self.assertEqual(ring[0], ring[-1])


class SeedTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Workspace(Path(self._tmp.name) / "demo").create()
        self.notebook = NotebookSession()
        self.notebook.load(self.workspace)

    def tearDown(self):
        self._tmp.cleanup()

    def test_seeding_writes_the_record_cell_the_prompt_cell_and_the_aoi_file(self):
        resolve.seed(self.workspace, self.notebook, workflows.get("segmentation"), SEGMENT_PARAMS)

        kinds = [cell["kind"] for cell in self.notebook.cells()]
        aoi_file = self.workspace.root / resolve.AOI_PATH

        self.assertEqual(kinds, ["markdown", "prompt"])
        self.assertTrue(aoi_file.is_file())
        self.assertIn("FeatureCollection", aoi_file.read_text(encoding="utf-8"))

    def test_the_prompt_cell_carries_the_workflow_and_the_answers(self):
        cells = resolve.seed(
            self.workspace, self.notebook, workflows.get("segmentation"), SEGMENT_PARAMS
        )

        prompt_cell = cells[-1]
        record = prompt_cell["metadata"]["geoai"]["workflow"]

        self.assertEqual(record["id"], "segmentation")
        self.assertIn("slimsam-77", record["answers"]["model"])
        self.assertEqual(record["aoi"]["mode"], "bbox")
        self.assertIn("75.8, 30.88, 75.92, 30.96", prompt_cell["source"])

    def test_a_drawn_area_of_interest_is_written_where_the_prompt_says(self):
        params = dict(SEGMENT_PARAMS, aoi={"mode": "drawn", "geojson": DRAWN})

        cells = resolve.seed(
            self.workspace, self.notebook, workflows.get("segmentation"), params
        )

        written = (self.workspace.root / resolve.AOI_PATH).read_text(encoding="utf-8")
        self.assertIn("75.8", written)
        self.assertIn("drawn", cells[-1]["source"])

    def test_the_workflow_block_survives_a_notebook_round_trip(self):
        resolve.seed(self.workspace, self.notebook, workflows.get("segmentation"), SEGMENT_PARAMS)
        self.notebook.save()

        reopened = NotebookSession()
        reopened.load(self.workspace)
        prompt_cell = [cell for cell in reopened.cells() if cell["kind"] == "prompt"][0]

        self.assertEqual(prompt_cell["workflow"]["id"], "segmentation")


class WorkflowRouteTests(unittest.TestCase):
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

    def test_the_route_lists_the_workflows_with_their_fields(self):
        payload = self.client.get("/api/workflows").json()
        listed = payload["workflows"]

        self.assertEqual(listed[0]["id"], "blank")
        self.assertEqual(listed[0]["params"], [])

        segmentation = next(item for item in listed if item["id"] == "segmentation")
        fields = {param["id"]: param for param in segmentation["params"]}

        self.assertEqual(fields["aoi"]["type"], "aoi")
        self.assertEqual(fields["model"]["type"], "model")
        self.assertEqual(fields["model"]["task"], "segmentation")
        self.assertTrue(any(option["recommended"] for option in fields["model"]["options"]))
        self.assertEqual(fields["model"]["default"], fields["model"]["options"][0]["value"])

    def test_creating_a_workspace_from_a_workflow_seeds_its_cells(self):
        response = self.client.post(
            "/api/workspace/new",
            json={"name": "wf-demo", "workflow": "segmentation", "params": SEGMENT_PARAMS},
        )

        self.assertEqual(response.status_code, 200, response.text)
        snapshot = response.json()

        self.assertEqual(snapshot["active_workspace"], "wf-demo")
        self.assertEqual([cell["kind"] for cell in snapshot["cells"]], ["markdown", "prompt"])
        self.assertIn("data/aoi.geojson", snapshot["files"])
        self.assertIn("75.8, 30.88, 75.92, 30.96", snapshot["cells"][-1]["source"])

    def test_a_plain_new_workspace_still_starts_empty(self):
        response = self.client.post("/api/workspace/new", json={"name": "plain"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["cells"], [])

    def test_a_missing_required_answer_is_a_400_with_the_field_name(self):
        params = dict(SEGMENT_PARAMS)
        params["aoi"] = None

        response = self.client.post(
            "/api/workspace/new",
            json={"name": "wf-bad", "workflow": "segmentation", "params": params},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("Area of interest is required", response.json()["detail"])

    def test_an_unknown_workflow_is_a_404(self):
        response = self.client.post(
            "/api/workspace/new", json={"name": "wf-nope", "workflow": "teleport"}
        )

        self.assertEqual(response.status_code, 404, response.text)

    def test_the_geocode_route_needs_a_query(self):
        self.assertEqual(self.client.get("/api/geocode").status_code, 422)

    def test_the_geocode_route_answers_with_a_box(self):
        with mock.patch(
            "spatial_intelligence.server.routers.workflows.places.geocode",
            return_value={"label": "Pune", "bbox": [73.7, 18.4, 73.9, 18.6], "center": None},
        ):
            response = self.client.get("/api/geocode", params={"q": "Pune"})

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["bbox"], [73.7, 18.4, 73.9, 18.6])

    def test_the_profile_route_accepts_a_repository_link(self):
        from spatial_intelligence.ai import profiles

        info = {
            "repo": "acme/segmenter",
            "revision": "abc",
            "pipeline": "image-segmentation",
            "library": "transformers",
            "tags": ["transformers", "safetensors"],
            "architectures": ["SegformerForSemanticSegmentation"],
            "labels": ["water"],
            "files": ["model.safetensors"],
        }
        profiles.clear_cache()

        with mock.patch.object(profiles.hub, "repo_info", return_value=info):
            response = self.client.post(
                "/api/models/profile",
                json={"model": "https://huggingface.co/acme/segmenter/tree/main", "task": "segmentation"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        answer = response.json()
        self.assertEqual(answer["repo"], "acme/segmenter")
        self.assertEqual(answer["revision"], "main")
        self.assertEqual(answer["labels"], ["water"])
        self.assertTrue(answer["supported"])

    def test_the_profile_route_refuses_a_foreign_link(self):
        response = self.client.post(
            "/api/models/profile",
            json={"model": "https://github.com/acme/thing", "task": "segmentation"},
        )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("only Hugging Face", response.json()["detail"])

    def test_creating_the_same_workspace_twice_is_refused(self):
        first = self.client.post("/api/workspace/new", json={"name": "twice"})
        self.assertEqual(first.status_code, 200, first.text)
        self.client.post("/api/cells", json={"kind": "markdown", "source": "mine"})
        self.client.post("/api/workspace/close")

        second = self.client.post("/api/workspace/new", json={"name": "twice"})

        self.assertEqual(second.status_code, 400, second.text)
        self.assertIn("already exists", second.json()["detail"])
        # and it must not have opened the old one either
        self.assertIsNone(self.client.get("/api/state").json()["active_workspace"])

    def test_a_workspace_name_cannot_be_empty(self):
        response = self.client.post("/api/workspace/new", json={"name": "   "})

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("enter a workspace name", response.json()["detail"])

    def test_a_workspace_name_cannot_escape_the_data_root(self):
        for name in ("../escape", "..\\escape", "sub/dir", ".hidden"):
            with self.subTest(name=name):
                response = self.client.post("/api/workspace/new", json={"name": name})
                self.assertEqual(response.status_code, 400, response.text)

    def test_a_failed_create_leaves_no_workspace_behind(self):
        # The name was free a moment ago, so the user must be able to fix the
        # mistake and try the same name again.
        params = dict(SEGMENT_PARAMS)
        params["model"] = {"ref": "hf:acme/unknown-thing", "target": ""}
        broken = self.client.post(
            "/api/workspace/new",
            json={"name": "retryable", "workflow": "segmentation", "params": params},
        )
        self.assertEqual(broken.status_code, 400, broken.text)
        # The answers are checked before anything is created.
        self.assertFalse((Path(os.environ["GEOAI_HOME"]) / "workspaces" / "retryable").exists())

        good = self.client.post(
            "/api/workspace/new",
            json={"name": "retryable", "workflow": "segmentation", "params": SEGMENT_PARAMS},
        )

        self.assertEqual(good.status_code, 200, good.text)
        self.assertEqual(good.json()["active_workspace"], "retryable")


if __name__ == "__main__":
    unittest.main()
