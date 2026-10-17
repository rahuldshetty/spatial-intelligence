"""Rendering the user's structured answers back into the notebook."""

from __future__ import annotations

import unittest

from spatial_intelligence.agent.interactions import answers_markdown


class AnswersMarkdownTests(unittest.TestCase):
    def test_an_aoi_answer_renders_mode_label_and_bbox(self):
        interaction = {
            "fields": [
                {"id": "area", "label": "Area to cover", "type": "aoi"},
            ]
        }
        text = answers_markdown(
            interaction,
            {
                "area": {
                    "mode": "view",
                    "bbox": [72.8585, 19.0169, 72.8742, 19.0297],
                    "geojson": None,
                    "label": None,
                }
            },
        )

        self.assertIn("**Area to cover:**", text)
        self.assertIn("Map view", text)
        self.assertIn("bbox [72.8585, 19.0169, 72.8742, 19.0297]", text)
        self.assertNotIn("geojson", text)

    def test_an_aoi_place_answer_uses_the_resolved_label(self):
        interaction = {
            "fields": [{"id": "area", "label": "Area", "type": "aoi"}]
        }
        text = answers_markdown(
            interaction,
            {
                "area": {
                    "mode": "place",
                    "bbox": [77.5, 12.9, 77.7, 13.1],
                    "label": "Bengaluru",
                }
            },
        )
        self.assertIn("Place", text)
        self.assertIn("Bengaluru", text)

    def test_radio_and_multi_select_keep_their_labels(self):
        interaction = {
            "fields": [
                {
                    "id": "scene",
                    "label": "Scene",
                    "type": "radio",
                    "options": [{"value": "after", "label": "After image"}],
                },
                {
                    "id": "extras",
                    "label": "Extras",
                    "type": "multi_select",
                    "options": [
                        {"value": "ndwi", "label": "Water index"},
                        {"value": "true", "label": "True colour"},
                    ],
                },
            ]
        }
        text = answers_markdown(
            interaction, {"scene": "after", "extras": ["ndwi", "true"]}
        )
        self.assertIn("**Scene:** After image", text)
        self.assertIn("**Extras:** Water index, True colour", text)

    def test_a_dict_answer_in_a_non_aoi_field_is_json_not_a_crash(self):
        interaction = {
            "fields": [{"id": "meta", "label": "Metadata", "type": "text"}]
        }
        text = answers_markdown(interaction, {"meta": {"part": 1, "kind": "x"}})
        self.assertIn('{"part": 1, "kind": "x"}', text)


if __name__ == "__main__":
    unittest.main()
