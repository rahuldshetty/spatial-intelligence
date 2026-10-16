"""The wizard's step visibility, which is layout and so cannot be unit-tested.

Two shipped bugs came from the same place: a step that is `hidden` but still laid
out sits on top of the step the user is looking at and silently swallows its
clicks — cards that "sometimes" select, and a dialog that looks like all its
steps at once. The mechanism that prevents it is a pair: the base rule hides,
`.active` shows, and the script toggles the class. These tests pin both halves
together so one cannot drift from the other.
"""

import re
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parents[2] / "spatial_intelligence" / "web"
CSS = WEB / "css" / "dialogs.css"
WIZARD = WEB / "js" / "components" / "workflow-dialog.js"

#: The wizard's step selector, and the two rules that must own its display.
STEP = "#dialog.wizard-dialog .wizard-step"
VISIBLE = STEP + ".active"

_BLOCK = re.compile(r"([^{}]+)\{([^{}]*)\}", re.S)
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
#: The step class itself, not `.wizard-steps` or `.wizard-step-chip`.
_STEP_CLASS = re.compile(r"\.wizard-step(?![\w-])")


def _rules(css: str) -> list[tuple[str, str]]:
    """Every ``(selector, body)`` pair, comments removed and spacing normalized."""
    without_comments = _COMMENT.sub("", css)
    return [
        (" ".join(selector.split()), body)
        for selector, body in _BLOCK.findall(without_comments)
    ]


def _selectors(css: str, selector: str) -> list[str]:
    """The ``display`` values declared by exactly this selector."""
    values: list[str] = []
    for block_selector, body in _rules(css):
        if block_selector != selector:
            continue
        for declaration in body.split(";"):
            name, _, value = declaration.partition(":")
            if name.strip() == "display":
                values.append(value.strip())
    return values


class WizardStepVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.css = CSS.read_text(encoding="utf-8")
        self.script = WIZARD.read_text(encoding="utf-8")

    def test_the_base_rule_hides_a_step(self):
        self.assertEqual(_selectors(self.css, STEP), ["none"])

    def test_only_the_active_class_shows_a_step(self):
        self.assertEqual(_selectors(self.css, VISIBLE), ["grid"])

    def test_no_other_rule_gives_a_step_a_display(self):
        # A bare `display` on the base (with `[hidden]` losing on specificity) is
        # exactly what stacked the steps and ate the card clicks.
        others = [
            block_selector
            for block_selector, body in _rules(self.css)
            if _STEP_CLASS.search(block_selector)
            and "display" in body
            and block_selector not in (STEP, VISIBLE)
        ]

        self.assertEqual(others, [])

    def test_the_model_panes_use_the_same_pair(self):
        panes = _selectors(self.css, ".model-pane")
        active = _selectors(self.css, ".model-pane.active")

        self.assertEqual(panes, ["none"])
        self.assertEqual(active, ["block"])

    def test_the_script_toggles_the_class_it_relies_on(self):
        self.assertIn('classList.toggle("active"', self.script)

    def test_the_script_shows_a_step_before_the_dialog_is_used(self):
        # Without this, the dialog opens with every step hidden and no way in.
        self.assertIn('show("type")', self.script)

    def test_a_step_hidden_from_the_accessibility_tree_matches_the_class(self):
        self.assertIn("node.hidden = !active", self.script)


if __name__ == "__main__":
    unittest.main()
