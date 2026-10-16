"""Human-in-the-loop tools rendered as structured forms by the web app."""

from __future__ import annotations

from pydantic_ai import CallDeferred

from ...contracts.forms import ChoiceOption, InteractionField
from ..runtime import ToolRuntime
from ..spec import ToolKind, pack, tool

# ``ChoiceOption``/``InteractionField`` live in ``contracts.forms`` so a
# workflow's wizard can ask for its parameters with the same schema this tool
# sends. They stay exported here for callers that already import them.


@pack(category="interaction")
class InteractionPack:
    """Asks the user a structured question by deferring the run."""

    def __init__(self, runtime: ToolRuntime) -> None:
        """The registry hands every pack its runtime; this tool needs none of it."""

    @tool(category="interaction", kind=ToolKind.INTERACTIVE, core=True)
    def request_user_input(
        self,
        title: str,
        prompt: str,
        fields: list[InteractionField],
        submit_label: str = "Continue",
        allow_cancel: bool = True,
    ) -> str:
        """Pause and ask the user for consequential or ambiguous choices.

        Use radio for exactly one choice, multi_select for several choices, text for
        custom input, and confirmation for yes/no. Include a recommended option when
        evidence supports one. Do not ask for minor reversible decisions.
        """
        form = {
            "title": title,
            "prompt": prompt,
            "fields": [field.model_dump(mode="json") for field in fields],
            "submit_label": submit_label,
            "allow_cancel": allow_cancel,
        }
        # The browser supplies this externally. Pydantic AI ends the current run
        # with DeferredToolRequests and consumes the submitted value on resume.
        raise CallDeferred(metadata={"interaction": form})


__all__ = ["ChoiceOption", "InteractionField", "InteractionPack"]
