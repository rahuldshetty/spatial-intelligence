"""Workflow wizards: a few fields in, one rendered prompt out.

``library`` declares the shipped workflows, ``resolve`` turns wizard answers
into ``data/aoi.geojson`` plus the prompt and cells to seed, and ``spec`` holds
the declaration type they share.
"""

from .library import BY_ID, WORKFLOWS, get
from .resolve import AOI_PATH, aoi_answer, prompt_text, seed
from .spec import WorkflowSpec

__all__ = [
    "AOI_PATH",
    "BY_ID",
    "WORKFLOWS",
    "WorkflowSpec",
    "aoi_answer",
    "get",
    "prompt_text",
    "seed",
]
