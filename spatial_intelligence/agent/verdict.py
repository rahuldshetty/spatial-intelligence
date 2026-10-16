"""Asking the session's model to judge a checkpoint the metadata cannot.

The wizard has to be able to say "this model will not do that task" before a run
starts. Metadata answers most cases; this answers the rest — a repository whose
task is undeclared, an unfamiliar architecture, or a heuristic that came out
negative and deserves a second opinion.

The call is small, cached per ``(ref, task, model)``, and never raises: if the
provider is unreachable the metadata's answer stands.

The answer is asked for as a JSON object in the text rather than as pydantic-ai
structured output: an OpenAI-compatible endpoint in reasoning mode (DeepSeek's,
for one) rejects the forced ``tool_choice`` that structured output uses, and the
verdict must work on whatever model the user configured.
"""

from __future__ import annotations

import json
import threading

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent

from .model import resolve_model

#: Tasks a verdict may name. Anything else is treated as "no fit".
TASKS = ("segmentation", "detection", "classification")

SYSTEM_PROMPT = """\
You judge whether a machine-learning checkpoint can do a geospatial task before
the user waits for a run. You are given facts from its Hugging Face repository.

Reply with one JSON object and nothing else — no prose, no code fences:
{"supported": true|false, "task": "segmentation|detection|classification|", \
"labels": ["..."], "promptable": true|false, "reason": "one sentence"}

- supported: true only when the checkpoint can carry out the requested task.
- task: the single best fit, or "" when none of them fits.
- labels: the class names the checkpoint already predicts. Empty when it is
  promptable (points, boxes, or text describe what to find) or unknown.
- promptable: true when it generalises to classes it was not trained on because
  the user prompts it — SAM-style segmentation, CLIP-style or grounding
  detectors. A fixed-class detector is not promptable.
- reason: one sentence, plain language, addressed to the user, saying what this
  checkpoint is good for. No hedging, no apologies.

Judge from what the weights are, not from what the name sounds like, and prefer
"I cannot tell" over a confident guess: say supported=false with a reason that
names the missing evidence.
"""


class CheckpointVerdict(BaseModel):
    """One judge's answer about one checkpoint."""

    supported: bool = False
    task: str = Field(default="", description="segmentation, detection, classification, or ''")
    labels: list[str] = Field(default_factory=list, description="classes it already predicts")
    promptable: bool = Field(default=False, description="takes prompts and generalises to new classes")
    reason: str = Field(default="", description="one plain sentence for the user")


_lock = threading.Lock()
_cache: dict[tuple[str, str, str], dict] = {}


def _prompt(facts: dict, task: str) -> str:
    """Describe the checkpoint and the task it is being considered for."""
    lines = [f"Requested task: {task or 'unspecified'}", "", "Repository facts:"]
    for key, value in facts.items():
        lines.append(f"- {key}: {value}")
    if not facts:
        lines.append("- (none)")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model answer."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def classify(model: str, ref: str, facts: dict, task: str) -> dict | None:
    """Judge ``ref`` for ``task``, or return ``None`` when the call fails.

    ``model`` is the session's model string; the verdict is cached, so the
    second glance at the same checkpoint in the same session is free.
    """
    key = (ref, task, str(model))
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    try:
        agent: Agent = Agent(resolve_model(model), system_prompt=SYSTEM_PROMPT)
        result = agent.run_sync(_prompt(facts, task))
    except Exception:
        return None
    payload = _extract_json(str(result.output))
    if payload is None:
        return None
    try:
        verdict = CheckpointVerdict.model_validate(payload)
    except ValidationError:
        return None
    answer = {
        "supported": bool(verdict.supported),
        "task": str(verdict.task or "").strip().lower(),
        "labels": [str(label) for label in (verdict.labels or []) if str(label).strip()],
        "promptable": bool(verdict.promptable),
        "reason": " ".join(str(verdict.reason or "").split()),
    }
    with _lock:
        _cache[key] = answer
    return answer


def clear_cache() -> None:
    """Forget every verdict (used by the tests)."""
    with _lock:
        _cache.clear()


__all__ = ["CheckpointVerdict", "classify", "clear_cache"]
