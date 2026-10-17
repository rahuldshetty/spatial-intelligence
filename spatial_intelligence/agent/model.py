"""Model resolution, so a custom OpenAI-compatible endpoint can be honored."""

from __future__ import annotations

import os
from dataclasses import dataclass

#: Model kinds served by an OpenAI-compatible endpoint. Each is routed through an
#: ``OpenAIProvider`` bound to ``OPENAI_BASE_URL``.
_OPENAI_KINDS = frozenset({"openai", "openai-chat", "openai-responses"})

#: The kind a bare ``openai:`` becomes once ``OPENAI_BASE_URL`` is set.
#:
#: Chat Completions is what self-hosted and third-party OpenAI-compatible servers
#: implement (llama.cpp, vLLM, LiteLLM, Ollama, OpenRouter, DeepSeek). pydantic-ai
#: would otherwise infer the Responses API for ``openai:``, which such a server
#: either rejects outright or streams in a shape pydantic-ai cannot parse — a
#: llama.cpp reasoning delta omits the ``content_index`` pydantic-ai indexes by.
#: ``openai-responses:`` opts back in for an endpoint that really speaks it.
_CUSTOM_ENDPOINT_KIND = "openai-chat"

#: Key sent to a custom endpoint that checks none. The OpenAI client insists on a
#: non-empty key even when the server ignores it; pydantic-ai substitutes its own
#: only when ``OPENAI_API_KEY`` is entirely absent from the environment, which an
#: empty ``.env`` line is not — so the placeholder is supplied here.
_KEYLESS_API_KEY = "not-needed"


def validate_model(model: object) -> None:
    """Raise when a model string cannot be resolved to a model.

    Settings must not accept a model that would only fail later, when a
    workspace is opened — which is exactly when a rebuild becomes possible.
    A ready-made ``Model`` instance passes through untouched.
    """
    if not isinstance(model, str):
        return
    resolved = resolve_model(model)
    if not isinstance(resolved, str):
        return
    from pydantic_ai.models import infer_model

    infer_model(resolved)


def _custom_base_url() -> str:
    """Return ``OPENAI_BASE_URL`` normalized to a base URL, or ``''`` when unset.

    pydantic-ai appends the route (``/chat/completions`` or ``/responses``) itself,
    so a user-supplied full endpoint is trimmed here rather than double-suffixed
    into a 404.
    """
    base_url = os.getenv("OPENAI_BASE_URL", "").strip().rstrip("/")
    if base_url.endswith("/chat/completions"):
        base_url = base_url[: -len("/chat/completions")].rstrip("/")
    return base_url


@dataclass(frozen=True, slots=True)
class OpenAIEndpoint:
    """Where an OpenAI-compatible model actually lives on the wire."""

    base_url: str
    model: str
    api_key: str
    api: str
    """``chat`` for Chat Completions, ``responses`` for the Responses API."""


def openai_endpoint(model: object) -> OpenAIEndpoint | None:
    """Return the HTTP endpoint for an OpenAI-compatible model string.

    ``None`` means this provider is not OpenAI-compatible, so image-input
    support cannot be checked through the same probe. A custom
    ``OPENAI_BASE_URL`` wins; without it the public OpenAI API is used.
    """
    if not isinstance(model, str):
        return None
    kind, separator, name = model.partition(":")
    if not separator or kind not in _OPENAI_KINDS or not name.strip():
        return None
    base_url = _custom_base_url() or "https://api.openai.com/v1"
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not _custom_base_url() and not api_key:
        # The public API needs a key; callers still get a useful endpoint so a
        # probe can report a credential failure instead of "not supported".
        api_key = ""
    elif not api_key:
        api_key = _KEYLESS_API_KEY
    return OpenAIEndpoint(
        base_url=base_url,
        model=name.strip(),
        api_key=api_key,
        api="responses" if kind == "openai-responses" else "chat",
    )


def resolve_model(model: str):
    """Resolve a model string to a ``Model`` instance, honoring a custom endpoint.

    A caller may also pass a ready-made ``Model`` instance, which is returned as is.

    With ``OPENAI_BASE_URL`` set, the ``openai`` kinds are built against that
    endpoint (Azure OpenAI, LiteLLM, vLLM, llama.cpp, OpenRouter, local proxy):
    ``openai:`` and ``openai-chat:`` speak Chat Completions, ``openai-responses:``
    speaks the Responses API. Without it the string is returned unchanged and
    pydantic-ai infers the provider and API, so plain OpenAI is untouched.
    """
    if not isinstance(model, str):
        return model
    kind, separator, name = model.partition(":")
    if not separator or kind not in _OPENAI_KINDS:
        return model
    base_url = _custom_base_url()
    if not base_url:
        return model

    from pydantic_ai.models import infer_model, infer_provider
    from pydantic_ai.providers.openai import OpenAIProvider

    api_key = os.getenv("OPENAI_API_KEY", "").strip() or _KEYLESS_API_KEY

    def provider_factory(provider_name: str):
        if provider_name in _OPENAI_KINDS:
            return OpenAIProvider(base_url=base_url, api_key=api_key)
        return infer_provider(provider_name)

    return infer_model(
        f"{kind if kind != 'openai' else _CUSTOM_ENDPOINT_KIND}:{name}",
        provider_factory=provider_factory,
    )
