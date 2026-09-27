"""Environment and data-root resolution for the app.

Ported from ``geoai/config.py``; the ``GEOAI_*`` variable names, the
``workspaces/`` + ``settings.json`` data-root layout, and the
``GEOAI_HOME``/XDG fallback are unchanged so an existing checkout keeps working.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL = "openai:gpt-4o"

# Repo root: <root>/spatial_intelligence/settings/env.py -> <root>/.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def app_root() -> Path:
    """Return the data root for ``workspaces/``, ``settings.json``, ``.env``.

    ``GEOAI_HOME`` wins when set. Otherwise a dev checkout (a ``.git`` dir or
    an existing ``workspaces/`` next to the package) keeps data beside the
    source; anything else (a pip-installed or frozen app) falls back to the
    XDG data home at ``~/.local/share/geo-ai``.
    """
    env = os.getenv("GEOAI_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    if (_PACKAGE_ROOT / ".git").exists() or (_PACKAGE_ROOT / "workspaces").is_dir():
        return _PACKAGE_ROOT
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "geo-ai"


def model_from_env() -> str:
    """Return the agent model string, overridable via ``GEOAI_MODEL``."""
    return os.getenv("GEOAI_MODEL", DEFAULT_MODEL)


# Model provider prefix -> env var holding its API key (None = no key required).
_PROVIDER_API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    # `openai-chat:` and `openai-responses:` pick the OpenAI-compatible API for a
    # custom endpoint (see `agent.model.resolve_model`); the key is the same one.
    "openai-chat": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google-gla": "GOOGLE_API_KEY",
    "google-gemini": "GOOGLE_API_KEY",
    "ollama": None,
}


def model_provider(model: str) -> str:
    """Return the provider prefix of a model string (default ``openai``)."""
    return model.split(":", 1)[0].lower() if ":" in model else "openai"


def api_key_env_for(model: str) -> str | None:
    """Return the env var holding the API key for ``model``'s provider.

    Returns ``None`` for providers that need no key (e.g. ``ollama``).
    """
    return _PROVIDER_API_KEY_ENV.get(model_provider(model))


def _key_is_optional(key_var: str) -> bool:
    """Return whether a custom endpoint makes ``key_var`` optional.

    pydantic-ai substitutes a placeholder key when ``OPENAI_BASE_URL`` is set and
    ``OPENAI_API_KEY`` is not, because a locally served OpenAI-compatible model
    usually wants none. Refusing to start would reject the deployment that works.
    """
    return key_var == "OPENAI_API_KEY" and bool(os.getenv("OPENAI_BASE_URL", "").strip())


def validate_env() -> None:
    """Refuse to start when the configured model's provider key is missing.

    ``<app_root>/.env`` is loaded here, so this check answers for both of the
    documented sources: the process environment and that file. Loading inside
    the check is deliberate — start-up used to validate before anything had read
    the file, which reported a key that is present in ``.env`` as missing.

    A custom ``OPENAI_BASE_URL`` makes the key optional (see ``_key_is_optional``).

    Raises ``SystemExit`` with a setup hint when the required variable is absent
    or empty, so a user never hits a late 500 while creating a workspace.
    """
    load_env()
    model = model_from_env()
    key_var = api_key_env_for(model)
    if key_var is None:
        return
    if os.getenv(key_var, "").strip():
        return
    if _key_is_optional(key_var):
        return
    env_file = app_root() / ".env"
    if env_file.is_file():
        detail = f"Set {key_var} in {env_file}, or export it in the environment."
    else:
        detail = (
            f"Copy .env.example to {env_file}, set {key_var} there, "
            f"or export {key_var} in the environment."
        )
    raise SystemExit(
        f"Spatial Intelligence: cannot start — {key_var} is not set "
        f"(model '{model}'). {detail}"
    )


def max_retries() -> int:
    """Return the transient provider attempt cap from ``GEOAI_MAX_RETRIES``.

    A prompt is replayed only for a transient provider failure before any tool
    changes the workspace or map. Defaults to 5; values are clamped to at least
    1.
    """
    raw = os.getenv("GEOAI_MAX_RETRIES", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def resolve_workspace_name(override: str | None = None) -> str:
    """Resolve the active workspace name.

    Precedence: explicit ``override`` (a notebook's ``WORKSPACE_NAME``) →
    ``GEOAI_WORKSPACE`` env var → the notebook's ``__file__`` stem → ``"default"``.
    """
    if override and override.strip():
        return override.strip()
    env_name = os.getenv("GEOAI_WORKSPACE", "").strip()
    if env_name:
        return env_name
    main = sys.modules.get("__main__")
    file = getattr(main, "__file__", None) if main is not None else None
    if file:
        return Path(file).stem
    return "default"

def server_base_url() -> str:
    """Return the app server's base URL for same-host file serving.

    The server binds ``127.0.0.1`` and serves workspace files from this origin
    (see ``/api/files/``), so the in-iframe map can fetch local rasters without
    the cross-origin failures of the geolibre static server's per-session tokens.
    The port mirrors the ``GEOAI_PORT`` env var used by the server's ``run()``.
    """
    port = os.getenv("GEOAI_PORT", "8000")
    return f"http://127.0.0.1:{port}/"


def workspace_root(name: str) -> Path:
    """Return the absolute workspace root for ``name`` (under ``app_root()``).

    Computed from the data root, never the process CWD.
    """
    return app_root() / "workspaces" / name


def list_workspaces() -> list[str]:
    """Return the names of existing workspace directories, sorted."""
    base = app_root() / "workspaces"
    if not base.is_dir():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


def load_env() -> None:
    """Load ``<app_root>/.env`` into ``os.environ``.

    Existing environment variables take precedence (dotenv's default) and the
    call is idempotent, so every entry point can safely resolve configuration
    from the file. When ``python-dotenv`` is not installed the file is ignored
    entirely, which is said out loud: in a frozen build that would otherwise
    surface much later as "the key is not set".
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        print(
            "spatial-intelligence: python-dotenv is missing, so "
            f"{app_root() / '.env'} will not be read",
            file=sys.stderr,
        )
        return
    path = app_root() / ".env"
    shadowed = _shadowed_keys(path)
    load_dotenv(path)
    if shadowed:
        # Silent shadowing is the confusing case: the file says one thing, the
        # process environment says another, and the process wins.
        print(
            "spatial-intelligence: these variables are already set in the "
            f"environment, so the values in {path} are ignored: "
            + ", ".join(sorted(shadowed)),
            file=sys.stderr,
        )


def _shadowed_keys(path: Path) -> set[str]:
    """Return the keys ``path`` sets that the environment already defines.

    Only names with a *different* value count: an identical value shadowing
    itself is not worth reporting.
    """
    if not path.is_file():
        return set()
    shadowed: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    for line in lines:
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not name or name.startswith("#"):
            continue
        current = os.environ.get(name)
        if current is not None and current != value.strip().strip("\"'"):
            shadowed.add(name)
    return shadowed
