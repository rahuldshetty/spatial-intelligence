"""The FastAPI application: static shell, JSON/SSE routes, and error mapping.

Every mutation lives behind a router that calls one ``AppState`` method; the
handlers here turn this package's error taxonomy into HTTP responses so routers
can raise domain errors instead of translating them one by one.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from ..contracts.errors import ToolInputError, WorkspaceError
from . import deps
from .assets import WEB_DIR
from .routers import events, files, state, workspace

NO_STORE = {"Cache-Control": "no-store"}


class NoCacheStaticFiles(StaticFiles):
    """Serve the app shell without letting the browser keep a stale bundle."""

    async def get_response(self, path: str, scope: dict) -> Response:
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


def _register_error_handlers(app: FastAPI) -> None:
    """Map domain errors onto status codes the UI already understands."""

    @app.exception_handler(ToolInputError)
    async def _bad_request(request: Request, exc: ToolInputError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(WorkspaceError)
    async def _workspace_error(request: Request, exc: WorkspaceError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(KeyError)
    async def _not_found(request: Request, exc: KeyError) -> JSONResponse:
        detail = exc.args[0] if exc.args else "not found"
        return JSONResponse({"detail": str(detail)}, status_code=404)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Run the app, then stop the process-wide session's worker on shutdown.

    Without this the agent's run worker thread outlives the serving loop: it is a
    daemon, so the process still exits, but nothing closes the workspace handles
    it holds and a managed restart would leave the previous session behind.
    """
    yield
    deps.close_app_state()


def create_app() -> FastAPI:
    """Build the application (routers, static mount, and the shell route)."""
    app = FastAPI(title="Spatial Intelligence", lifespan=_lifespan)
    _register_error_handlers(app)
    app.include_router(state.router)
    app.include_router(workspace.router)
    app.include_router(files.router)
    app.include_router(events.router)
    app.mount("/static", NoCacheStaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page shell."""
        return FileResponse(WEB_DIR / "index.html", headers=NO_STORE)

    return app


app = create_app()
