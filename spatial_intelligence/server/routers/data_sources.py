"""Data Source Registry: source list, enabled state, search, and add."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..deps import app_state

router = APIRouter(tags=["data-sources"])


class UpdateDataSources(BaseModel):
    """Enabled/disabled choices to save at one scope."""

    scope: Literal["global", "workspace"]
    config: dict = Field(default_factory=dict)


class DataQueryBody(BaseModel):
    """The optional common query fields plus source-specific filters."""

    text: str = ""
    bbox: list[float] | None = None
    start_date: str | None = None
    end_date: str | None = None
    limit: int = 20
    cursor: str | None = None
    filters: dict = Field(default_factory=dict)


class AddDatasetBody(BaseModel):
    asset_id: str | None = None
    name: str | None = None
    download: bool = True


class DownloadDatasetBody(BaseModel):
    asset_id: str | None = None
    filename: str | None = None


@router.get("/api/data-sources")
def api_get_data_sources() -> dict:
    """Return both scopes, the effective config, and every source."""
    return app_state().data_sources_payload()


@router.put("/api/data-sources/config")
def api_update_data_sources(body: UpdateDataSources) -> dict:
    """Persist enable/disable choices and reconfigure the live registry."""
    return app_state().update_data_source_config(body.scope, body.config)


@router.post("/api/data-sources/{source_id}/search")
def api_search_data_source(source_id: str, body: DataQueryBody) -> dict:
    """Search one enabled source."""
    return app_state().search_data_source(source_id, body.model_dump())


@router.post("/api/data-sources/{source_id}/datasets/{dataset_id}/add")
def api_add_data_source_dataset(
    source_id: str, dataset_id: str, body: AddDatasetBody
) -> dict:
    """Download when needed and add one dataset to the live map."""
    return app_state().add_data_source_dataset(
        source_id,
        dataset_id,
        asset_id=body.asset_id,
        name=body.name,
        download=body.download,
    )


@router.post("/api/data-sources/{source_id}/datasets/{dataset_id}/download")
def api_download_data_source_dataset(
    source_id: str, dataset_id: str, body: DownloadDatasetBody
) -> dict:
    """Download one dataset into the workspace without mapping it."""
    return app_state().download_data_source_dataset(
        source_id,
        dataset_id,
        asset_id=body.asset_id,
        filename=body.filename,
    )


__all__ = ["router"]
