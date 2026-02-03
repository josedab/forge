"""REST API for the feature catalog using FastAPI."""

from __future__ import annotations

import logging
from typing import Any

from forge.catalog.store import CatalogEntry, CatalogStore, FeatureCatalogStore
from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


def create_catalog_app(
    store: CatalogStore | None = None,
    title: str = "Forge Feature Catalog API",
) -> Any:
    """Create a FastAPI application for the feature catalog.

    Args:
        store: CatalogStore instance. Creates a new one if None.
        title: API title.

    Returns:
        FastAPI application instance.

    Example:
        >>> from forge.catalog.api import create_catalog_app
        >>> app = create_catalog_app()
        >>> # Run with: uvicorn module:app --host 0.0.0.0 --port 8000
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from pydantic import BaseModel
    except ImportError:
        raise MissingDependencyError("fastapi", "Feature Catalog REST API")

    catalog_store = store or FeatureCatalogStore()
    app = FastAPI(title=title, version="1.0.0")

    class EntryCreate(BaseModel):
        name: str
        description: str = ""
        dtype: str = "float64"
        source_columns: list[str] = []
        transformation: str = ""
        owner: str = ""
        tags: dict[str, str] = {}

    class EntryUpdate(BaseModel):
        description: str | None = None
        owner: str | None = None
        tags: dict[str, str] | None = None
        deprecated: bool | None = None
        deprecation_reason: str | None = None

    @app.get("/features")
    def list_features(
        query: str = Query("", description="Search query"),
        owner: str = Query("", description="Filter by owner"),
        include_deprecated: bool = Query(False),
    ) -> list[dict[str, Any]]:
        results = catalog_store.search(
            query=query,
            owner=owner if owner else None,
            include_deprecated=include_deprecated,
        )
        return [e.to_dict() for e in results]

    @app.get("/features/{name}")
    def get_feature(name: str) -> dict[str, Any]:
        entry = catalog_store.get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
        return entry.to_dict()

    @app.post("/features", status_code=201)
    def create_feature(body: EntryCreate) -> dict[str, Any]:
        entry = CatalogEntry(
            name=body.name,
            description=body.description,
            dtype=body.dtype,
            source_columns=body.source_columns,
            transformation=body.transformation,
            owner=body.owner,
            tags=body.tags,
        )
        result = catalog_store.add(entry)
        return result.to_dict()

    @app.put("/features/{name}")
    def update_feature(name: str, body: EntryUpdate) -> dict[str, Any]:
        entry = catalog_store.get(name)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
        if body.description is not None:
            entry.description = body.description
        if body.owner is not None:
            entry.owner = body.owner
        if body.tags is not None:
            entry.tags.update(body.tags)
        if body.deprecated is not None:
            entry.deprecated = body.deprecated
        if body.deprecation_reason is not None:
            entry.deprecation_reason = body.deprecation_reason
        catalog_store.add(entry)
        return entry.to_dict()

    @app.delete("/features/{name}")
    def delete_feature(name: str) -> dict[str, str]:
        if not catalog_store.delete(name):
            raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
        return {"status": "deleted", "name": name}

    @app.get("/statistics")
    def get_statistics() -> dict[str, Any]:
        return catalog_store.get_statistics()

    @app.post("/features/{name}/deprecate")
    def deprecate_feature(name: str, reason: str = "") -> dict[str, Any]:
        if not catalog_store.deprecate(name, reason):
            raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
        entry = catalog_store.get(name)
        return entry.to_dict() if entry else {}

    @app.post("/features/{name}/tags")
    def tag_feature(name: str, tags: dict[str, str]) -> dict[str, Any]:
        if not catalog_store.tag(name, tags):
            raise HTTPException(status_code=404, detail=f"Feature '{name}' not found")
        entry = catalog_store.get(name)
        return entry.to_dict() if entry else {}

    return app
