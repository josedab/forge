"""FastAPI-based REST API for real-time feature serving.

Provides HTTP endpoints for feature computation, health checks,
and batch serving. Requires FastAPI and uvicorn.

Usage:
    >>> from forge.serving.api import create_app
    >>> app = create_app(pipeline)
    >>> # Run with: uvicorn forge.serving.api:app --port 8000
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forge.exceptions import MissingDependencyError

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """Configuration for the serving API.

    Attributes:
        title: API title.
        version: API version string.
        enable_batch: Enable batch endpoint.
        enable_health: Enable health check endpoint.
        max_batch_size: Maximum rows in a batch request.
        request_timeout: Timeout per request in seconds.
    """

    title: str = "Forge Feature Server"
    version: str = "1.0.0"
    enable_batch: bool = True
    enable_health: bool = True
    max_batch_size: int = 10000
    request_timeout: float = 30.0


@dataclass
class APIMetrics:
    """Runtime metrics for the serving API."""

    total_requests: int = 0
    total_errors: int = 0
    total_rows_processed: int = 0
    avg_latency_ms: float = 0.0
    _latencies: list[float] = field(default_factory=list, repr=False)

    def record(self, latency_ms: float, n_rows: int = 1, error: bool = False) -> None:
        """Record a request metric."""
        self.total_requests += 1
        self.total_rows_processed += n_rows
        if error:
            self.total_errors += 1
        self._latencies.append(latency_ms)
        # Keep only last 1000 for rolling average
        if len(self._latencies) > 1000:
            self._latencies = self._latencies[-1000:]
        self.avg_latency_ms = sum(self._latencies) / len(self._latencies)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_rows_processed": self.total_rows_processed,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": round(float(np.percentile(self._latencies, 50)), 2) if self._latencies else 0.0,
            "p99_latency_ms": round(float(np.percentile(self._latencies, 99)), 2) if self._latencies else 0.0,
        }


def create_app(
    pipeline: Any,
    config: APIConfig | None = None,
    feature_names: list[str] | None = None,
) -> Any:
    """Create a FastAPI application for serving features.

    Args:
        pipeline: A fitted sklearn-compatible transformer with transform().
        config: API configuration.
        feature_names: Expected input feature names (for validation).

    Returns:
        A FastAPI application instance.

    Raises:
        MissingDependencyError: If fastapi is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError:
        raise MissingDependencyError("fastapi", "feature serving API")

    cfg = config or APIConfig()
    metrics = APIMetrics()

    app = FastAPI(title=cfg.title, version=cfg.version)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "pipeline_type": type(pipeline).__name__,
            "metrics": metrics.to_dict(),
        }

    @app.post("/transform")
    async def transform(request: dict[str, Any]) -> dict[str, Any]:
        """Transform a single row or small batch.

        Request body should have a "data" key with column->value mapping
        or a list of such mappings.
        """
        start = time.time()
        try:
            data = request.get("data")
            if data is None:
                raise HTTPException(status_code=400, detail="Missing 'data' field")

            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                raise HTTPException(status_code=400, detail="'data' must be dict or list")

            if feature_names:
                missing = set(feature_names) - set(df.columns)
                if missing:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing columns: {sorted(missing)}",
                    )

            result = pipeline.transform(df)
            if isinstance(result, pd.DataFrame):
                output = result.to_dict(orient="records")
            else:
                output = pd.DataFrame(result).to_dict(orient="records")

            latency = (time.time() - start) * 1000
            metrics.record(latency, n_rows=len(df))

            return {
                "features": output,
                "n_rows": len(df),
                "latency_ms": round(latency, 2),
            }
        except HTTPException:
            raise
        except Exception as e:
            latency = (time.time() - start) * 1000
            metrics.record(latency, error=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/batch")
    async def batch_transform(request: dict[str, Any]) -> dict[str, Any]:
        """Batch transform endpoint for larger datasets."""
        if not cfg.enable_batch:
            raise HTTPException(status_code=404, detail="Batch endpoint disabled")

        start = time.time()
        try:
            data = request.get("data")
            if not isinstance(data, list):
                raise HTTPException(
                    status_code=400, detail="'data' must be a list of records"
                )

            if len(data) > cfg.max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size {len(data)} exceeds max {cfg.max_batch_size}",
                )

            df = pd.DataFrame(data)
            result = pipeline.transform(df)
            if isinstance(result, pd.DataFrame):
                output = result.to_dict(orient="records")
            else:
                output = pd.DataFrame(result).to_dict(orient="records")

            latency = (time.time() - start) * 1000
            metrics.record(latency, n_rows=len(df))

            return {
                "features": output,
                "n_rows": len(df),
                "latency_ms": round(latency, 2),
            }
        except HTTPException:
            raise
        except Exception as e:
            latency = (time.time() - start) * 1000
            metrics.record(latency, error=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def get_metrics() -> dict[str, Any]:
        """Return serving metrics."""
        return metrics.to_dict()

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Pipeline information endpoint."""
        result: dict[str, Any] = {
            "pipeline_type": type(pipeline).__name__,
        }
        if hasattr(pipeline, "get_feature_names_out"):
            try:
                result["output_features"] = pipeline.get_feature_names_out()
            except Exception:
                pass
        if hasattr(pipeline, "n_features_in_"):
            result["n_features_in"] = pipeline.n_features_in_
        return result

    # Store references for testing
    app.state.metrics = metrics  # type: ignore[attr-defined]
    app.state.config = cfg  # type: ignore[attr-defined]

    return app
