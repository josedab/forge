"""Tests for the serving API module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.serving.api import APIConfig, APIMetrics, create_app


class MockPipeline:
    """Mock pipeline for testing the API."""

    def __init__(self, output_cols: list[str] | None = None) -> None:
        self.output_cols = output_cols or ["feat_1", "feat_2"]
        self.n_features_in_ = 3

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {col: np.random.randn(n) for col in self.output_cols}
        )

    def get_feature_names_out(self) -> list[str]:
        return self.output_cols


class TestAPIMetrics:
    """Tests for APIMetrics."""

    def test_record_increments_counts(self) -> None:
        m = APIMetrics()
        m.record(10.0, n_rows=5)
        assert m.total_requests == 1
        assert m.total_rows_processed == 5
        assert m.total_errors == 0

    def test_record_error(self) -> None:
        m = APIMetrics()
        m.record(10.0, error=True)
        assert m.total_errors == 1

    def test_avg_latency(self) -> None:
        m = APIMetrics()
        m.record(10.0)
        m.record(20.0)
        assert m.avg_latency_ms == 15.0

    def test_to_dict(self) -> None:
        m = APIMetrics()
        m.record(10.0)
        d = m.to_dict()
        assert "total_requests" in d
        assert "p50_latency_ms" in d
        assert "p99_latency_ms" in d


class TestCreateApp:
    """Tests for FastAPI app creation."""

    def test_create_app_returns_fastapi(self) -> None:
        pipeline = MockPipeline()
        app = create_app(pipeline)
        # FastAPI app has routes attribute
        assert hasattr(app, "routes")

    def test_app_has_endpoints(self) -> None:
        pipeline = MockPipeline()
        app = create_app(pipeline)
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/health" in paths
        assert "/transform" in paths
        assert "/batch" in paths
        assert "/metrics" in paths
        assert "/info" in paths

    def test_config_defaults(self) -> None:
        cfg = APIConfig()
        assert cfg.enable_batch
        assert cfg.enable_health
        assert cfg.max_batch_size == 10000


class TestAPIEndpoints:
    """Integration tests using FastAPI TestClient."""

    @pytest.fixture
    def client(self) -> Any:
        """Create a test client."""
        from fastapi.testclient import TestClient
        pipeline = MockPipeline()
        app = create_app(pipeline)
        return TestClient(app)

    def test_health_endpoint(self, client: Any) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    def test_transform_single_row(self, client: Any) -> None:
        resp = client.post("/transform", json={"data": {"a": 1, "b": 2, "c": 3}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_rows"] == 1
        assert len(data["features"]) == 1

    def test_transform_multiple_rows(self, client: Any) -> None:
        rows = [{"a": i, "b": i * 2, "c": i * 3} for i in range(5)]
        resp = client.post("/transform", json={"data": rows})
        assert resp.status_code == 200
        assert resp.json()["n_rows"] == 5

    def test_transform_missing_data(self, client: Any) -> None:
        resp = client.post("/transform", json={})
        assert resp.status_code == 400

    def test_batch_endpoint(self, client: Any) -> None:
        rows = [{"a": i, "b": i * 2} for i in range(10)]
        resp = client.post("/batch", json={"data": rows})
        assert resp.status_code == 200
        assert resp.json()["n_rows"] == 10

    def test_batch_exceeds_max(self, client: Any) -> None:
        from fastapi.testclient import TestClient
        pipeline = MockPipeline()
        app = create_app(pipeline, config=APIConfig(max_batch_size=5))
        c = TestClient(app)
        rows = [{"a": i} for i in range(10)]
        resp = c.post("/batch", json={"data": rows})
        assert resp.status_code == 400

    def test_metrics_endpoint(self, client: Any) -> None:
        # Make a request first
        client.post("/transform", json={"data": {"a": 1}})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] >= 1

    def test_info_endpoint(self, client: Any) -> None:
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pipeline_type"] == "MockPipeline"
        assert "output_features" in data


# Need to import Any for type hints in fixtures
from typing import Any
