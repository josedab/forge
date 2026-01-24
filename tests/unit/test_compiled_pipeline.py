"""Tests for the compiled pipeline and serving metrics."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forge.serving.compiled import (
    CompiledPipeline,
    ServingMetrics,
)


class TestServingMetrics:
    def test_record_and_stats(self):
        m = ServingMetrics()
        m.record(1.5)
        m.record(2.0)
        m.record(0.5)

        assert m.total_requests == 3
        assert m.avg_latency_ms == pytest.approx(4.0 / 3)
        assert m.error_rate == 0.0

    def test_error_recording(self):
        m = ServingMetrics()
        m.record(1.0)
        m.record(2.0, error=True)

        assert m.total_errors == 1
        assert m.error_rate == pytest.approx(0.5)

    def test_percentiles(self):
        m = ServingMetrics()
        for i in range(100):
            m.record(float(i))

        assert m.p50_latency_ms > 0
        assert m.p99_latency_ms > m.p50_latency_ms

    def test_to_dict(self):
        m = ServingMetrics()
        m.record(1.0)
        d = m.to_dict()

        assert "total_requests" in d
        assert "p50_latency_ms" in d
        assert "error_rate" in d

    def test_reset(self):
        m = ServingMetrics()
        m.record(1.0)
        m.reset()

        assert m.total_requests == 0
        assert m.avg_latency_ms == 0.0

    def test_empty_metrics(self):
        m = ServingMetrics()
        assert m.avg_latency_ms == 0.0
        assert m.p50_latency_ms == 0.0
        assert m.error_rate == 0.0


class TestCompiledPipeline:
    def test_compile_standard_scaler(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline.from_transformer(scaler)
        assert compiled.is_compiled
        assert len(compiled.graph.steps) >= 1

    def test_transform_single_with_scaler(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline.from_transformer(scaler)
        result = compiled.transform_single({"a": 2.5, "b": 25.0})

        assert "a" in result
        assert "b" in result
        # Standardized values should be near 0 for mean values
        assert abs(result["a"]) < 2.0
        assert abs(result["b"]) < 2.0

    def test_transform_batch(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        scaler = StandardScaler()
        scaler.fit(X_train)

        compiled = CompiledPipeline.from_transformer(scaler)

        X_test = pd.DataFrame({"a": [1.5, 2.5], "b": [4.5, 5.5]})
        result = compiled.transform_batch(X_test)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_compile_onehot_encoder(self):
        X = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(X)

        compiled = CompiledPipeline.from_transformer(enc)
        assert compiled.is_compiled
        assert len(compiled._onehot_maps) >= 1

    def test_transform_single_onehot(self):
        X = pd.DataFrame({"color": ["red", "blue", "green", "red"]})
        enc = OneHotEncoder(sparse_output=False)
        enc.fit(X)

        compiled = CompiledPipeline.from_transformer(enc)
        result = compiled.transform_single({"color": "red"})

        assert any("red" in k for k in result.keys())
        red_key = [k for k in result.keys() if "red" in k][0]
        assert result[red_key] == 1.0

    def test_metrics_tracking(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline.from_transformer(scaler)
        compiled.transform_single({"a": 2.0})
        compiled.transform_single({"a": 3.0})

        assert compiled.metrics.total_requests == 2
        assert compiled.metrics.avg_latency_ms > 0

    def test_health_check(self):
        compiled = CompiledPipeline()
        health = compiled.health_check()

        assert health["status"] == "not_compiled"
        assert not health["is_compiled"]

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline.from_transformer(scaler)
        health = compiled.health_check()

        assert health["status"] == "healthy"
        assert health["is_compiled"]

    def test_compile_sklearn_pipeline(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pipe = Pipeline([("scaler", StandardScaler())])
        pipe.fit(X)

        compiled = CompiledPipeline.from_transformer(pipe)
        assert compiled.is_compiled
        assert len(compiled.graph.steps) >= 1

    def test_transform_single_fallback(self):
        """Test fallback to full transform when no compiled params."""
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline(scaler)
        # Don't compile - force fallback
        compiled._is_compiled = True
        compiled._scale_params = {}
        compiled._feature_names = ["a"]

        result = compiled.transform_single({"a": 2.0})
        assert len(result) > 0

    def test_no_transformer(self):
        compiled = CompiledPipeline()
        compiled.compile()
        assert compiled.is_compiled

        result = compiled.transform_single({"a": 1.0})
        assert isinstance(result, dict)

    def test_graph_metadata(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        scaler = StandardScaler()
        scaler.fit(X)

        compiled = CompiledPipeline.from_transformer(scaler)
        graph = compiled.graph

        assert "compilation_time_ms" in graph.metadata
        assert graph.metadata["step_count"] >= 1
        assert graph.metadata["transformer_type"] == "StandardScaler"
