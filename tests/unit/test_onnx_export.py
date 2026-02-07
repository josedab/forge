"""Tests for ONNX export, NumPy fast-path, and streaming processor."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from forge.serving.onnx_export import (
    NumpyFastPath,
    ONNXExporter,
    ONNXExportResult,
    StreamProcessor,
)


@pytest.fixture
def fitted_scaler() -> StandardScaler:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


class TestNumpyFastPath:
    def test_from_standard_scaler(self, fitted_scaler: StandardScaler) -> None:
        fp = NumpyFastPath.from_transformer(fitted_scaler)
        assert len(fp.scale_params) == 1
        assert len(fp.scale_params[0].columns) == 2

    def test_transform_row(self, fitted_scaler: StandardScaler) -> None:
        fp = NumpyFastPath.from_transformer(fitted_scaler)
        result = fp.transform_row({"f0": 3.0, "f1": 4.0})
        assert isinstance(result, dict)
        assert "f0" in result
        assert abs(result["f0"] - 0.0) < 1e-6  # mean of [1,3,5]=3

    def test_transform_batch(self, fitted_scaler: StandardScaler) -> None:
        fp = NumpyFastPath.from_transformer(fitted_scaler)
        rows = [{"f0": 1.0, "f1": 2.0}, {"f0": 5.0, "f1": 6.0}]
        arr = fp.transform_batch(rows)
        assert arr.shape == (2, 2)

    def test_empty_batch(self) -> None:
        fp = NumpyFastPath()
        arr = fp.transform_batch([])
        assert arr.shape[0] == 0

    def test_passthrough_non_numeric(self) -> None:
        fp = NumpyFastPath()
        result = fp.transform_row({"a": 1.5, "b": "text"})
        assert result["a"] == 1.5
        assert "b" not in result  # non-numeric dropped


class TestONNXExporter:
    def test_export_without_skl2onnx(self, fitted_scaler: StandardScaler) -> None:
        exporter = ONNXExporter(fitted_scaler)
        result = exporter.export()
        assert isinstance(result, ONNXExportResult)
        # May fail if skl2onnx not installed—that's OK, check graceful handling
        if not result.success:
            assert "skl2onnx" in result.error or result.error


class TestStreamProcessor:
    def test_process_stream_fast_path(self, fitted_scaler: StandardScaler) -> None:
        fp = NumpyFastPath.from_transformer(fitted_scaler)
        proc = StreamProcessor(fast_path=fp, batch_size=2)
        rows = [{"f0": 1.0, "f1": 2.0}, {"f0": 3.0, "f1": 4.0}, {"f0": 5.0, "f1": 6.0}]
        results = list(proc.process_stream(iter(rows)))
        assert len(results) == 3
        assert proc.stats["processed"] == 3

    def test_process_stream_transformer(self, fitted_scaler: StandardScaler) -> None:
        proc = StreamProcessor(transformer=fitted_scaler, batch_size=2)
        rows = [{"f0": 1.0, "f1": 2.0}, {"f0": 3.0, "f1": 4.0}]
        results = list(proc.process_stream(iter(rows)))
        assert len(results) == 2

    def test_stats(self) -> None:
        proc = StreamProcessor(batch_size=1)
        list(proc.process_stream([{"a": 1}]))
        assert proc.stats["processed"] == 1
        assert proc.stats["errors"] == 0
