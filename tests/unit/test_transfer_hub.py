"""Tests for the transfer hub."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from forge.transfer.engine import ColumnFingerprint, DatasetFingerprint
from forge.transfer.hub import (
    DiscoveryMatch,
    TransferHub,
)


@pytest.fixture
def source_fingerprint():
    fp = DatasetFingerprint(name="source", n_rows=1000, n_columns=3)
    fp.columns = {
        "age": ColumnFingerprint(name="age", dtype="numeric", stats={"mean": 40, "std": 15}),
        "income": ColumnFingerprint(name="income", dtype="numeric", stats={"mean": 50000, "std": 15000}),
    }
    return fp


@pytest.fixture
def target_fingerprint():
    fp = DatasetFingerprint(name="target", n_rows=500, n_columns=3)
    fp.columns = {
        "age": ColumnFingerprint(name="age", dtype="numeric", stats={"mean": 38, "std": 14}),
        "salary": ColumnFingerprint(name="salary", dtype="numeric", stats={"mean": 48000, "std": 14000}),
    }
    return fp


@pytest.fixture
def fitted_scaler():
    X = pd.DataFrame({"age": [20, 30, 40, 50], "income": [30000, 50000, 70000, 90000]})
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


class TestTransferHub:
    def test_publish(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        pid = hub.publish("test_pipe", fitted_scaler, source_fingerprint)
        assert pid is not None
        assert hub.size == 1

    def test_list_pipelines(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        hub.publish("pipe1", fitted_scaler, source_fingerprint)
        pipelines = hub.list_pipelines()
        assert len(pipelines) == 1
        assert pipelines[0].name == "pipe1"

    def test_get(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        pid = hub.publish("pipe1", fitted_scaler, source_fingerprint)
        record = hub.get(pid)
        assert record is not None
        assert record.name == "pipe1"

    def test_get_missing(self):
        hub = TransferHub()
        assert hub.get("nonexistent") is None

    def test_discover(self, source_fingerprint, target_fingerprint, fitted_scaler):
        hub = TransferHub()
        hub.publish("test_pipe", fitted_scaler, source_fingerprint,
                     transform_names=["scale"])
        matches = hub.discover(target_fingerprint, min_similarity=0.0)
        assert len(matches) >= 1
        assert isinstance(matches[0], DiscoveryMatch)
        assert matches[0].similarity_score >= 0

    def test_discover_sorted_by_similarity(self, source_fingerprint, target_fingerprint, fitted_scaler):
        hub = TransferHub()
        hub.publish("pipe1", fitted_scaler, source_fingerprint)

        fp2 = DatasetFingerprint(name="other", n_rows=100, n_columns=2)
        fp2.columns = {
            "x": ColumnFingerprint(name="x", dtype="categorical", stats={}),
        }
        hub.publish("pipe2", fitted_scaler, fp2)

        matches = hub.discover(target_fingerprint, min_similarity=0.0)
        if len(matches) >= 2:
            assert matches[0].similarity_score >= matches[1].similarity_score

    def test_adapt_success(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        pid = hub.publish("test_pipe", fitted_scaler, source_fingerprint)

        target_df = pd.DataFrame({
            "age": [25, 35, 45],
            "income": [40000, 60000, 80000],
        })
        result = hub.adapt(pid, target_df)
        assert result.success
        assert result.output_features > 0

    def test_adapt_missing_pipeline(self):
        hub = TransferHub()
        result = hub.adapt("nonexistent", pd.DataFrame())
        assert not result.success
        assert "not found" in result.error

    def test_adapt_with_mapping(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        pid = hub.publish("test_pipe", fitted_scaler, source_fingerprint)

        target_df = pd.DataFrame({
            "user_age": [25, 35, 45],
            "user_income": [40000, 60000, 80000],
        })
        result = hub.adapt(pid, target_df, column_mapping={
            "age": "user_age",
            "income": "user_income",
        })
        assert result.success

    def test_persistence(self, source_fingerprint, fitted_scaler, tmp_path):
        hub1 = TransferHub(storage_path=tmp_path / "hub")
        hub1.publish("test", fitted_scaler, source_fingerprint)
        assert hub1.size == 1

        hub2 = TransferHub(storage_path=tmp_path / "hub")
        assert hub2.size == 1
        records = hub2.list_pipelines()
        assert records[0].name == "test"

    def test_publish_with_metadata(self, source_fingerprint, fitted_scaler):
        hub = TransferHub()
        pid = hub.publish(
            "test", fitted_scaler, source_fingerprint,
            transform_names=["scale", "interact"],
            score=0.85,
            metadata={"dataset": "kaggle_housing"},
        )
        record = hub.get(pid)
        assert record.score == 0.85
        assert record.metadata["dataset"] == "kaggle_housing"
        assert "scale" in record.transform_names
