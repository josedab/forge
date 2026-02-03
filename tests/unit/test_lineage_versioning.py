"""Tests for feature lineage versioning — events, snapshots, and diffs."""

from __future__ import annotations

import pytest

from forge.dashboard.lineage import FeatureLineageGraph
from forge.dashboard.versioning import (
    LineageDiff,
    LineageEvent,
    LineageEventRecorder,
    LineageSnapshot,
    LineageVersionStore,
)


class TestLineageEventRecorder:
    def test_record_create(self) -> None:
        recorder = LineageEventRecorder()
        recorder.record_create("log_price", ["price"], "log", "LogTransformer")
        assert len(recorder.events) == 1
        assert recorder.events[0].event_type == "create"
        assert recorder.events[0].feature_name == "log_price"

    def test_record_select(self) -> None:
        recorder = LineageEventRecorder()
        recorder.record_select("log_price", importance=0.85)
        assert len(recorder.events) == 1
        assert recorder.events[0].parameters["importance"] == 0.85

    def test_record_drop(self) -> None:
        recorder = LineageEventRecorder()
        recorder.record_drop("bad_feature", reason="low variance")
        assert recorder.events[0].event_type == "drop"

    def test_clear(self) -> None:
        recorder = LineageEventRecorder()
        recorder.record_create("f1", ["a"])
        recorder.record_create("f2", ["b"])
        recorder.clear()
        assert len(recorder.events) == 0

    def test_event_serialization(self) -> None:
        event = LineageEvent(
            feature_name="test",
            event_type="create",
            source_columns=["a", "b"],
            operation="multiply",
        )
        d = event.to_dict()
        assert d["feature_name"] == "test"
        assert d["source_columns"] == ["a", "b"]


class TestLineageSnapshot:
    def test_feature_names(self) -> None:
        snap = LineageSnapshot(
            version="v1",
            features={"b": {}, "a": {}, "c": {}},
        )
        assert snap.feature_names == ["a", "b", "c"]

    def test_checksum_deterministic(self) -> None:
        features = {"a": {"type": "source"}, "b": {"type": "generated"}}
        s1 = LineageSnapshot(version="v1", features=features)
        s2 = LineageSnapshot(version="v2", features=features)
        assert s1.checksum() == s2.checksum()

    def test_checksum_changes_with_features(self) -> None:
        s1 = LineageSnapshot(version="v1", features={"a": {"type": "source"}})
        s2 = LineageSnapshot(version="v2", features={"b": {"type": "source"}})
        assert s1.checksum() != s2.checksum()

    def test_to_dict(self) -> None:
        snap = LineageSnapshot(version="v1", features={"a": {}})
        d = snap.to_dict()
        assert d["version"] == "v1"
        assert "checksum" in d


class TestLineageVersionStore:
    def test_snapshot_and_retrieve(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {"type": "source"}})
        assert store.get("v1") is not None
        assert store.get("v1").version == "v1"

    def test_latest(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {}})
        store.snapshot("v2", {"a": {}, "b": {}})
        assert store.latest.version == "v2"

    def test_latest_empty(self) -> None:
        store = LineageVersionStore()
        assert store.latest is None

    def test_versions_order(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {})
        store.snapshot("v2", {})
        store.snapshot("v3", {})
        assert store.versions == ["v1", "v2", "v3"]

    def test_parent_version_tracking(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {}})
        snap2 = store.snapshot("v2", {"a": {}, "b": {}})
        assert snap2.parent_version == "v1"

    def test_diff_added(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {}})
        store.snapshot("v2", {"a": {}, "b": {}, "c": {}})
        diff = store.diff("v1", "v2")
        assert sorted(diff.added) == ["b", "c"]
        assert diff.removed == []

    def test_diff_removed(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {}, "b": {}, "c": {}})
        store.snapshot("v2", {"a": {}})
        diff = store.diff("v1", "v2")
        assert sorted(diff.removed) == ["b", "c"]
        assert diff.added == []

    def test_diff_modified(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {"importance": 0.5}})
        store.snapshot("v2", {"a": {"importance": 0.9}})
        diff = store.diff("v1", "v2")
        assert diff.modified == ["a"]

    def test_diff_no_changes(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {"x": 1}})
        store.snapshot("v2", {"a": {"x": 1}})
        diff = store.diff("v1", "v2")
        assert not diff.has_changes

    def test_diff_missing_version_raises(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {})
        with pytest.raises(KeyError):
            store.diff("v1", "v999")

    def test_history(self) -> None:
        store = LineageVersionStore()
        store.snapshot("v1", {"a": {}}, description="initial")
        store.snapshot("v2", {"a": {}, "b": {}}, description="added b")
        hist = store.history()
        assert len(hist) == 2
        assert hist[0]["version"] == "v1"
        assert hist[1]["n_features"] == 2


class TestLineageDiff:
    def test_summary(self) -> None:
        diff = LineageDiff(
            from_version="v1",
            to_version="v2",
            added=["new_feat"],
            removed=["old_feat"],
            modified=["changed_feat"],
        )
        s = diff.summary()
        assert "v1" in s
        assert "Added" in s
        assert "Removed" in s
        assert "Modified" in s

    def test_has_changes_true(self) -> None:
        diff = LineageDiff("v1", "v2", added=["x"])
        assert diff.has_changes

    def test_has_changes_false(self) -> None:
        diff = LineageDiff("v1", "v2")
        assert not diff.has_changes


class TestFeatureLineageGraphIntegration:
    def test_graph_to_versioned_snapshot(self) -> None:
        graph = FeatureLineageGraph()
        graph.add_source("age")
        graph.add_source("income")
        graph.add_generated("log_age", sources=["age"], generator="LogTransform")
        graph.add_selected("log_age", importance=0.85)

        # Convert graph to snapshot
        features = {}
        for node in graph.nodes:
            features[node.name] = node.to_dict()

        store = LineageVersionStore()
        store.snapshot("v1", features, description="initial pipeline")
        assert store.get("v1") is not None
        assert len(store.get("v1").features) == 3
