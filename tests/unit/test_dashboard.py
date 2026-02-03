"""Tests for the feature lineage dashboard."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from forge.dashboard.export import HTMLReportExporter
from forge.dashboard.lineage import FeatureLineageGraph, LineageEdge, LineageNode


class TestLineageNode:
    def test_creation(self):
        node = LineageNode(name="age", node_type="source")
        assert node.name == "age"
        assert node.node_type == "source"

    def test_to_dict(self):
        node = LineageNode(name="log_age", node_type="generated", generator="LogTransformer")
        d = node.to_dict()
        assert d["name"] == "log_age"
        assert d["generator"] == "LogTransformer"


class TestLineageEdge:
    def test_creation(self):
        edge = LineageEdge(source="age", target="log_age", transformation="log")
        assert edge.source == "age"
        assert edge.target == "log_age"

    def test_to_dict(self):
        edge = LineageEdge(source="a", target="b")
        d = edge.to_dict()
        assert d["source"] == "a"
        assert d["target"] == "b"


class TestFeatureLineageGraph:
    @pytest.fixture
    def graph(self):
        g = FeatureLineageGraph()
        g.add_source("age")
        g.add_source("income")
        g.add_generated(
            "log_age", sources=["age"], generator="LogTransformer",
            transformation="log", importance=0.85
        )
        g.add_generated(
            "age_income_ratio", sources=["age", "income"],
            generator="InteractionGenerator", transformation="divide"
        )
        g.add_selected("log_age", importance=0.85)
        g.add_dropped("income", reason="low importance")
        return g

    def test_node_count(self, graph):
        assert len(graph.nodes) == 4

    def test_edge_count(self, graph):
        assert len(graph.edges) == 3

    def test_get_sources(self, graph):
        sources = graph.get_sources("age_income_ratio")
        assert set(sources) == {"age", "income"}

    def test_get_sources_for_source_column(self, graph):
        sources = graph.get_sources("age")
        assert sources == ["age"]

    def test_get_derived(self, graph):
        derived = graph.get_derived("age")
        assert "log_age" in derived
        assert "age_income_ratio" in derived

    def test_importance_ranking(self, graph):
        ranking = graph.get_importance_ranking()
        assert len(ranking) >= 1
        assert ranking[0][0] == "log_age"
        assert ranking[0][1] == 0.85

    def test_summary(self, graph):
        summary = graph.summary()
        assert "Source columns" in summary
        assert "Generated features" in summary

    def test_to_dict(self, graph):
        d = graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 4

    def test_to_dataframe(self, graph):
        df = graph.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "name" in df.columns
        assert "type" in df.columns
        assert len(df) == 4

    def test_get_node(self, graph):
        node = graph.get_node("age")
        assert node is not None
        assert node.node_type == "source"

    def test_get_nonexistent_node(self, graph):
        assert graph.get_node("nonexistent") is None

    def test_add_selected_new_node(self):
        g = FeatureLineageGraph()
        g.add_selected("new_feat", importance=0.5)
        assert g.get_node("new_feat") is not None

    def test_add_dropped_new_node(self):
        g = FeatureLineageGraph()
        g.add_dropped("bad_feat", reason="constant")
        node = g.get_node("bad_feat")
        assert node is not None
        assert node.node_type == "dropped"

    def test_from_transformer(self):
        class MockTransformer:
            _input_columns = ["x", "y"]

            def get_feature_names_out(self):
                return ["x", "y", "x_squared"]

        t = MockTransformer()
        graph = FeatureLineageGraph.from_transformer(t)
        assert len(graph.nodes) >= 3


class TestHTMLReportExporter:
    def test_export_creates_file(self):
        graph = FeatureLineageGraph()
        graph.add_source("age")
        graph.add_generated("log_age", sources=["age"], generator="Log", importance=0.9)
        graph.add_selected("log_age", importance=0.9)

        exporter = HTMLReportExporter(title="Test Report")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            result = exporter.export(graph, path)
            assert result.exists()
            content = result.read_text()
            assert "Test Report" in content
            assert "log_age" in content
            assert "0.9000" in content

    def test_export_without_importance(self):
        graph = FeatureLineageGraph()
        graph.add_source("x")
        exporter = HTMLReportExporter(include_importance=False, include_table=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "report.html"
            result = exporter.export(graph, path)
            assert result.exists()
