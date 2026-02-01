"""Tests for cross-table graph features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forge.graph import EntityGraph, GraphBuilder, GraphFeatureGenerator


class TestEntityGraph:
    def test_add_nodes_and_edges(self):
        g = EntityGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B", 1.0)
        assert g.n_nodes == 2
        assert g.n_edges == 1

    def test_undirected_edge(self):
        g = EntityGraph()
        g.add_undirected_edge("A", "B", 2.0)
        assert g.n_edges == 2
        assert "B" in g.neighbors("A")
        assert "A" in g.neighbors("B")

    def test_degree(self):
        g = EntityGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        assert g.degree("A") == 2
        assert g.degree("B") == 0

    def test_in_degree(self):
        g = EntityGraph()
        g.add_edge("A", "B")
        g.add_edge("C", "B")
        assert g.in_degree("B") == 2

    def test_adjacency_matrix(self):
        g = EntityGraph()
        g.add_edge("A", "B", 1.0)
        g.add_edge("B", "C", 2.0)
        matrix, nodes = g.to_adjacency_matrix()
        assert matrix.shape[0] == 3
        # Non-zero entries should match edges
        assert np.sum(matrix > 0) == 2

    def test_weighted_neighbors(self):
        g = EntityGraph()
        g.add_edge("A", "B", 3.0)
        nbrs = g.weighted_neighbors("A")
        assert len(nbrs) == 1
        assert nbrs[0] == ("B", 3.0)


class TestGraphBuilder:
    def test_build_from_dataframe(self):
        df = pd.DataFrame({
            "user": ["U1", "U1", "U2", "U2"],
            "product": ["P1", "P2", "P1", "P3"],
        })
        builder = GraphBuilder(source_col="user", target_col="product")
        graph = builder.build(df)
        assert graph.n_nodes >= 4

    def test_build_with_weights(self):
        df = pd.DataFrame({
            "src": ["A", "B"],
            "tgt": ["B", "C"],
            "weight": [1.5, 2.5],
        })
        builder = GraphBuilder(source_col="src", target_col="tgt", weight_col="weight")
        graph = builder.build(df)
        nbrs = graph.weighted_neighbors("A")
        assert any(w == 1.5 for _, w in nbrs)

    def test_directed(self):
        df = pd.DataFrame({"src": ["A", "B"], "tgt": ["B", "C"]})
        builder = GraphBuilder(source_col="src", target_col="tgt", directed=True)
        graph = builder.build(df)
        assert "B" in graph.neighbors("A")
        assert "A" not in graph.neighbors("B")

    def test_from_edge_list(self):
        edges = [("A", "B", 1.0), ("B", "C", 1.0)]
        graph = GraphBuilder.from_edge_list(edges, directed=False)
        assert graph.n_nodes == 3


class TestGraphFeatureGenerator:
    @pytest.fixture
    def edge_df(self):
        return pd.DataFrame({
            "source": ["A", "A", "B", "B", "C", "D", "D", "E"],
            "target": ["B", "C", "C", "D", "D", "E", "A", "A"],
        })

    def test_fit_and_transform(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["degree", "pagerank"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(edge_df)
        assert any("degree" in c for c in result.columns)
        assert any("pagerank" in c for c in result.columns)

    def test_degree_feature(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["degree"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert "graph_degree" in result.columns
        # Node A has at least 2 connections
        assert result["graph_degree"].max() > 0

    def test_pagerank(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["pagerank"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert "graph_pagerank" in result.columns
        # PageRank values should be positive
        assert result["graph_pagerank"].min() >= 0

    def test_clustering_coefficient(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["clustering_coefficient"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert "graph_clustering" in result.columns
        # Clustering coefficients are between 0 and 1
        assert result["graph_clustering"].min() >= 0
        assert result["graph_clustering"].max() <= 1.0

    def test_community_detection(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["community"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert "graph_community" in result.columns

    def test_node2vec_embeddings(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["node2vec"],
            n_components=4,
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        emb_cols = [c for c in result.columns if "emb" in c]
        assert len(emb_cols) == 4

    def test_all_features(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            n_components=3,
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert len(result.columns) >= 8

    def test_unknown_entity(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["degree"],
        )
        gen.fit(edge_df)
        new_df = pd.DataFrame({"source": ["UNKNOWN"]})
        result = gen.transform(new_df)
        assert result["graph_degree"].iloc[0] == 0.0

    def test_custom_entity_col(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            entity_col="source", features=["degree"],
        )
        gen.fit(edge_df)
        result = gen.transform(edge_df)
        assert len(result) == len(edge_df)

    def test_get_feature_names_out(self, edge_df):
        gen = GraphFeatureGenerator(
            source_col="source", target_col="target",
            features=["degree", "pagerank"],
        )
        gen.fit(edge_df)
        names = gen.get_feature_names_out()
        assert isinstance(names, list)
        assert len(names) == 2
