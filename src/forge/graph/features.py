"""Graph feature generators: centrality, PageRank, community, embeddings."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError
from forge.graph.builder import EntityGraph, GraphBuilder

if TYPE_CHECKING:
    from typing_extensions import Self


class GraphFeatureGenerator(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Generate features from entity relationship graphs.

    Computes graph-based features for entities in a DataFrame by
    building an entity graph and extracting structural features.

    Args:
        source_col: Column for source entity IDs.
        target_col: Column for target entity IDs.
        entity_col: Column in transform data to match against graph nodes.
            Defaults to source_col.
        weight_col: Column for edge weights.
        features: List of features to compute. None for all.
        n_components: Number of components for embedding features.
        pagerank_alpha: PageRank damping factor.
        pagerank_iterations: Maximum PageRank iterations.
        directed: Whether the graph is directed.

    Available features:
        - degree: Number of connections.
        - in_degree: Incoming connections (directed).
        - weighted_degree: Sum of edge weights.
        - pagerank: PageRank centrality score.
        - clustering_coefficient: Local clustering coefficient.
        - n_triangles: Number of triangles the node participates in.
        - community: Community label from label propagation.
        - ego_edges: Number of edges in the node's ego graph.
        - node2vec: Random walk embedding features.

    Example:
        >>> gen = GraphFeatureGenerator(
        ...     source_col="user_id", target_col="product_id",
        ...     features=["degree", "pagerank", "community"]
        ... )
        >>> features = gen.fit_transform(transactions_df)
    """

    def __init__(
        self,
        source_col: str = "source",
        target_col: str = "target",
        entity_col: str | None = None,
        weight_col: str | None = None,
        features: list[str] | None = None,
        n_components: int = 8,
        pagerank_alpha: float = 0.85,
        pagerank_iterations: int = 100,
        directed: bool = False,
    ) -> None:
        self.source_col = source_col
        self.target_col = target_col
        self.entity_col = entity_col
        self.weight_col = weight_col
        self.features = features
        self.n_components = n_components
        self.pagerank_alpha = pagerank_alpha
        self.pagerank_iterations = pagerank_iterations
        self.directed = directed

        self._is_fitted = False
        self._graph: EntityGraph | None = None
        self._feature_names_out: list[str] = []
        self._node_features: dict[str, dict[str, float]] = {}

    def available_features(self) -> list[str]:
        """Return list of all available graph features."""
        return [
            "degree", "in_degree", "weighted_degree", "pagerank",
            "clustering_coefficient", "n_triangles", "community",
            "ego_edges", "node2vec",
        ]

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> Self:
        """Build the graph and compute features for all nodes.

        Args:
            X: DataFrame with edge data.
            y: Ignored.

        Returns:
            Self for method chaining.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        builder = GraphBuilder(
            source_col=self.source_col,
            target_col=self.target_col,
            weight_col=self.weight_col,
            directed=self.directed,
        )
        self._graph = builder.build(X)

        active = set(self.features) if self.features else set(self.available_features())
        self._node_features = {}

        for node in self._graph.nodes:
            feats: dict[str, float] = {}

            if "degree" in active:
                feats["graph_degree"] = float(self._graph.degree(node))

            if "in_degree" in active:
                feats["graph_in_degree"] = float(self._graph.in_degree(node))

            if "weighted_degree" in active:
                feats["graph_weighted_degree"] = sum(
                    w for _, w in self._graph.weighted_neighbors(node)
                )

            if "ego_edges" in active:
                feats["graph_ego_edges"] = float(self._count_ego_edges(node))

            self._node_features[node] = feats

        # Compute global features that need full-graph context
        if "pagerank" in active:
            pr = self._compute_pagerank()
            for node, score in pr.items():
                self._node_features.setdefault(node, {})["graph_pagerank"] = score

        if "clustering_coefficient" in active:
            cc = self._compute_clustering_coefficients()
            for node, coeff in cc.items():
                self._node_features.setdefault(node, {})["graph_clustering"] = coeff

        if "n_triangles" in active:
            tri = self._count_triangles()
            for node, count in tri.items():
                self._node_features.setdefault(node, {})["graph_triangles"] = float(count)

        if "community" in active:
            communities = self._label_propagation()
            for node, label in communities.items():
                self._node_features.setdefault(node, {})["graph_community"] = float(label)

        if "node2vec" in active:
            embeddings = self._node2vec_embeddings()
            for node, emb in embeddings.items():
                for i, val in enumerate(emb):
                    self._node_features.setdefault(node, {})[f"graph_emb_{i}"] = val

        # Determine output feature names
        if self._node_features:
            sample_node = next(iter(self._node_features))
            self._feature_names_out = sorted(self._node_features[sample_node].keys())
        else:
            self._feature_names_out = []

        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map graph features onto entities in the DataFrame.

        Args:
            X: DataFrame with an entity column matching graph nodes.

        Returns:
            DataFrame with graph features.
        """
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

        entity = self.entity_col or self.source_col
        if entity not in X.columns:
            raise ValidationError(f"Entity column '{entity}' not in DataFrame")

        records: list[dict[str, float]] = []
        for _, row in X.iterrows():
            node = str(row[entity])
            if node in self._node_features:
                records.append(self._node_features[node])
            else:
                records.append(dict.fromkeys(self._feature_names_out, 0.0))

        return pd.DataFrame(records, index=X.index, columns=self._feature_names_out)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)
        return self._feature_names_out.copy()

    # ── Graph algorithms ─────────────────────────────────────

    def _compute_pagerank(self) -> dict[str, float]:
        """Power-iteration PageRank."""
        graph = self._graph
        if graph is None or graph.n_nodes == 0:
            return {}

        nodes = graph.nodes
        n = len(nodes)
        pr: dict[str, float] = dict.fromkeys(nodes, 1.0 / n)

        for _ in range(self.pagerank_iterations):
            new_pr: dict[str, float] = {}
            for node in nodes:
                rank = (1 - self.pagerank_alpha) / n
                for src in nodes:
                    for nbr, _ in graph.adjacency.get(src, []):
                        if nbr == node and graph.degree(src) > 0:
                            rank += self.pagerank_alpha * pr[src] / graph.degree(src)
                new_pr[node] = rank
            pr = new_pr

        return pr

    def _compute_clustering_coefficients(self) -> dict[str, float]:
        """Local clustering coefficient for each node."""
        graph = self._graph
        if graph is None:
            return {}

        result: dict[str, float] = {}
        for node in graph.nodes:
            nbrs = set(graph.neighbors(node))
            k = len(nbrs)
            if k < 2:
                result[node] = 0.0
                continue
            # Count edges between neighbors
            links = 0
            for nbr in nbrs:
                for nbr2, _ in graph.adjacency.get(nbr, []):
                    if nbr2 in nbrs and nbr2 != nbr:
                        links += 1
            max_links = k * (k - 1)
            result[node] = links / max_links if max_links > 0 else 0.0
        return result

    def _count_triangles(self) -> dict[str, int]:
        """Count triangles each node participates in."""
        graph = self._graph
        if graph is None:
            return {}

        result: dict[str, int] = defaultdict(int)
        for node in graph.nodes:
            nbrs = set(graph.neighbors(node))
            for nbr in nbrs:
                for nbr2, _ in graph.adjacency.get(nbr, []):
                    if nbr2 in nbrs and nbr2 != node:
                        result[node] += 1
        # Each triangle counted twice per node
        return {k: v // 2 for k, v in result.items()}

    def _label_propagation(self, max_iter: int = 30) -> dict[str, int]:
        """Simple label propagation community detection."""
        graph = self._graph
        if graph is None:
            return {}

        labels: dict[str, int] = {node: i for i, node in enumerate(graph.nodes)}
        nodes = list(graph.nodes)

        for _ in range(max_iter):
            changed = False
            np.random.shuffle(nodes)
            for node in nodes:
                nbrs = graph.neighbors(node)
                if not nbrs:
                    continue
                # Most common label among neighbors
                label_counts: dict[int, int] = defaultdict(int)
                for nbr in nbrs:
                    label_counts[labels[nbr]] += 1
                best_label = max(label_counts, key=lambda k: label_counts[k])
                if labels[node] != best_label:
                    labels[node] = best_label
                    changed = True
            if not changed:
                break

        return labels

    def _node2vec_embeddings(self) -> dict[str, list[float]]:
        """Simple random walk based embeddings (lightweight node2vec)."""
        graph = self._graph
        if graph is None or graph.n_nodes == 0:
            return {}

        n_walks = 10
        walk_length = 20
        nodes = graph.nodes
        rng = np.random.RandomState(42)

        # Perform random walks
        walks: list[list[str]] = []
        for node in nodes:
            for _ in range(n_walks):
                walk = [node]
                current = node
                for _ in range(walk_length - 1):
                    nbrs = graph.neighbors(current)
                    if not nbrs:
                        break
                    current = nbrs[rng.randint(len(nbrs))]
                    walk.append(current)
                walks.append(walk)

        # Build co-occurrence matrix (simplified skip-gram target)
        node_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        dim = min(self.n_components, n)
        cooccur = np.zeros((n, n))
        window = 5
        for walk in walks:
            for i, node in enumerate(walk):
                start = max(0, i - window)
                end = min(len(walk), i + window + 1)
                for j in range(start, end):
                    if i != j:
                        cooccur[node_idx[node], node_idx[walk[j]]] += 1

        # SVD for dimensionality reduction
        cooccur_log = np.log1p(cooccur)
        try:
            from scipy.sparse.linalg import svds
            if dim < min(cooccur_log.shape) - 1:
                U, S, _ = svds(cooccur_log, k=dim)
                embeddings_matrix = U * np.sqrt(S)
            else:
                U, S, _ = np.linalg.svd(cooccur_log, full_matrices=False)
                embeddings_matrix = U[:, :dim] * np.sqrt(S[:dim])
        except Exception:
            U, S, _ = np.linalg.svd(cooccur_log, full_matrices=False)
            embeddings_matrix = U[:, :dim] * np.sqrt(S[:dim])

        result: dict[str, list[float]] = {}
        for node in nodes:
            idx = node_idx[node]
            result[node] = embeddings_matrix[idx].tolist()

        return result

    def _count_ego_edges(self, node: str) -> int:
        """Count edges in the 1-hop ego graph of a node."""
        graph = self._graph
        if graph is None:
            return 0
        nbrs = set(graph.neighbors(node))
        nbrs.add(node)
        count = 0
        for n in nbrs:
            for nbr, _ in graph.adjacency.get(n, []):
                if nbr in nbrs:
                    count += 1
        return count
