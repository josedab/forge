"""Entity graph construction from relational data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from forge.exceptions import ValidationError


@dataclass
class EntityGraph:
    """Graph representation of entity relationships.

    Stores an adjacency list, node attributes, and edge weights.
    Designed for efficient feature extraction without requiring
    external graph libraries.

    Attributes:
        adjacency: Node -> list of (neighbor, weight) pairs.
        node_attrs: Node -> attribute dictionary.
        n_nodes: Number of nodes.
        n_edges: Number of edges (directed).
    """

    adjacency: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    node_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.adjacency)

    @property
    def n_edges(self) -> int:
        return sum(len(neighbors) for neighbors in self.adjacency.values())

    @property
    def nodes(self) -> list[str]:
        return list(self.adjacency.keys())

    def add_node(self, node: str, **attrs: Any) -> None:
        """Add a node to the graph."""
        if node not in self.adjacency:
            self.adjacency[node] = []
        self.node_attrs[node] = attrs

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add a directed edge."""
        if source not in self.adjacency:
            self.adjacency[source] = []
        if target not in self.adjacency:
            self.adjacency[target] = []
        self.adjacency[source].append((target, weight))

    def add_undirected_edge(
        self, node1: str, node2: str, weight: float = 1.0
    ) -> None:
        """Add an undirected edge (both directions)."""
        self.add_edge(node1, node2, weight)
        self.add_edge(node2, node1, weight)

    def degree(self, node: str) -> int:
        """Return the out-degree of a node."""
        return len(self.adjacency.get(node, []))

    def in_degree(self, node: str) -> int:
        """Return the in-degree of a node."""
        count = 0
        for neighbors in self.adjacency.values():
            for nbr, _ in neighbors:
                if nbr == node:
                    count += 1
        return count

    def neighbors(self, node: str) -> list[str]:
        """Return neighbors of a node."""
        return [nbr for nbr, _ in self.adjacency.get(node, [])]

    def weighted_neighbors(self, node: str) -> list[tuple[str, float]]:
        """Return neighbors with weights."""
        return list(self.adjacency.get(node, []))

    def to_adjacency_matrix(self) -> tuple[np.ndarray[Any, Any], list[str]]:
        """Convert to dense adjacency matrix.

        Returns:
            Tuple of (matrix, node_list) where matrix[i][j] = weight of edge i->j.
        """
        nodes = self.nodes
        n = len(nodes)
        idx = {node: i for i, node in enumerate(nodes)}
        matrix = np.zeros((n, n))
        for node, neighbors in self.adjacency.items():
            i = idx[node]
            for nbr, w in neighbors:
                if nbr in idx:
                    matrix[i, idx[nbr]] = w
        return matrix, nodes


class GraphBuilder:
    """Build entity graphs from relational DataFrames.

    Constructs graphs by interpreting DataFrame rows as edges
    between entities identified by specified columns.

    Args:
        source_col: Column name for source nodes.
        target_col: Column name for target nodes.
        weight_col: Column for edge weights. None for unit weights.
        directed: Whether to create directed edges.

    Example:
        >>> builder = GraphBuilder(source_col="user_id", target_col="product_id")
        >>> graph = builder.build(transactions_df)
    """

    def __init__(
        self,
        source_col: str,
        target_col: str,
        weight_col: str | None = None,
        directed: bool = False,
    ) -> None:
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.directed = directed

    def build(self, df: pd.DataFrame) -> EntityGraph:
        """Build a graph from a DataFrame.

        Args:
            df: DataFrame where each row represents a relationship.

        Returns:
            EntityGraph instance.
        """
        if self.source_col not in df.columns:
            raise ValidationError(f"Source column '{self.source_col}' not in DataFrame")
        if self.target_col not in df.columns:
            raise ValidationError(f"Target column '{self.target_col}' not in DataFrame")

        graph = EntityGraph()

        for _, row in df.iterrows():
            src = str(row[self.source_col])
            tgt = str(row[self.target_col])
            weight = float(row[self.weight_col]) if self.weight_col else 1.0

            if self.directed:
                graph.add_edge(src, tgt, weight)
            else:
                graph.add_undirected_edge(src, tgt, weight)

        return graph

    @staticmethod
    def from_edge_list(
        edges: list[tuple[str, str, float]],
        directed: bool = False,
    ) -> EntityGraph:
        """Build a graph from a list of (source, target, weight) tuples."""
        graph = EntityGraph()
        for src, tgt, w in edges:
            if directed:
                graph.add_edge(src, tgt, w)
            else:
                graph.add_undirected_edge(src, tgt, w)
        return graph
