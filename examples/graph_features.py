#!/usr/bin/env python
"""Graph feature engineering example for Forge.

This example demonstrates EntityGraph, GraphBuilder, and
GraphFeatureGenerator for extracting structural features
from entity relationship data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forge.graph import EntityGraph, GraphBuilder, GraphFeatureGenerator


def create_social_network() -> pd.DataFrame:
    """Create a small social network as an edge list."""
    np.random.seed(42)
    users = [f"user_{i}" for i in range(20)]
    edges = []
    for user in users:
        n_friends = np.random.randint(1, 6)
        friends = np.random.choice(users, n_friends, replace=False)
        for friend in friends:
            if friend != user:
                edges.append({"source": user, "target": friend})
    return pd.DataFrame(edges)


def create_transaction_graph() -> pd.DataFrame:
    """Create transaction data between customers and merchants."""
    np.random.seed(42)
    n = 200
    customers = [f"cust_{i}" for i in range(30)]
    merchants = [f"merch_{i}" for i in range(10)]
    return pd.DataFrame({
        "customer": np.random.choice(customers, n),
        "merchant": np.random.choice(merchants, n),
        "amount": np.random.lognormal(3, 1, n),
    })


def example_entity_graph():
    """Example: Build and inspect an EntityGraph manually."""
    print("=" * 60)
    print("Example: EntityGraph Construction")
    print("=" * 60)

    graph = EntityGraph()

    # Add nodes and edges manually
    graph.add_node("Alice", role="admin")
    graph.add_node("Bob", role="user")
    graph.add_node("Charlie", role="user")
    graph.add_node("Diana", role="moderator")

    graph.add_undirected_edge("Alice", "Bob", weight=1.0)
    graph.add_undirected_edge("Alice", "Charlie", weight=0.8)
    graph.add_undirected_edge("Bob", "Charlie", weight=0.5)
    graph.add_undirected_edge("Bob", "Diana", weight=0.7)
    graph.add_undirected_edge("Charlie", "Diana", weight=0.3)

    print(f"\nGraph statistics:")
    print(f"  Nodes: {graph.n_nodes}")
    print(f"  Edges: {graph.n_edges}")

    print(f"\nNode details:")
    for node in graph.nodes:
        deg = graph.degree(node)
        in_deg = graph.in_degree(node)
        nbrs = graph.neighbors(node)
        print(f"  {node}: degree={deg}, in_degree={in_deg}, neighbors={nbrs}")

    # Convert to adjacency matrix
    matrix, node_list = graph.to_adjacency_matrix()
    print(f"\nAdjacency matrix shape: {matrix.shape}")
    print(f"Node order: {node_list}")


def example_graph_builder():
    """Example: Build a graph from a DataFrame with GraphBuilder."""
    print("\n" + "=" * 60)
    print("Example: GraphBuilder from DataFrame")
    print("=" * 60)

    edges_df = create_social_network()
    print(f"\nEdge list: {len(edges_df)} rows")
    print(f"Sample edges:\n{edges_df.head()}")

    builder = GraphBuilder(source_col="source", target_col="target", directed=False)
    graph = builder.build(edges_df)

    print(f"\nBuilt graph:")
    print(f"  Nodes: {graph.n_nodes}")
    print(f"  Edges: {graph.n_edges}")

    # Find most connected nodes
    degrees = [(node, graph.degree(node)) for node in graph.nodes]
    degrees.sort(key=lambda x: x[1], reverse=True)
    print(f"\nMost connected nodes:")
    for node, deg in degrees[:5]:
        print(f"  {node}: {deg} connections")


def example_graph_feature_generator():
    """Example: Generate graph features with GraphFeatureGenerator."""
    print("\n" + "=" * 60)
    print("Example: GraphFeatureGenerator")
    print("=" * 60)

    edges_df = create_social_network()

    gen = GraphFeatureGenerator(
        source_col="source",
        target_col="target",
        entity_col="source",
        features=["degree", "in_degree", "weighted_degree", "pagerank", "community"],
        directed=False,
    )

    # Fit builds the graph and computes features
    gen.fit(edges_df)

    # Transform maps features to entities in the DataFrame
    features = gen.transform(edges_df)

    print(f"\nGenerated {features.shape[1]} graph features for {features.shape[0]} rows")
    print(f"\nFeature names: {gen.get_feature_names_out()}")

    print(f"\nSample graph features (first 5 rows):")
    print(features.head().to_string())

    # Show unique PageRank values
    if "graph_pagerank" in features.columns:
        top_pr = features.groupby(edges_df["source"])["graph_pagerank"].first()
        top_pr = top_pr.sort_values(ascending=False).head(5)
        print(f"\nTop 5 nodes by PageRank:")
        for node, pr in top_pr.items():
            print(f"  {node}: {pr:.6f}")


if __name__ == "__main__":
    example_entity_graph()
    example_graph_builder()
    example_graph_feature_generator()

    print("\n" + "=" * 60)
    print("All graph feature examples completed!")
    print("=" * 60)
