"""Feature lineage graph — DAG representation of feature transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd


@dataclass
class LineageNode:
    """A node in the lineage graph representing a feature or source column.

    Args:
        name: Feature or column name.
        node_type: One of "source", "generated", "selected", "dropped".
        generator: Generator class name that produced this feature.
        importance: Feature importance score (if available).
        metadata: Arbitrary metadata about the node.
    """

    name: str
    node_type: str = "source"
    generator: str = ""
    importance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "node_type": self.node_type,
            "generator": self.generator,
            "importance": self.importance,
            "metadata": self.metadata,
        }


@dataclass
class LineageEdge:
    """A directed edge in the lineage graph.

    Args:
        source: Source node name.
        target: Target node name.
        transformation: Description of the transformation applied.
    """

    source: str
    target: str
    transformation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "transformation": self.transformation,
        }


class FeatureLineageGraph:
    """Directed acyclic graph tracking feature provenance.

    Builds a DAG of how source columns are transformed into
    generated features, which features are selected, and
    their importance scores.

    Example:
        >>> graph = FeatureLineageGraph()
        >>> graph.add_source("age")
        >>> graph.add_generated("log_age", sources=["age"], generator="LogTransformer")
        >>> graph.add_selected("log_age", importance=0.85)
        >>> print(graph.summary())
    """

    def __init__(self) -> None:
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._created_at = datetime.now(tz=timezone.utc)

    def add_source(self, name: str, **metadata: Any) -> None:
        """Add a source (raw input) column."""
        self._nodes[name] = LineageNode(
            name=name, node_type="source", metadata=metadata
        )

    def add_generated(
        self,
        name: str,
        sources: list[str],
        generator: str = "",
        transformation: str = "",
        importance: float | None = None,
        **metadata: Any,
    ) -> None:
        """Add a generated feature with edges to its source columns."""
        self._nodes[name] = LineageNode(
            name=name,
            node_type="generated",
            generator=generator,
            importance=importance,
            metadata=metadata,
        )
        for src in sources:
            self._edges.append(
                LineageEdge(source=src, target=name, transformation=transformation)
            )

    def add_selected(self, name: str, importance: float | None = None) -> None:
        """Mark a feature as selected (kept after selection)."""
        if name in self._nodes:
            self._nodes[name].node_type = "selected"
            if importance is not None:
                self._nodes[name].importance = importance
        else:
            self._nodes[name] = LineageNode(
                name=name, node_type="selected", importance=importance
            )

    def add_dropped(self, name: str, reason: str = "") -> None:
        """Mark a feature as dropped (removed by selection)."""
        if name in self._nodes:
            self._nodes[name].node_type = "dropped"
            self._nodes[name].metadata["drop_reason"] = reason
        else:
            self._nodes[name] = LineageNode(
                name=name, node_type="dropped", metadata={"drop_reason": reason}
            )

    def get_node(self, name: str) -> LineageNode | None:
        """Get a node by name."""
        return self._nodes.get(name)

    @property
    def nodes(self) -> list[LineageNode]:
        """Return all nodes."""
        return list(self._nodes.values())

    @property
    def edges(self) -> list[LineageEdge]:
        """Return all edges."""
        return list(self._edges)

    def get_sources(self, feature_name: str) -> list[str]:
        """Get all source columns that contribute to a feature."""
        sources: set[str] = set()
        self._trace_back(feature_name, sources)
        return sorted(sources)

    def _trace_back(self, name: str, sources: set[str]) -> None:
        """Recursively trace back to source columns."""
        incoming = [e for e in self._edges if e.target == name]
        if not incoming:
            sources.add(name)
            return
        for edge in incoming:
            self._trace_back(edge.source, sources)

    def get_derived(self, source_name: str) -> list[str]:
        """Get all features derived from a source column."""
        derived: set[str] = set()
        self._trace_forward(source_name, derived)
        return sorted(derived)

    def _trace_forward(self, name: str, derived: set[str]) -> None:
        """Recursively trace forward to derived features."""
        outgoing = [e for e in self._edges if e.source == name]
        for edge in outgoing:
            derived.add(edge.target)
            self._trace_forward(edge.target, derived)

    def get_importance_ranking(self) -> list[tuple[str, float]]:
        """Get features ranked by importance."""
        ranked = [
            (n.name, n.importance)
            for n in self._nodes.values()
            if n.importance is not None
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def summary(self) -> str:
        """Return a human-readable summary."""
        n_source = sum(1 for n in self._nodes.values() if n.node_type == "source")
        n_generated = sum(1 for n in self._nodes.values() if n.node_type == "generated")
        n_selected = sum(1 for n in self._nodes.values() if n.node_type == "selected")
        n_dropped = sum(1 for n in self._nodes.values() if n.node_type == "dropped")

        lines = [
            "Feature Lineage Graph",
            f"  Source columns:    {n_source}",
            f"  Generated features: {n_generated}",
            f"  Selected features: {n_selected}",
            f"  Dropped features:  {n_dropped}",
            f"  Total edges:       {len(self._edges)}",
        ]
        top = self.get_importance_ranking()[:5]
        if top:
            lines.append("  Top features by importance:")
            for name, imp in top:
                lines.append(f"    - {name}: {imp:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph to a dictionary."""
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
            "created_at": self._created_at.isoformat(),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert nodes to a DataFrame for analysis."""
        records = []
        for n in self._nodes.values():
            records.append({
                "name": n.name,
                "type": n.node_type,
                "generator": n.generator,
                "importance": n.importance,
                "n_sources": len(self.get_sources(n.name)),
            })
        return pd.DataFrame(records)

    @classmethod
    def from_transformer(cls, transformer: Any) -> FeatureLineageGraph:
        """Build lineage graph from a fitted AutoFeatureTransformer.

        Args:
            transformer: Fitted AutoFeatureTransformer or similar.

        Returns:
            Populated lineage graph.
        """
        graph = cls()

        # Extract input columns
        input_cols: list[str] = []
        if hasattr(transformer, "_input_columns"):
            input_cols = transformer._input_columns
        elif hasattr(transformer, "feature_names_in_"):
            input_cols = list(transformer.feature_names_in_)

        for col in input_cols:
            graph.add_source(col)

        # Extract generated features
        output_cols: list[str] = []
        if hasattr(transformer, "get_feature_names_out"):
            try:
                output_cols = transformer.get_feature_names_out()
            except Exception:
                pass

        for col in output_cols:
            if col not in input_cols:
                graph.add_generated(col, sources=input_cols[:1], generator="auto")
            else:
                graph.add_selected(col)

        # Extract importances
        if hasattr(transformer, "get_feature_importance"):
            try:
                imp_df = transformer.get_feature_importance()
                if imp_df is not None and hasattr(imp_df, "iterrows"):
                    for _, row in imp_df.iterrows():
                        fname = row.get("feature", row.get("name"))
                        score = row.get("importance", row.get("score"))
                        if fname and score is not None:
                            graph.add_selected(str(fname), importance=float(score))
            except Exception:
                pass

        return graph
