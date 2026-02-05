"""Deep feature synthesis for multi-table data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd

from forge.exceptions import ValidationError
from forge.multitable.relationships import (
    Relationship,
    RelationshipGraph,
    RelationshipType,
)


class PrimitiveType(str, Enum):
    """Types of feature synthesis primitives."""

    AGGREGATION = "aggregation"
    TRANSFORM = "transform"


@dataclass
class FeatureDefinition:
    """Definition of a synthesized feature."""

    name: str
    base_column: str
    source_table: str
    primitive: str
    path: list[str]
    depth: int
    dtype: str = "float64"


class Primitive(ABC):
    """Base class for feature synthesis primitives."""

    name: str
    input_type: str  # "numeric", "categorical", "datetime", "any"
    output_type: str

    @abstractmethod
    def apply(self, series: pd.Series) -> pd.Series | float:
        """Apply the primitive to a series."""
        pass


class AggregationPrimitive(Primitive):
    """Primitive for aggregating values from child tables."""

    def __init__(
        self,
        name: str,
        func: Callable[[pd.Series], float],
        input_type: str = "numeric",
    ) -> None:
        """Initialize aggregation primitive.

        Args:
            name: Primitive name.
            func: Aggregation function.
            input_type: Required input type.
        """
        self.name = name
        self.func = func
        self.input_type = input_type
        self.output_type = "numeric"

    def apply(self, series: pd.Series) -> float:
        """Apply aggregation to series."""
        try:
            return float(self.func(series))
        except (ValueError, TypeError):
            return np.nan


class TransformPrimitive(Primitive):
    """Primitive for transforming individual values."""

    def __init__(
        self,
        name: str,
        func: Callable[[pd.Series], pd.Series],
        input_type: str = "numeric",
        output_type: str = "numeric",
    ) -> None:
        """Initialize transform primitive.

        Args:
            name: Primitive name.
            func: Transform function.
            input_type: Required input type.
            output_type: Output type.
        """
        self.name = name
        self.func = func
        self.input_type = input_type
        self.output_type = output_type

    def apply(self, series: pd.Series) -> pd.Series:
        """Apply transform to series."""
        return self.func(series)


# Built-in aggregation primitives
AGGREGATION_PRIMITIVES: dict[str, AggregationPrimitive] = {
    "sum": AggregationPrimitive("sum", lambda s: s.sum(), "numeric"),
    "mean": AggregationPrimitive("mean", lambda s: s.mean(), "numeric"),
    "std": AggregationPrimitive("std", lambda s: s.std(), "numeric"),
    "min": AggregationPrimitive("min", lambda s: s.min(), "numeric"),
    "max": AggregationPrimitive("max", lambda s: s.max(), "numeric"),
    "count": AggregationPrimitive("count", lambda s: len(s), "any"),
    "nunique": AggregationPrimitive("nunique", lambda s: s.nunique(), "any"),
    "mode": AggregationPrimitive(
        "mode",
        lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else np.nan,
        "categorical",
    ),
    "first": AggregationPrimitive("first", lambda s: s.iloc[0] if len(s) > 0 else np.nan, "any"),
    "last": AggregationPrimitive("last", lambda s: s.iloc[-1] if len(s) > 0 else np.nan, "any"),
    "median": AggregationPrimitive("median", lambda s: s.median(), "numeric"),
    "skew": AggregationPrimitive("skew", lambda s: s.skew(), "numeric"),
    "percent_true": AggregationPrimitive(
        "percent_true",
        lambda s: s.sum() / len(s) if len(s) > 0 else np.nan,
        "boolean",
    ),
}

# Built-in transform primitives
TRANSFORM_PRIMITIVES: dict[str, TransformPrimitive] = {
    "log": TransformPrimitive("log", lambda s: np.log1p(s.clip(lower=0)), "numeric"),
    "sqrt": TransformPrimitive("sqrt", lambda s: np.sqrt(s.clip(lower=0)), "numeric"),
    "square": TransformPrimitive("square", lambda s: s ** 2, "numeric"),
    "abs": TransformPrimitive("abs", lambda s: s.abs(), "numeric"),
    "negate": TransformPrimitive("negate", lambda s: -s, "numeric"),
    "year": TransformPrimitive("year", lambda s: pd.to_datetime(s).dt.year, "datetime", "numeric"),
    "month": TransformPrimitive("month", lambda s: pd.to_datetime(s).dt.month, "datetime", "numeric"),
    "day": TransformPrimitive("day", lambda s: pd.to_datetime(s).dt.day, "datetime", "numeric"),
    "weekday": TransformPrimitive("weekday", lambda s: pd.to_datetime(s).dt.dayofweek, "datetime", "numeric"),
    "is_weekend": TransformPrimitive(
        "is_weekend",
        lambda s: pd.to_datetime(s).dt.dayofweek >= 5,
        "datetime",
        "boolean",
    ),
}


@dataclass
class SynthesisResult:
    """Result of deep feature synthesis."""

    features: pd.DataFrame
    feature_definitions: list[FeatureDefinition]
    synthesis_info: dict[str, Any] = field(default_factory=dict)


class DeepFeatureSynthesis:
    """Deep feature synthesis across related tables.

    This class generates features from related tables by traversing
    relationships and applying aggregation and transform primitives.

    Example:
        >>> graph = RelationshipGraph()
        >>> # ... add tables and relationships ...
        >>> dfs = DeepFeatureSynthesis(
        ...     graph,
        ...     target_table="customers",
        ...     max_depth=2,
        ... )
        >>> result = dfs.synthesize()
    """

    def __init__(
        self,
        graph: RelationshipGraph,
        target_table: str,
        agg_primitives: list[str] | None = None,
        trans_primitives: list[str] | None = None,
        max_depth: int = 2,
        max_features: int = 100,
        ignore_columns: list[str] | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialize deep feature synthesis.

        Args:
            graph: RelationshipGraph with tables and relationships.
            target_table: Target table to generate features for.
            agg_primitives: Aggregation primitives to use.
            trans_primitives: Transform primitives to use.
            max_depth: Maximum depth for feature synthesis.
            max_features: Maximum number of features to generate.
            ignore_columns: Columns to ignore in synthesis.
            verbose: Verbosity level.
        """
        self.graph = graph
        self.target_table = target_table
        self.max_depth = max_depth
        self.max_features = max_features
        self.ignore_columns = set(ignore_columns or [])
        self.verbose = verbose

        # Set up primitives
        if agg_primitives is None:
            agg_primitives = ["mean", "sum", "count", "std", "min", "max"]
        if trans_primitives is None:
            trans_primitives = []  # Default to no transforms for simplicity

        self.agg_primitives = {
            name: AGGREGATION_PRIMITIVES[name]
            for name in agg_primitives
            if name in AGGREGATION_PRIMITIVES
        }
        self.trans_primitives = {
            name: TRANSFORM_PRIMITIVES[name]
            for name in trans_primitives
            if name in TRANSFORM_PRIMITIVES
        }

        # Validation
        if target_table not in graph.get_tables():
            raise ValidationError(f"Target table '{target_table}' not in graph")

        self._feature_definitions: list[FeatureDefinition] = []

    def synthesize(self) -> SynthesisResult:
        """Run deep feature synthesis.

        Returns:
            SynthesisResult with generated features.
        """
        target_info = self.graph.get_table(self.target_table)
        result_df = target_info.dataframe.copy()

        self._feature_definitions = []
        features_generated = 0

        # Generate features from each depth level
        for depth in range(1, self.max_depth + 1):
            if features_generated >= self.max_features:
                break

            new_features = self._synthesize_at_depth(
                result_df, depth, self.max_features - features_generated
            )

            for col, values in new_features.items():
                result_df[col] = values
                features_generated += 1

            if self.verbose >= 1:
                print(f"Depth {depth}: generated {len(new_features)} features")

        # Separate original columns from generated features
        original_cols = list(target_info.dataframe.columns)
        feature_cols = [c for c in result_df.columns if c not in original_cols]

        return SynthesisResult(
            features=result_df[feature_cols],
            feature_definitions=self._feature_definitions,
            synthesis_info={
                "target_table": self.target_table,
                "max_depth": self.max_depth,
                "n_features": len(feature_cols),
            },
        )

    def _synthesize_at_depth(
        self, base_df: pd.DataFrame, depth: int, max_features: int
    ) -> dict[str, pd.Series]:
        """Synthesize features at a specific depth.

        Args:
            base_df: Current base DataFrame.
            depth: Current synthesis depth.
            max_features: Maximum features to generate.

        Returns:
            Dictionary of feature_name -> feature_values.
        """
        features: dict[str, pd.Series] = {}

        # Find all paths of the specified depth from target
        paths = self._find_paths_at_depth(depth)

        for path in paths:
            if len(features) >= max_features:
                break

            path_features = self._generate_features_for_path(base_df, path)

            for name, values in path_features.items():
                if len(features) >= max_features:
                    break
                features[name] = values

        return features

    def _find_paths_at_depth(self, depth: int) -> list[list[tuple[str, Relationship]]]:
        """Find all paths of exactly the specified depth from target."""
        if depth == 0:
            return []

        paths = []
        visited_paths: set[tuple[str, ...]] = set()

        def dfs(
            current: str,
            path: list[tuple[str, Relationship]],
            visited: set[str],
        ) -> None:
            if len(path) == depth:
                path_key = tuple(t for t, _ in path)
                if path_key not in visited_paths:
                    visited_paths.add(path_key)
                    paths.append(path.copy())
                return

            for neighbor, rel in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((neighbor, rel))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        visited = {self.target_table}
        dfs(self.target_table, [], visited)

        return paths

    def _generate_features_for_path(
        self,
        base_df: pd.DataFrame,
        path: list[tuple[str, Relationship]],
    ) -> dict[str, pd.Series]:
        """Generate features for a specific path.

        Args:
            base_df: Base DataFrame (target table).
            path: List of (table_name, relationship) tuples.

        Returns:
            Dictionary of feature_name -> feature_values.
        """
        features: dict[str, pd.Series] = {}

        if not path:
            return features

        # Get the final table in the path
        final_table_name = path[-1][0]
        final_table_info = self.graph.get_table(final_table_name)
        final_df = final_table_info.dataframe

        # Build the join path
        # We need to join from target through each intermediate table
        target_info = self.graph.get_table(self.target_table)
        target_pk = target_info.primary_key

        if target_pk is None:
            return features

        # Determine which relationship is the last one (connects to final table)
        last_rel = path[-1][1]

        # Determine join columns
        # If target is parent in relationship, use parent_column
        # If target is child in relationship, use child_column
        if last_rel.parent_table == self.target_table:
            target_join_col = last_rel.parent_column
            child_join_col = last_rel.child_column
        else:
            target_join_col = last_rel.child_column
            child_join_col = last_rel.parent_column

        # Verify join columns exist
        if target_join_col not in base_df.columns:
            return features
        if child_join_col not in final_df.columns:
            return features

        # For one-to-many relationships (parent -> child), aggregate child values
        # For many-to-one relationships (child -> parent), just join
        is_one_to_many = (
            last_rel.relationship_type == RelationshipType.ONE_TO_MANY
            and last_rel.parent_table == self.target_table
        )

        if is_one_to_many:
            # Generate aggregation features
            features = self._generate_aggregation_features(
                base_df, final_df, target_join_col, child_join_col, path
            )
        else:
            # Generate direct join features (for many-to-one)
            features = self._generate_direct_features(
                base_df, final_df, target_join_col, child_join_col, path
            )

        return features

    def _generate_aggregation_features(
        self,
        parent_df: pd.DataFrame,
        child_df: pd.DataFrame,
        parent_col: str,
        child_col: str,
        path: list[tuple[str, Relationship]],
    ) -> dict[str, pd.Series]:
        """Generate aggregation features from child to parent."""
        features: dict[str, pd.Series] = {}

        path_prefix = "__".join(t for t, _ in path)

        for col in child_df.columns:
            if col in self.ignore_columns or col == child_col:
                continue

            col_dtype = child_df[col].dtype

            for agg_name, primitive in self.agg_primitives.items():
                # Check type compatibility
                if primitive.input_type == "numeric" and not pd.api.types.is_numeric_dtype(col_dtype):
                    continue
                if primitive.input_type == "boolean" and not pd.api.types.is_bool_dtype(col_dtype):
                    continue

                feature_name = f"{path_prefix}__{col}__{agg_name}"

                try:
                    # Group by join column and aggregate
                    grouped = child_df.groupby(child_col)[col].agg(primitive.func)

                    # Map back to parent
                    feature_values = parent_df[parent_col].map(grouped)

                    # Fill missing with appropriate default
                    if agg_name == "count":
                        feature_values = feature_values.fillna(0)
                    else:
                        feature_values = feature_values.astype(float)

                    features[feature_name] = feature_values

                    # Record definition
                    self._feature_definitions.append(FeatureDefinition(
                        name=feature_name,
                        base_column=col,
                        source_table=path[-1][0],
                        primitive=agg_name,
                        path=[t for t, _ in path],
                        depth=len(path),
                    ))

                except Exception as e:
                    if self.verbose >= 2:
                        print(f"Failed to generate {feature_name}: {e}")
                    continue

        return features

    def _generate_direct_features(
        self,
        target_df: pd.DataFrame,
        source_df: pd.DataFrame,
        target_col: str,
        source_col: str,
        path: list[tuple[str, Relationship]],
    ) -> dict[str, pd.Series]:
        """Generate direct features by joining tables."""
        features: dict[str, pd.Series] = {}

        path_prefix = "__".join(t for t, _ in path)

        # Create a mapping from source
        for col in source_df.columns:
            if col in self.ignore_columns or col == source_col:
                continue

            feature_name = f"{path_prefix}__{col}"

            try:
                # Create mapping and apply
                mapping = source_df.set_index(source_col)[col].to_dict()
                feature_values = target_df[target_col].map(mapping)

                features[feature_name] = feature_values

                self._feature_definitions.append(FeatureDefinition(
                    name=feature_name,
                    base_column=col,
                    source_table=path[-1][0],
                    primitive="direct",
                    path=[t for t, _ in path],
                    depth=len(path),
                ))

            except Exception as e:
                if self.verbose >= 2:
                    print(f"Failed to generate {feature_name}: {e}")
                continue

        return features

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Get definitions of generated features."""
        return self._feature_definitions.copy()


def create_dfs(
    tables: dict[str, pd.DataFrame],
    target_table: str,
    max_depth: int = 2,
    **kwargs: Any,
) -> DeepFeatureSynthesis:
    """Convenience function to create a DeepFeatureSynthesis instance.

    This function automatically detects relationships between tables
    and sets up deep feature synthesis.

    Args:
        tables: Dictionary mapping table names to DataFrames.
        target_table: Target table for feature generation.
        max_depth: Maximum synthesis depth.
        **kwargs: Additional arguments for DeepFeatureSynthesis.

    Returns:
        Configured DeepFeatureSynthesis instance.
    """
    from forge.multitable.relationships import detect_relationships

    graph = detect_relationships(tables)
    return DeepFeatureSynthesis(
        graph=graph,
        target_table=target_table,
        max_depth=max_depth,
        **kwargs,
    )
