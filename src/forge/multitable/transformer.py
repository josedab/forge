"""sklearn-compatible transformer for multi-table feature synthesis."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from forge.exceptions import NotFittedError, ValidationError
from forge.multitable.relationships import (
    RelationshipGraph,
    detect_relationships,
    Relationship,
)
from forge.multitable.synthesis import (
    DeepFeatureSynthesis,
    FeatureDefinition,
    SynthesisResult,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class MultiTableTransformer(BaseEstimator, TransformerMixin):
    """sklearn-compatible transformer for multi-table feature engineering.

    This transformer generates features from multiple related tables using
    deep feature synthesis. It automatically detects relationships between
    tables and creates aggregation and direct features.

    Example:
        >>> transformer = MultiTableTransformer(
        ...     target_table="customers",
        ...     max_depth=2,
        ... )
        >>> X_train_features = transformer.fit_transform(
        ...     tables={
        ...         "customers": customers_df,
        ...         "orders": orders_df,
        ...         "products": products_df,
        ...     },
        ...     y=y_train,
        ... )
        >>> X_test_features = transformer.transform(test_tables)
    """

    def __init__(
        self,
        target_table: str,
        agg_primitives: list[str] | None = None,
        trans_primitives: list[str] | None = None,
        max_depth: int = 2,
        max_features: int = 100,
        relationships: list[dict[str, str]] | None = None,
        auto_detect_relationships: bool = True,
        include_target_columns: bool = True,
        ignore_columns: list[str] | None = None,
        n_jobs: int = 1,
        verbose: int = 0,
    ) -> None:
        """Initialize the multi-table transformer.

        Args:
            target_table: Name of the target table to generate features for.
            agg_primitives: List of aggregation primitive names to use.
            trans_primitives: List of transform primitive names to use.
            max_depth: Maximum depth for relationship traversal.
            max_features: Maximum number of features to generate.
            relationships: Manual relationship definitions. Each dict should have:
                - parent: Parent table name
                - parent_col: Parent column name
                - child: Child table name
                - child_col: Child column name
            auto_detect_relationships: Whether to auto-detect relationships.
            include_target_columns: Include original target table columns.
            ignore_columns: Columns to ignore during synthesis.
            n_jobs: Number of parallel jobs (not yet implemented).
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        """
        self.target_table = target_table
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.max_depth = max_depth
        self.max_features = max_features
        self.relationships = relationships
        self.auto_detect_relationships = auto_detect_relationships
        self.include_target_columns = include_target_columns
        self.ignore_columns = ignore_columns
        self.n_jobs = n_jobs
        self.verbose = verbose

        # Fitted state
        self._is_fitted = False
        self._graph: RelationshipGraph | None = None
        self._dfs: DeepFeatureSynthesis | None = None
        self._feature_definitions: list[FeatureDefinition] = []
        self._feature_names: list[str] = []
        self._target_columns: list[str] = []

    def fit(
        self,
        tables: dict[str, pd.DataFrame],
        y: pd.Series | None = None,
    ) -> Self:
        """Fit the transformer on training data.

        Args:
            tables: Dictionary mapping table names to DataFrames.
            y: Optional target variable (for target table).

        Returns:
            Self for method chaining.
        """
        self._validate_tables(tables)

        # Build or detect relationship graph
        if self.auto_detect_relationships:
            self._graph = detect_relationships(tables)

            # Add any manual relationships
            if self.relationships:
                for rel in self.relationships:
                    try:
                        self._graph.add_relationship(
                            parent=rel["parent"],
                            parent_col=rel["parent_col"],
                            child=rel["child"],
                            child_col=rel["child_col"],
                        )
                    except (ValidationError, KeyError):
                        pass
        else:
            # Build graph from manual relationships only
            self._graph = RelationshipGraph()
            for name, df in tables.items():
                self._graph.add_table(name, df)

            if self.relationships:
                for rel in self.relationships:
                    self._graph.add_relationship(
                        parent=rel["parent"],
                        parent_col=rel["parent_col"],
                        child=rel["child"],
                        child_col=rel["child_col"],
                    )

        if self.verbose >= 1:
            print(self._graph.summary())

        # Set up deep feature synthesis
        self._dfs = DeepFeatureSynthesis(
            graph=self._graph,
            target_table=self.target_table,
            agg_primitives=self.agg_primitives,
            trans_primitives=self.trans_primitives,
            max_depth=self.max_depth,
            max_features=self.max_features,
            ignore_columns=self.ignore_columns,
            verbose=self.verbose,
        )

        # Run synthesis to learn feature definitions
        result = self._dfs.synthesize()
        self._feature_definitions = result.feature_definitions
        self._feature_names = list(result.features.columns)
        self._target_columns = list(tables[self.target_table].columns)

        if self.verbose >= 1:
            print(f"Generated {len(self._feature_names)} features")

        self._is_fitted = True
        return self

    def transform(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Transform tables to generate features.

        Args:
            tables: Dictionary mapping table names to DataFrames.

        Returns:
            DataFrame with generated features.
        """
        self._check_is_fitted()
        self._validate_tables(tables)

        # Update graph with new data
        for name, df in tables.items():
            table_info = self._graph.get_table(name)
            table_info.dataframe = df

        # Run synthesis
        result = self._dfs.synthesize()

        if self.include_target_columns:
            target_df = tables[self.target_table]
            return pd.concat([target_df, result.features], axis=1)

        return result.features

    def fit_transform(
        self,
        tables: dict[str, pd.DataFrame],
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            tables: Dictionary mapping table names to DataFrames.
            y: Optional target variable.

        Returns:
            DataFrame with generated features.
        """
        self.fit(tables, y)
        return self.transform(tables)

    def get_feature_names_out(
        self, input_features: list[str] | None = None
    ) -> list[str]:
        """Get names of output features.

        Args:
            input_features: Ignored, for sklearn compatibility.

        Returns:
            List of output feature names.
        """
        self._check_is_fitted()
        names = self._feature_names.copy()
        if self.include_target_columns:
            names = self._target_columns + names
        return names

    def get_feature_definitions(self) -> list[FeatureDefinition]:
        """Get definitions of generated features.

        Returns:
            List of FeatureDefinition objects.
        """
        self._check_is_fitted()
        return self._feature_definitions.copy()

    def get_relationship_graph(self) -> RelationshipGraph:
        """Get the relationship graph.

        Returns:
            The fitted RelationshipGraph.
        """
        self._check_is_fitted()
        return self._graph

    def get_feature_lineage(self, feature_name: str) -> dict[str, Any]:
        """Get lineage information for a specific feature.

        Args:
            feature_name: Name of the feature.

        Returns:
            Dictionary with lineage information.
        """
        self._check_is_fitted()

        for defn in self._feature_definitions:
            if defn.name == feature_name:
                return {
                    "name": defn.name,
                    "base_column": defn.base_column,
                    "source_table": defn.source_table,
                    "primitive": defn.primitive,
                    "path": defn.path,
                    "depth": defn.depth,
                }

        raise ValueError(f"Feature '{feature_name}' not found")

    def save(self, path: str | Path) -> None:
        """Save the fitted transformer to disk.

        Args:
            path: Path to save the transformer.
        """
        self._check_is_fitted()

        state = {
            "params": self.get_params(),
            "feature_definitions": self._feature_definitions,
            "feature_names": self._feature_names,
            "target_columns": self._target_columns,
            "relationships": self._extract_relationships(),
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str | Path) -> MultiTableTransformer:
        """Load a fitted transformer from disk.

        Args:
            path: Path to load the transformer from.

        Returns:
            Loaded MultiTableTransformer.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        transformer = cls(**state["params"])
        transformer._feature_definitions = state["feature_definitions"]
        transformer._feature_names = state["feature_names"]
        transformer._target_columns = state["target_columns"]
        transformer._is_fitted = True

        # Note: The graph needs to be rebuilt on transform with actual data
        return transformer

    def _validate_tables(self, tables: dict[str, pd.DataFrame]) -> None:
        """Validate input tables."""
        if not isinstance(tables, dict):
            raise ValidationError("tables must be a dictionary")

        if self.target_table not in tables:
            raise ValidationError(
                f"Target table '{self.target_table}' not in tables. "
                f"Available: {list(tables.keys())}"
            )

        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                raise ValidationError(
                    f"Table '{name}' must be a DataFrame, got {type(df).__name__}"
                )
            if len(df) == 0:
                raise ValidationError(f"Table '{name}' is empty")

    def _check_is_fitted(self) -> None:
        """Check if the transformer has been fitted."""
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _extract_relationships(self) -> list[dict[str, str]]:
        """Extract relationships from graph for serialization."""
        if self._graph is None:
            return []

        relationships = []
        for rel in self._graph.get_relationships():
            relationships.append({
                "parent": rel.parent_table,
                "parent_col": rel.parent_column,
                "child": rel.child_table,
                "child_col": rel.child_column,
                "type": rel.relationship_type.value,
            })
        return relationships


def multi_table_features(
    tables: dict[str, pd.DataFrame],
    target_table: str,
    max_depth: int = 2,
    max_features: int = 100,
    **kwargs: Any,
) -> pd.DataFrame:
    """Convenience function to generate multi-table features.

    Args:
        tables: Dictionary mapping table names to DataFrames.
        target_table: Target table for feature generation.
        max_depth: Maximum depth for synthesis.
        max_features: Maximum features to generate.
        **kwargs: Additional arguments for MultiTableTransformer.

    Returns:
        DataFrame with generated features.

    Example:
        >>> features = multi_table_features(
        ...     tables={"customers": df1, "orders": df2},
        ...     target_table="customers",
        ... )
    """
    transformer = MultiTableTransformer(
        target_table=target_table,
        max_depth=max_depth,
        max_features=max_features,
        **kwargs,
    )
    return transformer.fit_transform(tables)
