"""Join-path optimization for cross-table feature synthesis.

Discovers optimal join paths between tables and generates
aggregated features with configurable depth and pruning.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class JoinPath:
    """A discovered path of joins between tables.

    Attributes:
        tables: Ordered list of table names in the path.
        join_keys: List of (left_col, right_col) for each join.
        depth: Number of joins (hops).
        estimated_cost: Estimated memory/compute cost.
    """

    tables: list[str]
    join_keys: list[tuple[str, str]]
    depth: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimated relative cost (higher = more expensive)."""
        return float(2 ** self.depth)

    def __str__(self) -> str:
        parts = []
        for i, table in enumerate(self.tables):
            parts.append(table)
            if i < len(self.join_keys):
                left, right = self.join_keys[i]
                parts.append(f"--[{left}={right}]-->")
        return " ".join(parts)


@dataclass
class AggregationSpec:
    """Specification for an aggregation across a join.

    Attributes:
        source_table: Table to aggregate from.
        target_table: Table to aggregate into.
        join_path: How to join the tables.
        agg_column: Column to aggregate.
        agg_functions: Aggregation functions to apply.
    """

    source_table: str
    target_table: str
    join_path: JoinPath
    agg_column: str
    agg_functions: list[str] = field(
        default_factory=lambda: ["count", "mean", "sum", "min", "max"]
    )


class JoinPathFinder:
    """Find optimal join paths between tables in a relational schema.

    Uses BFS to discover all valid join paths up to a maximum depth,
    then ranks them by estimated cost and relevance.

    Args:
        max_depth: Maximum number of joins (hops) to traverse.
        max_paths: Maximum number of paths to return per pair.

    Example:
        >>> finder = JoinPathFinder(max_depth=3)
        >>> relationships = {
        ...     ("customers", "orders"): ("customer_id", "customer_id"),
        ...     ("orders", "products"): ("product_id", "product_id"),
        ... }
        >>> paths = finder.find_paths(relationships, "customers", "products")
    """

    def __init__(self, max_depth: int = 3, max_paths: int = 5) -> None:
        self.max_depth = max_depth
        self.max_paths = max_paths

    def find_paths(
        self,
        relationships: dict[tuple[str, str], tuple[str, str]],
        source: str,
        target: str,
    ) -> list[JoinPath]:
        """Find all join paths from source to target table.

        Args:
            relationships: Map of (parent, child) → (parent_col, child_col).
            source: Source table name.
            target: Target table name.

        Returns:
            List of JoinPaths sorted by cost (cheapest first).
        """
        # Build adjacency list (bidirectional)
        adj: dict[str, list[tuple[str, str, str]]] = {}
        for (parent, child), (pcol, ccol) in relationships.items():
            adj.setdefault(parent, []).append((child, pcol, ccol))
            adj.setdefault(child, []).append((parent, ccol, pcol))

        paths: list[JoinPath] = []
        queue: deque[tuple[list[str], list[tuple[str, str]]]] = deque()
        queue.append(([source], []))

        while queue:
            current_path, current_keys = queue.popleft()
            current_table = current_path[-1]

            if current_table == target and len(current_path) > 1:
                paths.append(JoinPath(
                    tables=list(current_path),
                    join_keys=list(current_keys),
                    depth=len(current_keys),
                ))
                if len(paths) >= self.max_paths:
                    break
                continue

            if len(current_keys) >= self.max_depth:
                continue

            for neighbor, left_col, right_col in adj.get(current_table, []):
                if neighbor not in current_path:  # Avoid cycles
                    queue.append((
                        current_path + [neighbor],
                        current_keys + [(left_col, right_col)],
                    ))

        paths.sort(key=lambda p: p.estimated_cost)
        return paths[:self.max_paths]

    def find_all_reachable(
        self,
        relationships: dict[tuple[str, str], tuple[str, str]],
        source: str,
    ) -> dict[str, JoinPath]:
        """Find shortest path to all reachable tables from source.

        Args:
            relationships: Relationship map.
            source: Starting table name.

        Returns:
            Dict mapping table name → shortest JoinPath.
        """
        adj: dict[str, list[tuple[str, str, str]]] = {}
        for (parent, child), (pcol, ccol) in relationships.items():
            adj.setdefault(parent, []).append((child, pcol, ccol))
            adj.setdefault(child, []).append((parent, ccol, pcol))

        visited: dict[str, JoinPath] = {}
        queue: deque[tuple[list[str], list[tuple[str, str]]]] = deque()
        queue.append(([source], []))

        while queue:
            path, keys = queue.popleft()
            current = path[-1]

            if current in visited:
                continue

            if current != source:
                visited[current] = JoinPath(
                    tables=list(path), join_keys=list(keys), depth=len(keys)
                )

            if len(keys) >= self.max_depth:
                continue

            for neighbor, left_col, right_col in adj.get(current, []):
                if neighbor not in visited and neighbor not in [t for t in path]:
                    queue.append((
                        path + [neighbor],
                        keys + [(left_col, right_col)],
                    ))

        return visited


class CrossTableAggregator:
    """Generate aggregated features across table joins.

    Performs the actual join and aggregation operations along
    discovered join paths to create cross-table features.

    Args:
        agg_functions: Default aggregation functions.
        max_cardinality: Skip joins with cardinality above this.

    Example:
        >>> agg = CrossTableAggregator()
        >>> features = agg.aggregate(
        ...     tables, path, target_key="customer_id", agg_columns=["amount"]
        ... )
    """

    def __init__(
        self,
        agg_functions: list[str] | None = None,
        max_cardinality: int = 1_000_000,
    ) -> None:
        self.agg_functions = agg_functions or ["count", "mean", "sum", "min", "max", "std"]
        self.max_cardinality = max_cardinality

    def aggregate(
        self,
        tables: dict[str, pd.DataFrame],
        path: JoinPath,
        target_key: str,
        agg_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Generate aggregated features along a join path.

        Args:
            tables: Dictionary of table_name → DataFrame.
            path: Join path to follow.
            target_key: Column in the first table to aggregate by.
            agg_columns: Columns to aggregate from the last table.
                None to auto-detect numeric columns.

        Returns:
            DataFrame with aggregated features, indexed by target_key.
        """
        if len(path.tables) < 2:
            return pd.DataFrame()

        # Get the remote table
        remote_table = path.tables[-1]
        remote_df = tables[remote_table]

        if agg_columns is None:
            agg_columns = list(remote_df.select_dtypes(include=[np.number]).columns)

        if not agg_columns:
            return pd.DataFrame()

        # Build the joined view by following the path
        joined = self._follow_path(tables, path)

        if joined is None or joined.empty:
            return pd.DataFrame()

        # Check cardinality
        if len(joined) > self.max_cardinality:
            logger.warning(
                "Join cardinality %d exceeds limit %d, sampling",
                len(joined), self.max_cardinality,
            )
            joined = joined.sample(n=self.max_cardinality, random_state=42)

        # Aggregate
        result_frames: list[pd.DataFrame] = []
        prefix = "_".join(path.tables[1:])

        for col in agg_columns:
            if col not in joined.columns:
                continue

            for func in self.agg_functions:
                try:
                    if func == "count":
                        agg_result = joined.groupby(target_key)[col].count()
                    elif func == "std":
                        agg_result = joined.groupby(target_key)[col].std()
                    else:
                        agg_result = joined.groupby(target_key)[col].agg(func)

                    feat_name = f"{prefix}_{col}_{func}"
                    result_frames.append(
                        agg_result.rename(feat_name).to_frame()
                    )
                except Exception:
                    continue

        if not result_frames:
            return pd.DataFrame()

        result = pd.concat(result_frames, axis=1)
        return result.fillna(0)

    def _follow_path(
        self, tables: dict[str, pd.DataFrame], path: JoinPath
    ) -> pd.DataFrame | None:
        """Execute joins along a path to produce a single DataFrame."""
        if not path.tables:
            return None

        current = tables[path.tables[0]].copy()

        for i, (left_col, right_col) in enumerate(path.join_keys):
            next_table = path.tables[i + 1]
            right_df = tables[next_table]

            try:
                current = current.merge(
                    right_df,
                    left_on=left_col,
                    right_on=right_col,
                    how="left",
                    suffixes=("", f"_{next_table}"),
                )
            except Exception as exc:
                logger.warning("Join failed at %s: %s", next_table, exc)
                return None

        return current
