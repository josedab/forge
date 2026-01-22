"""Relationship management for multi-table feature synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from forge.exceptions import ConfigurationError, ValidationError


class RelationshipType(str, Enum):
    """Types of relationships between tables."""

    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass
class Relationship:
    """Represents a relationship between two tables.

    Attributes:
        parent_table: Name of the parent table.
        parent_column: Column in parent table (typically primary key).
        child_table: Name of the child table.
        child_column: Column in child table (foreign key).
        relationship_type: Type of relationship.
        confidence: Confidence score for auto-detected relationships.
    """

    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    relationship_type: RelationshipType = RelationshipType.ONE_TO_MANY
    confidence: float = 1.0

    def __str__(self) -> str:
        return (
            f"{self.parent_table}.{self.parent_column} "
            f"-> {self.child_table}.{self.child_column} "
            f"({self.relationship_type.value})"
        )

    def reverse(self) -> Relationship:
        """Create reversed relationship."""
        reversed_type = {
            RelationshipType.ONE_TO_ONE: RelationshipType.ONE_TO_ONE,
            RelationshipType.ONE_TO_MANY: RelationshipType.MANY_TO_ONE,
            RelationshipType.MANY_TO_ONE: RelationshipType.ONE_TO_MANY,
            RelationshipType.MANY_TO_MANY: RelationshipType.MANY_TO_MANY,
        }
        return Relationship(
            parent_table=self.child_table,
            parent_column=self.child_column,
            child_table=self.parent_table,
            child_column=self.parent_column,
            relationship_type=reversed_type[self.relationship_type],
            confidence=self.confidence,
        )


@dataclass
class TableInfo:
    """Information about a table in the relationship graph."""

    name: str
    dataframe: pd.DataFrame
    primary_key: str | None = None
    foreign_keys: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return list(self.dataframe.columns)

    @property
    def n_rows(self) -> int:
        return len(self.dataframe)


class RelationshipGraph:
    """Graph representing relationships between tables.

    This class manages a collection of tables and their relationships,
    providing methods for traversal, join path finding, and validation.

    Example:
        >>> graph = RelationshipGraph()
        >>> graph.add_table("customers", customers_df, primary_key="customer_id")
        >>> graph.add_table("orders", orders_df, primary_key="order_id")
        >>> graph.add_relationship(
        ...     parent="customers", parent_col="customer_id",
        ...     child="orders", child_col="customer_id"
        ... )
    """

    def __init__(self) -> None:
        """Initialize empty relationship graph."""
        self._tables: dict[str, TableInfo] = {}
        self._relationships: list[Relationship] = []
        self._adjacency: dict[str, list[tuple[str, Relationship]]] = {}

    def add_table(
        self,
        name: str,
        df: pd.DataFrame,
        primary_key: str | None = None,
        foreign_keys: list[str] | None = None,
    ) -> None:
        """Add a table to the graph.

        Args:
            name: Unique name for the table.
            df: The DataFrame.
            primary_key: Primary key column name.
            foreign_keys: List of foreign key column names.
        """
        if name in self._tables:
            raise ConfigurationError(f"Table '{name}' already exists")

        if primary_key and primary_key not in df.columns:
            raise ValidationError(f"Primary key '{primary_key}' not in table columns")

        self._tables[name] = TableInfo(
            name=name,
            dataframe=df,
            primary_key=primary_key,
            foreign_keys=foreign_keys or [],
        )
        self._adjacency[name] = []

    def add_relationship(
        self,
        parent: str,
        parent_col: str,
        child: str,
        child_col: str,
        relationship_type: RelationshipType | str = RelationshipType.ONE_TO_MANY,
    ) -> Relationship:
        """Add a relationship between tables.

        Args:
            parent: Parent table name.
            parent_col: Parent column (typically primary key).
            child: Child table name.
            child_col: Child column (foreign key).
            relationship_type: Type of relationship.

        Returns:
            The created Relationship.
        """
        if parent not in self._tables:
            raise ValidationError(f"Parent table '{parent}' not found")
        if child not in self._tables:
            raise ValidationError(f"Child table '{child}' not found")

        parent_df = self._tables[parent].dataframe
        child_df = self._tables[child].dataframe

        if parent_col not in parent_df.columns:
            raise ValidationError(f"Column '{parent_col}' not in table '{parent}'")
        if child_col not in child_df.columns:
            raise ValidationError(f"Column '{child_col}' not in table '{child}'")

        if isinstance(relationship_type, str):
            relationship_type = RelationshipType(relationship_type)

        relationship = Relationship(
            parent_table=parent,
            parent_column=parent_col,
            child_table=child,
            child_column=child_col,
            relationship_type=relationship_type,
        )

        self._relationships.append(relationship)
        self._adjacency[parent].append((child, relationship))
        self._adjacency[child].append((parent, relationship.reverse()))

        return relationship

    def get_table(self, name: str) -> TableInfo:
        """Get table information by name."""
        if name not in self._tables:
            raise ValidationError(f"Table '{name}' not found")
        return self._tables[name]

    def get_tables(self) -> dict[str, TableInfo]:
        """Get all tables."""
        return self._tables.copy()

    def get_relationships(self) -> list[Relationship]:
        """Get all relationships."""
        return self._relationships.copy()

    def get_neighbors(self, table_name: str) -> list[tuple[str, Relationship]]:
        """Get neighboring tables and their relationships.

        Args:
            table_name: Name of the table.

        Returns:
            List of (neighbor_name, relationship) tuples.
        """
        if table_name not in self._adjacency:
            raise ValidationError(f"Table '{table_name}' not found")
        return self._adjacency[table_name].copy()

    def find_path(
        self, source: str, target: str, max_depth: int = 5
    ) -> list[tuple[str, Relationship]] | None:
        """Find a path between two tables using BFS.

        Args:
            source: Source table name.
            target: Target table name.
            max_depth: Maximum path depth to search.

        Returns:
            List of (table_name, relationship) tuples or None if no path.
        """
        if source not in self._tables or target not in self._tables:
            return None

        if source == target:
            return []

        from collections import deque

        queue: deque[tuple[str, list[tuple[str, Relationship]]]] = deque([(source, [])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) >= max_depth:
                continue

            for neighbor, relationship in self._adjacency.get(current, []):
                if neighbor == target:
                    return path + [(neighbor, relationship)]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [(neighbor, relationship)]))

        return None

    def find_all_paths(
        self, source: str, target: str, max_depth: int = 3
    ) -> list[list[tuple[str, Relationship]]]:
        """Find all paths between two tables up to max_depth.

        Args:
            source: Source table name.
            target: Target table name.
            max_depth: Maximum path depth.

        Returns:
            List of paths, each path is a list of (table_name, relationship).
        """
        if source not in self._tables or target not in self._tables:
            return []

        paths: list[list[tuple[str, Relationship]]] = []

        def dfs(
            current: str,
            path: list[tuple[str, Relationship]],
            visited: set[str],
        ) -> None:
            if len(path) > max_depth:
                return

            if current == target and path:
                paths.append(path.copy())
                return

            for neighbor, relationship in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((neighbor, relationship))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)

        visited = {source}
        dfs(source, [], visited)
        return paths

    def validate(self) -> list[str]:
        """Validate the relationship graph.

        Returns:
            List of validation warnings/errors.
        """
        issues = []

        # Check for orphan tables
        connected = set()
        for rel in self._relationships:
            connected.add(rel.parent_table)
            connected.add(rel.child_table)

        for table in self._tables:
            if table not in connected and len(self._tables) > 1:
                issues.append(f"Table '{table}' has no relationships")

        # Check for referential integrity
        for rel in self._relationships:
            parent_df = self._tables[rel.parent_table].dataframe
            child_df = self._tables[rel.child_table].dataframe

            parent_values = set(parent_df[rel.parent_column].dropna())
            child_values = set(child_df[rel.child_column].dropna())

            orphan_children = child_values - parent_values
            if orphan_children and len(orphan_children) < 10:
                issues.append(
                    f"Relationship {rel}: {len(orphan_children)} orphan values in child"
                )
            elif orphan_children:
                pct = len(orphan_children) / len(child_values) * 100
                issues.append(
                    f"Relationship {rel}: {pct:.1f}% orphan values in child"
                )

        return issues

    def summary(self) -> str:
        """Generate a summary of the relationship graph."""
        lines = [
            f"RelationshipGraph: {len(self._tables)} tables, "
            f"{len(self._relationships)} relationships",
            "",
            "Tables:",
        ]

        for name, info in self._tables.items():
            pk = f", PK: {info.primary_key}" if info.primary_key else ""
            lines.append(f"  - {name}: {info.n_rows} rows, {len(info.columns)} cols{pk}")

        if self._relationships:
            lines.append("")
            lines.append("Relationships:")
            for rel in self._relationships:
                lines.append(f"  - {rel}")

        return "\n".join(lines)


def detect_relationships(
    tables: dict[str, pd.DataFrame],
    min_match_ratio: float = 0.8,
    id_suffixes: list[str] | None = None,
) -> RelationshipGraph:
    """Automatically detect relationships between tables.

    This function attempts to detect foreign key relationships by:
    1. Looking for columns with _id or _key suffixes
    2. Checking if column names match across tables
    3. Verifying value overlap between potential FK columns

    Args:
        tables: Dictionary mapping table names to DataFrames.
        min_match_ratio: Minimum ratio of child values in parent.
        id_suffixes: Suffixes indicating ID columns.

    Returns:
        RelationshipGraph with detected relationships.

    Example:
        >>> tables = {
        ...     "customers": customers_df,
        ...     "orders": orders_df,
        ... }
        >>> graph = detect_relationships(tables)
    """
    if id_suffixes is None:
        id_suffixes = ["_id", "_key", "_code", "id", "key"]

    graph = RelationshipGraph()

    # First pass: add all tables and detect primary keys
    primary_keys: dict[str, str | None] = {}

    for name, df in tables.items():
        # Try to detect primary key
        pk = _detect_primary_key(df, id_suffixes)
        primary_keys[name] = pk
        graph.add_table(name, df, primary_key=pk)

    # Second pass: detect relationships
    table_names = list(tables.keys())

    for i, parent_name in enumerate(table_names):
        parent_df = tables[parent_name]
        parent_pk = primary_keys[parent_name]

        for child_name in table_names[i + 1:]:
            child_df = tables[child_name]

            # Check for potential FK in child referencing parent
            relationships = _find_fk_relationships(
                parent_name, parent_df, parent_pk,
                child_name, child_df,
                min_match_ratio, id_suffixes,
            )

            for rel in relationships:
                try:
                    graph.add_relationship(
                        parent=rel["parent"],
                        parent_col=rel["parent_col"],
                        child=rel["child"],
                        child_col=rel["child_col"],
                        relationship_type=rel["type"],
                    )
                except (ValidationError, ConfigurationError):
                    continue

            # Check for potential FK in parent referencing child
            relationships = _find_fk_relationships(
                child_name, child_df, primary_keys[child_name],
                parent_name, parent_df,
                min_match_ratio, id_suffixes,
            )

            for rel in relationships:
                try:
                    graph.add_relationship(
                        parent=rel["parent"],
                        parent_col=rel["parent_col"],
                        child=rel["child"],
                        child_col=rel["child_col"],
                        relationship_type=rel["type"],
                    )
                except (ValidationError, ConfigurationError):
                    continue

    return graph


def _detect_primary_key(
    df: pd.DataFrame, id_suffixes: list[str]
) -> str | None:
    """Detect primary key column in a DataFrame."""
    for col in df.columns:
        col_lower = col.lower()

        # Check if it looks like a PK
        is_id_col = any(col_lower.endswith(suffix) for suffix in id_suffixes)
        is_id_col = is_id_col or col_lower in ("id", "pk", "key")

        if is_id_col:
            # Verify uniqueness
            if df[col].nunique() == len(df) and df[col].notna().all():
                return col

    # Fallback: find any unique column
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].notna().all():
            return col

    return None


def _find_fk_relationships(
    parent_name: str,
    parent_df: pd.DataFrame,
    parent_pk: str | None,
    child_name: str,
    child_df: pd.DataFrame,
    min_match_ratio: float,
    id_suffixes: list[str],
) -> list[dict[str, Any]]:
    """Find potential FK relationships from child to parent."""
    relationships = []

    for child_col in child_df.columns:
        child_col_lower = child_col.lower()

        # Skip if not an ID-like column
        is_id_col = any(child_col_lower.endswith(suffix) for suffix in id_suffixes)
        if not is_id_col:
            continue

        # Try to find matching parent column
        parent_col = None

        # First, check if column contains parent table name
        parent_name_lower = parent_name.lower().rstrip("s")  # Remove trailing 's'
        if parent_name_lower in child_col_lower:
            if parent_pk:
                parent_col = parent_pk
            else:
                # Look for matching column in parent
                for pcol in parent_df.columns:
                    if pcol.lower() == child_col_lower:
                        parent_col = pcol
                        break

        # Check for exact column name match
        if parent_col is None and child_col in parent_df.columns:
            parent_col = child_col

        # Check if parent PK name is in child
        if parent_col is None and parent_pk:
            if parent_pk.lower() == child_col_lower:
                parent_col = parent_pk

        if parent_col is None:
            continue

        # Verify the relationship by checking value overlap
        child_values = set(child_df[child_col].dropna())
        parent_values = set(parent_df[parent_col].dropna())

        if not child_values or not parent_values:
            continue

        overlap = child_values & parent_values
        match_ratio = len(overlap) / len(child_values)

        if match_ratio >= min_match_ratio:
            # Determine relationship type
            parent_unique = parent_df[parent_col].nunique() == len(parent_df)
            child_unique = child_df[child_col].nunique() == len(child_df)

            if parent_unique and child_unique:
                rel_type = RelationshipType.ONE_TO_ONE
            elif parent_unique:
                rel_type = RelationshipType.ONE_TO_MANY
            else:
                rel_type = RelationshipType.MANY_TO_MANY

            relationships.append({
                "parent": parent_name,
                "parent_col": parent_col,
                "child": child_name,
                "child_col": child_col,
                "type": rel_type,
                "confidence": match_ratio,
            })

    return relationships
