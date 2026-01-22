"""SQL transpiler: convert Forge pipeline specs to SQL queries.

Supports PostgreSQL, BigQuery, and Snowflake dialects.

Example:
    >>> from forge.dsl.sql_transpiler import SQLTranspiler
    >>> sql = SQLTranspiler(dialect="postgres").transpile(spec)
    >>> print(sql)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from forge.dsl.parser import FeatureSpec, PipelineSpec  # noqa: TC001

logger = logging.getLogger(__name__)

Dialect = Literal["postgres", "bigquery", "snowflake"]


@dataclass
class SQLTranspileResult:
    """Result of transpiling a pipeline to SQL.

    Attributes:
        sql: Generated SQL query.
        dialect: Target SQL dialect.
        feature_count: Number of features in the query.
        warnings: Any transpilation warnings.
    """

    sql: str
    dialect: str
    feature_count: int
    warnings: list[str]


_FUNC_MAP: dict[str, dict[str, str]] = {
    "log": {
        "postgres": "LN({col} + 1)",
        "bigquery": "LN({col} + 1)",
        "snowflake": "LN({col} + 1)",
    },
    "sqrt": {
        "postgres": "SQRT(GREATEST({col}, 0))",
        "bigquery": "SQRT(GREATEST({col}, 0))",
        "snowflake": "SQRT(GREATEST({col}, 0))",
    },
    "square": {
        "postgres": "POWER({col}, 2)",
        "bigquery": "POWER({col}, 2)",
        "snowflake": "POWER({col}, 2)",
    },
    "reciprocal": {
        "postgres": "CASE WHEN {col} != 0 THEN 1.0 / {col} ELSE NULL END",
        "bigquery": "SAFE_DIVIDE(1.0, {col})",
        "snowflake": "CASE WHEN {col} != 0 THEN 1.0 / {col} ELSE NULL END",
    },
    "bin": {
        "postgres": "NTILE({n_bins}) OVER (ORDER BY {col})",
        "bigquery": "NTILE({n_bins}) OVER (ORDER BY {col})",
        "snowflake": "NTILE({n_bins}) OVER (ORDER BY {col})",
    },
    "standard_scale": {
        "postgres": "({col} - AVG({col}) OVER ()) / NULLIF(STDDEV({col}) OVER (), 0)",
        "bigquery": "({col} - AVG({col}) OVER ()) / NULLIF(STDDEV({col}) OVER (), 0)",
        "snowflake": "({col} - AVG({col}) OVER ()) / NULLIF(STDDEV({col}) OVER (), 0)",
    },
    "min_max_scale": {
        "postgres": "({col} - MIN({col}) OVER ()) / NULLIF(MAX({col}) OVER () - MIN({col}) OVER (), 0)",
        "bigquery": "({col} - MIN({col}) OVER ()) / NULLIF(MAX({col}) OVER () - MIN({col}) OVER (), 0)",
        "snowflake": "({col} - MIN({col}) OVER ()) / NULLIF(MAX({col}) OVER () - MIN({col}) OVER (), 0)",
    },
    "interaction": {
        "postgres": "{col1} * {col2}",
        "bigquery": "{col1} * {col2}",
        "snowflake": "{col1} * {col2}",
    },
    "length": {
        "postgres": "LENGTH(CAST({col} AS TEXT))",
        "bigquery": "LENGTH(CAST({col} AS STRING))",
        "snowflake": "LENGTH(CAST({col} AS VARCHAR))",
    },
    "word_count": {
        "postgres": "ARRAY_LENGTH(STRING_TO_ARRAY(TRIM(CAST({col} AS TEXT)), ' '), 1)",
        "bigquery": "ARRAY_LENGTH(SPLIT(CAST({col} AS STRING), ' '))",
        "snowflake": "ARRAY_SIZE(SPLIT(CAST({col} AS VARCHAR), ' '))",
    },
    # Temporal
    "day_of_week": {
        "postgres": "EXTRACT(DOW FROM {col})",
        "bigquery": "EXTRACT(DAYOFWEEK FROM {col})",
        "snowflake": "DAYOFWEEK({col})",
    },
    "month": {
        "postgres": "EXTRACT(MONTH FROM {col})",
        "bigquery": "EXTRACT(MONTH FROM {col})",
        "snowflake": "EXTRACT(MONTH FROM {col})",
    },
    "year": {
        "postgres": "EXTRACT(YEAR FROM {col})",
        "bigquery": "EXTRACT(YEAR FROM {col})",
        "snowflake": "EXTRACT(YEAR FROM {col})",
    },
    "hour": {
        "postgres": "EXTRACT(HOUR FROM {col})",
        "bigquery": "EXTRACT(HOUR FROM {col})",
        "snowflake": "EXTRACT(HOUR FROM {col})",
    },
}


class SQLTranspiler:
    """Transpile a Forge PipelineSpec into a SQL SELECT statement.

    Parameters
    ----------
    dialect : Dialect
        Target SQL dialect.
    source_table : str
        Name of the source table in SQL.
    """

    SUPPORTED_DIALECTS: tuple[str, ...] = ("postgres", "bigquery", "snowflake")

    def __init__(
        self,
        dialect: Dialect = "postgres",
        source_table: str = "source_data",
    ) -> None:
        if dialect not in self.SUPPORTED_DIALECTS:
            raise ValueError(
                f"Unsupported dialect '{dialect}'. "
                f"Choose from {self.SUPPORTED_DIALECTS}"
            )
        self.dialect = dialect
        self.source_table = source_table

    def transpile(self, spec: PipelineSpec) -> SQLTranspileResult:
        """Transpile a pipeline spec into SQL.

        Args:
            spec: Parsed PipelineSpec.

        Returns:
            SQLTranspileResult with the generated query.
        """
        select_exprs: list[str] = []
        warnings: list[str] = []

        # Always include original columns via *
        select_exprs.append("*")

        for feat in spec.features:
            expr = self._transpile_feature(feat)
            if expr is not None:
                select_exprs.append(f"{expr} AS {self._quote(feat.name)}")
            else:
                warnings.append(
                    f"Feature '{feat.name}': transform '{feat.transform}' "
                    f"not supported in {self.dialect} dialect"
                )

        sql = self._build_query(select_exprs)

        return SQLTranspileResult(
            sql=sql,
            dialect=self.dialect,
            feature_count=len(spec.features),
            warnings=warnings,
        )

    def transpile_feature(self, feat: FeatureSpec) -> str | None:
        """Transpile a single feature spec to a SQL expression.

        Args:
            feat: Feature specification.

        Returns:
            SQL expression string, or None if unsupported.
        """
        return self._transpile_feature(feat)

    def _transpile_feature(self, feat: FeatureSpec) -> str | None:
        transform = feat.transform
        source = feat.source if isinstance(feat.source, list) else [feat.source]

        if transform in _FUNC_MAP and self.dialect in _FUNC_MAP[transform]:
            template = _FUNC_MAP[transform][self.dialect]
            col = self._quote(source[0])

            if transform == "interaction" and len(source) >= 2:
                return template.format(
                    col1=self._quote(source[0]),
                    col2=self._quote(source[1]),
                )
            elif transform == "bin":
                n_bins = feat.params.get("n_bins", 10)
                return template.format(col=col, n_bins=n_bins)
            else:
                return template.format(col=col)

        # Encoding transforms not directly expressible in SQL
        if transform in ("onehot", "target", "frequency", "ordinal", "tfidf"):
            return None

        return None

    def _build_query(self, select_exprs: list[str]) -> str:
        joined = ",\n  ".join(select_exprs)
        return f"SELECT\n  {joined}\nFROM {self._quote_table(self.source_table)}"

    def _quote(self, name: str) -> str:
        if self.dialect == "bigquery":
            return f"`{name}`"
        return f'"{name}"'

    def _quote_table(self, name: str) -> str:
        if self.dialect == "bigquery":
            return f"`{name}`"
        return f'"{name}"'
