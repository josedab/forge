"""Feature description and lineage tracking.

This module provides functionality to generate human-readable descriptions
for engineered features, helping users understand what each feature represents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureDescription:
    """Describes a generated feature's origin and meaning.

    Attributes:
    ----------
    name : str
        The feature name.
    description : str
        Human-readable description of the feature.
    source_columns : list[str]
        Original columns used to create this feature.
    generator : str
        Name of the generator that created this feature.
    parameters : dict
        Parameters used in feature generation.
    feature_type : str
        Type of feature (numeric, categorical, temporal, etc.).

    Examples:
    --------
    >>> desc = FeatureDescription(
    ...     name="price_quantity_multiply"
    ...     description="price multiplied by quantity"
    ...     source_columns=["price", "quantity"],
    ...     generator="InteractionGenerator"
    ...     parameters={"operation": "multiply"}
    ...     feature_type="numeric"
    ... )
    >>> print(desc.to_natural_language())
    price multiplied by quantity
    """

    name: str
    description: str
    source_columns: list[str]
    generator: str
    parameters: dict[str, Any] = field(default_factory=dict)
    feature_type: str = "derived"

    def to_natural_language(self) -> str:
        """Generate human-readable description.

        Returns:
        -------
        str
            Natural language description of the feature.
        """
        return self.description

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
        -------
        dict
            Dictionary with all feature metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "source_columns": self.source_columns,
            "generator": self.generator,
            "parameters": self.parameters,
            "feature_type": self.feature_type,
        }


class FeatureDescriber:
    """Generate natural language descriptions for features.

    This class provides templates and methods for creating human-readable
    descriptions of engineered features.

    Examples:
    --------
    >>> describer = FeatureDescriber()
    >>> desc = describer.describe_interaction(
    ...     feature_name="price_qty_multiply"
    ...     col1="price"
    ...     col2="quantity"
    ...     operation="multiply"
    ... )
    >>> print(desc.description)
    price multiplied by quantity
    """

    TEMPLATES = {
        # Interaction features
        "interaction_multiply": "{col1} multiplied by {col2}",
        "interaction_divide": "{col1} divided by {col2}",
        "interaction_add": "{col1} plus {col2}",
        "interaction_subtract": "{col1} minus {col2}",
        # Polynomial features
        "polynomial_2": "{col} squared (degree 2)",
        "polynomial_3": "{col} cubed (degree 3)",
        "polynomial_n": "{col} raised to power {degree}",
        # Transformations
        "log": "Natural logarithm of {col}",
        "log1p": "Natural logarithm of (1 + {col})",
        "sqrt": "Square root of {col}",
        "square": "{col} squared",
        "reciprocal": "Reciprocal (1/{col})",
        "abs": "Absolute value of {col}",
        # Binning
        "bin_quantile": "{col} binned into {n_bins} quantile-based bins",
        "bin_uniform": "{col} binned into {n_bins} equal-width bins",
        # Categorical encodings
        "onehot": "Binary indicator for {col} = {category}",
        "target_encoding": "Mean target value for each {col} category (smoothed)",
        "frequency_encoding": "Frequency of each {col} category in training data",
        "ordinal_encoding": "Ordinal integer encoding of {col}",
        "woe_encoding": "Weight of Evidence encoding of {col} for binary target",
        "catboost_encoding": "CatBoost-style ordered target encoding of {col}",
        "loo_encoding": "Leave-one-out target encoding of {col}",
        "hash_encoding": "Hash bucket {bucket} for {col}",
        # Temporal features
        "datetime_year": "Year extracted from {col}",
        "datetime_month": "Month (1-12) extracted from {col}",
        "datetime_day": "Day of month extracted from {col}",
        "datetime_hour": "Hour (0-23) extracted from {col}",
        "datetime_dayofweek": "Day of week (0=Monday) extracted from {col}",
        "datetime_quarter": "Quarter (1-4) extracted from {col}",
        "datetime_is_weekend": "Whether {col} is on a weekend",
        "lag": "{col} value from {n} periods ago",
        "rolling_mean": "{window}-period rolling mean of {col}",
        "rolling_std": "{window}-period rolling standard deviation of {col}",
        "rolling_min": "{window}-period rolling minimum of {col}",
        "rolling_max": "{window}-period rolling maximum of {col}",
        "diff": "Difference between current and previous {col} value",
        # Aggregations
        "groupby_mean": "Mean of {col} grouped by {group_col}",
        "groupby_sum": "Sum of {col} grouped by {group_col}",
        "groupby_std": "Standard deviation of {col} grouped by {group_col}",
        "groupby_count": "Count of {col} grouped by {group_col}",
        # Outlier handling
        "winsorized": "{col} with outliers capped at {lower}th and {upper}th percentiles",
        "iqr_capped": "{col} with outliers capped using IQR method (factor={factor})",
    }

    @classmethod
    def describe(
        cls,
        feature_name: str,
        feature_type: str,
        generator: str = "unknown",
        **kwargs: Any,
    ) -> FeatureDescription:
        """Generate description for a feature.

        Parameters
        ----------
        feature_name : str
            Name of the generated feature.
        feature_type : str
            Type key from TEMPLATES dict.
        generator : str
            Name of the generator class.
        **kwargs
            Template parameters (col, col1, col2, etc.).

        Returns:
        -------
        FeatureDescription
            Complete feature description object.
        """
        template = cls.TEMPLATES.get(feature_type, f"Derived feature from {kwargs}")

        try:
            description = template.format(**kwargs)
        except KeyError:
            description = f"Feature derived from: {', '.join(str(v) for v in kwargs.values())}"

        source_columns = []
        for key in ["col", "col1", "col2", "group_col"]:
            if key in kwargs:
                source_columns.append(kwargs[key])

        return FeatureDescription(
            name = feature_name,
            description = description,
            source_columns = source_columns,
            generator = generator,
            parameters = kwargs,
            feature_type=feature_type.split("_")[0] if "_" in feature_type else feature_type
        )

    @classmethod
    def describe_interaction(
        cls,
        feature_name: str,
        col1: str,
        col2: str,
        operation: str
    ) -> FeatureDescription:
        """Describe an interaction feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col1 : str
            First column name.
        col2 : str
            Second column name.
        operation : str
            Operation type (multiply, divide, add, subtract).

        Returns:
        -------
        FeatureDescription
            Description of the interaction feature.
        """
        return cls.describe(
            feature_name = feature_name,
            feature_type = f"interaction_{operation}",
            generator = "InteractionGenerator",
            col1 = col1,
            col2=col2
        )

    @classmethod
    def describe_polynomial(
        cls,
        feature_name: str,
        col: str,
        degree: int
    ) -> FeatureDescription:
        """Describe a polynomial feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Source column name.
        degree : int
            Polynomial degree.

        Returns:
        -------
        FeatureDescription
            Description of the polynomial feature.
        """
        if degree == 2:
            feature_type = "polynomial_2"
        elif degree == 3:
            feature_type = "polynomial_3"
        else:
            feature_type = "polynomial_n"

        return cls.describe(
            feature_name = feature_name,
            feature_type = feature_type,
            generator = "PolynomialGenerator",
            col = col,
            degree=degree
        )

    @classmethod
    def describe_encoding(
        cls,
        feature_name: str,
        col: str,
        encoding_type: str,
        **kwargs: Any,
    ) -> FeatureDescription:
        """Describe a categorical encoding feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Source column name.
        encoding_type : str
            Type of encoding (onehot, target, frequency, etc.).
        **kwargs
            Additional parameters (category for onehot, etc.).

        Returns:
        -------
        FeatureDescription
            Description of the encoding feature.
        """
        # Handle onehot specially - template key is 'onehot', not 'onehot_encoding'
        if encoding_type == "onehot":
            feature_type = "onehot"
        else:
            feature_type = f"{encoding_type}_encoding"

        return cls.describe(
            feature_name = feature_name,
            feature_type = feature_type,
            generator = f"{encoding_type.title()}Encoder",
            col = col,
            **kwargs,
        )

    @classmethod
    def describe_temporal(
        cls,
        feature_name: str,
        col: str,
        component: str
    ) -> FeatureDescription:
        """Describe a temporal feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Source datetime column name.
        component : str
            Temporal component (year, month, day, etc.).

        Returns:
        -------
        FeatureDescription
            Description of the temporal feature.
        """
        return cls.describe(
            feature_name = feature_name,
            feature_type = f"datetime_{component}",
            generator = "DateTimeComponents",
            col=col
        )

    @classmethod
    def describe_lag(
        cls,
        feature_name: str,
        col: str,
        n: int
    ) -> FeatureDescription:
        """Describe a lag feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Source column name.
        n : int
            Number of periods to lag.

        Returns:
        -------
        FeatureDescription
            Description of the lag feature.
        """
        return cls.describe(
            feature_name = feature_name,
            feature_type = "lag",
            generator = "LagGenerator",
            col = col,
            n=n
        )

    @classmethod
    def describe_rolling(
        cls,
        feature_name: str,
        col: str,
        window: int,
        statistic: str
    ) -> FeatureDescription:
        """Describe a rolling window feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.
        col : str
            Source column name.
        window : int
            Window size.
        statistic : str
            Rolling statistic (mean, std, min, max).

        Returns:
        -------
        FeatureDescription
            Description of the rolling feature.
        """
        return cls.describe(
            feature_name = feature_name,
            feature_type = f"rolling_{statistic}",
            generator = "RollingFeatures",
            col = col,
            window=window
        )


class FeatureLineage:
    """Track the lineage of engineered features.

    Maintains a record of how features were created, enabling
    traceability and understanding of the feature engineering pipeline.

    Examples:
    --------
    >>> lineage = FeatureLineage()
    >>> lineage.add_feature(
    ...     "price_qty",
    ...     sources=["price", "quantity"],
    ...     operation="multiply"
    ...     generator="InteractionGenerator"
    ... )
    >>> print(lineage.get_sources("price_qty"))
    ['price', 'quantity']
    """

    def __init__(self) -> None:
        """Initialize empty lineage tracker."""
        self._lineage: dict[str, dict[str, Any]] = {}
        self._descriptions: dict[str, FeatureDescription] = {}

    def add_feature(
        self,
        feature_name: str,
        sources: list[str],
        operation: str,
        generator: str,
        description: FeatureDescription | None = None,
        **metadata: Any,
    ) -> None:
        """Add a feature to the lineage.

        Parameters
        ----------
        feature_name : str
            Name of the new feature.
        sources : list[str]
            Source column names.
        operation : str
            Operation performed.
        generator : str
            Generator class name.
        description : FeatureDescription | None
            Optional description object.
        **metadata
            Additional metadata.
        """
        self._lineage[feature_name] = {
            "sources": sources,
            "operation": operation,
            "generator": generator,
            **metadata,
        }
        if description:
            self._descriptions[feature_name] = description

    def get_sources(self, feature_name: str) -> list[str]:
        """Get source columns for a feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.

        Returns:
        -------
        list[str]
            List of source column names.
        """
        if feature_name not in self._lineage:
            return [feature_name]  # Original feature
        return self._lineage[feature_name]["sources"]

    def get_description(self, feature_name: str) -> FeatureDescription | None:
        """Get description for a feature.

        Parameters
        ----------
        feature_name : str
            Name of the feature.

        Returns:
        -------
        FeatureDescription | None
            Description if available, None otherwise.
        """
        return self._descriptions.get(feature_name)

    def get_all_descriptions(self) -> dict[str, FeatureDescription]:
        """Get all feature descriptions.

        Returns:
        -------
        dict[str, FeatureDescription]
            Dictionary mapping feature names to descriptions.
        """
        return self._descriptions.copy()

    def trace_to_original(self, feature_name: str) -> list[str]:
        """Recursively trace feature back to original columns.

        Parameters
        ----------
        feature_name : str
            Name of the feature to trace.

        Returns:
        -------
        list[str]
            List of original column names.
        """
        if feature_name not in self._lineage:
            return [feature_name]

        originals: list[str] = []
        for source in self._lineage[feature_name]["sources"]:
            originals.extend(self.trace_to_original(source))

        return list(set(originals))

    def to_dataframe(self) -> Any:
        """Export lineage as a DataFrame.

        Returns:
        -------
        pd.DataFrame
            DataFrame with lineage information.
        """
        import pandas as pd

        records = []
        for name, info in self._lineage.items():
            record = {
                "feature": name,
                "sources": ", ".join(info["sources"]),
                "operation": info["operation"],
                "generator": info["generator"],
            }
            if name in self._descriptions:
                record["description"] = self._descriptions[name].description
            records.append(record)

        return pd.DataFrame(records)
