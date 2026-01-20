"""Input validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from forge.exceptions import ColumnNotFoundError, NotFittedError, ValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence


def validate_dataframe(
    X: Any,
    name: str = "X",
    allow_empty: bool = False,
    min_rows: int = 0,
    min_cols: int = 0
) -> pd.DataFrame:
    """Validate that input is a valid DataFrame.

    Args:
        X: Input to validate.,
        name: Name for error messages.,
        allow_empty: Whether to allow empty DataFrames.,
        min_rows: Minimum number of rows required.,
        min_cols: Minimum number of columns required.,

    Returns:
        Validated DataFrame.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        else:
            raise ValidationError(
                f"{name} must be a pandas DataFrame, got {type(X).__name__}"
            )

    if not allow_empty and X.empty:
        raise ValidationError(f"{name} cannot be empty")

    if len(X) < min_rows:
        raise ValidationError(
            f"{name} must have at least {min_rows} rows, got {len(X)}"
        )

    if len(X.columns) < min_cols:
        raise ValidationError(
            f"{name} must have at least {min_cols} columns, got {len(X.columns)}"
        )

    return X


def validate_columns_exist(
    X: pd.DataFrame,
    columns: Sequence[str],
    name: str = "X"
) -> None:
    """Validate that all specified columns exist in DataFrame.

    Args:
        X: Input DataFrame.,
        columns: Columns that must exist.,
        name: DataFrame name for error messages.,

    Raises:
        ColumnNotFoundError: If any column is missing.
    """
    missing = set(columns) - set(X.columns)
    if missing:
        raise ColumnNotFoundError(
            f"Columns not found in {name}: {sorted(missing)}. "
            f"Available columns: {sorted(X.columns)}"
        )


def validate_target(
    y: Any,
    X: pd.DataFrame | None = None,
    name: str = "y",
    allow_none: bool = False
) -> pd.Series | None:
    """Validate target variable.

    Args:
        y: Target to validate.,
        X: Optional DataFrame for length checking.,
        name: Name for error messages.,
        allow_none: Whether to allow None.,

    Returns:
        Validated Series or None.

    Raises:
        ValidationError: If validation fails.
    """
    if y is None:
        if allow_none:
            return None
        raise ValidationError(f"{name} cannot be None")

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValidationError(
            f"{name} must be a pandas Series or numpy array, got {type(y).__name__}"
        )

    if X is not None and len(y) != len(X):
        raise ValidationError(
            f"Length mismatch: {name} has {len(y)} samples, X has {len(X)}"
        )

    return y


def check_is_fitted(
    estimator: Any,
    attributes: Sequence[str] | None = None,
    msg: str | None = None
) -> None:
    """Check if estimator is fitted.

    Args:
        estimator: Estimator to check.,
        attributes: Attributes to check for. If None, checks _is_fitted.
        msg: Custom error message.,

    Raises:
        NotFittedError: If estimator is not fitted.
    """
    if attributes is None:
        attributes = ["_is_fitted"]

    fitted = any(
        hasattr(estimator, attr) and getattr(estimator, attr) is not None
        for attr in attributes
    )

    # Also check for _is_fitted == True specifically
    if hasattr(estimator, "_is_fitted") and estimator._is_fitted:
        fitted = True

    if not fitted:
        if msg is None:
            name = type(estimator).__name__
            msg = (
                f"This {name} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )
        raise NotFittedError(msg)


def validate_numeric_columns(
    X: pd.DataFrame,
    columns: Sequence[str] | None = None
) -> list[str]:
    """Get and validate numeric columns.

    Args:
        X: Input DataFrame.,
        columns: Specific columns to validate, or None for all numeric.

    Returns:
        List of validated numeric column names.

    Raises:
        ValidationError: If no numeric columns found or specified columns aren't numeric.
    """
    if columns is None:
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValidationError("No numeric columns found in DataFrame")
        return numeric_cols

    validate_columns_exist(X, columns)

    non_numeric = []
    for col in columns:
        if not np.issubdtype(X[col].dtype, np.number):
            non_numeric.append(col)

    if non_numeric:
        raise ValidationError(f"Columns must be numeric: {non_numeric}")

    return list(columns)


def validate_categorical_columns(
    X: pd.DataFrame,
    columns: Sequence[str] | None = None,
    max_cardinality: int | None = None
) -> list[str]:
    """Get and validate categorical columns.

    Args:
        X: Input DataFrame.,
        columns: Specific columns to validate, or None for all categorical.
        max_cardinality: Maximum allowed unique values.,

    Returns:
        List of validated categorical column names.

    Raises:
        ValidationError: If validation fails.
    """
    if columns is None:
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        if not categorical_cols:
            raise ValidationError("No categorical columns found in DataFrame")
        columns = categorical_cols

    validate_columns_exist(X, columns)

    if max_cardinality is not None:
        high_card = []
        for col in columns:
            n_unique = X[col].nunique()
            if n_unique > max_cardinality:
                high_card.append(f"{col} ({n_unique})")
        if high_card:
            raise ValidationError(
                f"Columns exceed max cardinality {max_cardinality}: {high_card}"
            )

    return list(columns)


def validate_feature_names(
    feature_names: Sequence[str],
    expected_length: int | None = None
) -> list[str]:
    """Validate feature names.

    Args:
        feature_names: Feature names to validate.,
        expected_length: Expected number of features.,

    Returns:
        List of validated feature names.

    Raises:
        ValidationError: If validation fails.
    """
    if not feature_names:
        raise ValidationError("feature_names cannot be empty")

    names = list(feature_names)

    # Check for duplicates
    if len(names) != len(set(names)):
        duplicates = [n for n in names if names.count(n) > 1]
        raise ValidationError(f"Duplicate feature names: {set(duplicates)}")

    if expected_length is not None and len(names) != expected_length:
        raise ValidationError(
            f"Expected {expected_length} feature names, got {len(names)}"
        )

    return names


def validate_positive_int(
    value: Any,
    name: str,
    allow_zero: bool = False,
    max_value: int | None = None
) -> int:
    """Validate positive integer parameter.

    Args:
        value: Value to validate.,
        name: Parameter name for error messages.,
        allow_zero: Whether to allow zero.,
        max_value: Maximum allowed value.,

    Returns:
        Validated integer.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")

    min_val = 0 if allow_zero else 1
    if value < min_val:
        raise ValidationError(f"{name} must be >= {min_val}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")

    return value


def validate_float_range(
    value: Any,
    name: str,
    min_value: float | None = None,
    max_value: float | None = None,
    inclusive: tuple[bool, bool] = (True, True),
) -> float:
    """Validate float parameter is within range.

    Args:
        value: Value to validate.,
        name: Parameter name for error messages.,
        min_value: Minimum allowed value.,
        max_value: Maximum allowed value.,
        inclusive: Tuple of (min_inclusive, max_inclusive).

    Returns:
        Validated float.

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")

    value = float(value)

    if min_value is not None:
        if inclusive[0]:
            if value < min_value:
                raise ValidationError(f"{name} must be >= {min_value}, got {value}")
        else:
            if value <= min_value:
                raise ValidationError(f"{name} must be > {min_value}, got {value}")

    if max_value is not None:
        if inclusive[1]:
            if value > max_value:
                raise ValidationError(f"{name} must be <= {max_value}, got {value}")
        else:
            if value >= max_value:
                raise ValidationError(f"{name} must be < {max_value}, got {value}")

    return value
