"""Custom exceptions for Forge."""

from __future__ import annotations


class ForgeError(Exception):
    """Base exception for all Forge errors."""

    pass


class NotFittedError(ForgeError):
    """Raised when a transformer is used before fitting."""

    def __init__(self, estimator_name: str) -> None:
        super().__init__(
            f"This {estimator_name} instance is not fitted yet. "
            f"Call 'fit' with appropriate arguments before using this estimator."
        )
        self.estimator_name = estimator_name


class ValidationError(ForgeError):
    """Raised when input validation fails."""

    pass


class ColumnNotFoundError(ValidationError):
    """Raised when a required column is not found in the DataFrame."""

    def __init__(self, column: str, available: list[str] | None = None) -> None:
        message = f"Column '{column}' not found in DataFrame"
        if available:
            message += f". Available columns: {available}"
        super().__init__(message)
        self.column = column
        self.available = available


class InvalidColumnTypeError(ValidationError):
    """Raised when a column has an unexpected type."""

    def __init__(self, column: str, expected: str, actual: str) -> None:
        super().__init__(
            f"Column '{column}' has type '{actual}', expected '{expected}'"
        )
        self.column = column
        self.expected = expected
        self.actual = actual


class ConfigurationError(ForgeError):
    """Raised when there's a configuration error."""

    pass


class FeatureGenerationError(ForgeError):
    """Raised when feature generation fails."""

    pass


class FeatureSelectionError(ForgeError):
    """Raised when feature selection fails."""

    pass


class MissingDependencyError(ForgeError):
    """Raised when an optional dependency is not installed."""

    def __init__(self, package: str, feature: str) -> None:
        super().__init__(
            f"Package '{package}' is required for {feature}. "
            f"Install it with: pip install {package}"
        )
        self.package = package
        self.feature = feature
