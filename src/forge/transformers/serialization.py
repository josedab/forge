"""Serialization utilities for Forge transformers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_transformer(
    transformer: Any,
    path: str | Path,
    format: str = "pickle"
) -> None:
    """Save a transformer to disk.

    Args:
        transformer: The transformer to save.,
        path: Path to save to.,
        format: Serialization format ("pickle" or "joblib").,

    Example:
        >>> from forge import AutoFeatureTransformer
        >>> transformer = AutoFeatureTransformer()
        >>> transformer.fit(X, y)
        >>> save_transformer(transformer, "transformer.pkl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "pickle":
        with open(path, "wb") as f:
            pickle.dump(transformer, f)
    elif format == "joblib":
        try:
            import joblib
            joblib.dump(transformer, path)
        except ImportError:
            raise ImportError("joblib is required for joblib format")
    else:
        raise ValueError(f"Unknown format: {format}")


def load_transformer(
    path: str | Path,
    format: str = "pickle"
) -> Any:
    """Load a transformer from disk.

    Args:
        path: Path to load from.,
        format: Serialization format ("pickle" or "joblib").,

    Returns:
        The loaded transformer.

    Example:
        >>> transformer = load_transformer("transformer.pkl")
        >>> X_transformed = transformer.transform(X_new)
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if format == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif format == "joblib":
        try:
            import joblib
            return joblib.load(path)
        except ImportError:
            raise ImportError("joblib is required for joblib format")
    else:
        raise ValueError(f"Unknown format: {format}")


def export_feature_names(
    transformer: Any,
    path: str | Path,
) -> None:
    """Export feature names to a JSON file.

    Args:
        transformer: Fitted transformer with get_feature_names_out.,
        path: Path to save JSON file.
    """
    path = Path(path)

    if not hasattr(transformer, "get_feature_names_out"):
        raise ValueError("Transformer must have get_feature_names_out method")

    feature_names = transformer.get_feature_names_out()

    with open(path, "w") as f:
        json.dump({"features": feature_names}, f, indent=2)


def export_feature_importance(
    transformer: Any,
    path: str | Path,
) -> None:
    """Export feature importance to a CSV file.

    Args:
        transformer: Fitted transformer with get_feature_importance.,
        path: Path to save CSV file.
    """
    path = Path(path)

    if not hasattr(transformer, "get_feature_importance"):
        raise ValueError("Transformer must have get_feature_importance method")

    importance = transformer.get_feature_importance()
    importance.to_csv(path, index=False)


class TransformerCheckpoint:
    """Checkpoint manager for long-running transformations.

    Saves intermediate state during fitting to allow resumption
    in case of failures.

    Example:
        >>> checkpoint = TransformerCheckpoint("checkpoints/")
        >>> if checkpoint.exists("my_transformer"):
        ...     transformer = checkpoint.load("my_transformer")
        ... else:
        ...     transformer = AutoFeatureTransformer()
        ...     transformer.fit(X, y)
        ...     checkpoint.save("my_transformer", transformer)
    """

    def __init__(self, directory: str | Path) -> None:
        """Initialize checkpoint manager.

        Args:
            directory: Directory to store checkpoints.
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, transformer: Any) -> Path:
        """Save a checkpoint.

        Args:
            name: Checkpoint name.,
            transformer: Transformer to save.,

        Returns:
            Path to saved checkpoint.
        """
        path = self.directory / f"{name}.pkl"
        save_transformer(transformer, path)
        return path

    def load(self, name: str) -> Any:
        """Load a checkpoint.

        Args:
            name: Checkpoint name.,

        Returns:
            Loaded transformer.
        """
        path = self.directory / f"{name}.pkl"
        return load_transformer(path)

    def exists(self, name: str) -> bool:
        """Check if a checkpoint exists.

        Args:
            name: Checkpoint name.,

        Returns:
            True if checkpoint exists.
        """
        path = self.directory / f"{name}.pkl"
        return path.exists()

    def delete(self, name: str) -> None:
        """Delete a checkpoint.

        Args:
            name: Checkpoint name.
        """
        path = self.directory / f"{name}.pkl"
        if path.exists():
            path.unlink()

    def list_checkpoints(self) -> list[str]:
        """List all checkpoints.

        Returns:
            List of checkpoint names.
        """
        return [p.stem for p in self.directory.glob("*.pkl")]
