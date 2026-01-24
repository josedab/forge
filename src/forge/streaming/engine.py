"""Streaming engine that orchestrates incremental feature generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from forge.exceptions import ConfigurationError, ValidationError
from forge.streaming.generators import (
    IncrementalAggregator,
    StreamingFeatureGenerator,
    StreamingWindowAggregator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class StreamingEngine:
    """Orchestrates multiple streaming generators into a unified pipeline.

    The engine manages a set of StreamingFeatureGenerator instances,
    feeding data through all of them and concatenating the results.

    Args:
        generators: List of streaming generators. If None, uses defaults.
        passthrough: Whether to include original columns in output.
        on_event: Optional callback invoked after each batch is processed.

    Example:
        >>> from forge.streaming import StreamingEngine, IncrementalAggregator
        >>> engine = StreamingEngine(generators=[
        ...     IncrementalAggregator(stats=["mean", "std"]),
        ...     StreamingWindowAggregator(window_size=50),
        ... ])
        >>> for batch in data_stream:
        ...     features = engine.process(batch)
        ...     model.predict(features)
    """

    def __init__(
        self,
        generators: list[StreamingFeatureGenerator] | None = None,
        passthrough: bool = True,
        on_event: Callable[[pd.DataFrame], None] | None = None,
    ) -> None:
        if generators is not None:
            self.generators = generators
        else:
            self.generators = [
                IncrementalAggregator(),
                StreamingWindowAggregator(window_size=100),
            ]
        self.passthrough = passthrough
        self.on_event = on_event
        self._n_batches_processed = 0
        self._n_events_processed = 0

    def process(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        """Process a batch of events through all generators.

        Each generator is partial_fit then partial_transform is called.
        Results are concatenated column-wise.

        Args:
            X: Batch of events as a DataFrame.
            y: Optional target variable.

        Returns:
            DataFrame with all generated features.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValidationError(f"Expected DataFrame, got {type(X).__name__}")

        frames: list[pd.DataFrame] = []
        if self.passthrough:
            frames.append(X)

        for gen in self.generators:
            gen.partial_fit(X, y)
            features = gen.partial_transform(X)
            if not features.empty:
                frames.append(features)

        self._n_batches_processed += 1
        self._n_events_processed += len(X)

        if self.on_event is not None:
            self.on_event(X)

        if not frames:
            return pd.DataFrame(index=X.index)

        return pd.concat(frames, axis=1)

    def process_stream(
        self,
        stream: Iterator[pd.DataFrame],
        y_stream: Iterator[pd.Series] | None = None,
    ) -> Iterator[pd.DataFrame]:
        """Process an iterator of batches, yielding feature DataFrames.

        Args:
            stream: Iterator of DataFrames (batches).
            y_stream: Optional iterator of target Series.

        Yields:
            DataFrame with generated features for each batch.
        """
        if y_stream is None:
            for batch in stream:
                yield self.process(batch)
        else:
            for batch, y_batch in zip(stream, y_stream):
                yield self.process(batch, y_batch)

    def get_feature_names_out(self) -> list[str]:
        """Get all output feature names across generators."""
        names: list[str] = []
        for gen in self.generators:
            if gen._is_fitted:
                names.extend(gen.get_feature_names_out())
        return names

    def reset(self) -> None:
        """Reset all generators and counters."""
        for gen in self.generators:
            gen.reset()
        self._n_batches_processed = 0
        self._n_events_processed = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Return processing statistics."""
        return {
            "n_batches_processed": self._n_batches_processed,
            "n_events_processed": self._n_events_processed,
            "n_generators": len(self.generators),
            "n_features": len(self.get_feature_names_out()),
        }

    def add_generator(self, generator: StreamingFeatureGenerator) -> None:
        """Add a generator to the pipeline.

        Args:
            generator: Generator to add.
        """
        if not isinstance(generator, StreamingFeatureGenerator):
            raise ConfigurationError(
                f"Generator must be a StreamingFeatureGenerator, "
                f"got {type(generator).__name__}"
            )
        self.generators.append(generator)
