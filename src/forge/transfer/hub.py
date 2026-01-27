"""Transfer hub for publishing and discovering reusable pipelines.

Manages a registry of fitted feature engineering pipelines that can
be adapted and applied to new datasets based on similarity matching.

Example:
    >>> from forge.transfer.hub import TransferHub
    >>> hub = TransferHub(storage_path="./transfer_hub")
    >>> hub.publish("user_pipeline", pipeline, source_fingerprint)
    >>> matches = hub.discover(target_fingerprint)
    >>> adapted = hub.adapt(matches[0].pipeline_id, target_df)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from forge.transfer.engine import (
    DatasetFingerprint,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineRecord:
    """A published pipeline in the transfer hub.

    Attributes:
        pipeline_id: Unique identifier.
        name: Human-readable name.
        fingerprint: Dataset fingerprint it was trained on.
        transform_names: List of transformation names applied.
        column_mappings: Source column name to fingerprint mapping.
        score: Performance score achieved (if available).
        created_at: Publication timestamp.
        metadata: Additional metadata.
    """

    pipeline_id: str
    name: str
    fingerprint: DatasetFingerprint
    transform_names: list[str] = field(default_factory=list)
    column_mappings: dict[str, str] = field(default_factory=dict)
    score: float | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryMatch:
    """Result of discovering a matching pipeline.

    Attributes:
        pipeline_id: ID of the matching pipeline.
        name: Pipeline name.
        similarity_score: Overall similarity to target dataset.
        column_matches: Mapping of source columns to target columns.
        confidence: Confidence in successful transfer.
    """

    pipeline_id: str
    name: str
    similarity_score: float
    column_matches: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class AdaptationResult:
    """Result of adapting a pipeline to a new dataset.

    Attributes:
        pipeline_id: ID of the adapted pipeline.
        adapted_columns: Mapping of original to adapted columns.
        skipped_columns: Columns that could not be adapted.
        output_features: Number of output features.
        success: Whether adaptation succeeded.
        error: Error message if failed.
    """

    pipeline_id: str
    adapted_columns: dict[str, str] = field(default_factory=dict)
    skipped_columns: list[str] = field(default_factory=list)
    output_features: int = 0
    success: bool = True
    error: str = ""


class TransferHub:
    """Registry for reusable feature engineering pipelines.

    Enables publishing fitted pipelines with their dataset fingerprints
    and discovering/adapting them for new, similar datasets.

    Parameters:
        storage_path: Path to persist the hub data.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._records: dict[str, PipelineRecord] = {}
        self._pipelines: dict[str, Any] = {}
        if self.storage_path:
            self._load()

    def publish(
        self,
        name: str,
        pipeline: Any,
        fingerprint: DatasetFingerprint,
        transform_names: list[str] | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish a pipeline to the hub.

        Args:
            name: Human-readable pipeline name.
            pipeline: Fitted sklearn-compatible pipeline.
            fingerprint: Dataset fingerprint it was trained on.
            transform_names: Names of transformations in the pipeline.
            score: Model performance score (optional).
            metadata: Additional metadata.

        Returns:
            Pipeline ID for future reference.
        """
        pipeline_id = self._generate_id(name, fingerprint)

        record = PipelineRecord(
            pipeline_id=pipeline_id,
            name=name,
            fingerprint=fingerprint,
            transform_names=transform_names or [],
            column_mappings={col: col for col in fingerprint.columns},
            score=score,
            metadata=metadata or {},
        )

        self._records[pipeline_id] = record
        self._pipelines[pipeline_id] = pipeline

        if self.storage_path:
            self._save()

        logger.info("Published pipeline '%s' (id=%s)", name, pipeline_id)
        return pipeline_id

    def discover(
        self,
        target_fingerprint: DatasetFingerprint,
        top_k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[DiscoveryMatch]:
        """Discover pipelines matching a target dataset.

        Args:
            target_fingerprint: Fingerprint of the target dataset.
            top_k: Maximum number of matches to return.
            min_similarity: Minimum similarity threshold.

        Returns:
            List of DiscoveryMatch objects sorted by similarity.
        """
        matches: list[DiscoveryMatch] = []

        for pid, record in self._records.items():
            similarity, col_matches = self._compute_similarity(
                record.fingerprint, target_fingerprint
            )

            if similarity >= min_similarity:
                confidence = similarity * (len(col_matches) / max(len(target_fingerprint.columns), 1))
                matches.append(DiscoveryMatch(
                    pipeline_id=pid,
                    name=record.name,
                    similarity_score=similarity,
                    column_matches=col_matches,
                    confidence=min(confidence, 1.0),
                ))

        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches[:top_k]

    def adapt(
        self,
        pipeline_id: str,
        target_df: pd.DataFrame,
        column_mapping: dict[str, str] | None = None,
    ) -> AdaptationResult:
        """Adapt a published pipeline to a new dataset.

        Renames columns to match the expected input and applies
        the pipeline, handling missing columns gracefully.

        Args:
            pipeline_id: ID of the pipeline to adapt.
            target_df: Target DataFrame.
            column_mapping: Explicit column mapping (source→target).

        Returns:
            AdaptationResult with details.
        """
        if pipeline_id not in self._records:
            return AdaptationResult(
                pipeline_id=pipeline_id,
                success=False,
                error=f"Pipeline '{pipeline_id}' not found",
            )

        record = self._records[pipeline_id]
        pipeline = self._pipelines.get(pipeline_id)
        if pipeline is None:
            return AdaptationResult(
                pipeline_id=pipeline_id,
                success=False,
                error="Pipeline object not available",
            )

        # Build column mapping
        if column_mapping is None:
            # Auto-map: use exact name matches
            source_cols = set(record.fingerprint.columns.keys())
            target_cols = set(target_df.columns)
            column_mapping = {c: c for c in source_cols & target_cols}

        adapted_columns = {}
        skipped_columns = []

        # Rename target columns to match source
        rename_map = {v: k for k, v in column_mapping.items()}
        adapted_df = target_df.rename(columns=rename_map)

        source_cols = list(record.fingerprint.columns.keys())
        for col in source_cols:
            if col in adapted_df.columns:
                adapted_columns[col] = column_mapping.get(col, col)
            else:
                skipped_columns.append(col)

        try:
            # Only transform columns the pipeline expects
            available_cols = [c for c in source_cols if c in adapted_df.columns]
            if available_cols:
                result_df = pipeline.transform(adapted_df[available_cols])
                n_features = result_df.shape[1] if hasattr(result_df, "shape") else 0
            else:
                n_features = 0

            return AdaptationResult(
                pipeline_id=pipeline_id,
                adapted_columns=adapted_columns,
                skipped_columns=skipped_columns,
                output_features=n_features,
                success=True,
            )
        except Exception as e:
            return AdaptationResult(
                pipeline_id=pipeline_id,
                adapted_columns=adapted_columns,
                skipped_columns=skipped_columns,
                success=False,
                error=str(e),
            )

    def get(self, pipeline_id: str) -> PipelineRecord | None:
        """Get a pipeline record by ID."""
        return self._records.get(pipeline_id)

    def list_pipelines(self) -> list[PipelineRecord]:
        """List all published pipelines."""
        return list(self._records.values())

    @property
    def size(self) -> int:
        """Number of published pipelines."""
        return len(self._records)

    def _compute_similarity(
        self,
        source_fp: DatasetFingerprint,
        target_fp: DatasetFingerprint,
    ) -> tuple[float, dict[str, str]]:
        """Compute similarity between two dataset fingerprints."""
        col_matches: dict[str, str] = {}
        total_similarity = 0.0

        for src_name, src_fp_col in source_fp.columns.items():
            best_score = 0.0
            best_match = ""

            for tgt_name, tgt_fp_col in target_fp.columns.items():
                if tgt_name in col_matches.values():
                    continue
                score = src_fp_col.similarity(tgt_fp_col)
                if score > best_score:
                    best_score = score
                    best_match = tgt_name

            if best_score > 0.3 and best_match:
                col_matches[src_name] = best_match
                total_similarity += best_score

        n_source = max(len(source_fp.columns), 1)
        avg_similarity = total_similarity / n_source

        return avg_similarity, col_matches

    def _generate_id(self, name: str, fingerprint: DatasetFingerprint) -> str:
        """Generate a unique pipeline ID."""
        content = f"{name}:{fingerprint.name}:{fingerprint.n_rows}"
        return hashlib.md5(content.encode()).hexdigest()[:12]  # noqa: S324

    def _save(self) -> None:
        """Persist hub metadata (not pipeline objects) to disk."""
        if self.storage_path is None:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        meta_path = self.storage_path / "hub_meta.json"

        data = {}
        for pid, record in self._records.items():
            data[pid] = {
                "pipeline_id": record.pipeline_id,
                "name": record.name,
                "transform_names": record.transform_names,
                "score": record.score,
                "created_at": record.created_at,
                "metadata": record.metadata,
                "fingerprint_name": record.fingerprint.name,
                "fingerprint_rows": record.fingerprint.n_rows,
                "fingerprint_cols": record.fingerprint.n_columns,
            }
        meta_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load hub metadata from disk."""
        if self.storage_path is None:
            return
        meta_path = self.storage_path / "hub_meta.json"
        if not meta_path.exists():
            return
        try:
            data = json.loads(meta_path.read_text())
            for pid, info in data.items():
                fp = DatasetFingerprint(
                    name=info.get("fingerprint_name", ""),
                    n_rows=info.get("fingerprint_rows", 0),
                    n_columns=info.get("fingerprint_cols", 0),
                )
                self._records[pid] = PipelineRecord(
                    pipeline_id=pid,
                    name=info["name"],
                    fingerprint=fp,
                    transform_names=info.get("transform_names", []),
                    score=info.get("score"),
                    created_at=info.get("created_at", ""),
                    metadata=info.get("metadata", {}),
                )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to load hub metadata: %s", e)
