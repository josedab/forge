"""Utility functions for Forge."""

from __future__ import annotations

from forge.utils.logging import configure_logging, get_logger
from forge.utils.memory import (
    MemoryEstimate,
    check_memory_and_warn,
    estimate_dataframe_size,
    estimate_feature_engineering_memory,
    get_available_memory,
    get_memory_usage_summary,
    process_in_chunks,
    suggest_dtype_optimizations,
)
from forge.utils.parallel import parallel_apply, parallel_transform
from forge.utils.validation import (
    check_is_fitted,
    validate_columns_exist,
    validate_dataframe,
)

__all__ = [
    # Validation
    "check_is_fitted",
    "validate_columns_exist",
    "validate_dataframe",
    # Logging
    "configure_logging",
    "get_logger",
    # Parallel processing
    "parallel_apply",
    "parallel_transform",
    # Memory utilities
    "MemoryEstimate",
    "check_memory_and_warn",
    "estimate_dataframe_size",
    "estimate_feature_engineering_memory",
    "get_available_memory",
    "get_memory_usage_summary",
    "process_in_chunks",
    "suggest_dtype_optimizations",
]
