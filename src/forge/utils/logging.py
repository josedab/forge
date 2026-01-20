"""Logging configuration for Forge."""

from __future__ import annotations

import logging
import sys
from typing import Literal

# Module-level logger
_logger: logging.Logger | None = None

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def get_logger(name: str = "forge") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name.,

    Returns:
        Logger instance.

    Example:
        >>> logger = get_logger("forge.generators")
        >>> logger.info("Generating features...")
    """
    return logging.getLogger(name)


def configure_logging(
    level: LogLevel | int = "INFO",
    format_string: str | None = None,
    stream: bool = True,
    filename: str | None = None
) -> logging.Logger:
    """Configure logging for Forge.

    Args:
        level: Logging level.,
        format_string: Custom format string.,
        stream: Whether to log to stdout.,
        filename: Optional file to log to.,

    Returns:
        Configured root forge logger.

    Example:
        >>> configure_logging(level="DEBUG")
        >>> logger = get_logger()
        >>> logger.debug("This will be shown")
    """
    global _logger

    logger = logging.getLogger("forge")

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Stream handler
    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # File handler
    if filename:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Don't propagate to root logger
    logger.propagate = False

    _logger = logger
    return logger


def set_verbosity(verbosity: int) -> None:
    """Set logging verbosity level.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG.

    Example:
        >>> set_verbosity(2)  # Enable debug logging
    """
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    level = level_map.get(verbosity, logging.DEBUG)

    logger = get_logger("forge")
    logger.setLevel(level)

    # Configure if not already configured
    if not logger.handlers:
        configure_logging(level=level)


class LogContext:
    """Context manager for temporary log level changes.

    Example:
        >>> with LogContext(level="DEBUG"):
        ...     # Debug logging enabled here
        ...     do_something()
        >>> # Back to original level
    """

    def __init__(self, level: LogLevel | int = "DEBUG", logger_name: str = "forge"):
        """Initialize context.

        Args:
            level: Temporary log level.,
            logger_name: Logger to modify.
        """
        self.level = level if isinstance(level, int) else getattr(logging, level.upper())
        self.logger_name = logger_name
        self._original_level: int | None = None

    def __enter__(self) -> "LogContext":
        logger = logging.getLogger(self.logger_name)
        self._original_level = logger.level
        logger.setLevel(self.level)
        return self

    def __exit__(self, *args: object) -> None:
        if self._original_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self._original_level)


class ProgressLogger:
    """Simple progress logging for long-running operations.

    Example:
        >>> progress = ProgressLogger(total=100, desc="Processing")
        >>> for i in range(100):
        ...     # do work
        ...     progress.update()
        >>> progress.close()
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        log_interval: int = 10,
        logger_name: str = "forge"
    ):
        """Initialize progress logger.

        Args:
            total: Total number of items.,
            desc: Description of operation.,
            log_interval: Log every N percent.,
            logger_name: Logger to use.
        """
        self.total = total
        self.desc = desc
        self.log_interval = log_interval
        self.logger = logging.getLogger(logger_name)
        self.current = 0
        self._last_logged_pct = -1

    def update(self, n: int = 1) -> None:
        """Update progress.

        Args:
            n: Number of items completed.
        """
        self.current += n

        if self.total <= 0:
            return

        pct = int(100 * self.current / self.total)
        pct_interval = (pct // self.log_interval) * self.log_interval

        if pct_interval > self._last_logged_pct:
            self._last_logged_pct = pct_interval
            self.logger.info(f"{self.desc}: {pct}% ({self.current}/{self.total})")

    def close(self) -> None:
        """Log completion message."""
        self.logger.info(f"{self.desc}: Complete ({self.current}/{self.total})")


def log_dataframe_info(
    df: "object",
    name: str = "DataFrame",
    logger_name: str = "forge"
) -> None:
    """Log basic DataFrame information.

    Args:
        df: DataFrame to log info about.,
        name: Name for logging.,
        logger_name: Logger to use.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return

    logger = logging.getLogger(logger_name)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    logger.info(
        f"{name}: {df.shape[0]:,} rows x {df.shape[1]:,} columns "
        f"({memory_mb:.2f} MB)"
    )


def log_transform_info(
    name: str,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    logger_name: str = "forge"
) -> None:
    """Log transformation information.

    Args:
        name: Transformer name.,
        input_shape: Input shape (rows, cols).
        output_shape: Output shape (rows, cols).
        logger_name: Logger to use.
    """
    logger = logging.getLogger(logger_name)
    col_diff = output_shape[1] - input_shape[1]
    sign = "+" if col_diff >= 0 else ""

    logger.info(
        f"{name}: {input_shape} -> {output_shape} ({sign}{col_diff} columns)"
    )


# Initialize default logger on import
_default_logger = logging.getLogger("forge")
if not _default_logger.handlers:
    # Add a null handler to prevent "No handlers" warnings
    _default_logger.addHandler(logging.NullHandler())
