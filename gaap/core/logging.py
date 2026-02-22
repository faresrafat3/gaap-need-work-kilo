"""
Centralized Logging Module for GAAP

Provides unified logging configuration with:

Features:
    - Structured logging support (structlog)
    - JSON output for production environments
    - Console output for development
    - Correlation IDs for request tracing
    - Graceful degradation if structlog unavailable

Usage:
    from gaap.core.logging import get_logger, configure_logging

    # Configure global logging
    configure_logging(level=logging.INFO, json_format=False)

    # Get logger
    logger = get_logger("gaap.provider.groq")
    logger.info("Processing request", model="llama-3.3-70b", tokens=100)

Environment Variables:
    GAAP_LOG_LEVEL: Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    GAAP_LOG_FORMAT: Set format (console, json)
"""

import logging
import os
import sys
from datetime import datetime, timezone
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

# =============================================================================
# Optional Dependencies
# =============================================================================

if TYPE_CHECKING:
    import structlog
else:

    class StructLogProcessor(Protocol):
        def __call__(
            self, logger: Any, method_name: str, event_dict: dict[str, Any]
        ) -> dict[str, Any]: ...

    class StructLogger(Protocol):
        def debug(self, msg: str, **kwargs: Any) -> None: ...
        def info(self, msg: str, **kwargs: Any) -> None: ...
        def warning(self, msg: str, **kwargs: Any) -> None: ...
        def error(self, msg: str, **kwargs: Any) -> None: ...

    class structlog:
        contextvars: Any
        processors: Any
        dev: Any

        @staticmethod
        def configure(**kwargs: Any) -> None: ...
        @staticmethod
        def get_logger(name: str) -> StructLogger: ...
        @staticmethod
        def make_filtering_bound_logger(level: int) -> type: ...

        class PrintLoggerFactory:
            def __init__(self) -> None: ...


try:
    import structlog as _structlog

    STRUCTLOG_AVAILABLE = True
    structlog = _structlog
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None

try:
    import orjson

    JSON_LIB: Any = orjson
except ImportError:
    import json

    JSON_LIB = json

# =============================================================================
# Constants
# =============================================================================

DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_FORMAT = "console"
MAX_CACHE_SIZE = 128


# =============================================================================
# Logger Class
# =============================================================================


class GAAPLogger:
    """
    Centralized logger for GAAP system.

    Provides unified logging interface with support for:
    - Standard logging levels (debug, info, warning, error, critical)
    - Exception logging with traceback
    - Structured data via kwargs
    - JSON formatting for production

    Attributes:
        name: Logger name
        level: Logging level
        json_format: Whether JSON formatting is enabled

    Usage:
        >>> logger = get_logger("gaap.my_module")
        >>> logger.info("Processing started", task_id="123")
        >>> logger.error("Task failed", error="timeout")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        json_format: bool = False,
    ) -> None:
        """
        Initialize GAAP logger.

        Args:
            name: Logger name (typically module name like "gaap.provider.groq")
            level: Logging level (default: INFO)
            json_format: Enable JSON formatting (default: False)
        """
        self.name = name
        self.level = level
        self.json_format = json_format or os.getenv("GAAP_LOG_FORMAT", DEFAULT_LOG_FORMAT) == "json"
        self._logger = logging.getLogger(name)
        self._setup_logger()

        if STRUCTLOG_AVAILABLE and structlog:
            self._setup_structlog()

    def _setup_logger(self) -> None:
        """
        Configure standard Python logger.

        Sets up:
        - Stream handler (stdout)
        - Formatter (JSON or console)
        - Log level
        """
        if self._logger.handlers:
            return

        handler = logging.StreamHandler(sys.stdout)

        formatter: logging.Formatter
        if self.json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(self.level)

    def _setup_structlog(self) -> None:
        """
        Configure structlog if available.

        Provides enhanced logging with:
        - Context variables
        - Stack info rendering
        - Timestamps
        - Console or JSON output
        """
        if not structlog:
            return

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                (
                    structlog.processors.JSONRenderer()
                    if self.json_format
                    else structlog.dev.ConsoleRenderer()
                ),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(self.level),
            logger_factory=structlog.PrintLoggerFactory(),
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug message.

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> logger.debug("Cache miss", key="user_123", ttl=300)
        """
        self._logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info message.

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> logger.info("Request processed", tokens=100, latency_ms=250)
        """
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning message.

        Use for:
        - Deprecated features
        - Recoverable errors
        - Unexpected but handled conditions

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> logger.warning("Rate limit approaching", remaining=10)
        """
        self._logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error message.

        Use for:
        - Operation failures
        - Recoverable errors
        - Issues requiring attention

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> logger.error("API call failed", status_code=500, retry_after=60)
        """
        self._logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """
        Log critical message.

        Use for:
        - System failures
        - Data corruption
        - Immediate action required

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> logger.critical("Database connection lost", db="primary")
        """
        self._logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """
        Log exception with full traceback.

        Automatically includes exception details from current context.
        Should be called within an except block.

        Args:
            message: Log message string
            **kwargs: Additional structured data to include

        Example:
            >>> try:
            ...     process_data()
            ... except Exception as e:
            ...     logger.exception("Processing failed", task_id="123")
        """
        self._logger.exception(message, extra=kwargs)


# =============================================================================
# Formatters
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Produces JSON-formatted log entries suitable for:
    - Production logging systems (ELK, Splunk, etc.)
    - Automated log analysis
    - Structured querying

    Example output:
        {
            "timestamp": "2026-02-17T10:30:00.000000",
            "level": "INFO",
            "logger": "gaap.provider.groq",
            "message": "Request processed",
            "extra": {"tokens": 100, "latency_ms": 250}
        }
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extra_fields = {
            k: v
            for k, v in record.__dict__.items()
            if k
            not in (
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "asctime",
            )
        }
        if extra_fields:
            log_data["extra"] = extra_fields

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(JSON_LIB, "dumps"):
            result = JSON_LIB.dumps(log_data)
            if isinstance(result, bytes):
                return result.decode()
            return str(result)
        return str(JSON_LIB.dumps(log_data))


# =============================================================================
# Factory Functions
# =============================================================================


@lru_cache(maxsize=MAX_CACHE_SIZE)
def get_logger(name: str, level: int | None = None) -> GAAPLogger:
    """
    Get or create a logger instance.

    Uses LRU cache for efficient logger retrieval.

    Args:
        name: Logger name (typically __name__ of calling module)
        level: Optional log level override (default: from GAAP_LOG_LEVEL env)

    Returns:
        Configured GAAPLogger instance

    Example:
        >>> logger = get_logger("gaap.provider.groq")
        >>> logger.info("Request processed", tokens=100, latency_ms=250)
    """
    log_level = level or getattr(
        logging, os.getenv("GAAP_LOG_LEVEL", "INFO").upper(), DEFAULT_LOG_LEVEL
    )
    json_format = os.getenv("GAAP_LOG_FORMAT", DEFAULT_LOG_FORMAT) == "json"

    return GAAPLogger(name, log_level, json_format)


def set_log_level(level: int) -> None:
    """
    Set global log level for all GAAP loggers.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> set_log_level(logging.DEBUG)  # Enable debug logging
        >>> set_log_level(logging.WARNING)  # Only warnings and above
    """
    logging.getLogger("gaap").setLevel(level)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """
    Configure GAAP logging globally.

    Should be called once at application startup.

    Args:
        level: Log level (default: INFO)
        json_format: Enable JSON formatting for production (default: False)
        include_timestamp: Include timestamp in console format (default: True)

    Example:
        >>> # Development logging
        >>> configure_logging(level=logging.DEBUG, json_format=False)

        >>> # Production logging
        >>> configure_logging(level=logging.INFO, json_format=True)
    """
    root_logger = logging.getLogger("gaap")
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        fmt = (
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            if include_timestamp
            else "%(name)s - %(levelname)s - %(message)s"
        )
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "get_logger",
    "set_log_level",
    "configure_logging",
    "GAAPLogger",
    "JSONFormatter",
]


def get_standard_logger(name: str, level: int | None = None) -> logging.Logger:
    """
    Get a standard Python logger with GAAP formatting.

    This is a lightweight alternative to GAAPLogger for modules
    that need simple logging without structured logging features.

    Args:
        name: Logger name (typically __name__ of calling module)
        level: Optional log level override

    Returns:
        Standard logging.Logger instance with GAAP formatting

    Example:
        >>> from gaap.core.logging import get_standard_logger
        >>> logger = get_standard_logger(__name__)
        >>> logger.info("Processing started")
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_level = level or getattr(logging, os.getenv("GAAP_LOG_LEVEL", "INFO").upper(), logging.INFO)
    logger.setLevel(log_level)

    return logger


__all__.append("get_standard_logger")
