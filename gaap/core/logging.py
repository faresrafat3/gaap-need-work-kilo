"""
Centralized Logging Module for GAAP

Provides unified logging configuration with:
- Structured logging support (structlog)
- JSON output for production
- Console output for development
- Correlation IDs for request tracing
"""

import logging
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import Any

try:
    import structlog

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import orjson

    JSON_LIB = orjson
except ImportError:
    import json

    JSON_LIB = json


class GAAPLogger:
    """
    Centralized logger for GAAP system

    Usage:
        from gaap.core.logging import get_logger

        logger = get_logger("gaap.provider.groq")
        logger.info("Processing request", model="llama-3.3-70b")
    """

    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        json_format: bool = False,
    ):
        self.name = name
        self.level = level
        self.json_format = json_format or os.getenv("GAAP_LOG_FORMAT", "console") == "json"
        self._logger = logging.getLogger(name)
        self._setup_logger()

        if STRUCTLOG_AVAILABLE:
            self._setup_structlog()

    def _setup_logger(self) -> None:
        """Configure standard Python logger"""
        if self._logger.handlers:
            return

        handler = logging.StreamHandler(sys.stdout)

        if self.json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(self.level)

    def _setup_structlog(self) -> None:
        """Configure structlog if available"""
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

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self._logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self._logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self._logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self._logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self._logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback"""
        self._logger.exception(message, extra=kwargs)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra"):
            log_data["extra"] = record.extra

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(JSON_LIB, "dumps"):
            return (
                JSON_LIB.dumps(log_data).decode()
                if isinstance(JSON_LIB.dumps(log_data), bytes)
                else JSON_LIB.dumps(log_data)
            )
        return JSON_LIB.dumps(log_data)


@lru_cache(maxsize=128)
def get_logger(name: str, level: int | None = None) -> GAAPLogger:
    """
    Get or create a logger instance

    Args:
        name: Logger name (usually module name)
        level: Optional log level override

    Returns:
        Configured GAAPLogger instance

    Example:
        >>> logger = get_logger("gaap.provider.groq")
        >>> logger.info("Request processed", tokens=100)
    """
    log_level = level or getattr(logging, os.getenv("GAAP_LOG_LEVEL", "INFO").upper())
    json_format = os.getenv("GAAP_LOG_FORMAT", "console") == "json"

    return GAAPLogger(name, log_level, json_format)


def set_log_level(level: int) -> None:
    """Set global log level for all GAAP loggers"""
    logging.getLogger("gaap").setLevel(level)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    include_timestamp: bool = True,
) -> None:
    """
    Configure GAAP logging globally

    Args:
        level: Log level (default: INFO)
        json_format: Use JSON format for logs
        include_timestamp: Include timestamp in logs
    """
    root_logger = logging.getLogger("gaap")
    root_logger.setLevel(level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

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


# Convenience exports
__all__ = [
    "get_logger",
    "set_log_level",
    "configure_logging",
    "GAAPLogger",
]
