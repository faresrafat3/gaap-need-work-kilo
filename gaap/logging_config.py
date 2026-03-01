"""
GAAP Logging Configuration
==========================

Structured JSON logging with rotation, correlation IDs, and module-level configuration.

Features:
- Structured JSON logging
- Log rotation by size and time
- Different log levels per module
- Correlation IDs for distributed tracing
- Colored console output for development
- Async logging support

Usage:
    from gaap.logging_config import get_logger, get_correlation_id, set_correlation_id

    # Get a structured logger
    logger = get_logger("gaap.api.chat")

    # Set correlation ID for request tracing
    set_correlation_id("req-12345")

    # Log with structured data
    logger.info("Processing request", extra={
        "user_id": "user-123",
        "session_id": "sess-456",
        "request_path": "/api/chat"
    })
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Context variable for correlation ID
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_request_id: ContextVar[str] = ContextVar("request_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")


@dataclass
class LogContext:
    """Logging context for correlation and tracing."""

    correlation_id: str = ""
    request_id: str = ""
    session_id: str = ""
    user_id: str = ""
    client_ip: str = ""
    service: str = "gaap"
    version: str = "1.0.0"
    environment: str = "development"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id or get_correlation_id(),
            "request_id": self.request_id or get_request_id(),
            "session_id": self.session_id or get_session_id(),
            "user_id": self.user_id,
            "client_ip": self.client_ip,
            "service": self.service,
            "version": self.version,
            "environment": self.environment,
            **self.extra,
        }


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        indent: Optional[int] = None,
    ):
        super().__init__(fmt, datefmt, style, validate)
        self.indent = indent

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source": {
                "file": record.filename,
                "line": record.lineno,
                "function": record.funcName,
                "module": record.module,
            },
        }

        # Add correlation IDs
        correlation_id = get_correlation_id()
        if correlation_id:
            log_dict["correlation_id"] = correlation_id

        request_id = get_request_id()
        if request_id:
            log_dict["request_id"] = request_id

        session_id = get_session_id()
        if session_id:
            log_dict["session_id"] = session_id

        # Add extra fields from record
        if hasattr(record, "user_id") and record.user_id:
            log_dict["user_id"] = record.user_id

        if hasattr(record, "extra") and record.extra:
            log_dict.update(record.extra)

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        # Add stack info if present
        if record.stack_info:
            log_dict["stack"] = self.formatStack(record.stack_info)

        return json.dumps(log_dict, default=str, indent=self.indent)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        """Format with colors."""
        levelname = record.levelname

        if self.use_colors and levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        formatted = super().format(record)
        record.levelname = levelname  # Restore original

        return formatted


class ContextFilter(logging.Filter):
    """Filter to inject correlation IDs into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation IDs to record."""
        record.correlation_id = get_correlation_id()
        record.request_id = get_request_id()
        record.session_id = get_session_id()
        return True


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID."""
    _correlation_id.set(correlation_id)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return f"corr-{uuid.uuid4().hex[:12]}"


def get_request_id() -> str:
    """Get current request ID."""
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    """Set request ID."""
    _request_id.set(request_id)


def generate_request_id() -> str:
    """Generate a new request ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


def get_session_id() -> str:
    """Get current session ID."""
    return _session_id.get()


def set_session_id(session_id: str) -> None:
    """Set session ID."""
    _session_id.set(session_id)


@contextmanager
def correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """
    Context manager for correlation IDs.

    Usage:
        with correlation_context(request_id="req-123"):
            logger.info("Processing request")
            # All logs in this block will have the request_id
    """
    old_corr = get_correlation_id()
    old_req = get_request_id()
    old_sess = get_session_id()

    if correlation_id:
        set_correlation_id(correlation_id)
    if request_id:
        set_request_id(request_id)
    if session_id:
        set_session_id(session_id)

    try:
        yield
    finally:
        set_correlation_id(old_corr)
        set_request_id(old_req)
        set_session_id(old_sess)


# Module log levels configuration
DEFAULT_MODULE_LEVELS = {
    "gaap": "INFO",
    "gaap.api": "INFO",
    "gaap.core": "WARNING",
    "gaap.providers": "INFO",
    "gaap.metrics": "WARNING",
    "gaap.observability": "WARNING",
    "gaap.memory": "WARNING",
    "gaap.healing": "INFO",
    "gaap.routing": "WARNING",
    "uvicorn": "INFO",
    "fastapi": "WARNING",
    "sqlalchemy": "WARNING",
    "asyncio": "WARNING",
}


def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    module_levels: Optional[dict[str, str]] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    colored_console: bool = True,
) -> None:
    """
    Setup structured logging.

    Args:
        level: Default log level
        json_format: Use JSON formatting for file logs
        log_dir: Directory for log files
        max_bytes: Max log file size before rotation
        backup_count: Number of backup files to keep
        module_levels: Dict of module names to log levels
        enable_console: Enable console logging
        enable_file: Enable file logging
        colored_console: Use colored console output
    """
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Context filter
    context_filter = ContextFilter()

    # Module levels
    levels = {**DEFAULT_MODULE_LEVELS, **(module_levels or {})}
    for module, mod_level in levels.items():
        logging.getLogger(module).setLevel(getattr(logging, mod_level.upper()))

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        if colored_console and not json_format:
            console_formatter = ColoredFormatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        elif json_format:
            console_formatter = JSONFormatter()
        else:
            console_formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(context_filter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Main log file
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / "gaap.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_formatter = (
            JSONFormatter()
            if json_format
            else logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(context_filter)
        root_logger.addHandler(file_handler)

        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            log_path / "gaap.error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        error_handler.addFilter(context_filter)
        root_logger.addHandler(error_handler)

    logging.info(f"Logging configured (level={level}, json={json_format})")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


class StructuredLogger:
    """Wrapper for structured logging with context."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        """Internal log method with structured data."""
        log_extra = {"extra": {**(extra or {}), **kwargs}}
        self._logger.log(level, message, extra=log_extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, extra={"extra": kwargs})


def get_structured_logger(name: str) -> StructuredLogger:
    """Get a structured logger."""
    return StructuredLogger(get_logger(name))


# Convenience function for quick setup
def configure_logging(
    environment: str = "development",
    log_level: Optional[str] = None,
) -> None:
    """
    Quick setup for logging based on environment.

    Args:
        environment: Environment name (development, staging, production)
        log_level: Override log level
    """
    is_prod = environment.lower() == "production"

    setup_logging(
        level=log_level or ("INFO" if is_prod else "DEBUG"),
        json_format=is_prod,
        log_dir=os.environ.get("GAAP_LOG_DIR", "/var/log/gaap" if is_prod else "./logs"),
        enable_console=not is_prod
        or os.environ.get("GAAP_ENABLE_CONSOLE_LOG", "false").lower() == "true",
        colored_console=not is_prod,
        module_levels={
            "gaap": "DEBUG" if not is_prod else "INFO",
        },
    )


__all__ = [
    "LogContext",
    "JSONFormatter",
    "ColoredFormatter",
    "ContextFilter",
    "StructuredLogger",
    "get_correlation_id",
    "set_correlation_id",
    "generate_correlation_id",
    "get_request_id",
    "set_request_id",
    "generate_request_id",
    "get_session_id",
    "set_session_id",
    "correlation_context",
    "setup_logging",
    "configure_logging",
    "get_logger",
    "get_structured_logger",
    "DEFAULT_MODULE_LEVELS",
]
