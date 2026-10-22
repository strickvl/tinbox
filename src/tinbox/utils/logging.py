"""Logging configuration for Tinbox."""

import sys

import structlog
from rich.console import Console

console = Console()


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure structured logging for the application.

    Args:
        level: The logging level to use. Defaults to "INFO".
        json: Whether to output logs in JSON format. Defaults to False.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.set_exc_info
            if not json
            else structlog.processors.format_exc_info,
            structlog.processors.dict_tracebacks
            if json
            else structlog.processors.ExceptionPrettyPrinter(),
            structlog.processors.JSONRenderer()
            if json
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a logger instance.

    Args:
        name: Optional name for the logger. Defaults to None.

    Returns:
        A structured logger instance.
    """
    return structlog.get_logger(name)
