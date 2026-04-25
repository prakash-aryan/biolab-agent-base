"""structlog + stdlib logging wired up for JSON output."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str | None = None) -> None:
    effective = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, effective, logging.INFO),
        force=True,
    )

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, effective, logging.INFO),
        ),
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
