# spotify-streampulse/src/streampulse/logging_setup.py
"""
Logging configuration — rotating file logger with structured JSON lines.
R-LOG-01: One logging setup writing to logs/streampulse.log.
R-LOG-03: Never log passwords, hashes or full stack traces to the UI.
"""
from __future__ import annotations
import json
import logging
import logging.handlers
from datetime import datetime
from typing import Any

from streampulse.config import LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON for Admin System Logs page."""

    def format(self, record: logging.LogRecord) -> str:
        from datetime import timezone
        extra = getattr(record, "_extra", {})
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        payload.update(extra)
        if record.exc_info:
            # Only log exception type and message, not full traceback to file
            exc = record.exc_info[1]
            payload["exc_type"] = type(exc).__name__ if exc else None
            payload["exc_msg"] = str(exc) if exc else None
        return json.dumps(payload, default=str)


def setup_logging() -> None:
    """Initialise the root logger. Call once at app startup."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(logging.INFO)  # library DEBUG chatter (asyncio, PIL, ...) is noise
    logging.getLogger("streampulse").setLevel(logging.DEBUG)

    # Rotating file handler — JSON lines
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(_JsonFormatter())
    file_handler.setLevel(logging.DEBUG)
    root.addHandler(file_handler)

    # Console handler — human-readable for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(module)s: %(message)s")
    )
    console_handler.setLevel(logging.INFO)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call setup_logging() once before using loggers."""
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: str,
    user: str | None = None,
    role: str | None = None,
    **details: Any,
) -> None:
    """Structured event log helper.

    Args:
        logger: Logger instance.
        level: logging.INFO / WARNING / ERROR etc.
        event: Event catalogue name (e.g. 'LOGIN_SUCCESS').
        message: Human-readable message.
        user: Authenticated username (omitted if None).
        role: Role string (omitted if None).
        **details: Small dict of additional context (R-LOG-03: no passwords/hashes).
    """
    extra: dict[str, Any] = {"event": event}
    if user is not None:
        extra["user"] = user
    if role is not None:
        extra["role"] = role
    if details:
        extra["details"] = details

    record = logger.makeRecord(
        logger.name, level, "(log_event)", 0, message, (), None
    )
    record._extra = extra  # type: ignore[attr-defined]
    logger.handle(record)
