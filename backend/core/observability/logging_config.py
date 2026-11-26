"""Modular rotating file logging configuration for API, Agent, and Data Pipelines.

Configures stdout streaming alongside per-domain RotatingFileHandler outputs
(logs/api.log, logs/agent.log, logs/pipeline.log) using one shared handler instance
per file to avoid multi-handler rotation races.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_DOMAIN_LOG_FILES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "api.log",
        (
            "backend.main",
            "backend.routes",
            "backend.services",
            "backend.security",
            "backend.deps",
            "corporate_filing_intelligence.api",
        ),
    ),
    ("agent.log", ("backend.ai",)),
    ("pipeline.log", ("pipelines", "corporate_filing_cli")),
)

_configured_log_dir: Path | None = None


def configure_logging(
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    force: bool = False,
) -> dict[str, Path]:
    """Configure modular rotating file logs and stdout logging idempotently."""
    global _configured_log_dir
    target_dir = Path(log_dir)

    if _configured_log_dir == target_dir and not force:
        return {filename: target_dir / filename for filename, _ in _DOMAIN_LOG_FILES}

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    created_paths: dict[str, Path] = {}
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        root_logger.warning("Could not create log directory '%s': %s", target_dir, exc)
        return created_paths

    for filename, logger_names in _DOMAIN_LOG_FILES:
        file_path = target_dir / filename
        resolved_path = file_path.resolve()
        created_paths[filename] = file_path

        if force:
            for logger_name in logger_names:
                domain_logger = logging.getLogger(logger_name)
                for h in list(domain_logger.handlers):
                    if isinstance(h, RotatingFileHandler):
                        domain_logger.removeHandler(h)
                        h.close()

        # Reuse one shared RotatingFileHandler instance across all loggers targeting the same file
        shared_handler: RotatingFileHandler | None = None
        for logger_name in logger_names:
            for h in logging.getLogger(logger_name).handlers:
                if isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == resolved_path:
                    shared_handler = h
                    break
            if shared_handler is not None:
                break

        if shared_handler is None:
            shared_handler = RotatingFileHandler(
                file_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            shared_handler.setLevel(level)
            shared_handler.setFormatter(formatter)

        for logger_name in logger_names:
            domain_logger = logging.getLogger(logger_name)
            domain_logger.setLevel(level)
            if shared_handler not in domain_logger.handlers:
                domain_logger.addHandler(shared_handler)

    _configured_log_dir = target_dir
    return created_paths
