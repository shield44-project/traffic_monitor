"""Centralised logging setup.

Call :func:`get_logger` anywhere in the codebase to obtain a configured logger
that writes to both the console and a rotating file under ``logs/``. Handlers
are attached only once per process to avoid duplicate log lines.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

import config

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("traffic_ai")
    root.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))
    root.propagate = False

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        file_handler = RotatingFileHandler(
            config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # If the log file can't be opened (e.g. read-only FS) keep console only.
        root.warning("Could not open log file %s; logging to console only.",
                     config.LOG_FILE)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger, configuring the root once."""
    _configure_root()
    return logging.getLogger(f"traffic_ai.{name}")
