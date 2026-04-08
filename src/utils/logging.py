"""Lightweight structured logging for experiments."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger writing to stderr.

    Parameters
    ----------
    name
        Logger name (typically ``__name__``).
    level
        Logging level.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def configure_root(level: int = logging.INFO) -> None:
    """Set root logging level (optional convenience)."""
    logging.basicConfig(level=level)
