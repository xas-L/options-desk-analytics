"""Structured logging utility for the ODX library.


Example
-------
>>> from odx.logging import get_logger
>>> log = get_logger(__name__)
>>> log.info("Surface fitted in %.2f s", elapsed)
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def _configure_root() -> None:
    """One-time setup of the root ``odx`` logger."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger("odx")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``odx`` namespace.
    """
    _configure_root()
    if not name.startswith("odx"):
        name = f"odx.{name}"
    return logging.getLogger(name)
