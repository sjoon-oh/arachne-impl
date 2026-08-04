"""arachne.workload.logging_utils

Central logging setup for arachne.workload. Every module in this package
logs through a child of the shared "arachne.workload" logger (via
get_logger(__name__)), so a single call to configure_logging() in a driver
script controls verbosity for the whole pipeline (cluster fit/assign
progress, per-pool apportionment, groundtruth batch progress, pipeline
stage transitions, etc.) without touching library code.

Library modules never attach handlers themselves (standard logging
practice) -- only configure_logging() does, and only when a caller
explicitly asks for console output.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "arachne.workload"

# A library should not print anything until a caller opts in; attaching a
# NullHandler up front silences Python's "no handlers found" warning if
# configure_logging() is never called.
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())


def get_logger(module_name: str) -> logging.Logger:
    """Returns a child logger under the shared "arachne.workload" namespace.

    Call as get_logger(__name__) from any module in this package.
    """
    if module_name.startswith(LOGGER_NAME):
        return logging.getLogger(module_name)
    return logging.getLogger(f"{LOGGER_NAME}.{module_name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Attaches a single timestamped stream handler to the shared
    "arachne.workload" logger, printing to stdout. Call this once at the
    start of a script (e.g. before constructing a StreamingWorkloadOrganizer) to see
    progress/debug output; safe to call more than once (does not attach
    duplicate handlers).
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    already_configured = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    )
    if not already_configured:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.propagate = False
