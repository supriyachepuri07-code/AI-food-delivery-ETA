"""
Centralized logging configuration for the project.
"""

import logging
from pathlib import Path

from config.paths import LOGS_DIR
from config.settings import LOG_LEVEL


# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log file path
LOG_FILE = LOGS_DIR / "application.log"

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for the given module.
    """
    return logging.getLogger(name)