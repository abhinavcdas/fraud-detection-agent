"""Production Structured Logging using Loguru.

Features:
- Dual sinks: Colored console output for development + structured JSON file sink for production.
- Automated log rotation (10MB limit), compression (zip), and retention policy (14 days).
- Contextual binding (transaction_id, customer_id, component).
- Thread-safe and async-safe.
"""

import os
import sys
from pathlib import Path
from loguru import logger

# Ensure logs directory exists
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / "fraud_pipeline.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Clear default handler
logger.remove()

# 1. Human-readable colored console sink
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level> "
    "<yellow>{extra}</yellow>"
)

logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=CONSOLE_FORMAT,
    colorize=True,
    backtrace=True,
    diagnose=True,
)

# 2. Production JSON file sink with rotation
logger.add(
    str(LOG_FILE),
    level="DEBUG",
    rotation="10 MB",
    retention="14 days",
    compression="zip",
    serialize=True, # Structured JSON lines
    enqueue=True,    # Thread-safe and asynchronous logging
    backtrace=True,
    diagnose=False,  # Avoid leaking sensitive memory state in production logs
)

def get_logger(component_name: str = None):
    """Obtain a logger instance bound to a specific component/module name."""
    if component_name:
        return logger.bind(component=component_name)
    return logger

def bind_tx_context(base_logger, transaction_id: str, customer_id: str = None):
    """Bind transaction metadata for end-to-end distributed tracing."""
    context = {"transaction_id": transaction_id}
    if customer_id:
        context["customer_id"] = customer_id
    return base_logger.bind(**context)
