"""Production Structured Logging using Loguru.

Features:
- Dual sinks: Colored console output for development + structured JSON file sink for production.
- Automated log rotation (10MB limit), compression (zip), and retention policy (14 days).
- Contextual binding (transaction_id, customer_id, component).
- Thread-safe and async-safe.
"""

import re
import os
import sys
from pathlib import Path
from loguru import logger

# PCI-DSS 3.4: Mask Primary Account Numbers (PAN) across all sinks
PAN_REGEX = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

def mask_pan(match: re.Match) -> str:
    raw = match.group(0)
    digits = re.sub(r"\D", "", raw)
    if 13 <= len(digits) <= 19:
        return f"{digits[:6]}******{digits[-4:]}"
    return raw

def sanitize_record(record):
    """Patcher to redact unmasked card numbers from message and extra payloads."""
    if record.get("message"):
        record["message"] = PAN_REGEX.sub(mask_pan, str(record["message"]))
    for k, v in record.get("extra", {}).items():
        if isinstance(v, str):
            record["extra"][k] = PAN_REGEX.sub(mask_pan, v)
        elif isinstance(v, dict):
            for dk, dv in v.items():
                if isinstance(dv, str):
                    v[dk] = PAN_REGEX.sub(mask_pan, dv)

# Ensure logs directory exists defensively
LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
file_sink_available = False
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    file_sink_available = True
except Exception:
    pass

LOG_FILE = LOGS_DIR / "fraud_pipeline.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Clear default handler
logger.remove()

# Configure global sanitizer patcher
logger.configure(patcher=sanitize_record)

# 1. Human-readable colored console sink (thread-safe and non-blocking via enqueue)
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
    enqueue=True, # Thread-safe non-blocking console logging
)

# 2. Production JSON file sink with rotation
if file_sink_available:
    try:
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
    except Exception:
        pass

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
