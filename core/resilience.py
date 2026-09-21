"""Failure Isolation, Dead-Letter Queue (DLQ), and Resilience Utilities.

Guarantees that individual component failures (e.g., Groq API timeout, DB hiccup,
or corrupted JSON payload) never crash the stream or drop transactions.
"""

import functools
import json
import traceback
from typing import Callable, Any, Dict, Optional
from pathlib import Path
from datetime import datetime, timezone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.logger import get_logger

logger = get_logger("resilience")

DLQ_DIR = Path(__file__).resolve().parent.parent / "logs" / "dlq"
DLQ_DIR.mkdir(parents=True, exist_ok=True)

class DeadLetterQueue:
    """Dead-Letter Queue handler capturing poisoned or unprocessable messages."""

    def __init__(self, dlq_file: Path = None):
        self.dlq_file = dlq_file or (DLQ_DIR / "poisoned_messages.jsonl")

    def route_to_dlq(self, payload: Any, error: Exception, context: Dict[str, Any] = None):
        """Record poisoned message with error details without halting execution."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "context": context or {}
        }
        logger.error(
            "Transaction routed to DLQ | error={error_type}: {error_msg} | context={context}",
            error_type=entry["error_type"],
            error_msg=entry["error_message"],
            context=entry["context"]
        )
        with open(self.dlq_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

# Global DLQ instance
global_dlq = DeadLetterQueue()

def isolated_async_execution(fallback_factory: Callable[[Exception, Dict[str, Any]], Any]):
    """Decorator for async functions: catches exceptions, logs them, and returns a graceful fallback."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                func_name = func.__name__
                logger.warning(
                    "Failure isolated in async component '{func}' | Error: {exc}. Invoking fallback.",
                    func=func_name,
                    exc=str(exc)
                )
                return fallback_factory(exc, kwargs)
        return wrapper
    return decorator

# Reusable exponential backoff retry decorator for external network calls (e.g. Groq)
retry_external_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True
)
