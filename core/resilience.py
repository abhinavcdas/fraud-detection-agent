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

import threading
import concurrent.futures
import inspect

class DeadLetterQueue:
    """Thread-safe, non-blocking Dead-Letter Queue handler capturing poisoned messages."""

    def __init__(self, dlq_file: Path = None):
        self.dlq_file = dlq_file or (DLQ_DIR / "poisoned_messages.jsonl")
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="dlq_writer")

    def _sync_append(self, line: str):
        with self._lock:
            try:
                self.dlq_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.dlq_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.error("Failed to write to DLQ file: {err}", err=str(e))

    def route_to_dlq(self, payload: Any = None, error: Exception = None, context: Dict[str, Any] = None, **kwargs):
        """Record poisoned message with error details asynchronously without halting execution."""
        actual_payload = payload if payload is not None else kwargs.pop("poison_message", None)
        actual_error = error if error is not None else kwargs.pop("error", None)
        ctx = dict(context or {})
        ctx.update(kwargs)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": actual_payload,
            "error_type": type(actual_error).__name__ if actual_error else "UnknownError",
            "error_message": str(actual_error) if actual_error else "",
            "stack_trace": traceback.format_exc(),
            "context": ctx
        }
        logger.error(
            "Transaction routed to DLQ | error={error_type}: {error_msg} | context={context}",
            error_type=entry["error_type"],
            error_msg=entry["error_message"],
            context=entry["context"]
        )
        line = json.dumps(entry, default=str)
        self._executor.submit(self._sync_append, line)

    def close(self):
        """Flush and shutdown the writer executor cleanly."""
        self._executor.shutdown(wait=True)

# Global DLQ instance
global_dlq = DeadLetterQueue()

def isolated_async_execution(fallback_factory: Callable[..., Any]):
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
                try:
                    sig = inspect.signature(fallback_factory)
                    if len(sig.parameters) >= 3:
                        return fallback_factory(exc, args, kwargs)
                except Exception:
                    pass
                return fallback_factory(exc, kwargs)
        return wrapper
    return decorator

# Reusable exponential backoff retry decorator for external network calls (e.g. Groq)
retry_external_call = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
    reraise=True
)
