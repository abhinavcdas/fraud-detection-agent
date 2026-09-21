"""Stream Operator Factory.

Instantiates:
- 'kafka'  -> KafkaStreamOperator (Redpanda / Apache Kafka)
- 'memory' -> MemoryStreamOperator (Async In-Memory Queue for local dev & testing)
"""

import os
import threading
from typing import Dict
from core.interfaces import BaseStreamOperator
from operators.stream.memory_operator import MemoryStreamOperator
from operators.stream.kafka_operator import KafkaStreamOperator
from core.logger import get_logger

logger = get_logger("stream_factory")

_stream_instances: Dict[str, BaseStreamOperator] = {}
_lock = threading.Lock()

def get_stream_operator(backend: str = None) -> BaseStreamOperator:
    """Factory method to get the configured streaming operator (thread-safe singleton)."""
    selected = (backend or os.getenv("STREAM_BACKEND", "memory")).lower().strip()
    
    with _lock:
        if selected in _stream_instances:
            return _stream_instances[selected]

        logger.info("Initializing stream operator: {backend}", backend=selected)
        
        if selected == "kafka":
            operator = KafkaStreamOperator()
        elif selected == "memory":
            operator = MemoryStreamOperator()
        else:
            logger.warning("Unknown stream backend '{backend}'. Falling back to 'memory'.", backend=selected)
            operator = MemoryStreamOperator()
            selected = "memory"
            
        _stream_instances[selected] = operator
        return operator

def reset_stream_operators() -> None:
    """Reset cached singleton stream operators (used mainly in testing)."""
    with _lock:
        _stream_instances.clear()

