"""Stream Operator Factory.

Instantiates:
- 'kafka'  -> KafkaStreamOperator (Redpanda / Apache Kafka)
- 'memory' -> MemoryStreamOperator (Async In-Memory Queue for local dev & testing)
"""

import os
from core.interfaces import BaseStreamOperator
from operators.stream.memory_operator import MemoryStreamOperator
from operators.stream.kafka_operator import KafkaStreamOperator
from core.logger import get_logger

logger = get_logger("stream_factory")

def get_stream_operator(backend: str = None) -> BaseStreamOperator:
    """Factory method to get the configured streaming operator."""
    selected = (backend or os.getenv("STREAM_BACKEND", "memory")).lower().strip()
    
    logger.info("Initializing stream operator: {backend}", backend=selected)
    
    if selected == "kafka":
        return KafkaStreamOperator()
    elif selected == "memory":
        return MemoryStreamOperator()
    else:
        logger.warning("Unknown stream backend '{backend}'. Falling back to 'memory'.", backend=selected)
        return MemoryStreamOperator()
