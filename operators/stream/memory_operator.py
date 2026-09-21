"""In-Memory Streaming Operator using asyncio Queues.

Provides an asynchronous event stream for standalone execution, unit testing,
and local verification without external Kafka/Redpanda infrastructure.
"""

import asyncio
from typing import Dict, Any, AsyncGenerator
from core.interfaces import BaseStreamOperator
from core.logger import get_logger

logger = get_logger("memory_stream")

class MemoryStreamOperator(BaseStreamOperator):
    """In-memory asyncio queue based event bus."""

    def __init__(self):
        self._topics: Dict[str, asyncio.Queue] = {}
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("Started In-Memory stream operator.")

    def _get_queue(self, topic: str) -> asyncio.Queue:
        if topic not in self._topics:
            self._topics[topic] = asyncio.Queue()
        return self._topics[topic]

    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        if not self._running:
            await self.start()
        queue = self._get_queue(topic)
        await queue.put(message.copy())
        logger.debug("Published event to memory topic '{topic}' | id={tx_id}",
                     topic=topic, tx_id=message.get("transaction_id"))

    async def consume(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._running:
            await self.start()
        queue = self._get_queue(topic)
        while self._running:
            try:
                # Wait for next message or timeout to check running flag
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield message
                queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def close(self) -> None:
        self._running = False
        logger.info("Closed In-Memory stream operator.")
