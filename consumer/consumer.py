"""Asynchronous Stream Consumer Service.

Consumes transaction messages from Kafka/Redpanda or In-Memory stream operator,
queries customer history from storage, executes real-time feature engineering,
and persists both raw transactions and engineered features to PostgreSQL or SQLite.

Features:
- Complete decoupling via Strategy Pattern (BaseStreamOperator + BaseStorageOperator).
- Dead-Letter Queue (DLQ) protection against poisoned payloads.
- Structured Loguru distributed tracing with transaction metadata binding.
- Graceful shutdown handles.
"""

import os
import sys
import json
import asyncio
import argparse
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from core.logger import get_logger, bind_tx_context
from core.interfaces import BaseStreamOperator, BaseStorageOperator
from core.resilience import global_dlq
from operators.stream.stream_factory import get_stream_operator
from operators.storage.storage_factory import get_storage_operator
from consumer.feature_engineering import compute_rolling_features

load_dotenv()

logger = get_logger("stream_consumer")

DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")

class AsyncStreamConsumer:
    """Production streaming consumer with real-time feature engineering."""

    def __init__(
        self,
        stream_operator: Optional[BaseStreamOperator] = None,
        storage_operator: Optional[BaseStorageOperator] = None,
        topic: str = DEFAULT_TOPIC
    ):
        self.stream = stream_operator or get_stream_operator()
        self.storage = storage_operator or get_storage_operator()
        self.topic = topic
        self._running = False

    async def start(self) -> None:
        """Initialize underlying stream and storage connections."""
        await self.storage.initialize()
        await self.stream.start()
        self._running = True
        logger.info("AsyncStreamConsumer active on topic '{topic}'", topic=self.topic)

    async def process_message(self, raw_tx: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event: history lookup -> feature engineering -> persistence."""
        tx_id = raw_tx.get("transaction_id")
        cust_id = raw_tx.get("customer_id")

        if not tx_id or not cust_id:
            raise ValueError("Malformed transaction message: missing 'transaction_id' or 'customer_id'")

        tx_logger = bind_tx_context(logger, tx_id, cust_id)
        tx_logger.debug("Processing transaction event | amount=${amt}", amt=raw_tx.get("amount"))

        # 1. Fetch historical transactions for rolling calculation
        history = await self.storage.get_customer_history(cust_id, limit=50)

        # 2. Compute rolling features
        features = compute_rolling_features(raw_tx, history)

        # 3. Persist raw transaction and engineered features concurrently
        await self.storage.save_raw_transaction(raw_tx)
        await self.storage.save_engineered_features(features)

        tx_logger.info(
            "Features engineered & stored | v5m={v5m} | v60m={v60m} | dev={dev:.2f} | geo_dist={geo:.1f}km",
            v5m=features["velocity_5m"],
            v60m=features["velocity_60m"],
            dev=features["amount_deviation"],
            geo=features["geo_distance_km"]
        )

        return features

    async def run(self, max_events: Optional[int] = None) -> int:
        """Main streaming consumption loop with fault isolation."""
        if not self._running:
            await self.start()

        processed_count = 0
        logger.info("Starting consumption loop (max_events={max_events})", max_events=max_events or "unlimited")

        async for message in self.stream.consume(self.topic):
            if not self._running:
                break
            try:
                await self.process_message(message)
                processed_count += 1

                if max_events and processed_count >= max_events:
                    logger.info("Reached target limit of {count} events. Halting consumer.", count=processed_count)
                    break

            except Exception as exc:
                # Isolate failure to DLQ so the stream consumer loop NEVER halts!
                global_dlq.route_to_dlq(
                    payload=message,
                    error=exc,
                    context={"topic": self.topic, "processed_count": processed_count}
                )
                logger.error("Error processing stream message. Isolated to DLQ; consumer continuing.")

        logger.info("Consumer loop finished. Total events processed: {count}", count=processed_count)
        return processed_count

    async def close(self) -> None:
        """Gracefully close consumer and storage."""
        self._running = False
        await self.stream.close()
        await self.storage.close()
        logger.info("AsyncStreamConsumer shut down cleanly.")

async def main():
    parser = argparse.ArgumentParser(description="Consume transactions from stream and engineer features")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Stream topic")
    parser.add_argument("--max-events", type=int, default=50, help="Max events to process")
    args = parser.parse_args()

    consumer = AsyncStreamConsumer(topic=args.topic)
    try:
        await consumer.run(max_events=args.max_events)
    finally:
        await consumer.close()

if __name__ == "__main__":
    asyncio.run(main())
