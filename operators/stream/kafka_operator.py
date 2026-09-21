"""Kafka / Redpanda Streaming Operator for production streaming workloads."""

import os
import json
import asyncio
from typing import Dict, Any, AsyncGenerator
from core.interfaces import BaseStreamOperator
from core.logger import get_logger

logger = get_logger("kafka_stream")

class KafkaStreamOperator(BaseStreamOperator):
    """Kafka/Redpanda stream operator with async non-blocking wrappers."""

    def __init__(self, bootstrap_servers: str = None):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        self.producer = None
        self.consumer = None
        self._running = False

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        def _init_producer():
            try:
                from kafka import KafkaProducer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    acks="all",
                    retries=3
                )
                self._running = True
                logger.info("Connected to Kafka producer at: {servers}", servers=self.bootstrap_servers)
            except Exception as e:
                logger.error("Failed to connect to Kafka at {servers}: {err}",
                             servers=self.bootstrap_servers, err=str(e))
                raise
        await loop.run_in_executor(None, _init_producer)

    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        if not self.producer:
            await self.start()
        loop = asyncio.get_running_loop()
        def _sync_send():
            future = self.producer.send(topic, message)
            self.producer.flush()
            return future.get(timeout=5)
        await loop.run_in_executor(None, _sync_send)

    async def consume(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        from kafka import KafkaConsumer
        loop = asyncio.get_running_loop()
        
        def _get_consumer():
            return KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="fraud_detection_group",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=1000
            )

        self.consumer = await loop.run_in_executor(None, _get_consumer)
        self._running = True
        logger.info("Kafka consumer subscribed to topic '{topic}'", topic=topic)

        while self._running:
            def _poll():
                try:
                    return self.consumer.poll(timeout_ms=500)
                except Exception as e:
                    logger.warning("Error polling Kafka topic {topic}: {err}", topic=topic, err=str(e))
                    return {}

            msg_pack = await loop.run_in_executor(None, _poll)
            for tp, messages in msg_pack.items():
                for msg in messages:
                    yield msg.value

    async def close(self) -> None:
        self._running = False
        loop = asyncio.get_running_loop()
        def _shutdown():
            if self.producer:
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            logger.info("Closed Kafka stream connections.")
        await loop.run_in_executor(None, _shutdown)
