"""Kafka / Redpanda Streaming Operator for production streaming workloads."""

import os
import json
import asyncio
from typing import Dict, Any, AsyncGenerator, Optional
from core.interfaces import BaseStreamOperator
from core.logger import get_logger
from core.resilience import global_dlq

logger = get_logger("kafka_stream")

class KafkaStreamOperator(BaseStreamOperator):
    """Kafka/Redpanda stream operator with async non-blocking wrappers,
    poison-pill DLQ isolation, partition keying, and explicit offset commit support."""

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
                    retries=3,
                    linger_ms=10,  # Enable batching
                    batch_size=32768  # 32KB batch
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
        # Key on customer_id for deterministic partition ordering
        customer_id = message.get("customer_id")
        key_bytes = str(customer_id).encode("utf-8") if customer_id else None

        def _sync_send():
            # Send without per-message flush to leverage Kafka's internal batching
            future = self.producer.send(topic, key=key_bytes, value=message)
            return future.get(timeout=10)
        await loop.run_in_executor(None, _sync_send)

    async def consume(self, topic: str) -> AsyncGenerator[Dict[str, Any], None]:
        from kafka import KafkaConsumer
        loop = asyncio.get_running_loop()
        
        def _get_consumer():
            # NOTE: DO NOT pass value_deserializer here.
            # An unhandled deserialization error inside poll() crashes the consumer loop
            # and prevents offset progression (poison pill). Raw bytes are safely parsed below.
            return KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="fraud_detection_group",
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
                    try:
                        raw_bytes = msg.value
                        if isinstance(raw_bytes, (bytes, bytearray)):
                            payload = json.loads(raw_bytes.decode("utf-8"))
                        elif isinstance(raw_bytes, str):
                            payload = json.loads(raw_bytes)
                        else:
                            payload = raw_bytes
                        yield payload
                    except Exception as exc:
                        # Poison pill encountered: Route directly to DLQ to prevent blocking consumer loop
                        logger.error(
                            "Poison pill detected on {topic} partition {part} offset {offset}: {err}",
                            topic=msg.topic,
                            part=msg.partition,
                            offset=msg.offset,
                            err=str(exc)
                        )
                        global_dlq.route_to_dlq(
                            poison_message={
                                "topic": msg.topic,
                                "partition": msg.partition,
                                "offset": msg.offset,
                                "raw_content": str(msg.value)
                            },
                            error=exc,
                            pipeline_stage="kafka_consumer_deserialization"
                        )

    async def commit_offset(self, topic: str, partition: int, offset: int) -> None:
        """Explicitly commit offset for a topic partition."""
        if not self.consumer:
            return
        from kafka import TopicPartition, OffsetAndMetadata
        tp = TopicPartition(topic, partition)
        loop = asyncio.get_running_loop()
        def _sync_commit():
            self.consumer.commit({tp: OffsetAndMetadata(offset, "")})
        await loop.run_in_executor(None, _sync_commit)

    async def close(self) -> None:
        self._running = False
        loop = asyncio.get_running_loop()
        def _shutdown():
            if self.producer:
                try:
                    self.producer.flush(timeout=5)
                except Exception:
                    pass
                self.producer.close()
            if self.consumer:
                self.consumer.close()
            logger.info("Closed Kafka stream connections.")
        await loop.run_in_executor(None, _shutdown)

