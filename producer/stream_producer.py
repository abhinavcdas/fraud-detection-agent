"""Asynchronous Transaction Stream Producer.

Reads raw credit card transactions (sorted by time), injects deterministic synthetic
entities (customer_id, merchant_id, lat/lon coordinates, ISO8601 timestamp), and
produces streaming JSON events to Redpanda / Kafka or In-Memory stream operator.
"""

import os
import sys
import csv
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, AsyncGenerator
from dotenv import load_dotenv

from core.logger import get_logger, bind_tx_context
from core.interfaces import BaseStreamOperator
from operators.stream.stream_factory import get_stream_operator

load_dotenv()

logger = get_logger("stream_producer")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_CSV = DATA_DIR / "creditcard.csv"
DEFAULT_TOPIC = os.getenv("KAFKA_TOPIC_TRANSACTIONS", "transactions")

# Base simulation anchor timestamp
SIMULATION_START = datetime(2026, 9, 18, 0, 0, 0, tzinfo=timezone.utc)

def generate_synthetic_metadata(row_index: int, amount: float, is_fraud: int) -> Dict[str, Any]:
    """Generate deterministic synthetic customer_id, merchant_id, and coordinates.
    
    A controlled customer pool (~2,500 customers) ensures recurrence density
    needed for rolling velocity and deviation features to have meaningful signal.
    """
    # Recurrence pool: 2,500 customers
    customer_id = f"CUST_{row_index % 2500:04d}"
    merchant_id = f"MERCH_{int((row_index * 7) % 500):03d}"
    
    # Base coords around NY metro area with small jitter
    base_lat, base_lon = 40.7128, -74.0060
    lat = base_lat + ((row_index % 100) - 50) * 0.01
    lon = base_lon + (((row_index * 3) % 100) - 50) * 0.01

    # Inject geographic anomaly for fraud cases (simulate cross-country or international jump)
    if is_fraud == 1 and row_index % 2 == 0:
        lat += 10.0 # Far jump
        lon -= 15.0

    return {
        "customer_id": customer_id,
        "merchant_id": merchant_id,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
    }

def format_transaction_event(row: Dict[str, str], row_index: int) -> Dict[str, Any]:
    """Convert raw CSV row to structured streaming transaction event."""
    time_step = float(row.get("Time", 0.0))
    amount = float(row.get("Amount", 0.0))
    is_fraud = int(row.get("Class", 0))

    meta = generate_synthetic_metadata(row_index, amount, is_fraud)
    tx_timestamp = (SIMULATION_START + timedelta(seconds=time_step)).isoformat()

    event = {
        "transaction_id": f"TX_{row_index:07d}",
        "timestamp": tx_timestamp,
        "time_step": time_step,
        "customer_id": meta["customer_id"],
        "merchant_id": meta["merchant_id"],
        "amount": round(amount, 2),
        "lat": meta["lat"],
        "lon": meta["lon"],
        "is_fraud": is_fraud,
    }

    # Include PCA features V1-V28
    for i in range(1, 29):
        col = f"V{i}"
        if col in row:
            event[f"v{i}"] = round(float(row[col]), 6)

    return event

class AsyncStreamProducer:
    """Production stream producer supporting pluggable streaming operators."""

    def __init__(self, stream_operator: Optional[BaseStreamOperator] = None, topic: str = DEFAULT_TOPIC):
        self.stream = stream_operator or get_stream_operator()
        self.topic = topic
        self._running = False

    async def start(self) -> None:
        """Initialize stream operator."""
        await self.stream.start()
        self._running = True
        logger.info("AsyncStreamProducer started. Target topic: '{topic}'", topic=self.topic)

    async def stream_csv(
        self,
        csv_path: Optional[Path] = None,
        delay_sec: float = 0.01,
        limit: Optional[int] = None,
        batch_size: int = 1
    ) -> int:
        """Replay transactions from CSV into the streaming topic."""
        path = csv_path or DEFAULT_CSV
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {path}. Run 'python data/download_data.py' first."
            )

        if not self._running:
            await self.start()

        logger.info("Streaming from {path} | delay={delay}s | limit={limit}",
                    path=path, delay=delay_sec, limit=limit or "unlimited")

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_index, row in enumerate(reader):
                if not self._running:
                    break
                if limit and count >= limit:
                    break

                event = format_transaction_event(row, row_index)
                tx_id = event["transaction_id"]
                cust_id = event["customer_id"]

                tx_logger = bind_tx_context(logger, tx_id, cust_id)
                tx_logger.debug("Producing transaction event | amount=${amt}", amt=event["amount"])

                await self.stream.publish(self.topic, event)
                count += 1

                if count % 100 == 0:
                    logger.info("Stream progress: {count} events published to '{topic}'",
                                count=count, topic=self.topic)

                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

        logger.info("Streaming complete. Total events published: {count}", count=count)
        return count

    async def close(self) -> None:
        """Graceful shutdown."""
        self._running = False
        await self.stream.close()
        logger.info("AsyncStreamProducer closed.")

async def main():
    parser = argparse.ArgumentParser(description="Replay fraud dataset to event stream")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV), help="Path to creditcard.csv")
    parser.add_argument("--limit", type=int, default=50, help="Max events to stream")
    parser.add_argument("--delay", type=float, default=0.01, help="Delay between events in seconds")
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC, help="Target stream topic")
    args = parser.parse_args()

    producer = AsyncStreamProducer(topic=args.topic)
    try:
        await producer.stream_csv(
            csv_path=Path(args.csv),
            delay_sec=args.delay,
            limit=args.limit
        )
    finally:
        await producer.close()

if __name__ == "__main__":
    asyncio.run(main())
