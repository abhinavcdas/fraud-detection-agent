"""Phase 1 Automated Test Suite: Streaming Data Pipeline & Feature Engineering."""

import pytest
import asyncio
from pathlib import Path
from producer.stream_producer import (
    generate_synthetic_metadata,
    format_transaction_event,
    AsyncStreamProducer
)
from consumer.feature_engineering import (
    haversine_distance,
    compute_rolling_features,
    parse_time_seconds
)
from consumer.consumer import AsyncStreamConsumer
from operators.storage.memory_operator import MemoryStorageOperator
from operators.stream.memory_operator import MemoryStreamOperator

def test_producer_event_formatting():
    """Verify raw CSV row is properly enriched with metadata and formatted as JSON event."""
    raw_row = {
        "Time": "120.5",
        "Amount": "149.99",
        "Class": "0",
        "V1": "-1.2345",
        "V2": "0.6789"
    }
    event = format_transaction_event(raw_row, row_index=42)

    assert event["transaction_id"] == "TX_0000042"
    assert event["customer_id"] == "CUST_0042"
    assert event["merchant_id"].startswith("MERCH_")
    assert event["amount"] == 149.99
    assert event["time_step"] == 120.5
    assert "timestamp" in event
    assert event["v1"] == -1.2345
    assert event["v2"] == 0.6789
    assert event["is_fraud"] == 0

def test_haversine_calculation():
    """Verify geographic distance is accurately calculated."""
    # NYC to Philadelphia is ~130 km
    dist = haversine_distance(40.7128, -74.0060, 39.9526, -75.1652)
    assert 120.0 <= dist <= 140.0

def test_dynamic_rolling_features_evolution():
    """Verify windowed features (velocity_5m, velocity_60m, deviation) evolve accurately."""
    cust_id = "CUST_TEST_VELOCITY"

    # Transaction 1 at t=0s
    tx1 = {
        "transaction_id": "TX_01",
        "customer_id": cust_id,
        "amount": 50.0,
        "time_step": 0.0,
        "lat": 40.7128,
        "lon": -74.0060
    }
    feat1 = compute_rolling_features(tx1, history=[])
    assert feat1["velocity_5m"] == 0
    assert feat1["velocity_60m"] == 0
    assert feat1["amount_deviation"] == 0.0
    assert feat1["geo_distance_km"] == 0.0

    # Transaction 2 at t=60s (1 min later)
    tx2 = {
        "transaction_id": "TX_02",
        "customer_id": cust_id,
        "amount": 75.0,
        "time_step": 60.0,
        "lat": 40.7130,
        "lon": -74.0062
    }
    feat2 = compute_rolling_features(tx2, history=[tx1])
    assert feat2["velocity_5m"] == 1
    assert feat2["velocity_60m"] == 1
    assert feat2["time_since_last_tx_sec"] == 60.0

    # Transaction 3 at t=180s (3 min later) - velocity burst!
    tx3 = {
        "transaction_id": "TX_03",
        "customer_id": cust_id,
        "amount": 100.0,
        "time_step": 180.0,
        "lat": 40.7135,
        "lon": -74.0065
    }
    feat3 = compute_rolling_features(tx3, history=[tx2, tx1])
    assert feat3["velocity_5m"] == 2
    assert feat3["velocity_60m"] == 2
    assert feat3["time_since_last_tx_sec"] == 120.0

    # Transaction 4 at t=700s (> 10 mins later)
    # tx1 (t=0), tx2 (t=60), tx3 (t=180) are outside 5 min (300s) window, but inside 60 min (3600s) window!
    tx4 = {
        "transaction_id": "TX_04",
        "customer_id": cust_id,
        "amount": 60.0,
        "time_step": 700.0,
        "lat": 40.7140,
        "lon": -74.0070
    }
    feat4 = compute_rolling_features(tx4, history=[tx3, tx2, tx1])
    assert feat4["velocity_5m"] == 0   # Past 5 mins has zero prior txs
    assert feat4["velocity_60m"] == 3  # Past 60 mins has 3 prior txs
    assert feat4["time_since_last_tx_sec"] == 520.0

def test_geo_jump_distance_spike():
    """Verify impossible geographic travel jump triggers large haversine distance."""
    tx_nyc = {
        "transaction_id": "TX_NY",
        "customer_id": "CUST_FLY",
        "amount": 100.0,
        "time_step": 100.0,
        "lat": 40.7128,
        "lon": -74.0060 # NYC
    }
    tx_la = {
        "transaction_id": "TX_LA",
        "customer_id": "CUST_FLY",
        "amount": 2500.0,
        "time_step": 160.0, # 1 minute later
        "lat": 34.0522,
        "lon": -118.2437 # Los Angeles
    }
    feat = compute_rolling_features(tx_la, history=[tx_nyc])
    assert feat["geo_distance_km"] > 3900.0 # ~3,935 km between NYC and LA
    assert feat["time_since_last_tx_sec"] == 60.0

@pytest.mark.asyncio
async def test_end_to_end_streaming_pipeline():
    """Verify producer -> stream -> consumer -> storage end-to-end integration."""
    stream = MemoryStreamOperator()
    storage = MemoryStorageOperator()

    producer = AsyncStreamProducer(stream_operator=stream, topic="test_stream_pipe")
    consumer = AsyncStreamConsumer(stream_operator=stream, storage_operator=storage, topic="test_stream_pipe")

    await consumer.start()

    # Produce 10 synthetic transactions
    test_events = []
    for i in range(10):
        event = {
            "transaction_id": f"TX_PIPE_{i:03d}",
            "customer_id": "CUST_PIPE_USER",
            "amount": 50.0 + (i * 10),
            "time_step": float(i * 30), # Every 30 seconds
            "lat": 40.7128,
            "lon": -74.0060,
            "is_fraud": 0
        }
        test_events.append(event)
        await stream.publish("test_stream_pipe", event)

    # Consume all 10 events
    processed = await consumer.run(max_events=10)

    assert processed == 10
    assert len(storage.raw_transactions) == 10
    assert len(storage.engineered_features) == 10

    # Verify that the 10th transaction has accumulated velocity
    tenth_tx = storage.engineered_features["TX_PIPE_009"]
    assert tenth_tx["velocity_5m"] == 9

    await consumer.close()

@pytest.mark.asyncio
async def test_consumer_dlq_resilience():
    """Verify corrupted messages are diverted to DLQ while valid stream continues."""
    stream = MemoryStreamOperator()
    storage = MemoryStorageOperator()
    consumer = AsyncStreamConsumer(stream_operator=stream, storage_operator=storage, topic="test_dlq_pipe")

    await consumer.start()

    # Publish 1 poison pill (missing customer_id), then 1 valid event
    await stream.publish("test_dlq_pipe", {"transaction_id": "POISON_TX"})
    await stream.publish("test_dlq_pipe", {
        "transaction_id": "CLEAN_TX_01",
        "customer_id": "CUST_CLEAN",
        "amount": 25.0,
        "time_step": 10.0
    })

    processed = await consumer.run(max_events=1)

    assert processed == 1
    assert "CLEAN_TX_01" in storage.raw_transactions

    await consumer.close()
