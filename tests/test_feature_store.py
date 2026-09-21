"""Unit tests for the Redis Sliding-Window Feature Store Operator."""

import pytest
import time
from features.redis_store import RedisFeatureStore, calculate_haversine_distance

@pytest.mark.asyncio
async def test_haversine_distance_calculation():
    # New York (40.7128, -74.0060) to London (51.5074, -0.1278) ~ 5570 km
    dist = calculate_haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)
    assert 5500 < dist < 5650

@pytest.mark.asyncio
async def test_sliding_window_counts_and_speed():
    store = RedisFeatureStore(redis_url="redis://localhost:6379/15")
    cust = "CUST_TEST_VELOCITY"
    base_ts = time.time()

    # Record 3 events spaced across 30 seconds
    # Event 1: 30s ago in NYC
    await store.record_event(cust, timestamp=base_ts - 30.0, amount=100.0, lat=40.7128, lon=-74.0060)
    # Event 2: 15s ago in NYC
    await store.record_event(cust, timestamp=base_ts - 15.0, amount=120.0, lat=40.7130, lon=-74.0065)
    # Event 3: Now in London (Impossible travel jump!)
    await store.record_event(cust, timestamp=base_ts, amount=1500.0, lat=51.5074, lon=-0.1278)

    features = await store.get_sliding_window_features(
        customer_id=cust,
        current_timestamp=base_ts,
        current_lat=51.5074,
        current_lon=-0.1278
    )

    assert features["customer_id"] == cust
    assert features["velocity_60s"] == 3
    assert features["velocity_5m"] == 3
    # Time since last tx was 15 seconds, distance was ~5570km => travel speed is extreme
    assert features["geo_distance_km"] > 5000.0
    assert features["travel_speed_kmh"] > 10000.0
    assert features["lookup_latency_ms"] < 25.0  # Sub-25ms even in python emulation

    await store.close()
