"""Unit tests for pluggable operators (Storage, Stream, LLM, Features, Scoring)."""

import pytest
import asyncio
from operators.storage.memory_operator import MemoryStorageOperator
from operators.storage.sqlite_operator import SqliteStorageOperator
from operators.stream.memory_operator import MemoryStreamOperator
from operators.llm.mock_operator import MockLLMOperator
from operators.features.rolling_feature_operator import RollingFeatureOperator
from operators.scoring.heuristic_operator import HeuristicScorerOperator

@pytest.mark.asyncio
async def test_memory_storage_operator():
    storage = MemoryStorageOperator()
    await storage.initialize()

    tx = {"transaction_id": "TX_MEM_01", "customer_id": "CUST_99", "amount": 100.0, "timestamp": "2026-09-18T10:00:00"}
    await storage.save_raw_transaction(tx)
    
    history = await storage.get_customer_history("CUST_99")
    assert len(history) == 1
    assert history[0]["transaction_id"] == "TX_MEM_01"

    feat = {"transaction_id": "TX_MEM_01", "customer_id": "CUST_99", "velocity_5m": 1}
    await storage.save_engineered_features(feat)

    audit = {"transaction_id": "TX_MEM_01", "is_flagged": True, "fraud_score": 0.85}
    await storage.save_audit_log(audit)

    flagged = await storage.get_recent_flagged_transactions()
    assert len(flagged) == 1
    assert flagged[0]["transaction_id"] == "TX_MEM_01"

    await storage.close()

@pytest.mark.asyncio
async def test_sqlite_storage_operator(tmp_path):
    db_file = str(tmp_path / "test_fraud.db")
    storage = SqliteStorageOperator(db_path=db_file)
    await storage.initialize()

    tx = {"transaction_id": "TX_SQLITE_01", "customer_id": "CUST_55", "amount": 250.0, "timestamp": "2026-09-18T10:00:00", "time_step": 100.0}
    await storage.save_raw_transaction(tx)

    history = await storage.get_customer_history("CUST_55")
    assert len(history) == 1
    assert history[0]["customer_id"] == "CUST_55"

    audit = {"transaction_id": "TX_SQLITE_01", "is_flagged": True, "fraud_score": 0.92, "agent_decision": "DECLINE"}
    await storage.save_audit_log(audit)

    flagged = await storage.get_recent_flagged_transactions()
    assert len(flagged) == 1
    assert flagged[0]["agent_decision"] == "DECLINE"

    await storage.close()

@pytest.mark.asyncio
async def test_memory_stream_operator():
    stream = MemoryStreamOperator()
    await stream.start()

    test_payload = {"transaction_id": "TX_STREAM_01", "amount": 50.0}
    await stream.publish("test_topic", test_payload)

    # Consume single message
    async for msg in stream.consume("test_topic"):
        assert msg["transaction_id"] == "TX_STREAM_01"
        break

    await stream.close()

@pytest.mark.asyncio
async def test_mock_llm_operator():
    llm = MockLLMOperator()
    assert await llm.health_check() is True

    suspicious_tx = {
        "transaction_id": "TX_LLM_01",
        "customer_id": "CUST_11",
        "amount": 2500.0,
        "velocity_5m": 4
    }
    result = await llm.investigate(suspicious_tx)
    assert result["report"]["recommendation"] == "DECLINE"
    assert result["guardrails"]["status"] == "PASSED"
    assert result["latency_ms"] >= 0.0

def test_rolling_feature_operator():
    extractor = RollingFeatureOperator()
    current_tx = {
        "transaction_id": "TX_02",
        "customer_id": "CUST_A",
        "amount": 500.0,
        "time_step": 360.0, # 6 minutes
        "lat": 40.7128,
        "lon": -74.0060
    }
    history = [
        {
            "transaction_id": "TX_01",
            "customer_id": "CUST_A",
            "amount": 100.0,
            "time_step": 120.0, # 2 minutes
            "lat": 40.7128,
            "lon": -74.0060
        }
    ]
    features = extractor.extract_features(current_tx, history)
    assert features["transaction_id"] == "TX_02"
    assert features["time_since_last_tx_sec"] == 240.0
    assert features["velocity_60m"] == 1 # within 1 hour
    assert features["geo_distance_km"] == 0.0

def test_heuristic_scorer():
    scorer = HeuristicScorerOperator()
    benign_tx = {"amount": 25.0, "velocity_5m": 0, "geo_distance_km": 2.0}
    assert scorer.predict_proba(benign_tx) < 0.2

    fraud_tx = {"amount": 6000.0, "velocity_5m": 4, "geo_distance_km": 800.0}
    assert scorer.predict_proba(fraud_tx) >= 0.8
