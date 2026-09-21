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

@pytest.mark.asyncio
async def test_sqlite_storage_batch_and_wal(tmp_path):
    db_file = str(tmp_path / "test_batch.db")
    storage = SqliteStorageOperator(db_path=db_file)
    await storage.initialize()

    # Verify WAL mode
    with storage._get_connection() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"

    # Batch raw transactions
    tx_batch = [
        {"transaction_id": f"TX_B_{i}", "customer_id": "CUST_B", "amount": 10.0 * i, "timestamp": "2026-09-18T12:00:00"}
        for i in range(5)
    ]
    await storage.save_raw_transactions_batch(tx_batch)
    history = await storage.get_customer_history("CUST_B", limit=10)
    assert len(history) == 5

    # Batch engineered features
    feat_batch = [
        {"transaction_id": f"TX_B_{i}", "customer_id": "CUST_B", "amount": 10.0 * i, "velocity_5m": i}
        for i in range(5)
    ]
    await storage.save_engineered_features_batch(feat_batch)

    await storage.close()

def test_storage_and_stream_factory_singletons():
    from operators.storage.storage_factory import get_storage_operator, reset_storage_operators
    from operators.stream.stream_factory import get_stream_operator, reset_stream_operators

    reset_storage_operators()
    s1 = get_storage_operator("sqlite")
    s2 = get_storage_operator("sqlite")
    assert s1 is s2
    reset_storage_operators()
    s3 = get_storage_operator("sqlite")
    assert s1 is not s3

    reset_stream_operators()
    st1 = get_stream_operator("memory")
    st2 = get_stream_operator("memory")
    assert st1 is st2
    reset_stream_operators()
    st3 = get_stream_operator("memory")
    assert st1 is not st3

@pytest.mark.asyncio
async def test_kafka_stream_operator_mocked(monkeypatch):
    import sys
    from unittest.mock import MagicMock
    from operators.stream.kafka_operator import KafkaStreamOperator

    mock_producer = MagicMock()
    mock_future = MagicMock()
    mock_future.get.return_value = MagicMock()
    mock_producer.send.return_value = mock_future

    mock_consumer = MagicMock()
    mock_consumer.poll.return_value = {}

    fake_kafka_mod = MagicMock()
    fake_kafka_mod.KafkaProducer = MagicMock(return_value=mock_producer)
    fake_kafka_mod.KafkaConsumer = MagicMock(return_value=mock_consumer)
    fake_kafka_mod.TopicPartition = MagicMock()
    fake_kafka_mod.OffsetAndMetadata = MagicMock()
    monkeypatch.setitem(sys.modules, "kafka", fake_kafka_mod)

    kafka_op = KafkaStreamOperator(bootstrap_servers="localhost:9092")
    await kafka_op.start()

    # Test publish with partition key
    await kafka_op.publish("transactions", {"transaction_id": "TX_K_1", "customer_id": "CUST_77", "amount": 100.0})
    mock_producer.send.assert_called_once()
    call_kwargs = mock_producer.send.call_args
    assert call_kwargs[1]["key"] == b"CUST_77"

    # Test commit_offset
    kafka_op.consumer = mock_consumer
    await kafka_op.commit_offset("transactions", 0, 100)
    mock_consumer.commit.assert_called_once()

    await kafka_op.close()


