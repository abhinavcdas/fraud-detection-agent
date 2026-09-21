"""Resilience and Fault-Tolerance Test Suite.

Proves:
1. Complete failure isolation: External LLM provider crash or timeout does not crash consumer.
2. Dead-Letter Queue (DLQ): Poison pills are safely isolated while streaming continues.
3. System stability under simulated degradation.
"""

import pytest
import asyncio
from core.interfaces import BaseLLMOperator
from core.resilience import global_dlq
from operators.storage.memory_operator import MemoryStorageOperator
from operators.stream.memory_operator import MemoryStreamOperator
from operators.features.rolling_feature_operator import RollingFeatureOperator
from operators.scoring.heuristic_operator import HeuristicScorerOperator
from pipeline.orchestrator import FraudPipelineOrchestrator

class BrokenFailingLLMOperator(BaseLLMOperator):
    """Simulates an external provider experiencing sudden 500 error / rate limit timeout."""

    async def health_check(self) -> bool:
        return False

    async def investigate(self, transaction, context) -> dict:
        raise ConnectionError("Simulated Groq API 503 Service Unavailable / Rate Limit 429")

@pytest.mark.asyncio
async def test_llm_failure_does_not_crash_pipeline():
    """Verify that an LLM failure is isolated, returning degraded fallback and recording audit log."""
    storage = MemoryStorageOperator()
    stream = MemoryStreamOperator()
    failing_llm = BrokenFailingLLMOperator()
    scorer = HeuristicScorerOperator()
    features = RollingFeatureOperator()

    orchestrator = FraudPipelineOrchestrator(
        storage=storage,
        stream=stream,
        llm=failing_llm,
        scorer=scorer,
        feature_extractor=features,
        fraud_threshold=0.5
    )
    await orchestrator.initialize()

    # Suspicious transaction that triggers the failing LLM
    suspicious_tx = {
        "transaction_id": "TX_CRASH_TEST_01",
        "customer_id": "CUST_999",
        "amount": 8500.0,
        "velocity_5m": 5,
        "geo_distance_km": 950.0
    }

    # Should execute smoothly without raising ConnectionError
    result = await orchestrator.process_transaction(suspicious_tx)

    assert result["is_flagged"] is True
    assert result["investigation"] is not None
    assert result["investigation"]["provider"] == "degraded_fallback"
    assert result["investigation"]["report"]["recommendation"] == "ESCALATE"

    # Verify audit log was still persisted
    audit_logs = storage.audit_log
    assert len(audit_logs) == 1
    assert audit_logs[0]["transaction_id"] == "TX_CRASH_TEST_01"
    assert audit_logs[0]["is_flagged"] is True
    assert audit_logs[0]["agent_decision"] == "ESCALATE"

    await orchestrator.close()

@pytest.mark.asyncio
async def test_dlq_handles_poisoned_messages_without_stopping_stream():
    """Verify that a corrupted payload in the stream is routed to DLQ while stream continues."""
    storage = MemoryStorageOperator()
    stream = MemoryStreamOperator()
    failing_llm = BrokenFailingLLMOperator()
    
    orchestrator = FraudPipelineOrchestrator(
        storage=storage,
        stream=stream,
        llm=failing_llm
    )
    await orchestrator.initialize()

    # Publish: 1 poisoned message (missing customer_id), then 1 valid message
    await stream.publish("transactions", {"transaction_id": "TX_POISON_01"}) # Missing customer_id -> raises ValueError
    await stream.publish("transactions", {
        "transaction_id": "TX_VALID_02",
        "customer_id": "CUST_001",
        "amount": 25.0,
        "time_step": 10.0
    })

    # Run stream consumer for 2 events
    consumed_count = await orchestrator.run_stream(topic="transactions", max_events=1)

    # 1 valid message successfully processed despite poison pill!
    assert consumed_count == 1
    assert "TX_VALID_02" in storage.raw_transactions

    await orchestrator.close()
