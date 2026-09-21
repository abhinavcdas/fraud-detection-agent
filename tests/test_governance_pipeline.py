"""Unit and Integration Tests for Phase 5 Model Risk, Fairness, and Governance Layer."""

import asyncio
from pathlib import Path
import pytest

from governance.audit import compute_input_hash, build_audit_record
from governance.run_fairness_audit import run_fairness_audit
from operators.storage.sqlite_operator import SqliteStorageOperator
from operators.storage.memory_operator import MemoryStorageOperator


def test_audit_input_hash_deterministic():
    """Verify SHA-256 hash is invariant to dict ordering and sensitive to content changes."""
    tx1 = {"transaction_id": "TX_TEST_01", "amount": 150.0, "customer_id": "CUST_01"}
    tx2 = {"customer_id": "CUST_01", "transaction_id": "TX_TEST_01", "amount": 150.0}
    tx3 = {"transaction_id": "TX_TEST_01", "amount": 150.01, "customer_id": "CUST_01"}

    hash1 = compute_input_hash(tx1)
    hash2 = compute_input_hash(tx2)
    hash3 = compute_input_hash(tx3)

    assert len(hash1) == 64
    assert hash1 == hash2
    assert hash1 != hash3


def test_build_audit_record_structure():
    """Verify build_audit_record correctly constructs compliant schema payloads."""
    tx = {"transaction_id": "TX_AUDIT_99", "amount": 9950.0}
    agent_mock = {
        "report": {
            "risk_level": "CRITICAL",
            "recommendation": "DECLINE",
            "summary": "High risk detected"
        },
        "guardrails": {
            "status": "PASSED",
            "faithfulness_score": 1.0
        }
    }

    record = build_audit_record(
        transaction=tx,
        fraud_score=0.92,
        is_flagged=True,
        model_version="fraud-xgb-v1",
        agent_result=agent_mock,
        latency_ms=12.5
    )

    assert record["transaction_id"] == "TX_AUDIT_99"
    assert record["fraud_score"] == 0.92
    assert record["is_flagged"] is True
    assert record["agent_decision"] == "DECLINE"
    assert record["guardrail_status"] == "PASSED"
    assert record["faithfulness_score"] == 1.0
    assert len(record["input_hash"]) == 64


@pytest.mark.asyncio
async def test_memory_storage_audit_trail():
    """Verify in-memory operator supports complete audit trail lifecycle."""
    storage = MemoryStorageOperator()
    await storage.initialize()

    tx = {"transaction_id": "TX_MEM_01", "amount": 250.0}
    record = build_audit_record(tx, fraud_score=0.88, is_flagged=True)
    await storage.save_audit_log(record)

    fetched = await storage.get_audit_record_by_tx("TX_MEM_01")
    assert fetched is not None
    assert fetched["transaction_id"] == "TX_MEM_01"
    assert fetched["fraud_score"] == 0.88

    flagged = await storage.get_recent_flagged_transactions(limit=10)
    assert len(flagged) == 1
    assert flagged[0]["transaction_id"] == "TX_MEM_01"

    await storage.close()


@pytest.mark.asyncio
async def test_sqlite_storage_audit_trail(tmp_path):
    """Verify SQLite operator persists and retrieves immutable audit logs."""
    db_file = tmp_path / "test_audit.db"
    storage = SqliteStorageOperator(db_path=str(db_file))
    await storage.initialize()

    tx = {"transaction_id": "TX_SQLITE_01", "amount": 420.0}
    record = build_audit_record(tx, fraud_score=0.75, is_flagged=True)
    await storage.save_audit_log(record)

    fetched = await storage.get_audit_record_by_tx("TX_SQLITE_01")
    assert fetched is not None
    assert fetched["transaction_id"] == "TX_SQLITE_01"
    assert fetched["fraud_score"] == 0.75
    assert fetched["is_flagged"] == 1

    flagged = await storage.get_recent_flagged_transactions(limit=5)
    assert len(flagged) == 1
    assert flagged[0]["transaction_id"] == "TX_SQLITE_01"

    await storage.close()


def test_fairness_audit_execution(tmp_path):
    """Verify fairness audit computes disparate impact ratios across segments."""
    output_md = tmp_path / "test_fairness_audit.md"
    results = run_fairness_audit(max_rows=1000, threshold=0.38, output_md=output_md)

    assert "amount_audit" in results
    assert "merchant_audit" in results
    assert "time_audit" in results
    assert output_md.exists()
    assert output_md.stat().st_size > 1000

    for band, metrics in results["amount_audit"].items():
        assert "dir" in metrics
        assert "fpr" in metrics
        assert "four_fifths_pass" in metrics


def test_model_card_structure():
    """Verify model card exists and contains key governance sections."""
    model_card = Path(__file__).resolve().parent.parent / "governance" / "model_card.md"
    assert model_card.exists()

    content = model_card.read_text(encoding="utf-8")
    assert "Intended Use" in content
    assert "Factors & Demographic Proxy Segments" in content
    assert "Training Data & Imbalance Mitigation" in content
    assert "Quantitative Performance" in content
    assert "Explainability & Interpretability (SHAP)" in content
    assert "LLM Agent Investigation & Guardrail Assurance" in content
    assert "Continuous Monitoring & Maintenance Plan" in content
