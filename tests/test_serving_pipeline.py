"""Unit and Integration Tests for Phase 6 Serving, API Endpoints, and Monitoring."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from monitoring.drift_report import generate_drift_report

client = TestClient(app)


def test_health_endpoint():
    """Verify FastAPI service health probe returns healthy status and metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "fraud-detection-api"
    assert "operational_threshold" in data
    assert "model_version" in data


def test_score_endpoint_benign_transaction():
    """Verify routine legitimate transaction is approved without triggering agent."""
    tx = {
        "transaction_id": "TX_TEST_BENIGN_01",
        "customer_id": "CUST_0180",
        "merchant_id": "MERCH_005",
        "amount": 35.00,
        "velocity_5m": 1,
        "velocity_60m": 1,
        "amount_deviation": 0.1,
        "geo_distance_km": 1.2,
        "time_since_last_tx_sec": 7200.0
    }
    response = client.post("/score", json=tx)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX_TEST_BENIGN_01"
    assert data["is_flagged"] is False
    assert data["action"] == "APPROVE"
    assert data["investigation_dossier"] is None
    assert len(data["input_hash"]) == 64
    assert data["latency_ms"] > 0


def test_score_endpoint_suspicious_transaction_triggers_investigation():
    """Verify high-risk transaction triggers multi-turn agent investigation and SHAP factors."""
    tx = {
        "transaction_id": "TX_TEST_FRAUD_01",
        "customer_id": "CUST_0042",
        "merchant_id": "MERCH_002",
        "amount": 2850.00,
        "velocity_5m": 5,
        "velocity_60m": 8,
        "amount_deviation": 4.5,
        "geo_distance_km": 720.0,
        "time_since_last_tx_sec": 30.0
    }
    response = client.post("/score", json=tx)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX_TEST_FRAUD_01"
    assert data["is_flagged"] is True
    assert data["action"] == "INVESTIGATE"
    assert data["investigation_dossier"] is not None
    assert "report" in data["investigation_dossier"]
    assert "guardrails" in data["investigation_dossier"]
    assert data["fraud_score"] >= data["threshold"]


def test_get_flagged_transactions_endpoint():
    """Verify operations queue retrieves recently flagged transactions."""
    response = client.get("/transactions/flagged?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "flagged_count" in data
    assert "transactions" in data
    assert isinstance(data["transactions"], list)
    assert data["flagged_count"] >= 1


def test_get_audit_log_reconstruction():
    """Verify audit-log endpoint reconstructs complete forensic decision by transaction ID."""
    response = client.get("/audit-log/TX_TEST_FRAUD_01")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX_TEST_FRAUD_01"
    assert data["is_flagged"] == 1
    assert "input_hash" in data
    assert len(data["input_hash"]) == 64
    assert data["fraud_score"] > 0


def test_get_audit_log_not_found():
    """Verify non-existent transaction returns structured 404 error."""
    response = client.get("/audit-log/TX_NON_EXISTENT_99999")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_drift_monitoring_report_generation(tmp_path):
    """Verify Evidently drift report generation runs and produces HTML."""
    out_html = tmp_path / "test_drift.html"
    result = generate_drift_report(output_html=str(out_html), max_sample=50)

    assert result["status"] == "SUCCESS"
    assert out_html.exists()
    assert out_html.stat().st_size > 1000


def test_deterministic_hard_rule_block_in_score():
    """Verify sanctions or compromised cards are blocked immediately with zero ML latency."""
    tx = {
        "transaction_id": "TX_OFAC_BLOCKED_01",
        "customer_id": "CUST_9999",
        "amount": 250.0,
        "country": "KP",  # Sanctioned country
        "card_number": "424242424242"
    }
    response = client.post("/score", json=tx)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "DECLINE"
    assert data["is_flagged"] is True
    assert data["pipeline_stage"] == "DETERMINISTIC_HARD_RULE_BLOCK"
    assert data["rule_decision"]["passed"] is False


def test_rules_evaluate_endpoint():
    """Verify direct POST /rules/evaluate endpoint."""
    response = client.post("/rules/evaluate", json={"amount": 15000.0, "country": "US"})
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "STEP_UP"
    assert "RULE_HARD_CAP_EXCEEDED" in data["triggered_rules"]


def test_mule_ring_graph_endpoint():
    """Verify GET /graph/mule-ring/{customer_id} returns cluster and graph visualization data."""
    response = client.get("/graph/mule-ring/CUST_0042")
    assert response.status_code == 200
    data = response.json()
    assert "analysis" in data
    assert "graph_data" in data
    assert data["analysis"]["mule_ring_detected"] is True
    assert "DEV_FARM_01" in data["analysis"]["shared_devices"]


def test_score_endpoint_async_triage():
    """Verify async_triage=True returns immediately without waiting for LLM investigation."""
    tx = {
        "transaction_id": "TX_ASYNC_TRIAGE_01",
        "customer_id": "CUST_0042",
        "merchant_id": "MERCH_002",
        "amount": 2850.00,
        "velocity_5m": 5,
        "velocity_60m": 8,
        "amount_deviation": 4.5,
        "geo_distance_km": 720.0,
        "time_since_last_tx_sec": 30.0
    }
    response = client.post("/score?async_triage=true", json=tx)
    assert response.status_code == 200
    data = response.json()

    assert data["transaction_id"] == "TX_ASYNC_TRIAGE_01"
    assert data["is_flagged"] is True
    assert data["action"] == "INVESTIGATE_ASYNC"
    assert data["pipeline_stage"] == "HOT_PATH_SCORING_COMPLETE_ASYNC_TRIAGE_QUEUED"
    assert data["investigation_dossier"]["status"] == "QUEUED_FOR_TRIAGE"


def test_score_endpoint_pan_sanitization():
    """Verify PCI-DSS 3.4 Card PAN is sanitized/masked."""
    from api.main import sanitize_card_pan
    masked = sanitize_card_pan("4242424242421234")
    assert masked == "424242******1234"

    tx = {
        "transaction_id": "TX_PAN_TEST_01",
        "customer_id": "CUST_0180",
        "card_number": "4111112233445566",
        "amount": 25.00
    }
    response = client.post("/score", json=tx)
    assert response.status_code == 200


