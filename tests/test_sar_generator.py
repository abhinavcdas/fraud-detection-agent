"""Unit tests for the FinCEN Regulatory SAR Generator and Guardrails."""

import pytest
from agent.agent_loop import run_investigation

def test_high_risk_transaction_generates_valid_sar():
    suspicious_tx = {
        "transaction_id": "TX_SAR_TEST_01",
        "customer_id": "CUST_0042",  # Linked to mule ring in entity graph
        "merchant_id": "MERCH_012",   # Prohibited crypto/onramp
        "amount": 4950.00,
        "velocity_5m": 5,
        "velocity_60m": 9,
        "amount_deviation": 5.2,
        "geo_distance_km": 850.0,
        "time_since_last_tx_sec": 30.0,
        "device_id": "DEV_FARM_01",
        "ip_address": "198.51.100.77"
    }

    result = run_investigation(suspicious_tx)
    report = result["report"]

    assert report["risk_level"] in ["HIGH", "CRITICAL"]
    assert report["recommendation"] in ["ESCALATE", "DECLINE"]
    assert "fin_cen_sar" in report
    sar = report["fin_cen_sar"]
    assert sar is not None

    # Verify Part I: Subject Information
    assert sar["part_i_subject"]["customer_id"] == "CUST_0042"
    assert sar["part_i_subject"]["device_fingerprint"] == "DEV_FARM_01"
    assert sar["part_i_subject"]["ip_address"] == "198.51.100.77"

    # Verify Part II: Suspicious Activity
    assert "$4,950.00" in sar["part_ii_suspicious_activity"]["transaction_amount"]
    assert "MERCH_012" in sar["part_ii_suspicious_activity"]["merchant_mcc"]

    # Verify Part III: Financial Institution
    assert sar["part_iii_financial_institution"]["model_version"] == "fraud-xgb-v1"
    assert sar["part_iii_financial_institution"]["calibrated_threshold"] == 0.38

    # Verify Part IV: Narrative
    assert len(sar["part_iv_narrative"]) > 50
    assert "CUST_0042" in sar["part_iv_narrative"]

    # Verify Guardrail status
    assert result["guardrails"]["status"] in ["PASSED", "WARNING"]
