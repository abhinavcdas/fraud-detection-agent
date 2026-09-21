"""Automated Test Suite for Phase 3: LLM Investigation Agent & Guardrails."""

import pytest
from agent.tools import (
    get_customer_history,
    check_known_patterns,
    get_merchant_risk_score,
    TOOL_DEFINITIONS
)
from agent.guardrails import validate_agent_report
from agent.agent_loop import run_investigation
from operators.storage.memory_operator import MemoryStorageOperator

def test_tool_definitions_schema():
    """Verify tool schemas conform to OpenAI / Groq function specification."""
    assert len(TOOL_DEFINITIONS) == 4
    tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
    assert "get_customer_history" in tool_names
    assert "check_known_patterns" in tool_names
    assert "get_merchant_risk_score" in tool_names
    assert "analyze_mule_ring_network" in tool_names


def test_tool_check_known_patterns_triggers():
    """Verify heuristic rule engine triggers structuring, velocity, and geo anomalies."""
    # Structuring & Velocity burst
    structuring_tx = {
        "amount": 9500.00,
        "velocity_5m": 4,
        "velocity_60m": 9,
        "geo_distance_km": 600.0,
        "time_since_last_tx_sec": 30.0,
        "amount_deviation": 4.0
    }
    result = check_known_patterns(structuring_tx)
    assert result["triggered_count"] >= 4
    assert result["heuristic_severity_score"] >= 0.80
    assert result["heuristic_risk_assessment"] == "CRITICAL"

    # Benign routine tx
    benign_tx = {
        "amount": 35.00,
        "velocity_5m": 0,
        "velocity_60m": 1,
        "geo_distance_km": 2.0,
        "time_since_last_tx_sec": 1800.0,
        "amount_deviation": 0.2
    }
    benign_res = check_known_patterns(benign_tx)
    assert benign_res["triggered_count"] == 0
    assert benign_res["heuristic_risk_assessment"] == "LOW"

def test_tool_merchant_risk_catalog():
    """Verify merchant risk categorization and chargeback rates."""
    merch_high = get_merchant_risk_score("MERCH_002") # electronics
    assert merch_high["category"] == "electronics_digital_goods"
    assert merch_high["merchant_risk_tier"] in {"HIGH", "CRITICAL", "MEDIUM"}

    merch_low = get_merchant_risk_score("MERCH_006") # supermarket
    assert merch_low["category"] == "supermarket_groceries"
    assert merch_low["merchant_risk_tier"] == "LOW"
    assert merch_low["historical_chargeback_rate_pct"] < 1.0

def test_guardrail_verifies_grounded_facts():
    """Verify guardrail passes when cited claims exist in tool data."""
    tool_data = [
        {"customer_id": "CUST_0042", "rolling_avg_amount": 75.50, "prior_fraud_flags": 0},
        {"triggered_rules": ["HIGH_VELOCITY_BURST: 4 transactions in last 5 minutes"]}
    ]
    report = {
        "risk_level": "HIGH",
        "recommendation": "DECLINE",
        "evidence": ["Velocity burst detected"],
        "cited_facts": [
            "rolling_avg_amount: $75.50",
            "HIGH_VELOCITY_BURST: 4 transactions in last 5 minutes"
        ]
    }
    val = validate_agent_report(report, tool_data)
    assert val["status"] == "PASSED"
    assert val["faithfulness_score"] == 1.0
    assert len(val["unverified_facts"]) == 0

def test_guardrail_intercepts_hallucinations():
    """Verify guardrail intercepts fabricated claims as NEEDS_REVIEW."""
    tool_data = [
        {"customer_id": "CUST_0042", "rolling_avg_amount": 75.50}
    ]
    hallucinated_report = {
        "risk_level": "HIGH",
        "recommendation": "DECLINE",
        "evidence": ["Customer had stolen identity"],
        "cited_facts": [
            "Customer account was banned by Interpol for money laundering in 2021"
        ]
    }
    val = validate_agent_report(hallucinated_report, tool_data)
    assert val["status"] == "NEEDS_REVIEW"
    assert val["faithfulness_score"] == 0.0
    assert len(val["unverified_facts"]) == 1

def test_investigation_fraud_scenario():
    """Verify investigation produces structured dossier recommending DECLINE for extreme fraud."""
    fraud_tx = {
        "transaction_id": "TX_TEST_FRAUD",
        "customer_id": "CUST_0042",
        "merchant_id": "MERCH_000", # crypto exchange
        "amount": 8900.00,
        "velocity_5m": 5,
        "velocity_60m": 10,
        "geo_distance_km": 1200.0,
        "time_since_last_tx_sec": 45.0,
        "amount_deviation": 5.2
    }
    result = run_investigation(fraud_tx)

    report = result["report"]
    assert report["risk_level"] in {"HIGH", "CRITICAL"}
    assert report["recommendation"] in {"DECLINE", "ESCALATE"}
    assert len(report["evidence"]) > 0
    assert len(report["cited_facts"]) > 0
    assert result["guardrails"]["status"] == "PASSED"

def test_investigation_benign_scenario():
    """Verify investigation handles routine benign transactions without dramatic escalation."""
    benign_tx = {
        "transaction_id": "TX_TEST_BENIGN",
        "customer_id": "CUST_0099",
        "merchant_id": "MERCH_006", # supermarket
        "amount": 28.50,
        "velocity_5m": 0,
        "velocity_60m": 1,
        "geo_distance_km": 1.5,
        "time_since_last_tx_sec": 7200.0,
        "amount_deviation": 0.1
    }
    result = run_investigation(benign_tx)

    report = result["report"]
    assert report["risk_level"] == "LOW"
    assert report["recommendation"] == "APPROVE"
    assert result["guardrails"]["status"] == "PASSED"
