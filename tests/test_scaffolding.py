"""Scaffolding and Baseline Unit Tests for Phase 0."""

import os
from pathlib import Path
import pytest

from agent.prompts import SYSTEM_PROMPT
from agent.tools import get_customer_history, check_known_patterns, get_merchant_risk_score
from agent.guardrails import validate_agent_report, verify_fact_against_evidence
from consumer.feature_engineering import haversine_distance, compute_rolling_features
from producer.stream_producer import generate_synthetic_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_directory_structure_exists():
    """Verify all project skeleton directories exist."""
    expected_dirs = [
        "data/raw",
        "data/processed",
        "producer",
        "consumer",
        "db",
        "model",
        "agent",
        "eval",
        "governance",
        "api",
        "dashboard",
        "monitoring",
        "tests",
        ".github/workflows",
    ]
    for d in expected_dirs:
        dir_path = PROJECT_ROOT / d
        assert dir_path.is_dir(), f"Expected directory missing: {d}"

def test_key_files_exist():
    """Verify critical architecture and configuration files exist."""
    expected_files = [
        "docker-compose.yml",
        ".env.example",
        "requirements.txt",
        "db/schema.sql",
        "data/download_data.py",
        "governance/model_card.md",
        "governance/fairness_audit.md",
        "eval/eval_set.csv",
    ]
    for f in expected_files:
        file_path = PROJECT_ROOT / f
        assert file_path.is_file(), f"Expected file missing: {f}"

def test_haversine_distance():
    """Verify haversine distance calculation is accurate."""
    # NYC coords: 40.7128, -74.0060; Philadelphia coords: 39.9526, -75.1652 (~130 km)
    dist = haversine_distance(40.7128, -74.0060, 39.9526, -75.1652)
    assert 120.0 < dist < 145.0

def test_synthetic_metadata_generation():
    """Verify synthetic entity assignment produces consistent attributes."""
    meta = generate_synthetic_metadata(row_index=42, amount=150.0, is_fraud=0)
    assert meta["customer_id"] == "CUST_0042"
    assert "lat" in meta and "lon" in meta
    assert meta["merchant_id"].startswith("MERCH_")

def test_check_known_patterns():
    """Verify rule engine correctly flags high velocity and threshold structuring."""
    suspicious_tx = {
        "amount": 9950.00,
        "velocity_5m": 4,
        "geo_distance_km": 600.0
    }
    result = check_known_patterns(suspicious_tx)
    assert result["triggered_count"] == 3
    assert result["heuristic_score"] > 0.8

def test_guardrail_validation_passed():
    """Verify guardrail passes when cited facts exist in tool outputs."""
    tool_outputs = [
        {"prior_fraud_flags": 0, "rolling_avg_amount": 75.50},
        {"triggered_rules": ["HIGH_VELOCITY_BURST: >= 3 transactions in 5 minutes"]}
    ]
    report = {
        "risk_level": "HIGH",
        "recommendation": "DECLINE",
        "evidence": ["High velocity detected"],
        "cited_facts": [
            "prior_fraud_flags: 0",
            "HIGH_VELOCITY_BURST: >= 3 transactions in 5 minutes"
        ]
    }
    val = validate_agent_report(report, tool_outputs)
    assert val["status"] == "PASSED"
    assert val["faithfulness_score"] == 1.0

def test_guardrail_validation_hallucination_caught():
    """Verify guardrail flags hallucinated or ungrounded claims as NEEDS_REVIEW."""
    tool_outputs = [
        {"customer_id": "CUST_0042", "total_tx_count_30d": 5}
    ]
    hallucinated_report = {
        "risk_level": "HIGH",
        "recommendation": "DECLINE",
        "evidence": ["Customer IP originates from Tor exit node"],
        "cited_facts": [
            "Customer IP was flagged on DarkWeb blacklist 12 times" # Fake ungrounded claim
        ]
    }
    val = validate_agent_report(hallucinated_report, tool_outputs)
    assert val["status"] == "NEEDS_REVIEW"
    assert len(val["unverified_facts"]) == 1
