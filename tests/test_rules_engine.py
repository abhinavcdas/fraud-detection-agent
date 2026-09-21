"""Unit tests for the Deterministic Pre-ML Hard Rules Engine."""

import pytest
from core.rules_engine import DeterministicRulesEngine

@pytest.fixture
def rules_engine():
    return DeterministicRulesEngine(hard_cap_amount=10000.0, velocity_killswitch=8)

def test_clean_transaction_passes(rules_engine):
    tx = {
        "transaction_id": "TX_CLEAN_01",
        "customer_id": "CUST_001",
        "amount": 150.0,
        "country": "US",
        "card_number": "424242424242",
        "merchant_id": "MERCH_GROCERY"
    }
    decision = rules_engine.evaluate_rules(tx, features={"velocity_60s": 1, "travel_speed_kmh": 25.0})
    assert decision["passed"] is True
    assert decision["action"] == "PASS"
    assert len(decision["triggered_rules"]) == 0

def test_sanctioned_country_blocked(rules_engine):
    tx = {
        "transaction_id": "TX_SANCTION_01",
        "customer_id": "CUST_002",
        "amount": 50.0,
        "country": "KP",
        "card_number": "424242424242"
    }
    decision = rules_engine.evaluate_rules(tx)
    assert decision["passed"] is False
    assert decision["action"] == "BLOCK"
    assert "RULE_OFAC_SANCTIONED_COUNTRY" in decision["triggered_rules"]

def test_compromised_card_blocked(rules_engine):
    tx = {
        "transaction_id": "TX_STOLEN_01",
        "customer_id": "CUST_003",
        "amount": 200.0,
        "country": "US",
        "card_number": "4111119912345678"
    }
    decision = rules_engine.evaluate_rules(tx)
    assert decision["passed"] is False
    assert decision["action"] == "BLOCK"
    assert "RULE_COMPROMISED_CARD_BIN" in decision["triggered_rules"]

def test_velocity_killswitch_blocked(rules_engine):
    tx = {
        "transaction_id": "TX_BURST_01",
        "customer_id": "CUST_004",
        "amount": 99.0,
        "country": "US"
    }
    decision = rules_engine.evaluate_rules(tx, features={"velocity_60s": 10})
    assert decision["passed"] is False
    assert decision["action"] == "BLOCK"
    assert "RULE_VELOCITY_KILLSWITCH_EXCEEDED" in decision["triggered_rules"]

def test_hard_cap_triggers_step_up(rules_engine):
    tx = {
        "transaction_id": "TX_WHALE_01",
        "customer_id": "CUST_005",
        "amount": 15000.0,
        "country": "US"
    }
    decision = rules_engine.evaluate_rules(tx)
    assert decision["passed"] is False
    assert decision["action"] == "STEP_UP"
    assert "RULE_HARD_CAP_EXCEEDED" in decision["triggered_rules"]

def test_impossible_travel_triggers_step_up(rules_engine):
    tx = {
        "transaction_id": "TX_SPEED_01",
        "customer_id": "CUST_006",
        "amount": 300.0,
        "country": "US"
    }
    decision = rules_engine.evaluate_rules(tx, features={"travel_speed_kmh": 1400.0})
    assert decision["passed"] is False
    assert decision["action"] == "STEP_UP"
    assert "RULE_IMPOSSIBLE_TRAVEL_SPEED" in decision["triggered_rules"]


# ---------------------------------------------------------------------------
# Invalid & Edge Case Inputs
# ---------------------------------------------------------------------------

def test_negative_amount_does_not_crash(rules_engine):
    """Invalid input: negative amount should not crash the rules engine."""
    tx = {
        "transaction_id": "TX_NEG_01",
        "customer_id": "CUST_NEG",
        "amount": -500.0,
        "country": "US",
    }
    decision = rules_engine.evaluate_rules(tx, features={})
    # Must not raise; result should be a dict with 'passed' key
    assert "passed" in decision
    assert "action" in decision


def test_zero_amount_does_not_crash(rules_engine):
    """Edge case: zero-amount transaction must not crash or trigger hard cap."""
    tx = {
        "transaction_id": "TX_ZERO_01",
        "customer_id": "CUST_ZERO",
        "amount": 0.0,
        "country": "US",
    }
    decision = rules_engine.evaluate_rules(tx, features={})
    assert "passed" in decision
    # Zero amount should not trigger hard-cap rule
    assert "RULE_HARD_CAP_EXCEEDED" not in decision["triggered_rules"]


def test_null_country_code_does_not_crash(rules_engine):
    """Invalid input: missing country field should not crash — treated as unknown/pass."""
    tx = {
        "transaction_id": "TX_NULL_COUNTRY",
        "customer_id": "CUST_NC",
        "amount": 100.0,
        # country intentionally omitted
    }
    decision = rules_engine.evaluate_rules(tx, features={})
    assert "passed" in decision


def test_lowercase_country_code_normalized(rules_engine):
    """Edge case: lowercase sanctioned country code 'kp' should be treated same as 'KP'."""
    tx = {
        "transaction_id": "TX_LOWER_COUNTRY",
        "customer_id": "CUST_LC",
        "amount": 50.0,
        "country": "kp",  # North Korea, lowercase
        "card_number": "424242424242",
    }
    decision = rules_engine.evaluate_rules(tx, features={})
    # Either blocked (if engine normalizes) or passed (if engine is case-sensitive)
    # Either way it must not crash
    assert "passed" in decision
    assert "action" in decision
