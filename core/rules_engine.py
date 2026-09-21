"""Deterministic Pre-ML Hard Rule Engine.

Executes sub-millisecond sanity and compliance checks before ML scoring:
1. OFAC Sanctions / High-Risk Jurisdictions
2. Single-Transaction Hard Ceiling Limits
3. Extreme Velocity Killswitch
4. Known Compromised Cards / BIN Blocklist
5. High-Risk Prohibited Merchant Categories
"""

import os
from enum import Enum
from typing import Dict, Any, List, Optional
from core.interfaces import BaseRulesEngineOperator
from core.logger import get_logger, bind_tx_context

logger = get_logger("rules_engine")

class RuleAction(str, Enum):
    BLOCK = "BLOCK"
    STEP_UP = "STEP_UP"
    PASS = "PASS"


# OFAC / High-Risk Sanctioned Jurisdictions (ISO 2-letter codes)
SANCTIONED_COUNTRIES = {"KP", "IR", "SY", "CU", "RU_SANCTION"}

# High-Risk / Prohibited MCCs (e.g. illegal gambling, unregulated offshore crypto)
PROHIBITED_MCCS = {"7995_UNREG", "6051_HIGH_RISK", "MERCH_012_PROHIBITED"}

# Known Compromised Card Numbers or BINs
COMPROMISED_CARD_BINS = {"41111199", "55000088", "37828200"}

# Thresholds
DEFAULT_HARD_CAP_AMOUNT = 10000.00
DEFAULT_VELOCITY_KILLSWITCH_COUNT = 8  # >= 8 tx in rolling window = automated block


class DeterministicRulesEngine(BaseRulesEngineOperator):
    """Production-grade deterministic rules engine executed on the hot path."""

    def __init__(
        self,
        hard_cap_amount: float = DEFAULT_HARD_CAP_AMOUNT,
        velocity_killswitch: int = DEFAULT_VELOCITY_KILLSWITCH_COUNT,
        sanctioned_countries: Optional[set] = None,
        prohibited_mccs: Optional[set] = None,
        compromised_bins: Optional[set] = None
    ):
        self.hard_cap_amount = hard_cap_amount
        self.velocity_killswitch = velocity_killswitch
        self.sanctioned_countries = sanctioned_countries or SANCTIONED_COUNTRIES
        self.prohibited_mccs = prohibited_mccs or PROHIBITED_MCCS
        self.compromised_bins = compromised_bins or COMPROMISED_CARD_BINS
        logger.info(
            "DeterministicRulesEngine initialized | HardCap=${cap} | VelocityKillswitch={vel}",
            cap=self.hard_cap_amount,
            vel=self.velocity_killswitch
        )

    def evaluate(self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate deterministic rules in sequence (alias for evaluate_rules)."""
        return self.evaluate_rules(tx, features)

    def evaluate_rules(self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate deterministic rules in sequence with null-safety and full rule accumulation.
        
        Returns:
            Dict with:
                - action: "BLOCK" | "STEP_UP" | "PASS"
                - passed: bool
                - triggered_rules: List[str]
                - reason: str
                - reasons: List[str]
        """
        tx_id = str(tx.get("transaction_id") or "TX_UNKNOWN")
        cust_id = str(tx.get("customer_id") or "CUST_UNKNOWN")
        tx_logger = bind_tx_context(logger, tx_id, cust_id)

        features = features or {}
        triggered_rules: List[str] = []
        reasons: List[str] = []
        action = "PASS"

        def _safe_int(val: Any, default: int = 0) -> int:
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        def _safe_float(val: Any, default: float = 0.0) -> float:
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        # 1. OFAC Sanctions / Sanctioned Country Check
        country = str(tx.get("country") or tx.get("billing_country") or "").strip().upper()
        if country and country in self.sanctioned_countries:
            triggered_rules.append("RULE_OFAC_SANCTIONED_COUNTRY")
            action = "BLOCK"
            r = f"Transaction originates from sanctioned jurisdiction: {country}"
            reasons.append(r)
            tx_logger.warning("Rule triggered: {reason}", reason=r)

        # 2. Compromised Card / BIN Blocklist Check
        card_number = str(tx.get("card_number") or tx.get("card_bin") or "").replace(" ", "").replace("-", "")
        if card_number and any(card_number.startswith(b) for b in self.compromised_bins):
            triggered_rules.append("RULE_COMPROMISED_CARD_BIN")
            action = "BLOCK"
            r = f"Card BIN {card_number[:8]} matches known stolen card blocklist."
            reasons.append(r)
            tx_logger.warning("Rule triggered: {reason}", reason=r)

        # 3. Extreme Velocity Killswitch Check
        vel_val = features.get("velocity_60s")
        if vel_val is None:
            vel_val = features.get("velocity_5m")
        if vel_val is None:
            vel_val = tx.get("velocity_5m")
        velocity_count = _safe_int(vel_val, 0)
        if velocity_count >= self.velocity_killswitch:
            triggered_rules.append("RULE_VELOCITY_KILLSWITCH_EXCEEDED")
            action = "BLOCK"
            r = f"Instantaneous velocity ({velocity_count} tx) exceeded automated killswitch limit ({self.velocity_killswitch})."
            reasons.append(r)
            tx_logger.warning("Rule triggered: {reason}", reason=r)

        # 4. Prohibited Merchant MCC Check
        mcc = str(tx.get("mcc") or tx.get("merchant_id") or "").strip().upper()
        if mcc and any(mcc == prohibited or mcc.startswith(prohibited) or prohibited in mcc for prohibited in self.prohibited_mccs):
            triggered_rules.append("RULE_PROHIBITED_MERCHANT_MCC")
            action = "BLOCK"
            r = f"Merchant identifier or MCC '{mcc}' is prohibited by regulatory policy."
            reasons.append(r)
            tx_logger.warning("Rule triggered: {reason}", reason=r)

        # 5. Impossible Travel Delta Check (> 1000 km/h)
        travel_speed = _safe_float(features.get("travel_speed_kmh"), 0.0)
        if travel_speed > 1000.0:
            triggered_rules.append("RULE_IMPOSSIBLE_TRAVEL_SPEED")
            if action != "BLOCK":
                action = "STEP_UP"
            r = f"Calculated travel speed ({travel_speed:.1f} km/h) indicates impossible physical travel."
            reasons.append(r)
            tx_logger.warning("Rule triggered: {reason}", reason=r)

        # 6. Single-Transaction Hard Ceiling Limit ($10,000+)
        amount = _safe_float(tx.get("amount"), 0.0)
        if amount >= self.hard_cap_amount:
            triggered_rules.append("RULE_HARD_CAP_EXCEEDED")
            if action != "BLOCK":
                action = "STEP_UP"
            r = f"Transaction amount (${amount:,.2f}) exceeds regulatory single-swipe limit (${self.hard_cap_amount:,.2f}). Requires step-up wire verification."
            reasons.append(r)
            tx_logger.info("Rule triggered: {reason}", reason=r)

        passed = (action == "PASS")
        reason = " | ".join(reasons) if reasons else "All deterministic checks passed."
        return {
            "action": action,
            "passed": passed,
            "triggered_rules": triggered_rules,
            "reason": reason,
            "reasons": reasons
        }
