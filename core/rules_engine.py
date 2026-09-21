"""Deterministic Pre-ML Hard Rule Engine.

Executes sub-millisecond sanity and compliance checks before ML scoring:
1. OFAC Sanctions / High-Risk Jurisdictions
2. Single-Transaction Hard Ceiling Limits
3. Extreme Velocity Killswitch
4. Known Compromised Cards / BIN Blocklist
5. High-Risk Prohibited Merchant Categories
"""

import os
from typing import Dict, Any, List, Optional
from core.interfaces import BaseRulesEngineOperator
from core.logger import get_logger, bind_tx_context

logger = get_logger("rules_engine")

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

    def evaluate_rules(self, tx: Dict[str, Any], features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate deterministic rules in sequence.
        
        Returns:
            Dict with:
                - action: "BLOCK" | "STEP_UP" | "PASS"
                - passed: bool
                - triggered_rules: List[str]
                - reason: str
        """
        tx_id = tx.get("transaction_id", "TX_UNKNOWN")
        cust_id = tx.get("customer_id", "CUST_UNKNOWN")
        tx_logger = bind_tx_context(logger, tx_id, cust_id)

        features = features or {}
        triggered_rules: List[str] = []
        action = "PASS"
        reason = "All deterministic checks passed."

        # 1. OFAC Sanctions / Sanctioned Country Check
        country = str(tx.get("country", tx.get("billing_country", ""))).upper()
        if country and country in self.sanctioned_countries:
            triggered_rules.append("RULE_OFAC_SANCTIONED_COUNTRY")
            action = "BLOCK"
            reason = f"Transaction originates from sanctioned jurisdiction: {country}"
            tx_logger.warning("Rule triggered: {reason}", reason=reason)
            return {
                "action": action,
                "passed": False,
                "triggered_rules": triggered_rules,
                "reason": reason
            }

        # 2. Compromised Card / BIN Blocklist Check
        card_number = str(tx.get("card_number", tx.get("card_bin", "")))
        if any(card_number.startswith(b) for b in self.compromised_bins):
            triggered_rules.append("RULE_COMPROMISED_CARD_BIN")
            action = "BLOCK"
            reason = f"Card BIN {card_number[:8]} matches known stolen card blocklist."
            tx_logger.warning("Rule triggered: {reason}", reason=reason)
            return {
                "action": action,
                "passed": False,
                "triggered_rules": triggered_rules,
                "reason": reason
            }

        # 3. Extreme Velocity Killswitch Check
        velocity_count = int(features.get("velocity_60s", features.get("velocity_5m", tx.get("velocity_5m", 0))))
        if velocity_count >= self.velocity_killswitch:
            triggered_rules.append("RULE_VELOCITY_KILLSWITCH_EXCEEDED")
            action = "BLOCK"
            reason = f"Instantaneous velocity ({velocity_count} tx) exceeded automated killswitch limit ({self.velocity_killswitch})."
            tx_logger.warning("Rule triggered: {reason}", reason=reason)
            return {
                "action": action,
                "passed": False,
                "triggered_rules": triggered_rules,
                "reason": reason
            }

        # 4. Prohibited Merchant MCC Check
        mcc = str(tx.get("mcc", tx.get("merchant_id", "")))
        if any(prohibited in mcc for prohibited in self.prohibited_mccs):
            triggered_rules.append("RULE_PROHIBITED_MERCHANT_MCC")
            action = "BLOCK"
            reason = f"Merchant identifier or MCC '{mcc}' is prohibited by regulatory policy."
            tx_logger.warning("Rule triggered: {reason}", reason=reason)
            return {
                "action": action,
                "passed": False,
                "triggered_rules": triggered_rules,
                "reason": reason
            }

        # 5. Impossible Travel Delta Check (> 1000 km/h)
        travel_speed = float(features.get("travel_speed_kmh", 0.0))
        if travel_speed > 1000.0:
            triggered_rules.append("RULE_IMPOSSIBLE_TRAVEL_SPEED")
            action = "STEP_UP"
            reason = f"Calculated travel speed ({travel_speed:.1f} km/h) indicates impossible physical travel."
            tx_logger.warning("Rule triggered: {reason}", reason=reason)

        # 6. Single-Transaction Hard Ceiling Limit ($10,000+)
        amount = float(tx.get("amount", 0.0))
        if amount >= self.hard_cap_amount:
            triggered_rules.append("RULE_HARD_CAP_EXCEEDED")
            if action != "BLOCK":
                action = "STEP_UP"
            reason = f"Transaction amount (${amount:,.2f}) exceeds regulatory single-swipe limit (${self.hard_cap_amount:,.2f}). Requires step-up wire verification."
            tx_logger.info("Rule triggered: {reason}", reason=reason)

        passed = (action == "PASS")
        return {
            "action": action,
            "passed": passed,
            "triggered_rules": triggered_rules,
            "reason": reason
        }
