"""Investigation Agent Tools for Fraud Analysis.

Provides grounded tools for the Groq investigation agent:
1. get_customer_history: queries active storage for real customer transaction activity.
2. check_known_patterns: heuristic rule engine for structuring, velocity bursts, and geo jumps.
3. get_merchant_risk_score: merchant category and historical chargeback catalog lookup.
"""

import math
import asyncio
from typing import Dict, Any, List, Optional
from operators.storage.storage_factory import get_storage_operator
from core.logger import get_logger

logger = get_logger("agent_tools")

# Synthetic Merchant Catalog
MERCHANT_CATALOG = {
    0: {"category": "crypto_exchange_onramp", "chargeback_rate_pct": 5.8, "risk_tier": "CRITICAL"},
    1: {"category": "wire_remittance_transfer", "chargeback_rate_pct": 4.2, "risk_tier": "HIGH"},
    2: {"category": "electronics_digital_goods", "chargeback_rate_pct": 2.9, "risk_tier": "HIGH"},
    3: {"category": "luxury_jewelry_retail", "chargeback_rate_pct": 2.1, "risk_tier": "MEDIUM"},
    4: {"category": "airline_travel_booking", "chargeback_rate_pct": 1.8, "risk_tier": "MEDIUM"},
    5: {"category": "fuel_convenience_store", "chargeback_rate_pct": 0.9, "risk_tier": "LOW"},
    6: {"category": "supermarket_groceries", "chargeback_rate_pct": 0.4, "risk_tier": "LOW"},
    7: {"category": "utility_telecom_services", "chargeback_rate_pct": 0.2, "risk_tier": "LOW"},
}

import concurrent.futures

_ASYNC_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent_tools_pool")

def _run_async_or_sync(coro):
    """Safely execute async storage call whether inside or outside an active event loop."""
    try:
        loop = asyncio.get_running_loop()
        # If running inside an existing loop, run in a dedicated worker thread
        return _ASYNC_POOL.submit(lambda: asyncio.run(coro)).result()
    except RuntimeError:
        return asyncio.run(coro)

def get_customer_history(customer_id: str) -> Dict[str, Any]:
    """Retrieve historical transaction activity and rolling statistics for a customer from storage.
    
    Args:
        customer_id: The unique customer identifier (e.g., 'CUST_0042').
        
    Returns:
        Dict containing transaction count, rolling average amount, standard deviation, and prior fraud flags.
    """
    cid = str(customer_id or "CUST_UNKNOWN")
    try:
        storage = get_storage_operator()
        
        async def _fetch():
            await storage.initialize()
            return await storage.get_customer_history(cid, limit=50)

        history = _run_async_or_sync(_fetch())
    except Exception as e:
        logger.warning("Could not query storage for customer {cid}: {err}", cid=cid, err=str(e))
        history = []

    if history:
        amounts = [float(tx.get("amount", 0.0) or 0.0) for tx in history]
        avg_amt = sum(amounts) / len(amounts)
        var_amt = sum((x - avg_amt) ** 2 for x in amounts) / len(amounts)
        std_amt = math.sqrt(var_amt)
        prior_fraud = sum(1 for tx in history if int(tx.get("is_fraud", 0) or 0) == 1)
        last_tx_time = str(history[0].get("timestamp", ""))

        return {
            "customer_id": cid,
            "total_prior_transactions": len(history),
            "rolling_avg_amount": round(avg_amt, 2),
            "rolling_std_amount": round(std_amt, 2),
            "max_prior_amount": round(max(amounts), 2),
            "prior_fraud_flags": prior_fraud,
            "last_active_timestamp": last_tx_time,
            "profile_status": "ESTABLISHED"
        }

    # Baseline for cold-start customer
    return {
        "customer_id": cid,
        "total_prior_transactions": 0,
        "rolling_avg_amount": 0.0,
        "rolling_std_amount": 0.0,
        "max_prior_amount": 0.0,
        "prior_fraud_flags": 0,
        "last_active_timestamp": "NONE",
        "profile_status": "NEW_ACCOUNT"
    }

def check_known_patterns(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate deterministic fraud heuristics and rule-based red flags.
    
    Checks:
    - High velocity burst (>= 3 transactions in 5 minutes)
    - Structuring suspicion (amount just under $10,000 threshold)
    - Impossible travel velocity (> 500 km within 1 hour, requiring previous transaction)
    - Severe amount deviation (> 3.5 sigma above rolling average)
    
    Args:
        transaction: Dictionary containing current transaction attributes.
        
    Returns:
        Dict detailing triggered rules, pattern count, and heuristic severity score.
    """
    if not isinstance(transaction, dict):
        transaction = {}

    amount = float(transaction.get("amount") or 0.0)
    velocity_5m = int(transaction.get("velocity_5m") or 0)
    velocity_60m = int(transaction.get("velocity_60m") or 0)
    geo_distance = float(transaction.get("geo_distance_km") or 0.0)
    time_since_last = float(transaction.get("time_since_last_tx_sec") or 0.0)
    amount_deviation = float(transaction.get("amount_deviation") or 0.0)

    triggered_rules = []
    
    if velocity_5m >= 3:
        triggered_rules.append(f"HIGH_VELOCITY_BURST: {velocity_5m} transactions in last 5 minutes")
    if velocity_60m >= 8:
        triggered_rules.append(f"HOURLY_VELOCITY_SPIKE: {velocity_60m} transactions in last 60 minutes")
    if 9000.0 <= amount <= 9999.0:
        triggered_rules.append(f"STRUCTURING_SUSPICION: Amount ${amount:.2f} just under $10,000 reporting threshold")
    # Impossible travel requires previous activity; cold-start accounts with 0 velocity and 0 time delta cannot have traveled
    is_cold_start = (velocity_5m == 0 and velocity_60m == 0 and time_since_last <= 0.0)
    if geo_distance > 500.0 and not is_cold_start:
        delta_sec = time_since_last if time_since_last > 0 else (300.0 if velocity_5m > 0 else 3600.0)
        if delta_sec <= 3600.0:
            speed_kmh = (geo_distance / max(delta_sec, 60)) * 3600
            triggered_rules.append(f"IMPOSSIBLE_TRAVEL: {geo_distance:.1f} km jump within {delta_sec:.0f} seconds (~{speed_kmh:.0f} km/h)")
    if amount_deviation >= 3.5:
        triggered_rules.append(f"EXTREME_AMOUNT_DEVIATION: Spend is {amount_deviation:.1f} standard deviations above normal")

    severity = min(1.0, len(triggered_rules) * 0.30)
    risk_assessment = "CRITICAL" if severity >= 0.8 else ("HIGH" if severity >= 0.5 else ("MEDIUM" if severity > 0 else "LOW"))

    return {
        "triggered_count": len(triggered_rules),
        "triggered_rules": triggered_rules,
        "heuristic_score": round(severity, 2),
        "heuristic_severity_score": round(severity, 2),
        "heuristic_risk_assessment": risk_assessment
    }

def get_merchant_risk_score(merchant_id: str) -> Dict[str, Any]:
    """Lookup merchant baseline risk index, business category, and chargeback rates.
    
    Args:
        merchant_id: Unique merchant identifier (e.g., 'MERCH_012').
        
    Returns:
        Dict with merchant category, chargeback rate, and assigned risk tier.
    """
    m_id_str = str(merchant_id or "MERCH_002")
    try:
        clean_id = "".join(filter(str.isdigit, m_id_str))
        m_idx = int(clean_id) % len(MERCHANT_CATALOG) if clean_id else 2
    except Exception:
        m_idx = 2

    info = MERCHANT_CATALOG[m_idx]
    return {
        "merchant_id": m_id_str,
        "category": info["category"],
        "historical_chargeback_rate_pct": info["chargeback_rate_pct"],
        "merchant_risk_tier": info["risk_tier"]
    }

def analyze_mule_ring_network(
    customer_id: str,
    device_id: Optional[str] = None,
    ip_address: Optional[str] = None
) -> Dict[str, Any]:
    """Inspect entity graph resolution network for multi-account money mule rings and shared hardware/IPs.
    
    Args:
        customer_id: Unique customer profile ID.
        device_id: Optional hardware/canvas device fingerprint hash.
        ip_address: Optional client IP address.
        
    Returns:
        Dict containing mule ring detection flag, cluster risk level, connected customer accounts, and shared devices/IPs.
    """
    cid = str(customer_id or "CUST_UNKNOWN")
    try:
        from graph.entity_graph import entity_graph
        # Read-only query: do not mutate graph state during investigative inspection
        return entity_graph.analyze_customer_syndicate(cid)
    except Exception as e:
        logger.warning("Mule ring graph analysis failed: {err}", err=str(e))
        return {
            "customer_id": cid,
            "mule_ring_detected": False,
            "cluster_risk_level": "LOW",
            "connected_customers": [cid],
            "shared_devices": [],
            "shared_ips": [],
            "cluster_size": 1,
            "summary": "Graph engine offline; defaulting to isolated account profile."
        }

# Tool schemas for Groq / OpenAI tool calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Retrieve historical transaction activity, 30-day rolling spend averages, and prior fraud flags for a customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Unique customer identifier, e.g., 'CUST_0042'"
                    }
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_known_patterns",
            "description": "Evaluate deterministic fraud heuristics including velocity bursts, structuring under $10k, impossible travel, and amount deviations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction": {
                        "type": "object",
                        "description": "Dictionary of transaction features including amount, velocity_5m, geo_distance_km, and time_since_last_tx_sec"
                    }
                },
                "required": ["transaction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_merchant_risk_score",
            "description": "Lookup merchant category reputation, chargeback rate percentage, and assigned baseline risk tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "merchant_id": {
                        "type": "string",
                        "description": "Unique merchant identifier, e.g., 'MERCH_012'"
                    }
                },
                "required": ["merchant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_mule_ring_network",
            "description": "Perform entity resolution graph mining to detect money mule rings, device farms, and multi-account syndicates sharing devices or IP subnets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer identifier, e.g., 'CUST_0042'"
                    },
                    "device_id": {
                        "type": "string",
                        "description": "Optional device hardware fingerprint hash, e.g., 'DEV_FARM_01'"
                    },
                    "ip_address": {
                        "type": "string",
                        "description": "Optional IP address string, e.g., '198.51.100.77'"
                    }
                },
                "required": ["customer_id"]
            }
        }
    }
]

