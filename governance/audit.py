"""Audit Logging & Cryptographic Integrity Utilities for Model Risk Management.

Ensures every model inference and agent investigation is permanently recorded
with a tamper-evident SHA-256 input hash, model version, risk scores, and guardrails.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.logger import get_logger

logger = get_logger("governance_audit")


def compute_input_hash(transaction: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of transaction input payload.
    
    Args:
        transaction: Transaction dictionary.
        
    Returns:
        Hex-encoded SHA-256 string for audit trail verification.
    """
    # Exclude dynamic audit metadata fields if present
    filtered = {
        k: v for k, v in transaction.items()
        if k not in ("audit_id", "timestamp", "created_at", "raw_json")
    }
    canonical_json = json.dumps(filtered, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def build_audit_record(
    transaction: Dict[str, Any],
    fraud_score: float,
    is_flagged: bool,
    model_version: str = "fraud-xgb-v1",
    agent_result: Optional[Dict[str, Any]] = None,
    latency_ms: float = 0.0
) -> Dict[str, Any]:
    """Construct an immutable audit record matching the audit_log schema.
    
    Args:
        transaction: Scored transaction dictionary.
        fraud_score: Model predicted fraud probability (0.0 to 1.0).
        is_flagged: Whether transaction breached operational threshold.
        model_version: Serialized model identifier.
        agent_result: Output dossier from LLM investigation agent (if triggered).
        latency_ms: Total end-to-end processing latency.
        
    Returns:
        Dict conforming to audit_log table specifications.
    """
    input_hash = compute_input_hash(transaction)
    tx_id = transaction.get("transaction_id", "UNKNOWN")
    now_iso = datetime.now(timezone.utc).isoformat()

    agent_decision = "CLEARED"
    agent_report = None
    guardrail_status = "SKIPPED"
    faithfulness_score = None

    if agent_result:
        report = agent_result.get("report", {})
        guardrails = agent_result.get("guardrails", {})
        agent_decision = report.get("recommendation", "MANUAL_REVIEW")
        agent_report = report
        guardrail_status = guardrails.get("status", "PASSED")
        faithfulness_score = guardrails.get("faithfulness_score", 1.0)
    elif is_flagged:
        agent_decision = "FLAGGED_PENDING_REVIEW"

    return {
        "transaction_id": tx_id,
        "timestamp": now_iso,
        "input_hash": input_hash,
        "model_version": model_version,
        "fraud_score": round(float(fraud_score), 4),
        "is_flagged": bool(is_flagged),
        "agent_decision": agent_decision,
        "agent_report": agent_report,
        "guardrail_status": guardrail_status,
        "faithfulness_score": faithfulness_score,
        "latency_ms": round(float(latency_ms), 2)
    }
