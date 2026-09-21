"""Mock LLM Operator for testing and deterministic offline execution."""

import time
from typing import Dict, Any
from core.interfaces import BaseLLMOperator
from core.logger import get_logger
from agent.guardrails import validate_agent_report

logger = get_logger("mock_llm")

class MockLLMOperator(BaseLLMOperator):
    """Deterministic offline LLM operator."""

    async def health_check(self) -> bool:
        return True

    async def investigate(self, transaction: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        start = time.time()
        amount = float(transaction.get("amount", 0.0))
        velocity_5m = int(transaction.get("velocity_5m", 0))
        
        is_suspicious = (amount > 1000.0) or (velocity_5m >= 3)
        risk_level = "CRITICAL" if (amount > 5000 or velocity_5m >= 5) else ("HIGH" if is_suspicious else "LOW")
        recommendation = "DECLINE" if risk_level in {"CRITICAL", "HIGH"} else "APPROVE"

        cited_facts = [
            f"Transaction amount: ${amount:.2f}",
            f"velocity_5m count: {velocity_5m}"
        ]

        tool_outputs = [
            {"amount": amount, "velocity_5m": velocity_5m},
            {"customer_history_checked": True, "prior_fraud_flags": 0}
        ]

        report = {
            "risk_level": risk_level,
            "recommendation": recommendation,
            "summary": f"Mock investigation verdict: {recommendation} based on heuristic criteria.",
            "evidence": [
                f"Evaluated amount (${amount:.2f}) and rolling velocity ({velocity_5m})."
            ],
            "cited_facts": cited_facts
        }

        guardrails = validate_agent_report(report, tool_outputs)
        latency_ms = round((time.time() - start) * 1000, 2)

        logger.info("Completed Mock LLM investigation for tx={tx_id} | decision={dec}",
                    tx_id=transaction.get("transaction_id"), dec=recommendation)

        return {
            "report": report,
            "guardrails": guardrails,
            "latency_ms": latency_ms,
            "provider": "mock"
        }
