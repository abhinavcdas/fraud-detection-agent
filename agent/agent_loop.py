"""Agent Loop using Groq for Real-Time Fraud Investigation.

Orchestrates multi-turn tool calling using Groq's high-speed inference engine (llama-3.3-70b-versatile),
executes selected tools against active storage and heuristic engines,
and enforces strict guardrail faithfulness validation on the final report.
"""

import os
import sys
import re
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.logger import get_logger, bind_tx_context
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    TOOL_DEFINITIONS,
    get_customer_history,
    check_known_patterns,
    get_merchant_risk_score,
    analyze_mule_ring_network
)
from agent.guardrails import validate_agent_report

load_dotenv()

logger = get_logger("agent_loop")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

AVAILABLE_TOOLS = {
    "get_customer_history": get_customer_history,
    "check_known_patterns": check_known_patterns,
    "get_merchant_risk_score": get_merchant_risk_score,
    "analyze_mule_ring_network": analyze_mule_ring_network
}


def execute_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool function safely."""
    func = AVAILABLE_TOOLS.get(name)
    if not func:
        return {"error": f"Tool '{name}' not found"}
    try:
        return func(**arguments)
    except Exception as e:
        logger.warning("Tool '{name}' execution failed: {err}", name=name, err=str(e))
        return {"error": f"Tool execution error: {str(e)}"}

def _run_deterministic_investigation(transaction: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Deterministic grounded investigation when running offline or when API key is unconfigured."""
    cust_id = transaction.get("customer_id", "CUST_UNKNOWN")
    merch_id = transaction.get("merchant_id", "MERCH_UNKNOWN")
    amount = float(transaction.get("amount", 0.0))
    dev_id = transaction.get("device_id", "DEV_UNKNOWN")
    ip_addr = transaction.get("ip_address", transaction.get("ip", "IP_UNKNOWN"))

    # Execute all tools directly
    history_out = get_customer_history(cust_id)
    pattern_out = check_known_patterns(transaction)
    merchant_out = get_merchant_risk_score(merch_id)
    mule_out = analyze_mule_ring_network(cust_id, device_id=dev_id, ip_address=ip_addr)

    tool_outputs = [
        {"tool": "get_customer_history", "output": history_out},
        {"tool": "check_known_patterns", "output": pattern_out},
        {"tool": "get_merchant_risk_score", "output": merchant_out},
        {"tool": "analyze_mule_ring_network", "output": mule_out}
    ]

    # Grounded decision heuristic
    sev_score = pattern_out.get("heuristic_severity_score", 0.0)
    merchant_tier = merchant_out.get("merchant_risk_tier", "LOW")
    mule_detected = mule_out.get("mule_ring_detected", False)

    if sev_score >= 0.6 or merchant_tier == "CRITICAL" or mule_detected or amount > 5000.0:
        risk_level = "CRITICAL"
        recommendation = "DECLINE"
    elif sev_score >= 0.3 or merchant_tier == "HIGH":
        risk_level = "HIGH"
        recommendation = "ESCALATE"
    elif amount > 500.0 or merchant_tier == "MEDIUM":
        risk_level = "MEDIUM"
        recommendation = "MONITOR"
    else:
        risk_level = "LOW"
        recommendation = "APPROVE"

    evidence = []
    cited_facts = []

    # Ground facts strictly in tool outputs
    if pattern_out.get("triggered_rules"):
        for rule in pattern_out["triggered_rules"]:
            evidence.append(f"Heuristic Red Flag: {rule}")
            cited_facts.append(rule)

    if history_out.get("total_prior_transactions", 0) > 0:
        evidence.append(f"Customer historical rolling average is ${history_out['rolling_avg_amount']:.2f}")
        cited_facts.append(f"rolling_avg_amount: ${history_out['rolling_avg_amount']:.2f}")
    else:
        evidence.append(f"Customer account is new with 0 prior transactions on record.")
        cited_facts.append(f"total_prior_transactions: 0")

    evidence.append(f"Merchant category '{merchant_out['category']}' exhibits {merchant_out['historical_chargeback_rate_pct']:.1f}% chargeback incidence (Tier: {merchant_tier}).")
    cited_facts.append(f"merchant_risk_tier: {merchant_tier}")

    if mule_detected:
        evidence.append(f"Entity Graph Alert: {mule_out['summary']}")
        cited_facts.append(f"mule_ring_detected: True (Cluster size: {mule_out['cluster_size']})")

    fin_cen_sar = None
    if risk_level in ["HIGH", "CRITICAL"]:
        vel_desc = f"{transaction.get('velocity_5m', 0)} transactions in 5 minutes"
        geo_desc = f"{transaction.get('geo_distance_km', 0.0)} km geo-jump"
        narrative = (
            f"Subject {cust_id} attempted an unauthorized transaction of ${amount:,.2f} with {merchant_out.get('category')} "
            f"(Merchant Tier: {merchant_tier}). Anomaly signals include {vel_desc} and {geo_desc}. "
            f"{mule_out.get('summary', '')} "
            f"Autonomous Model Risk Management issued action {recommendation} and recommends permanent account restriction."
        )
        fin_cen_sar = {
            "filing_type": "INITIAL_REPORT",
            "part_i_subject": {
                "customer_id": cust_id,
                "device_fingerprint": dev_id,
                "ip_address": ip_addr,
                "account_status": "RESTRICTED" if recommendation == "DECLINE" else "MONITORED"
            },
            "part_ii_suspicious_activity": {
                "transaction_amount": f"${amount:,.2f}",
                "activity_date": str(transaction.get("timestamp", "RECENT")),
                "merchant_mcc": f"{merch_id} ({merchant_out.get('category')})",
                "velocity_indicator": vel_desc,
                "impossible_travel_jump": geo_desc
            },
            "part_iii_financial_institution": {
                "institution_name": "Autonomous Fintech Fraud Operations Bank",
                "model_version": "fraud-xgb-v1",
                "calibrated_threshold": 0.38,
                "scoring_engine_action": f"STEP_UP_{recommendation}"
            },
            "part_iv_narrative": narrative
        }

    report = {
        "risk_level": risk_level,
        "recommendation": recommendation,
        "summary": f"Automated risk investigation recommends {recommendation} (Risk: {risk_level}).",
        "evidence": evidence,
        "cited_facts": cited_facts,
        "fin_cen_sar": fin_cen_sar
    }

    raw_tool_data = [history_out, pattern_out, merchant_out, mule_out]
    guardrail_result = validate_agent_report(report, raw_tool_data)
    latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "report": report,
        "tool_history": tool_outputs,
        "guardrails": guardrail_result,
        "latency_ms": latency_ms,
        "provider": "deterministic_investigator"
    }


def run_investigation(transaction: Dict[str, Any], max_turns: int = 4) -> Dict[str, Any]:
    """Run full investigation workflow on a flagged transaction using Groq.
    
    Args:
        transaction: Flagged transaction dictionary.
        max_turns: Maximum conversation turns for tool calling.
        
    Returns:
        Dict containing structured report, collected tool outputs, guardrail audit, and latency.
    """
    start_time = time.time()
    tx_id = transaction.get("transaction_id", "TX_UNKNOWN")
    cust_id = transaction.get("customer_id", "CUST_UNKNOWN")

    tx_logger = bind_tx_context(logger, tx_id, cust_id)

    # Fallback to deterministic grounded investigator if API key is not configured
    if not GROQ_API_KEY:
        tx_logger.info("GROQ_API_KEY not set. Using deterministic grounded investigator.")
        return _run_deterministic_investigation(transaction, start_time)

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Investigate suspicious flagged transaction:\n{json.dumps(transaction, indent=2)}"}
        ]

        collected_tool_outputs = []

        for turn in range(max_turns):
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.1
            )

            msg = response.choices[0].message

            # Process tool calls (native API tool_calls or inline pseudo-tool calling syntax)
            raw_content = msg.content or ""
            inline_called = False

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments or "{}")
                    out = execute_tool_call(fn_name, fn_args)
                    collected_tool_outputs.append(out)

                    tx_logger.debug("Executed agent tool: {fn} | args={args}", fn=fn_name, args=fn_args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": fn_name,
                        "content": json.dumps(out)
                    })
            elif "<tool_call>" in raw_content or "<function=" in raw_content:
                # Handle models emitting inline pseudo-tool calling syntax
                match = re.search(r"<function=([a-zA-Z0-9_]+)>(.*?)(?:</function>|</tool_call>|$)", raw_content, re.DOTALL)
                if match:
                    fn_name = match.group(1).strip()
                    args_str = match.group(2).strip()
                    try:
                        fn_args = json.loads(args_str) if args_str else {}
                    except json.JSONDecodeError:
                        fn_args = {}
                    out = execute_tool_call(fn_name, fn_args)
                    collected_tool_outputs.append(out)
                    tx_logger.debug("Executed inline agent tool: {fn} | args={args}", fn=fn_name, args=fn_args)
                    messages.append({"role": "assistant", "content": raw_content})
                    messages.append({
                        "role": "user",
                        "content": f"Tool '{fn_name}' execution result: {json.dumps(out)}. Synthesize final JSON report now."
                    })
                    inline_called = True

            if inline_called:
                continue

            if not msg.tool_calls:
                # Terminal answer reached
                try:
                    # Strip any markdown code fences if model enclosed JSON in ```json ... ```
                    cleaned_content = raw_content.strip()
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content.split("\n", 1)[1]
                        if cleaned_content.endswith("```"):
                            cleaned_content = cleaned_content.rsplit("\n", 1)[0]
                    # Also strip any stray thought or tool tags before parsing
                    cleaned_content = re.sub(r"<thought>.*?</thought>", "", cleaned_content, flags=re.DOTALL).strip()
                    report = json.loads(cleaned_content)
                except json.JSONDecodeError:
                    # Sanitize any raw XML or leaked tool tags to ensure clean audit dossiers
                    sanitized_summary = re.sub(r"<[^>]+>", " ", raw_content).strip()
                    sanitized_summary = re.sub(r"\s+", " ", sanitized_summary)
                    if not sanitized_summary or len(sanitized_summary) < 10:
                        sanitized_summary = "Automated risk investigation recommends ESCALATE for forensic manual review."
                    else:
                        sanitized_summary = sanitized_summary[:300]

                    report = {
                        "risk_level": "HIGH",
                        "recommendation": "ESCALATE",
                        "summary": sanitized_summary,
                        "evidence": ["Unstructured model response; sanitized and routed to forensic review."],
                        "cited_facts": []
                    }

                guardrails = validate_agent_report(report, collected_tool_outputs)
                latency_ms = round((time.time() - start_time) * 1000, 2)

                tx_logger.info(
                    "Groq investigation complete | decision={dec} | risk={risk} | guardrail={gr} ({ms}ms)",
                    dec=report.get("recommendation"),
                    risk=report.get("risk_level"),
                    gr=guardrails.get("status"),
                    ms=latency_ms
                )

                return {
                    "report": report,
                    "tool_history": collected_tool_outputs,
                    "guardrails": guardrails,
                    "latency_ms": latency_ms,
                    "provider": "groq",
                    "model": GROQ_MODEL
                }

        # Max turns exceeded
        fallback_report = {
            "risk_level": "HIGH",
            "recommendation": "ESCALATE",
            "summary": "Investigation turn limit reached without final convergence.",
            "evidence": ["Max tool execution turns exceeded."],
            "cited_facts": []
        }
        return {
            "report": fallback_report,
            "tool_history": collected_tool_outputs,
            "guardrails": validate_agent_report(fallback_report, collected_tool_outputs),
            "latency_ms": round((time.time() - start_time) * 1000, 2),
            "provider": "groq_max_turns"
        }

    except Exception as exc:
        tx_logger.warning("Groq API error encountered ({err}). Falling back to deterministic investigation.", err=str(exc))
        return _run_deterministic_investigation(transaction, start_time)

if __name__ == "__main__":
    sample_flagged_tx = {
        "transaction_id": "TX_LIVE_DEMO_01",
        "customer_id": "CUST_0042",
        "merchant_id": "MERCH_002",
        "amount": 2450.00,
        "velocity_5m": 4,
        "velocity_60m": 7,
        "amount_deviation": 4.5,
        "geo_distance_km": 680.0,
        "time_since_last_tx_sec": 45.0
    }
    result = run_investigation(sample_flagged_tx)
    logger.info("Investigation Dossier:\n{dossier}", dossier=json.dumps(result, indent=2))
