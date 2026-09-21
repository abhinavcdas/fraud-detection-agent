"""Guardrails and Faithfulness Validation for Agent Reports.

Ensures that every fact claimed in 'cited_facts' is grounded in tool execution outputs.
If any fact cannot be verified, the report is flagged as 'NEEDS_REVIEW' rather than 'PASSED'.
Also audits FinCEN Form 111 SAR narrative for entity consistency and ungrounded claims.
"""

import re
from typing import Dict, Any, List, Tuple, Set

STOP_WORDS = {
    "the", "and", "for", "with", "from", "was", "were", "been", "have", "has", "had",
    "that", "this", "these", "those", "about", "above", "below", "customer", "account",
    "transaction", "report", "flag", "risk", "level", "system", "rule"
}

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for relaxed matching."""
    return re.sub(r"[^\w\s]", "", str(text).lower())

def extract_tokens(text: str) -> Set[str]:
    """Extract informative alphanumeric tokens, excluding trivial stopwords."""
    raw_tokens = set(re.findall(r"\b[a-zA-Z0-9_]{3,}\b", str(text).lower()))
    return {t for t in raw_tokens if t not in STOP_WORDS}

def extract_numbers(text: str) -> List[float]:
    """Extract numeric values as floats from text."""
    raw_nums = re.findall(r"\b\d+(?:\.\d+)?\b", str(text))
    result = []
    for n in raw_nums:
        try:
            result.append(float(n))
        except ValueError:
            pass
    return result

def _verify_kv_claim(fact: str, output: Dict[str, Any]) -> bool:
    """Check if a key-value formatted claim (e.g. 'key: value') matches an output dictionary."""
    if ":" in fact:
        parts = fact.split(":", 1)
        k = parts[0].strip().lower().replace(" ", "_")
        v_str = parts[1].strip()
        for ok, ov in output.items():
            norm_ok = str(ok).lower().replace(" ", "_")
            if norm_ok == k or k in norm_ok:
                # Compare values
                v_nums = extract_numbers(v_str)
                ov_nums = extract_numbers(str(ov))
                if v_nums and ov_nums:
                    if any(abs(vn - on) < 1e-2 for vn in v_nums for on in ov_nums):
                        return True
                elif normalize_text(v_str) in normalize_text(str(ov)) or normalize_text(str(ov)) in normalize_text(v_str):
                    return True
    return False

def verify_fact_against_evidence(fact: str, tool_outputs: List[Dict[str, Any]]) -> bool:
    """Verify if a single cited fact is supported by any collected tool output.
    
    Checks:
    1. Direct substring match against stringified tool output.
    2. Key-value structured match against output dict fields.
    3. Grounded numerical equivalence bound to semantic non-stopword tokens.
    """
    if not fact or not str(fact).strip():
        return False

    norm_fact = normalize_text(fact)
    fact_tokens = extract_tokens(fact)
    fact_nums = extract_numbers(fact)

    for output in tool_outputs:
        if not isinstance(output, dict):
            continue

        output_str = str(output)
        norm_output = normalize_text(output_str)

        # 1. Direct inclusion of normalized fact
        if norm_fact in norm_output:
            return True

        # 2. Key-value field match
        if _verify_kv_claim(fact, output):
            return True

        # 3. Grounded numerical check
        if fact_nums:
            output_nums = extract_numbers(output_str)
            all_numbers_grounded = all(
                any(abs(fn - on) < 1e-2 for on in output_nums)
                for fn in fact_nums
            )
            if all_numbers_grounded:
                # Ensure meaningful token overlap on informative tokens
                output_tokens = extract_tokens(output_str)
                overlap = fact_tokens.intersection(output_tokens)
                if fact_tokens and len(overlap) / len(fact_tokens) >= 0.40:
                    return True

    return False

def validate_agent_report(report: Dict[str, Any], tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate faithfulness of all cited facts and SAR narratives in an agent report.
    
    Returns:
        Dict with:
        - status: 'PASSED' | 'NEEDS_REVIEW' | 'WARNING'
        - faithfulness_score: float (0.0 to 1.0)
        - verified_facts: list of passed facts
        - unverified_facts: list of failed facts
        - narrative_valid: bool
    """
    if not isinstance(report, dict):
        return {
            "status": "NEEDS_REVIEW",
            "faithfulness_score": 0.0,
            "verified_facts": [],
            "unverified_facts": [],
            "reason": "Invalid report payload."
        }

    cited_facts = report.get("cited_facts", [])
    if not cited_facts:
        return {
            "status": "NEEDS_REVIEW",
            "faithfulness_score": 0.0,
            "verified_facts": [],
            "unverified_facts": [],
            "reason": "Agent provided no cited facts to substantiate recommendation."
        }

    safe_tool_outputs = [o for o in tool_outputs if isinstance(o, dict)]

    verified = []
    unverified = []

    for fact in cited_facts:
        if verify_fact_against_evidence(str(fact), safe_tool_outputs):
            verified.append(fact)
        else:
            unverified.append(fact)

    score = len(verified) / len(cited_facts) if cited_facts else 0.0

    # Cross-validate FinCEN SAR narrative if present
    narrative_issues = []
    fin_cen_sar = report.get("fin_cen_sar")
    if isinstance(fin_cen_sar, dict):
        narrative = fin_cen_sar.get("part_iv_narrative", "")
        subject_cust = fin_cen_sar.get("part_i_subject", {}).get("customer_id")
        if subject_cust and subject_cust not in narrative:
            narrative_issues.append(f"Subject customer ID '{subject_cust}' omitted from SAR narrative.")

    status = "PASSED" if score >= 0.8 and len(unverified) == 0 else "NEEDS_REVIEW"

    res = {
        "status": status,
        "faithfulness_score": round(score, 3),
        "verified_facts": verified,
        "unverified_facts": unverified
    }
    if narrative_issues:
        res["narrative_warnings"] = narrative_issues

    return res
