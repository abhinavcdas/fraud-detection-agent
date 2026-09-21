"""Guardrails and Faithfulness Validation for Agent Reports.

Ensures that every fact claimed in 'cited_facts' is grounded in tool execution outputs.
If any fact cannot be verified, the report is flagged as 'NEEDS_REVIEW' rather than 'PASSED'.
"""

import re
from typing import Dict, Any, List, Tuple

def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation for relaxed matching."""
    return re.sub(r"[^\w\s]", "", str(text).lower())

def extract_tokens(text: str) -> set:
    """Extract informative alphanumeric tokens."""
    return set(re.findall(r"\b\w{3,}\b", str(text).lower()))

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

def verify_fact_against_evidence(fact: str, tool_outputs: List[Dict[str, Any]]) -> bool:
    """Verify if a single cited fact is supported by any collected tool output.
    
    Checks:
    1. Direct substring match against stringified tool output.
    2. Numerical equivalence & token overlap.
    """
    fact_tokens = extract_tokens(fact)
    fact_nums = extract_numbers(fact)
    
    for output in tool_outputs:
        output_str = str(output)
        
        # Check direct inclusion
        if normalize_text(fact) in normalize_text(output_str):
            return True
            
        # Check numerical equivalence
        if fact_nums:
            output_nums = extract_numbers(output_str)
            all_numbers_grounded = all(
                any(abs(fn - on) < 1e-2 for on in output_nums)
                for fn in fact_nums
            )
            if all_numbers_grounded:
                # Check semantic token overlap
                output_tokens = extract_tokens(output_str)
                overlap = fact_tokens.intersection(output_tokens)
                if len(overlap) / max(len(fact_tokens), 1) >= 0.25:
                    return True

    return False

def validate_agent_report(report: Dict[str, Any], tool_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate faithfulness of all cited facts in an agent report.
    
    Returns:
        Dict with:
        - status: 'PASSED' | 'NEEDS_REVIEW'
        - faithfulness_score: float (0.0 to 1.0)
        - verified_facts: list of passed facts
        - unverified_facts: list of failed facts
    """
    cited_facts = report.get("cited_facts", [])
    if not cited_facts:
        return {
            "status": "NEEDS_REVIEW",
            "faithfulness_score": 0.0,
            "verified_facts": [],
            "unverified_facts": [],
            "reason": "Agent provided no cited facts to substantiate recommendation."
        }

    verified = []
    unverified = []

    for fact in cited_facts:
        if verify_fact_against_evidence(fact, tool_outputs):
            verified.append(fact)
        else:
            unverified.append(fact)

    score = len(verified) / len(cited_facts)
    status = "PASSED" if score >= 0.8 and len(unverified) == 0 else "NEEDS_REVIEW"

    return {
        "status": status,
        "faithfulness_score": round(score, 3),
        "verified_facts": verified,
        "unverified_facts": unverified
    }
