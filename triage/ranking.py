"""Ranks triage candidates by urgency, confidence, and completeness for case workflow ordering.

Deterministic by design, same as eligibility.py — no LLM judgment here.
Each case gets a priority tier (1 = solve first) plus a written reason,
mirroring the "rule_fired" transparency pattern already used in eligibility.
"""

from __future__ import annotations

from typing import Any

from rules.eligibility import check_crisis_eligibility, check_regular_eligibility


# Lower number = higher priority = solve first.
# Lower number = higher priority = solve first.
TIER_CRISIS_ELIGIBLE = 1
TIER_URGENT_INCOMPLETE = 2
TIER_CONTRADICTION = 3
TIER_REGULAR_ELIGIBLE = 4
TIER_INCOMPLETE_NO_URGENCY = 5
TIER_INELIGIBLE = 6
 
 
def _case_confidence(audit: dict[str, Any]) -> float:
    """Weakest-link confidence: the lowest confidence among fields actually
    present in the audit. A case is only as trustworthy as its shakiest
    extracted field — averaging would let one bad field hide behind
    several confident ones.
 
    Returns 1.0 for an audit with no fields at all (nothing to distrust),
    so an empty audit doesn't artificially sort as "worst confidence."
    """
    confidences = [
        entry["confidence"]
        for entry in audit.values()
        if isinstance(entry, dict) and "confidence" in entry
    ]
    if not confidences:
        return 1.0
    return min(confidences)
 
 
def classify_case(pipeline_result: dict[str, Any]) -> dict[str, Any]:
    """Assign a priority tier and reason to a single pipeline_result
    (the dict returned by run_extraction_pipeline).
 
    Runs eligibility itself when the case is complete enough to evaluate,
    so callers only need to hand in the pipeline output, not pre-run
    eligibility separately.
    """
    status = pipeline_result["status"]
    urgency_flag = pipeline_result["next_action"]["urgency_flag"]
 
    if status == "needs_followup":
        confidence = _case_confidence(pipeline_result["partial_audit"])
        if urgency_flag:
            return {
                "tier": TIER_URGENT_INCOMPLETE,
                "reason": "urgent_signal_present_but_required_fields_missing: " +
                          f"missing={pipeline_result['next_action']['missing_fields']}",
                "confidence": confidence,
                "eligibility": None,
            }
        return {
            "tier": TIER_INCOMPLETE_NO_URGENCY,
            "reason": "required_fields_missing_no_urgency_signal: " +
                      f"missing={pipeline_result['next_action']['missing_fields']}",
            "confidence": confidence,
            "eligibility": None,
        }
 
    if status == "needs_review":
        # Text and document evidence disagree beyond tolerance. This is
        # NOT the same as missing data — the case is complete but
        # untrustworthy until a caseworker resolves which source is
        # correct. Ranked above regular-eligible cases because an unresolved
        # contradiction could flip the eligibility outcome once settled.
        confidence = _case_confidence(pipeline_result["partial_audit"])
        contradiction_fields = [c["field"] for c in pipeline_result.get("contradictions", [])]
        return {
            "tier": TIER_CONTRADICTION,
            "reason": f"unresolved_contradiction_between_text_and_document: fields={contradiction_fields}",
            "confidence": confidence,
            "eligibility": None,
        }
 
    # status == "ready_for_eligibility" — evaluate against the real rules.
    extraction_output = pipeline_result["extraction_output"]
    confidence = _case_confidence(extraction_output["audit"])
 
    regular = check_regular_eligibility(extraction_output)
    crisis = check_crisis_eligibility(extraction_output)
 
    if crisis["eligible"]:
        return {
            "tier": TIER_CRISIS_ELIGIBLE,
            "reason": f"crisis_eligible: {crisis['rule_fired']}",
            "confidence": confidence,
            "eligibility": {"regular": regular, "crisis": crisis},
        }
 
    if regular["eligible"]:
        return {
            "tier": TIER_REGULAR_ELIGIBLE,
            "reason": f"regular_eligible_not_crisis: {regular['rule_fired']}",
            "confidence": confidence,
            "eligibility": {"regular": regular, "crisis": crisis},
        }
 
    return {
        "tier": TIER_INELIGIBLE,
        "reason": f"ineligible: {regular['rule_fired']}",
        "confidence": confidence,
        "eligibility": {"regular": regular, "crisis": crisis},
    }
 
 
def rank_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified = []
    for case in cases:
        triage_info = classify_case(case)
        classified.append({**case, "triage": triage_info})
 
    classified.sort(key=lambda c: (c["triage"]["tier"], c["triage"]["confidence"]))
    return classified
 
