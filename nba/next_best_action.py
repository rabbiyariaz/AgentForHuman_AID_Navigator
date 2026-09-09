"""Creates follow-up questions, urgency signals, and next-step suggestions
for incomplete or ambiguous cases.

Deterministic by design, same as the eligibility rules — this module makes
no LLM calls. It only inspects the already-extracted (possibly incomplete)
evidence and decides what should happen next. This is the layer that turns
"the LLM couldn't confidently fill this field" into "here is a specific
question a caseworker or applicant needs answered," rather than letting
that gap surface as an unhandled schema validation crash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Fields the deterministic eligibility engine cannot function without.
# These are the ones that are non-Optional on ExtractedApplicantFields
# because they're lookup keys or gating conditions, not because the
# applicant is guaranteed to have stated them.
REQUIRED_FOR_ELIGIBILITY: dict[str, str] = {
    "household_size": "How many people live in your household?",
    "annual_income": "What is your household's annual income?",
    "home_type": "Do you live in a single-family home or a multi-family/apartment building?",
    "heating_fuel_type": "What type of heat do you use — gas, oil, electric, or is heat included in your rent?",
    "disconnection_status": "Has your utility service been disconnected, or do you have a disconnection notice?",
}

# Confidence below this on a *present* field is treated the same as
# missing for follow-up purposes — a low-confidence guess is not something
# the eligibility engine should silently trust.
LOW_CONFIDENCE_THRESHOLD = 0.5

# Free-text urgency phrases that should raise case priority regardless of
# eligibility outcome. Kept intentionally small and literal rather than an
# LLM judgment call — expand this list from real intake data, don't guess.
URGENCY_SIGNALS = {
    "immediately", "today", "asap", "right away", "urgent",
    "shutoff", "shut off", "disconnected", "disconnection",
}


@dataclass
class NextBestAction:
    """Result of evaluating a partially or fully extracted case."""

    status: str  # "needs_followup" | "ready_for_eligibility" | "flag_urgent"
    missing_fields: list[str] = field(default_factory=list)
    followup_questions: list[str] = field(default_factory=list)
    urgency_flag: bool = False
    urgency_reason: str | None = None


def _get_confidence(confidences: dict[str, Any], field_name: str) -> float:
    try:
        return float(confidences.get(field_name, 0.0))
    except (TypeError, ValueError):
        return 0.0


def find_missing_required_fields(
    raw_fields: dict[str, Any],
    confidences: dict[str, Any],
) -> list[str]:
    """Return the subset of REQUIRED_FOR_ELIGIBILITY that is either null
    or present with confidence below LOW_CONFIDENCE_THRESHOLD.

    Operates on the raw LLM output dict, before Pydantic validation, so
    this can run even when ExtractedApplicantFields.model_validate would
    reject the payload outright.
    """
    missing: list[str] = []
    for field_name in REQUIRED_FOR_ELIGIBILITY:
        value = raw_fields.get(field_name)
        if value is None:
            missing.append(field_name)
            continue
        if _get_confidence(confidences, field_name) < LOW_CONFIDENCE_THRESHOLD:
            missing.append(field_name)
    return missing


def check_urgency(
    raw_fields: dict[str, Any],
    original_text: str | None = None,
) -> tuple[bool, str | None]:
    """Flag urgency from structured evidence first, free text second.

    disconnection_status=True is the authoritative structured signal.
    Free-text scanning is a fallback for cases where disconnection_status
    itself is one of the missing fields (so we still don't want to lose
    urgency awareness while a follow-up question is pending).
    """
    if raw_fields.get("disconnection_status") is True:
        return True, "disconnection_status is true"

    deadline_text = (raw_fields.get("stated_deadline_or_urgency") or "").lower()
    for phrase in URGENCY_SIGNALS:
        if phrase in deadline_text:
            return True, f"urgency phrase in stated_deadline_or_urgency: {phrase!r}"

    if original_text:
        lowered = original_text.lower()
        for phrase in URGENCY_SIGNALS:
            if phrase in lowered:
                return True, f"urgency phrase in original text: {phrase!r}"

    return False, None


def evaluate_next_action(
    raw_fields: dict[str, Any],
    confidences: dict[str, Any],
    original_text: str | None = None,
) -> NextBestAction:
    """Single entry point: decide what should happen next for this case.

    Call this instead of letting a missing required field surface as an
    unhandled ExtractedApplicantFields validation error. If status is
    "needs_followup", do not proceed to eligibility evaluation — surface
    followup_questions to the caseworker/applicant instead.
    """
    missing = find_missing_required_fields(raw_fields, confidences)
    is_urgent, urgency_reason = check_urgency(raw_fields, original_text)

    if missing:
        questions = [REQUIRED_FOR_ELIGIBILITY[f] for f in missing]
        return NextBestAction(
            status="needs_followup",
            missing_fields=missing,
            followup_questions=questions,
            urgency_flag=is_urgent,
            urgency_reason=urgency_reason,
        )

    if is_urgent:
        return NextBestAction(
            status="flag_urgent",
            urgency_flag=True,
            urgency_reason=urgency_reason,
        )

    return NextBestAction(status="ready_for_eligibility")