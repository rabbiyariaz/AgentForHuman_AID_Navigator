from __future__ import annotations

import json
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from dataclasses import asdict
from uuid import uuid4

from nba.next_best_action import evaluate_next_action


from groq import Groq
from strands.tools import tool
from dotenv import load_dotenv
from schema.models import (
    EvidenceField,
    EvidenceSource,
    ExtractionAudit,
    ExtractionOutput,
    ExtractedApplicantFields,
)
load_dotenv()

_tool_state: ContextVar[dict[str, Any] | None] = ContextVar("aid_navigator_tool_state", default=None)


def begin_tool_state() -> None:
    _tool_state.set({"evidence": {}, "contradictions": {}, "calls": []})


def _current_tool_state() -> dict[str, Any]:
    state = _tool_state.get()
    if state is None:
        begin_tool_state()
        state = _tool_state.get()
    assert state is not None
    return state


def _store_evidence(payload: dict[str, Any]) -> str:
    reference = f"evidence_{uuid4().hex[:10]}"
    _current_tool_state()["evidence"][reference] = payload
    return reference


def get_tool_state() -> dict[str, Any]:
    return _current_tool_state()


def _claim_tool_call(name: str) -> None:
    calls = _current_tool_state()["calls"]
    if name in calls:
        raise ValueError(f"Tool {name} was already called for this case.")
    if len(calls) >= 4:
        raise ValueError("Maximum of four tool calls reached for this case.")
    calls.append(name)
# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------
# Groq is used directly here so the extraction call is explicit, auditable,
# and easy to swap models on.

_GROQ_MODEL_ID = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
_groq_client = None



def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it in your environment before running."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


EXTRACTION_FIELDS_SPEC = """
Extract exactly these fields from the applicant's text. Return ONLY a
single JSON object, no prose, no markdown fences.

Fields:
- household_size: integer, number of people in the household
- annual_income: number, stated annual income in dollars
- home_type: one of "SF" (single-family) or "MF" (multi-family/apartment)
- heating_fuel_type: one of "gas", "oil", "electric", "HIR"
  (HIR = heat included in rent, use only if that is the actual heating
  arrangement, not just an incidental mention)
- disconnection_status: boolean, true if utility service has been or is
  about to be disconnected
- arrearage_amount: number or null, amount owed/past due if stated
- fuel_tank_percent: number or null, oil tank fill percentage if stated
  (only relevant for oil heating)

- heat_included_in_rent: boolean or null. true only if the applicant
  explicitly states heat is included in their rent; false only if the
  applicant explicitly states it is NOT included. If rent or heat
  inclusion is never mentioned at all, this must be null — do not
  default to false just because the topic wasn't raised. (This can be
  true independent of heating_fuel_type if HIR was not chosen as the
  fuel type.)

- stated_deadline_or_urgency: string or null, any deadline or urgency
  language used (e.g. "by Friday", "immediately")
- applicant_name: string or null, the applicant's name if stated

For each field, also return a confidence score between 0.0 and 1.0
reflecting how explicitly and unambiguously that field was stated in
the text. A field that is directly and clearly stated should score
0.9-1.0. A field that is inferred or implied should score lower
(0.4-0.7). A field that is not mentioned at all must be null, with
confidence 0.0.

Return this exact JSON shape:
{
  "fields": {
    "household_size": <int or null>,
    "annual_income": <number or null>,
    "home_type": <"SF" | "MF" | null>,
    "heating_fuel_type": <"gas" | "oil" | "electric" | "HIR" | null>,
    "disconnection_status": <bool>,
    "arrearage_amount": <number or null>,
    "fuel_tank_percent": <number or null>,
    "heat_included_in_rent": <bool or null>,
    "stated_deadline_or_urgency": <string or null>,
    "applicant_name": <string or null>
  },
  "confidence": {
    "household_size": <0.0-1.0>,
    "annual_income": <0.0-1.0>,
    "home_type": <0.0-1.0>,
    "heating_fuel_type": <0.0-1.0>,
    "disconnection_status": <0.0-1.0>,
    "arrearage_amount": <0.0-1.0>,
    "fuel_tank_percent": <0.0-1.0>,
    "heat_included_in_rent": <0.0-1.0>,
    "stated_deadline_or_urgency": <0.0-1.0>,
    "applicant_name": <0.0-1.0>
  }
}

Do not invent values that are not stated or clearly implied in the text.
If a field is genuinely absent, use null and confidence 0.0. Extract
what is stated even if it seems internally inconsistent with other
statements — do not resolve contradictions yourself, just report what
was said.
"""


def _call_llm(input_text: str) -> dict[str, Any]:
    """Call Groq at temperature 0 and return the parsed JSON response.

    Raises ValueError if the model does not return valid, parseable JSON.
    """
    client = _get_groq_client()

    prompt = f"{EXTRACTION_FIELDS_SPEC}\n\nApplicant text:\n\"\"\"\n{input_text}\n\"\"\""

    response = client.chat.completions.create(
        model=_GROQ_MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )

    usage = response.usage
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    print(f"[LLM call tokens] input={input_tokens}  output={output_tokens}")

    raw_text = response.choices[0].message.content.strip()

    # Guard against the model wrapping output in markdown fences despite
    # instructions not to (response_format=json_object should prevent this,
    # but keep the guard — cheap insurance).
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[len("json"):].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM extraction did not return valid JSON. Raw response: {raw_text!r}"
        ) from exc

    if "fields" not in parsed or "confidence" not in parsed:
        raise ValueError(
            f"LLM extraction response missing required 'fields'/'confidence' keys: {parsed!r}"
        )

    return parsed


def _read_document_text(document_path_or_text: str) -> str:
    candidate = Path(document_path_or_text)
    if candidate.suffix and not candidate.exists():
        raise FileNotFoundError(
            f"'{document_path_or_text}' looks like a file path but does not exist. "
            f"Resolved to: {candidate.resolve()}"
        )
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return str(document_path_or_text)



def _build_extraction_output(
    llm_response: dict[str, Any],
    source: EvidenceSource,
    original_text: str | None = None,
) -> dict[str, Any]:
    """Evaluate next-best-action first, then validate against the strict
    schema only when the required fields are actually present with
    sufficient confidence.

    Returns a dict (not ExtractionOutput directly) because an incomplete
    case has no valid ExtractedApplicantFields/ExtractionAudit to return —
    forcing that shape on a partial case is the same mistake as the
    original schema requiring every field to exist. Partial cases still
    carry every field that *did* extract successfully; nothing gets
    silently dropped just because two fields are missing.
    """
    fields = llm_response["fields"]
    confidences = llm_response["confidence"]

    next_action = evaluate_next_action(fields, confidences, original_text=original_text)

    # Always build the audit for whatever fields are present, regardless
    # of whether the case is complete — this is the "partial data" the
    # caseworker sees alongside follow-up questions.
    partial_audit: dict[str, dict[str, Any]] = {}
    for field_name in fields:
        value = fields.get(field_name)
        if value is None:
            continue
        confidence = float(confidences.get(field_name, 0.0))
        partial_audit[field_name] = EvidenceField(
            value=value,
            confidence=confidence,
            source=source,
        ).model_dump(mode="python")

    if next_action.status == "needs_followup":
        return {
            "next_action": asdict(next_action),
            "extracted": None,
            "audit": None,
            "partial_fields": fields,
            "partial_audit": partial_audit,
        }

    # Required fields are present with sufficient confidence — attempt
    # full validation. Still wrapped, since confidence-based screening in
    # NBA is a heuristic, not a guarantee the value is schema-valid
    # (e.g. a stray type mismatch NBA doesn't check for).
    try:
        extracted = ExtractedApplicantFields.model_validate(fields)
    except Exception as exc:
        raise ValueError(
            f"LLM-extracted fields passed NBA's required-field check but "
            f"failed schema validation anyway. Raw fields: {fields!r}. Error: {exc}"
        ) from exc

    audit = ExtractionAudit.model_validate(partial_audit)

    output = ExtractionOutput(extracted=extracted, audit=audit)
    return {
        "next_action": asdict(next_action),
        "extracted": output.extracted.model_dump(mode="python"),
        "audit": output.audit.model_dump(mode="python"),
        "partial_fields": fields,
        "partial_audit": partial_audit,
    }


@tool(description="Extract applicant evidence from a raw text intake note using an LLM.", name="extract_from_text")
def extract_from_text(text: str) -> dict[str, Any]:
    _claim_tool_call("extract_from_text")
    normalised = (text or "").strip()
    if not normalised:
        raise ValueError("Input text is empty; no applicant evidence could be extracted.")

    llm_response = _call_llm(normalised)
    evidence = _build_extraction_output(llm_response, "text", original_text=normalised)
    reference = _store_evidence(evidence)
    next_action = evidence["next_action"]
    return {
        "evidence_ref": reference,
        "source": "text",
        "status": next_action["status"],
        "fields_present": sorted(k for k, v in evidence.get("partial_fields", {}).items() if v is not None),
        "missing_fields": next_action.get("missing_fields", []),
        "urgency_flag": next_action.get("urgency_flag", False),
    }


@tool(description="Extract applicant evidence from a document file or raw document text using an LLM.", name="extract_from_document")
def extract_from_document(document_path_or_text: str) -> dict[str, Any]:
    _claim_tool_call("extract_from_document")
    document_text = _read_document_text(document_path_or_text)
    normalised = (document_text or "").strip()
    if not normalised:
        raise ValueError("Document text is empty; no applicant evidence could be extracted.")

    llm_response = _call_llm(normalised)
    evidence = _build_extraction_output(llm_response, "document", original_text=normalised)
    reference = _store_evidence(evidence)
    next_action = evidence["next_action"]
    return {
        "evidence_ref": reference,
        "source": "document",
        "status": next_action["status"],
        "fields_present": sorted(k for k, v in evidence.get("partial_fields", {}).items() if v is not None),
        "missing_fields": next_action.get("missing_fields", []),
        "urgency_flag": next_action.get("urgency_flag", False),
    }

@tool(description="Compare text and document evidence to find direct contradictions or mismatches.", name="check_contradiction")
def check_contradiction(
    text_evidence: dict[str, Any],
    document_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _claim_tool_call("check_contradiction")
    """Flag fields present in both sources that disagree beyond tolerance.

    Tolerances:
    - annual_income: 5% relative difference allowed before flagging
    - household_size, disconnection_status, home_type, heating_fuel_type,
      applicant_name: exact match required
    """

    def _payload(evidence: Any) -> dict[str, Any]:
        if not isinstance(evidence, dict):
            return {}
        if isinstance(evidence.get("evidence_ref"), str):
            evidence = _current_tool_state()["evidence"][evidence["evidence_ref"]]
        extracted = evidence.get("extracted")
        if isinstance(extracted, dict):
            return extracted
        # Incomplete case: "extracted" is None, but partial_fields still
        # holds whatever was successfully pulled from this source.
        partial = evidence.get("partial_fields")
        if isinstance(partial, dict):
            return partial
        return evidence

    if document_evidence is None:
        reference = f"contradictions_{uuid4().hex[:10]}"
        _current_tool_state()["contradictions"][reference] = []
        return {"contradictions_ref": reference, "count": 0, "fields": []}

    text_payload = _payload(text_evidence)
    doc_payload = _payload(document_evidence)

    mismatches: list[dict[str, Any]] = []

    for field in ExtractedApplicantFields.model_fields:
        text_value = text_payload.get(field)
        doc_value = doc_payload.get(field)

        if text_value is None or doc_value is None:
            continue

        if field == "applicant_name":
            text_str = str(text_value).strip().lower()
            doc_str = str(doc_value).strip().lower()
            if text_str in doc_str or doc_str in text_str:
                continue
            mismatches.append(
                {
                    "field": field,
                    "text_value": text_value,
                    "document_value": doc_value,
                    "status": "mismatch",
                }
            )
            continue

        if field == "annual_income":
            try:
                text_num = float(text_value)
                doc_num = float(doc_value)
            except (TypeError, ValueError):
                continue
            if text_num == 0 and doc_num == 0:
                continue
            relative_diff = abs(text_num - doc_num) / max(abs(text_num), abs(doc_num), 1e-9)
            if relative_diff > 0.05:
                mismatches.append(
                    {
                        "field": field,
                        "text_value": text_value,
                        "document_value": doc_value,
                        "relative_diff": relative_diff,
                        "status": "mismatch",
                    }
                )
            continue

        if text_value != doc_value:
            mismatches.append(
                {
                    "field": field,
                    "text_value": text_value,
                    "document_value": doc_value,
                    "status": "mismatch",
                }
            )

    reference = f"contradictions_{uuid4().hex[:10]}"
    _current_tool_state()["contradictions"][reference] = mismatches
    return {
        "contradictions_ref": reference,
        "count": len(mismatches),
        "fields": [item["field"] for item in mismatches],
    }


@tool(description="Create precise follow-up questions for required evidence that is missing or low confidence.", name="request_missing_information")
def request_missing_information(missing_fields: list[str]) -> dict[str, Any]:
    _claim_tool_call("request_missing_information")
    questions = {
        "household_size": "How many people live in your household?",
        "annual_income": "What is your household's annual income?",
        "home_type": "Do you live in a single-family home or a multi-family/apartment building?",
        "heating_fuel_type": "What type of heat do you use: gas, oil, electric, or heat included in rent?",
        "disconnection_status": "Has your utility service been disconnected, or do you have a disconnection notice?",
    }
    return {
        "workflow": "needs_followup",
        "missing_fields": missing_fields,
        "followup_questions": [questions[field] for field in missing_fields if field in questions],
    }


@tool(description="Route a case to a human caseworker when evidence sources conflict.", name="create_caseworker_review")
def create_caseworker_review(contradictions: list[dict[str, Any]] | dict[str, Any]) -> dict[str, Any]:
    _claim_tool_call("create_caseworker_review")
    if isinstance(contradictions, dict):
        reference = contradictions.get("contradictions_ref")
        contradictions = _current_tool_state()["contradictions"].get(reference, [])
    review_ref = f"contradictions_{uuid4().hex[:10]}"
    _current_tool_state()["contradictions"][review_ref] = contradictions
    return {
        "workflow": "needs_review",
        "reason": "conflicting evidence requires human resolution",
        "contradictions_ref": review_ref,
        "count": len(contradictions),
    }
