from __future__ import annotations

import json
import os
from typing import Any
from dataclasses import asdict

from strands import Agent
from strands.models.openai import OpenAIModel
from nba.next_best_action import evaluate_next_action

from agent.tools import (
    begin_tool_state,
    check_contradiction,
    create_caseworker_review,
    extract_from_document,
    extract_from_text,
    get_tool_state,
    request_missing_information,
)
from schema.models import (
    EvidenceField,
    ExtractionAudit,
    ExtractionOutput,
    ExtractedApplicantFields,
)


def _normalise_extraction_output(candidate: Any) -> ExtractionOutput:
    if isinstance(candidate, ExtractionOutput):
        return candidate
    if isinstance(candidate, dict):
        if "extracted" in candidate and "audit" in candidate:
            return ExtractionOutput.model_validate(candidate)
        if set(candidate).issubset(ExtractedApplicantFields.model_fields):
            extracted = ExtractedApplicantFields.model_validate(candidate)
            audit_payload: dict[str, EvidenceField] = {}
            for field_name, value in candidate.items():
                if value is None:
                    continue
                audit_payload[field_name] = EvidenceField(value=value, confidence=0.9, source="text")
            audit = ExtractionAudit.model_validate(audit_payload)
            return ExtractionOutput(extracted=extracted, audit=audit)
    raise TypeError("Expected an ExtractionOutput instance or a dict representing extracted applicant fields.")


def _fields_and_audit(evidence: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pull (fields, audit) dicts out of either a complete or incomplete
    extraction result, whichever shape it's in.
    """
    if isinstance(evidence, ExtractionOutput):
        return evidence.extracted.model_dump(), evidence.audit.model_dump(mode="python")
    if isinstance(evidence, dict):
        extracted = evidence.get("extracted")
        audit = evidence.get("audit")
        if isinstance(extracted, dict) and isinstance(audit, dict):
            return extracted, audit
        partial_fields = evidence.get("partial_fields")
        partial_audit = evidence.get("partial_audit")
        if isinstance(partial_fields, dict):
            return partial_fields, partial_audit or {}
        if set(evidence).issubset(ExtractedApplicantFields.model_fields):
            return evidence, {}
    raise TypeError(
        "Expected an ExtractionOutput, or a dict with extracted/audit "
        "or partial_fields/partial_audit."
    )


def merge_evidence(text_evidence: Any, document_evidence: Any | None = None) -> dict[str, Any]:
    """Merge text and document evidence into one combined result.

    Returns a plain {"fields": ..., "audit": ...} dict, NOT an
    ExtractionOutput — the merge may still be incomplete (e.g. neither
    source states home_type), and forcing Pydantic validation here is
    exactly the bug that just crashed.

    Fields present and agreeing in both sources get source="both" and the
    higher of the two confidences. Fields present in only one source keep
    that source. Contradicting fields keep the higher-confidence value;
    check_contradiction() is the source of truth for flagging the
    disagreement, not this function.
    """
    text_fields, text_audit = _fields_and_audit(text_evidence)
    doc_fields, doc_audit = _fields_and_audit(document_evidence) if document_evidence is not None else ({}, {})

    merged_values: dict[str, Any] = {}
    merged_audit: dict[str, dict[str, Any]] = {}

    for field_name in ExtractedApplicantFields.model_fields:
        values: list[tuple[Any, float, str]] = []

        text_value = text_fields.get(field_name)
        if text_value is not None:
            entry = text_audit.get(field_name)
            confidence = float(entry["confidence"]) if entry else 0.9
            values.append((text_value, confidence, "text"))

        doc_value = doc_fields.get(field_name)
        if doc_value is not None:
            entry = doc_audit.get(field_name)
            confidence = float(entry["confidence"]) if entry else 0.9
            values.append((doc_value, confidence, "document"))

        if not values:
            continue

        if len(values) == 1:
            chosen_value, chosen_confidence, chosen_source = values[0]
        else:
            first_value, first_confidence, _ = values[0]
            second_value, second_confidence, _ = values[1]
            if first_value == second_value:
                chosen_value = first_value
                chosen_confidence = max(first_confidence, second_confidence)
                chosen_source = "both"
            else:
                chosen_value, chosen_confidence, chosen_source = max(values, key=lambda v: v[1])

        merged_values[field_name] = chosen_value
        merged_audit[field_name] = {
            "value": chosen_value,
            "confidence": float(chosen_confidence),
            "source": chosen_source if chosen_source in {"text", "document", "both"} else "text",
        }

    return {"fields": merged_values, "audit": merged_audit}


def _build_groq_orchestrator_model() -> OpenAIModel:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to .env before running the Strands orchestrator."
        )

    configured_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    model_id = (
        "openai/gpt-oss-120b"
        if "gpt-oss" in configured_model.lower()
        else configured_model
    )
    return OpenAIModel(
        client_args={
            "api_key": api_key,
            "base_url": "https://api.groq.com/openai/v1",
        },
        model_id=model_id,
        params={
            "temperature": 0,
            "max_tokens": 1200,
            "extra_body": {
                "reasoning_format": "hidden",
                "reasoning_effort": "low",
            },
        },
    )


def build_extraction_agent(model: Any | None = None) -> Agent:
    # Groq keeps local development independent of Bedrock access. A caller can
    # still provide a Bedrock or AgentCore model explicitly for deployment.
    selected_model = model if model is not None else _build_groq_orchestrator_model()
    agent = Agent(
        model=selected_model,
        tools=[
            extract_from_text,
            extract_from_document,
            check_contradiction,
            request_missing_information,
            create_caseworker_review,
        ],
        system_prompt=(
            "You are the Aid Navigator orchestration agent for DC LIHEAP intake. "
            "Autonomously choose the evidence tools needed for this case. "
            "Call extract_from_text for the intake, extract_from_document when a document is supplied, "
            "and check_contradiction when both sources exist. "
            "If extraction reports missing required fields, call request_missing_information. "
            "If sources conflict, call create_caseworker_review. "
            "Tool results contain compact references; never ask a tool to repeat full evidence or audit data. "
            "Call each applicable tool at most once, then return the final JSON. "
            "Return tool results without changing extracted values. "
            "Never decide eligibility or benefit amounts; those belong to the deterministic rule engine."
        ),
    )
    return agent


def _unwrap_direct_tool_result(result: Any) -> Any:
    """Recover a direct tool's Python payload from the Strands ToolResult envelope."""
    if not isinstance(result, dict) or "content" not in result:
        return result

    content = result["content"]
    if not isinstance(content, list) or not content:
        raise ValueError(f"Strands tool returned an empty result: {result!r}")

    first = content[0]
    if not isinstance(first, dict) or "text" not in first:
        raise ValueError(f"Strands tool returned an unsupported result: {result!r}")

    import json

    try:
        return json.loads(first["text"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Strands tool returned invalid JSON: {result!r}") from exc


def _parse_agent_evidence(result: Any) -> dict[str, Any]:
    """Parse and locally validate the final response from the Groq-backed agent."""
    raw = str(result).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"Strands agent did not return an evidence package: {raw!r}")
    try:
        payload = json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Strands agent returned invalid evidence JSON: {raw!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Strands agent response is not a JSON object.")
    if "text_evidence_ref" in payload:
        if not isinstance(payload["text_evidence_ref"], str):
            raise ValueError("Strands agent returned an invalid text evidence reference.")
        for key in ("document_evidence_ref", "contradictions_ref"):
            if payload.get(key) is not None and not isinstance(payload[key], str):
                raise ValueError(f"Strands agent returned an invalid {key}.")
    elif "text_evidence" not in payload or not isinstance(payload["text_evidence"], dict):
        raise ValueError("Strands agent response is missing text evidence.")
    return payload


def _orchestrate_extraction(
    text: str,
    document_path_or_text: str | None,
    model: Any | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    """Let Strands autonomously choose tools while Python owns full evidence state."""
    begin_tool_state()
    agent = build_extraction_agent(model=model)
    document_context = (
        f"A supporting document is available at this exact path/value: {document_path_or_text!r}. "
        "Use extract_from_document when it is relevant."
        if document_path_or_text
        else "No supporting document is available; do not call extract_from_document."
    )
    prompt = f"""
Process this DC LIHEAP intake as an evidence workflow. Use the registered tools autonomously.

Intake:
{text}

{document_context}

After using the tools, return ONLY one compact JSON object with this shape:
{{
  "text_evidence_ref": <evidence_ref from extract_from_text>,
  "document_evidence_ref": <evidence_ref from extract_from_document or null>,
  "contradictions_ref": <contradictions_ref from check_contradiction or null>,
  "workflow_recommendation": <compact workflow result or null>
}}

Do not calculate eligibility, benefits, or crisis qualification. Do not narrate your reasoning.
"""
    try:
        agent_result = agent(prompt, limits={"turns": 5, "output_tokens": 1200, "total_tokens": 6000})
    except TypeError as exc:
        if "limits" not in str(exc):
            raise
        agent_result = agent(prompt)
    payload = _parse_agent_evidence(agent_result)
    state = get_tool_state()
    if "text_evidence_ref" in payload:
        text_output = state["evidence"][payload["text_evidence_ref"]]
        document_ref = payload.get("document_evidence_ref")
        document_output = state["evidence"].get(document_ref) if document_ref else None
        contradiction_ref = payload.get("contradictions_ref")
        contradictions = state["contradictions"].get(contradiction_ref, []) if contradiction_ref else []
    else:
        text_output = payload["text_evidence"]
        document_output = payload.get("document_evidence")
        contradictions = payload.get("contradictions", [])
    trace = ["Strands agent selected evidence tools"]
    recommendation = payload.get("workflow_recommendation")
    if isinstance(recommendation, dict):
        trace.append(f"Strands recommended {recommendation.get('workflow', 'workflow review')}")
    return text_output, document_output, contradictions, trace



def run_extraction_pipeline(
    text: str,
    document_path_or_text: str | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    text_output, document_output, contradictions, agent_trace = _orchestrate_extraction(
        text=text,
        document_path_or_text=document_path_or_text,
        model=model,
    )

    merged = merge_evidence(text_output, document_output)
    merged_fields = merged["fields"]
    merged_audit = merged["audit"]
    merged_confidences = {name: entry["confidence"] for name, entry in merged_audit.items()}

    next_action = evaluate_next_action(merged_fields, merged_confidences)

    if next_action.status == "needs_followup":
        return {
            "status": "needs_followup",
            "next_action": asdict(next_action),
            "contradictions": contradictions,
            "agent_trace": agent_trace,
            "partial_fields": merged_fields,
            "partial_audit": merged_audit,
        }

    # Material disagreement between text and document evidence must stop
    # the pipeline before eligibility, not just get logged alongside it.
    # merge_evidence already picked a value by confidence, but a picked
    # value is not the same as a resolved disagreement — a caseworker
    # needs to see and settle this, not have it silently decided for them.
    if contradictions:
        return {
            "status": "needs_review",
            "next_action": asdict(next_action),
            "contradictions": contradictions,
            "agent_trace": agent_trace,
            "partial_fields": merged_fields,
            "partial_audit": merged_audit,
        }

    extracted = ExtractedApplicantFields.model_validate(merged_fields)
    audit = ExtractionAudit.model_validate(merged_audit)
    output = ExtractionOutput(extracted=extracted, audit=audit)

    return {
        "status": "ready_for_eligibility",
        "next_action": asdict(next_action),
        "contradictions": contradictions,
        "agent_trace": agent_trace,
        "extraction_output": output.model_dump(mode="python"),
    }
